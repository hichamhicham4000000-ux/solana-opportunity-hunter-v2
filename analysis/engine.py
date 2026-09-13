"""Main scoring orchestrator — security + opportunity analysis pipeline."""

import asyncio
import logging
import time
from typing import Optional

from analysis.criteria import CriterionResult
from analysis.criteria.contract import score_contract
from analysis.criteria.liquidity import score_liquidity
from analysis.criteria.holders import score_holders
from analysis.criteria.dev_wallet import score_dev_wallet
from analysis.criteria.volume import score_volume
from analysis.criteria.social import score_social
from analysis.criteria.metadata import score_metadata
from analysis.criteria.smart_money import score_smart_money

from analysis.scoring import calculate_composite_score
from analysis.opportunity import (
    calculate_opportunity_score,
    result_to_dict,
)

from data.dexscreener import (
    get_token_data,
    extract_market_data,
    extract_social_data,
)
from data.helius_client import get_asset, get_asset_creator
from data.solana_rpc import (
    get_token_largest_accounts,
    get_token_supply,
)
from data.cache import cache

from config.settings import settings
from config.constants import (
    CACHE_TTL_ANALYSIS,
    PUMP_FUN_AUTHORITY,
    PUMP_FUN_PROGRAM,
)

logger = logging.getLogger(__name__)

# Per-criterion timeout (seconds).
# Slow criteria don't kill fast ones.
CRITERION_TIMEOUT = 10

# Cache namespace version.
# Increment when the analysis result structure changes.
ANALYSIS_CACHE_VERSION = "v2"


async def _safe_criterion(
    name: str,
    coro,
) -> CriterionResult:
    """Run a single criterion with timeout and error handling.

    Returns a conservative low score on failure.
    Never returns a fake neutral 50.
    """
    try:
        return await asyncio.wait_for(
            coro,
            timeout=CRITERION_TIMEOUT,
        )

    except asyncio.TimeoutError:
        logger.warning(
            f"Criterion {name} timed out "
            f"({CRITERION_TIMEOUT}s)"
        )

        return CriterionResult(
            name=name.replace("_", " ").title(),
            score=20,
            flags=[
                "❌ Data unavailable (timed out)"
            ],
            estimated=True,
        )

    except Exception as e:
        logger.warning(
            f"Criterion {name} error: {e}"
        )

        return CriterionResult(
            name=name.replace("_", " ").title(),
            score=20,
            flags=[
                "❌ Data unavailable (error)"
            ],
            estimated=True,
        )


async def analyze_token(
    token_mint: str,
) -> dict:
    """Run a full security + opportunity analysis.

    Pipeline:

    1. Check versioned cache
    2. Parallel market/on-chain data fetch
    3. Resolve creator
    4. Detect Pump.fun
    5. Run all security criteria in parallel
    6. Calculate security composite score
    7. Extract market intelligence
    8. Calculate opportunity score
    9. Combine security + opportunity results
    10. Cache and return result

    No trading execution is performed here.
    """

    start_time = time.time()

    # ---------------------------------------------------------
    # Phase 0: Cache
    # ---------------------------------------------------------

    cache_key = (
        f"analysis:{ANALYSIS_CACHE_VERSION}:{token_mint}"
    )

    cached = cache.get(cache_key)

    if cached:
        cached["from_cache"] = True
        return cached

    # ---------------------------------------------------------
    # Phase 1: Parallel data fetch
    # ---------------------------------------------------------

    timeout = settings.analysis_timeout_seconds

    try:
        (
            dex_data,
            asset_data,
            top_holders,
            supply_data,
            account_info,
        ) = await asyncio.wait_for(
            asyncio.gather(
                get_token_data(token_mint),
                get_asset(token_mint),
                get_token_largest_accounts(token_mint),
                get_token_supply(token_mint),
                _safe_get_account_info(token_mint),
                return_exceptions=True,
            ),
            timeout=timeout,
        )

    except asyncio.TimeoutError:
        logger.warning(
            f"Data fetch timeout for {token_mint}"
        )

        (
            dex_data,
            asset_data,
            top_holders,
            supply_data,
            account_info,
        ) = (
            None,
            None,
            None,
            None,
            None,
        )

    # ---------------------------------------------------------
    # Handle individual fetch exceptions
    # ---------------------------------------------------------

    fetched_data = [
        ("DexScreener", dex_data),
        ("Helius", asset_data),
        ("Top holders", top_holders),
        ("Supply", supply_data),
        ("Account info", account_info),
    ]

    for label, value in fetched_data:
        if isinstance(value, Exception):
            logger.warning(
                f"{label} error: {value}"
            )

    if isinstance(dex_data, Exception):
        dex_data = None

    if isinstance(asset_data, Exception):
        asset_data = None

    if isinstance(top_holders, Exception):
        top_holders = None

    if isinstance(supply_data, Exception):
        supply_data = None

    if isinstance(account_info, Exception):
        account_info = None

    # ---------------------------------------------------------
    # Total supply
    # ---------------------------------------------------------

    total_supply = 0.0

    if (
        supply_data
        and isinstance(supply_data, dict)
    ):
        try:
            total_supply = float(
                supply_data.get(
                    "uiAmount",
                    0,
                )
                or 0
            )
        except (
            TypeError,
            ValueError,
        ):
            total_supply = 0.0

    # ---------------------------------------------------------
    # Creator
    # ---------------------------------------------------------

    creator = None

    try:
        creator = await get_asset_creator(
            token_mint
        )

    except Exception as e:
        logger.warning(
            f"Creator fetch error: {e}"
        )

    # ---------------------------------------------------------
    # Pump.fun detection
    # ---------------------------------------------------------

    is_pumpfun = _detect_pumpfun(
        dex_data,
        asset_data,
    )

    # ---------------------------------------------------------
    # Token metadata + social
    # ---------------------------------------------------------

    token_metadata = _extract_social_metadata(
        asset_data
    )

    dex_socials = extract_social_data(
        dex_data
    )

    for key in (
        "twitter",
        "telegram",
        "website",
    ):
        if (
            not token_metadata.get(key)
            and dex_socials.get(key)
        ):
            token_metadata[key] = (
                dex_socials[key]
            )

    token_name = token_metadata.get(
        "name",
        "Unknown",
    )

    token_symbol = token_metadata.get(
        "symbol",
        "???",
    )

    # ---------------------------------------------------------
    # Phase 2: Security criteria
    # ---------------------------------------------------------

    criteria_names = [
        "contract",
        "liquidity",
        "holders",
        "dev_wallet",
        "volume",
        "social",
        "metadata",
        "smart_money",
    ]

    results = await asyncio.gather(
        _safe_criterion(
            "contract",
            score_contract(
                token_mint,
                account_info=account_info,
                asset_data=asset_data,
            ),
        ),

        _safe_criterion(
            "liquidity",
            score_liquidity(
                dex_data,
                is_pumpfun=is_pumpfun,
            ),
        ),

        _safe_criterion(
            "holders",
            score_holders(
                token_mint,
                dex_data,
                top_holders=top_holders,
                supply_data=supply_data,
                is_pumpfun=is_pumpfun,
            ),
        ),

        _safe_criterion(
            "dev_wallet",
            score_dev_wallet(
                creator,
                dex_data=dex_data,
            ),
        ),

        _safe_criterion(
            "volume",
            score_volume(
                dex_data
            ),
        ),

        _safe_criterion(
            "social",
            score_social(
                token_metadata
            ),
        ),

        _safe_criterion(
            "metadata",
            score_metadata(
                asset_data
            ),
        ),

        _safe_criterion(
            "smart_money",
            score_smart_money(
                token_mint,
                top_holders,
                total_supply,
            ),
        ),
    )

    criteria_results = dict(
        zip(
            criteria_names,
            results,
        )
    )

    # ---------------------------------------------------------
    # Phase 3: Security composite score
    # ---------------------------------------------------------

    security_analysis = (
        calculate_composite_score(
            criteria_results
        )
    )

    # ---------------------------------------------------------
    # Phase 4: Market intelligence
    # ---------------------------------------------------------

    market = extract_market_data(
        dex_data
    )

    # ---------------------------------------------------------
    # Phase 5: Opportunity score
    # ---------------------------------------------------------

    opportunity_result = (
        calculate_opportunity_score(
            market
        )
    )

    opportunity_data = result_to_dict(
        opportunity_result
    )

    # ---------------------------------------------------------
    # Phase 6: Final combined result
    # ---------------------------------------------------------

    elapsed = round(
        time.time() - start_time,
        1,
    )

    analysis = dict(
        security_analysis
    )

    analysis.update(
        {
            # -----------------------------
            # Token identity
            # -----------------------------
            "token_mint": token_mint,
            "token_name": token_name,
            "token_symbol": token_symbol,

            # -----------------------------
            # Security
            # -----------------------------
            "security_score": security_analysis.get(
                "total_score"
            ),
            "security_risk_label": security_analysis.get(
                "risk_label"
            ),
            "security_risk_emoji": security_analysis.get(
                "risk_emoji"
            ),
            "security_risk_desc": security_analysis.get(
                "risk_desc"
            ),

            # -----------------------------
            # Market
            # -----------------------------
            "price_usd": market.get(
                "price_usd"
            ),
            "market_cap": market.get(
                "market_cap"
            ),
            "liquidity_usd": market.get(
                "liquidity_usd"
            ),
            "volume_24h": market.get(
                "volume_24h"
            ),
            "volume_6h": market.get(
                "volume_6h"
            ),
            "volume_1h": market.get(
                "volume_1h"
            ),

            # -----------------------------
            # Price momentum
            # -----------------------------
            "price_change_5m": market.get(
                "price_change_5m"
            ),
            "price_change_1h": market.get(
                "price_change_1h"
            ),
            "price_change_6h": market.get(
                "price_change_6h"
            ),
            "price_change_24h": market.get(
                "price_change_24h"
            ),

            # -----------------------------
            # Market activity
            # -----------------------------
            "txns_24h_buys": market.get(
                "txns_24h_buys"
            ),
            "txns_24h_sells": market.get(
                "txns_24h_sells"
            ),
            "maker_count": market.get(
                "maker_count"
            ),

            # -----------------------------
            # Pair
            # -----------------------------
            "pair_address": market.get(
                "pair_address"
            ),
            "dex_id": market.get(
                "dex_id"
            ),
            "pair_url": market.get(
                "pair_url"
            ),
            "pair_created_at": market.get(
                "pair_created_at"
            ),

            # -----------------------------
            # Creator / chain
            # -----------------------------
            "creator_address": creator,
            "is_pumpfun": is_pumpfun,

            # -----------------------------
            # Opportunity
            # -----------------------------
            "opportunity_score": opportunity_data.get(
                "opportunity_score"
            ),
            "opportunity_label": opportunity_data.get(
                "opportunity_label"
            ),
            "opportunity_data_quality": opportunity_data.get(
                "data_quality"
            ),
            "opportunity_reasons": opportunity_data.get(
                "reasons",
                [],
            ),
            "opportunity_signals": opportunity_data.get(
                "signals",
                [],
            ),

            # -----------------------------
            # Performance
            # -----------------------------
            "analysis_time": elapsed,
            "from_cache": False,
            "analysis_version": ANALYSIS_CACHE_VERSION,
        }
    )

    # ---------------------------------------------------------
    # Cache
    # ---------------------------------------------------------

    cache.set(
        cache_key,
        analysis,
        ttl=CACHE_TTL_ANALYSIS,
    )

    logger.info(
        f"Analyzed {token_symbol} "
        f"({token_mint[:8]}...): "
        f"security={analysis.get('total_score')}, "
        f"opportunity={analysis.get('opportunity_score')}, "
        f"label={analysis.get('opportunity_label')}, "
        f"time={elapsed}s"
    )

    return analysis


def _detect_pumpfun(
    dex_data: Optional[dict],
    asset_data: Optional[dict],
) -> bool:
    """Detect if a token is a Pump.fun token."""

    # Signal 1:
    # DexScreener reports dexId as "pumpfun"
    if (
        dex_data
        and dex_data.get("primary_pair")
    ):
        if (
            dex_data["primary_pair"].get(
                "dexId"
            )
            == "pumpfun"
        ):
            return True

    # Signal 2:
    # Helius DAS authorities contain Pump.fun address
    if asset_data:
        pf_addrs = {
            PUMP_FUN_AUTHORITY,
            PUMP_FUN_PROGRAM,
        }

        for auth in asset_data.get(
            "authorities",
            [],
        ):
            if (
                auth.get("address")
                in pf_addrs
            ):
                return True

        # Signal 3:
        # Metadata URI points to Pump.fun
        json_uri = (
            asset_data
            .get("content", {})
            .get("json_uri", "")
        )

        if (
            isinstance(json_uri, str)
            and "pump.fun" in json_uri
        ):
            return True

    return False


async def _safe_get_account_info(
    token_mint: str,
):
    """Fetch raw account info for contract scoring."""
    from data.solana_rpc import (
        get_account_info
    )

    return await get_account_info(
        token_mint
    )


async def quick_score(
    token_mint: str,
) -> dict:
    """Get a quick security + opportunity score.

    Quick analysis uses:
    - Contract
    - Liquidity
    - Holders
    - Market opportunity signals

    It is intended for fast candidate filtering,
    not final investment evaluation.
    """

    cache_key = (
        f"quick:{ANALYSIS_CACHE_VERSION}:{token_mint}"
    )

    cached = cache.get(
        cache_key
    )

    if cached:
        cached["from_cache"] = True
        return cached

    start_time = time.time()

    # ---------------------------------------------------------
    # Essential data
    # ---------------------------------------------------------

    (
        dex_data,
        top_holders,
        supply_data,
    ) = await asyncio.gather(
        get_token_data(token_mint),
        get_token_largest_accounts(
            token_mint
        ),
        get_token_supply(
            token_mint
        ),
        return_exceptions=True,
    )

    if isinstance(
        dex_data,
        Exception,
    ):
        dex_data = None

    if isinstance(
        top_holders,
        Exception,
    ):
        top_holders = None

    if isinstance(
        supply_data,
        Exception,
    ):
        supply_data = None

    # ---------------------------------------------------------
    # Detect Pump.fun
    # ---------------------------------------------------------

    try:
        asset = await get_asset(
            token_mint
        )
    except Exception:
        asset = None

    is_pumpfun = _detect_pumpfun(
        dex_data,
        asset,
    )

    # ---------------------------------------------------------
    # Critical security criteria
    # ---------------------------------------------------------

    (
        contract_result,
        liquidity_result,
        holders_result,
    ) = await asyncio.gather(
        _safe_criterion(
            "contract",
            score_contract(
                token_mint
            ),
        ),

        _safe_criterion(
            "liquidity",
            score_liquidity(
                dex_data,
                is_pumpfun=is_pumpfun,
            ),
        ),

        _safe_criterion(
            "holders",
            score_holders(
                token_mint,
                dex_data,
                top_holders=top_holders,
                supply_data=supply_data,
                is_pumpfun=is_pumpfun,
            ),
        ),
    )

    # ---------------------------------------------------------
    # Quick security score
    # ---------------------------------------------------------

    scores = [
        contract_result.score,
        liquidity_result.score,
        holders_result.score,
    ]

    quick_security = (
        round(
            sum(scores) / len(scores),
            1,
        )
        if scores
        else 25.0
    )

    # ---------------------------------------------------------
    # Market intelligence
    # ---------------------------------------------------------

    market = extract_market_data(
        dex_data
    )

    # ---------------------------------------------------------
    # Quick opportunity score
    # ---------------------------------------------------------

    opportunity_result = (
        calculate_opportunity_score(
            market
        )
    )

    opportunity_data = result_to_dict(
        opportunity_result
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    token_metadata = (
        _extract_social_metadata(
            asset
        )
    )

    dex_socials = extract_social_data(
        dex_data
    )

    for key in (
        "twitter",
        "telegram",
        "website",
    ):
        if (
            not token_metadata.get(key)
            and dex_socials.get(key)
        ):
            token_metadata[key] = (
                dex_socials[key]
            )

    # ---------------------------------------------------------
    # Risk label
    # ---------------------------------------------------------

    from analysis.scoring import (
        get_risk_label
    )

    risk = get_risk_label(
        quick_security
    )

    elapsed = round(
        time.time() - start_time,
        1,
    )

    result = {
        # Security
        "total_score": quick_security,
        "security_score": quick_security,
        "risk_label": risk["label"],
        "risk_emoji": risk["emoji"],
        "risk_desc": risk["desc"],

        # Token
        "token_mint": token_mint,
        "token_name": token_metadata.get(
            "name",
            "Unknown",
        ),
        "token_symbol": token_metadata.get(
            "symbol",
            "???",
        ),

        # Market
        "price_usd": market.get(
            "price_usd"
        ),
        "market_cap": market.get(
            "market_cap"
        ),
        "liquidity_usd": market.get(
            "liquidity_usd"
        ),
        "volume_24h": market.get(
            "volume_24h"
        ),

        # Momentum
        "price_change_5m": market.get(
            "price_change_5m"
        ),
        "price_change_1h": market.get(
            "price_change_1h"
        ),
        "price_change_6h": market.get(
            "price_change_6h"
        ),
        "price_change_24h": market.get(
            "price_change_24h"
        ),

        # Activity
        "txns_24h_buys": market.get(
            "txns_24h_buys"
        ),
        "txns_24h_sells": market.get(
            "txns_24h_sells"
        ),
        "maker_count": market.get(
            "maker_count"
        ),

        # Pair
        "pair_address": market.get(
            "pair_address"
        ),
        "dex_id": market.get(
            "dex_id"
        ),
        "pair_url": market.get(
            "pair_url"
        ),

        # Chain
        "is_pumpfun": is_pumpfun,

        # Opportunity
        "opportunity_score": opportunity_data.get(
            "opportunity_score"
        ),
        "opportunity_label": opportunity_data.get(
            "opportunity_label"
        ),
        "opportunity_data_quality": opportunity_data.get(
            "data_quality"
        ),
        "opportunity_reasons": opportunity_data.get(
            "reasons",
            [],
        ),
        "opportunity_signals": opportunity_data.get(
            "signals",
            [],
        ),

        # Performance
        "analysis_time": elapsed,
        "is_quick": True,
        "from_cache": False,
        "analysis_version": ANALYSIS_CACHE_VERSION,
    }

    # ---------------------------------------------------------
    # Cache
    # ---------------------------------------------------------

    cache.set(
        cache_key,
        result,
        ttl=CACHE_TTL_ANALYSIS,
    )

    return result


def _extract_social_metadata(
    asset_data: Optional[dict],
) -> dict:
    """Extract social links and token info from Helius asset data."""

    if not asset_data:
        return {}

    content = asset_data.get(
        "content",
        {}
    )

    metadata = content.get(
        "metadata",
        {}
    )

    links = content.get(
        "links",
        {}
    )

    result = {
        "name": (
            metadata.get(
                "name",
                "",
            )
            or asset_data.get(
                "name",
                "",
            )
        ),

        "symbol": (
            metadata.get(
                "symbol",
                "",
            )
            or asset_data.get(
                "symbol",
                "",
            )
        ),

        "description": metadata.get(
            "description",
            "",
        ),
    }

    # ---------------------------------------------------------
    # Metadata attributes
    # ---------------------------------------------------------

    attributes = metadata.get(
        "attributes",
        [],
    )

    if isinstance(
        attributes,
        list,
    ):
        for attr in attributes:
            if not isinstance(
                attr,
                dict,
            ):
                continue

            trait = str(
                attr.get(
                    "trait_type",
                    "",
                )
            ).lower()

            value = attr.get(
                "value",
                "",
            )

            if (
                "twitter" in trait
                or "x.com" in trait
            ):
                result["twitter"] = value

            elif "telegram" in trait:
                result["telegram"] = value

            elif (
                "website" in trait
                or "url" in trait
            ):
                result["website"] = value

    # ---------------------------------------------------------
    # Helius links object
    # ---------------------------------------------------------

    if isinstance(
        links,
        dict,
    ):
        if not result.get(
            "twitter"
        ):
            result["twitter"] = (
                links.get("twitter")
                or links.get("x")
            )

        if not result.get(
            "telegram"
        ):
            result["telegram"] = (
                links.get("telegram")
            )

        if not result.get(
            "website"
        ):
            result["website"] = (
                links.get("website")
                or links.get(
                    "external_url"
                )
            )

    # ---------------------------------------------------------
    # Top-level metadata
    # ---------------------------------------------------------

    if not result.get(
        "website"
    ):
        result["website"] = (
            metadata.get(
                "external_url"
            )
        )

    return result
