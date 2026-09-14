"""
Smart Money Intelligence for Solana Opportunity Hunter.

This module builds intelligence from:
- known smart-money wallets;
- token largest-holder data;
- token-account owner resolution;
- recent wallet transaction activity.

Important:
- It does NOT execute trades.
- It does NOT claim that a wallet is profitable without evidence.
- It does NOT fabricate BUY/SELL activity.
- General wallet activity is not treated as token-specific BUY/SELL evidence.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from smart_money.wallet_list import get_smart_money_set
from data.solana_rpc import (
    get_signatures_for_address,
    get_token_account_owner,
    get_token_largest_accounts,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

DEFAULT_TOP_HOLDERS = 20
DEFAULT_SIGNATURE_LIMIT = 20
MAX_WALLETS_TO_ANALYZE = 10

STRONG_SMART_MONEY = 75.0
POSITIVE_SMART_MONEY = 60.0
NEUTRAL_SMART_MONEY = 45.0


# -------------------------------------------------------------------
# Data models
# -------------------------------------------------------------------

@dataclass
class WalletActivity:
    """Recent network activity observed for a wallet."""

    address: str
    signature_count: int = 0
    recent_activity: bool = False
    oldest_signature_age_seconds: Optional[float] = None
    newest_signature_age_seconds: Optional[float] = None
    error: Optional[str] = None


@dataclass
class SmartMoneyWalletSignal:
    """Smart-money signal for one actual wallet."""

    address: str
    amount: float = 0.0
    known: bool = False
    label: Optional[str] = None
    source: Optional[str] = None
    activity: Optional[WalletActivity] = None
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SmartMoneyAssessment:
    """
    Aggregated Smart Money intelligence for a token.

    confidence means confidence in DATA QUALITY,
    not probability of profit.
    """

    mint: str

    score: float = 0.0

    confidence: float = 0.0

    state: str = "UNKNOWN"

    known_wallet_count: int = 0

    active_known_wallet_count: int = 0

    total_known_holder_amount: float = 0.0

    holder_coverage: float = 0.0

    wallets: list[SmartMoneyWalletSignal] = field(
        default_factory=list
    )

    reasons: list[str] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)

    metrics: dict[str, Any] = field(default_factory=dict)

    generated_at: float = field(default_factory=time.time)


# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------

def _safe_float(value: Any) -> float:
    """Safely convert a value to float."""

    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    """Safely convert a value to int."""

    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _signature_timestamp(
    signature: dict[str, Any],
) -> Optional[float]:
    """Extract a transaction timestamp when available."""

    value = signature.get("blockTime")

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _activity_score(
    activity: Optional[WalletActivity],
) -> float:
    """
    Convert observed wallet activity into a bounded score.

    This measures activity intensity only.
    It does NOT mean buying or selling.
    """

    if activity is None:
        return 0.0

    count = activity.signature_count

    if count >= 20:
        return 100.0

    if count >= 10:
        return 80.0

    if count >= 5:
        return 60.0

    if count >= 2:
        return 40.0

    if count == 1:
        return 20.0

    return 0.0


def _state_from_score(
    score: float,
    known_wallet_count: int,
) -> str:
    """Convert the Smart Money score into a state."""

    if known_wallet_count == 0:
        return "NONE"

    if score >= STRONG_SMART_MONEY:
        return "STRONG"

    if score >= POSITIVE_SMART_MONEY:
        return "POSITIVE"

    if score >= NEUTRAL_SMART_MONEY:
        return "NEUTRAL"

    return "WEAK"


# -------------------------------------------------------------------
# Wallet activity
# -------------------------------------------------------------------

async def inspect_wallet_activity(
    wallet_address: str,
    limit: int = DEFAULT_SIGNATURE_LIMIT,
) -> WalletActivity:
    """
    Inspect recent network activity for a wallet.

    This intentionally measures wallet activity only.

    It does NOT classify transactions as BUY or SELL because
    the available RPC helper does not provide reliable,
    token-specific trade classification by itself.
    """

    activity = WalletActivity(
        address=wallet_address
    )

    try:
        signatures = await get_signatures_for_address(
            wallet_address,
            limit=max(1, min(limit, 100)),
        )

        if not signatures:
            return activity

        activity.signature_count = len(signatures)
        activity.recent_activity = True

        timestamps: list[float] = []

        now = time.time()

        for item in signatures:
            if not isinstance(item, dict):
                continue

            timestamp = _signature_timestamp(item)

            if timestamp is None:
                continue

            age = max(0.0, now - timestamp)

            timestamps.append(age)

        if timestamps:
            activity.newest_signature_age_seconds = min(
                timestamps
            )

            activity.oldest_signature_age_seconds = max(
                timestamps
            )

        return activity

    except Exception as exc:
        activity.error = str(exc)

        logger.warning(
            "Failed to inspect wallet activity %s: %s",
            wallet_address[:12],
            exc,
        )

        return activity


# -------------------------------------------------------------------
# Holder owner resolution
# -------------------------------------------------------------------

async def resolve_holder_owner(
    holder: dict[str, Any],
) -> Optional[str]:
    """
    Resolve the actual wallet owner of a token account.

    getTokenLargestAccounts returns token-account addresses,
    not necessarily the wallet addresses controlling them.
    """

    address = holder.get("address")

    if not isinstance(address, str) or not address:
        return None

    # Prefer on-chain owner resolution.
    try:
        owner = await get_token_account_owner(address)

        if owner:
            return owner
    except Exception as exc:
        logger.debug(
            "Owner resolution failed for %s: %s",
            address[:12],
            exc,
        )

    # Some upstream sources may already expose the owner.
    direct_owner = holder.get("owner")

    if isinstance(direct_owner, str) and direct_owner:
        return direct_owner

    return None


async def resolve_top_holder_owners(
    mint: str,
    top_holders: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """
    Resolve wallet owners for the token's largest accounts.

    Returns records containing:
    - token_account
    - owner
    - amount
    """

    if top_holders is None:
        top_holders = (
            await get_token_largest_accounts(mint)
            or []
        )

    if not top_holders:
        return []

    limited = top_holders[:DEFAULT_TOP_HOLDERS]

    async def resolve(
        holder: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        owner = await resolve_holder_owner(holder)

        if not owner:
            return None

        return {
            "token_account": holder.get("address"),
            "owner": owner,
            "amount": _safe_float(
                holder.get("uiAmount")
            ),
        }

    results = await asyncio.gather(
        *(resolve(holder) for holder in limited),
        return_exceptions=True,
    )

    resolved: list[dict[str, Any]] = []

    for result in results:
        if isinstance(result, dict):
            resolved.append(result)

    return resolved


# -------------------------------------------------------------------
# Known Smart Money matching
# -------------------------------------------------------------------

async def collect_known_smart_money(
    mint: str,
    top_holders: Optional[list[dict[str, Any]]] = None,
) -> list[SmartMoneyWalletSignal]:
    """
    Find known Smart Money wallets among the token's largest holders.

    IMPORTANT:
    getTokenLargestAccounts returns token-account addresses.

    Therefore the matching flow is:

        token account
              ↓
        actual owner
              ↓
        known Smart Money set

    This prevents incorrectly treating a token-account address
    as a wallet address.
    """

    del mint  # Reserved for future token-specific activity analysis.

    try:
        smart_wallets = get_smart_money_set()

    except Exception as exc:
        logger.warning(
            "Failed to load Smart Money wallet set: %s",
            exc,
        )

        return []

    if not smart_wallets:
        return []

    resolved_holders = await resolve_top_holder_owners(
        mint=mint,
        top_holders=top_holders,
    )

    if not resolved_holders:
        return []

    signals: list[SmartMoneyWalletSignal] = []

    seen_wallets: set[str] = set()

    for holder in resolved_holders:
        owner = holder.get("owner")

        if not isinstance(owner, str) or not owner:
            continue

        if owner in seen_wallets:
            continue

        if owner not in smart_wallets:
            continue

        seen_wallets.add(owner)

        signals.append(
            SmartMoneyWalletSignal(
                address=owner,
                amount=_safe_float(
                    holder.get("amount")
                ),
                known=True,
                reasons=[
                    "محفظة Smart Money معروفة ظهرت ضمن كبار الحائزين"
                ],
            )
        )

    return signals


# -------------------------------------------------------------------
# Activity enrichment
# -------------------------------------------------------------------

async def enrich_wallet_signals(
    signals: list[SmartMoneyWalletSignal],
) -> list[SmartMoneyWalletSignal]:
    """Add recent network activity to Smart Money signals."""

    signals = signals[:MAX_WALLETS_TO_ANALYZE]

    async def enrich(
        signal: SmartMoneyWalletSignal,
    ) -> SmartMoneyWalletSignal:

        signal.activity = await inspect_wallet_activity(
            signal.address
        )

        if signal.activity.error:
            signal.warnings.append(
                "تعذر قراءة نشاط المحفظة بالكامل"
            )

        elif signal.activity.signature_count > 0:
            signal.reasons.append(
                "المحفظة لديها نشاط حديث على الشبكة"
            )

        else:
            signal.warnings.append(
                "لم يظهر نشاط حديث في نافذة الفحص"
            )

        # Explicitly prevent unsupported trade claims.
        signal.warnings.append(
            "النشاط الشبكي لا يثبت وحده شراء أو بيع هذا التوكن"
        )

        return signal

    results = await asyncio.gather(
        *(enrich(signal) for signal in signals),
        return_exceptions=True,
    )

    enriched: list[SmartMoneyWalletSignal] = []

    for result in results:
        if isinstance(
            result,
            SmartMoneyWalletSignal,
        ):
            enriched.append(result)

    return enriched


# -------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------

def calculate_smart_money_score(
    signals: list[SmartMoneyWalletSignal],
    holder_count: int,
) -> tuple[
    float,
    float,
    list[str],
    list[str],
]:
    """
    Calculate Smart Money score and data confidence.

    Score components:
    - presence of known Smart Money;
    - amount concentration among matched holders;
    - recent network activity;
    - holder coverage.

    This is an analytical signal, not a profit probability.
    """

    reasons: list[str] = []
    warnings: list[str] = []

    if not signals:
        return (
            0.0,
            20.0 if holder_count else 10.0,
            reasons,
            [
                "لم يتم العثور على محفظة Smart Money معروفة ضمن كبار الحائزين"
            ],
        )

    known_count = len(signals)

    # ---------------------------------------------------------------
    # Presence
    # ---------------------------------------------------------------

    presence_score = min(
        100.0,
        35.0 + (known_count * 12.0),
    )

    # ---------------------------------------------------------------
    # Amount concentration
    # ---------------------------------------------------------------

    total_amount = sum(
        max(signal.amount, 0.0)
        for signal in signals
    )

    amount_values = [
        max(signal.amount, 0.0)
        for signal in signals
    ]

    total_top_holder_amount = sum(
        amount_values
    )

    if total_top_holder_amount > 0:
        concentration_ratio = (
            total_amount
            / total_top_holder_amount
        )

        concentration_score = min(
            100.0,
            40.0 + (concentration_ratio * 60.0),
        )

    else:
        concentration_score = 0.0

        warnings.append(
            "لا توجد كمية موثوقة كافية لحساب تركيز Smart Money"
        )

    # ---------------------------------------------------------------
    # Activity
    # ---------------------------------------------------------------

    activity_scores = [
        _activity_score(signal.activity)
        for signal in signals
        if signal.activity is not None
    ]

    if activity_scores:
        activity_score = (
            sum(activity_scores)
            / len(activity_scores)
        )

    else:
        activity_score = 0.0

    if activity_score >= 60:
        reasons.append(
            "يوجد نشاط حديث لدى جزء من محافظ Smart Money"
        )

    elif activity_score == 0:
        warnings.append(
            "لا توجد بيانات نشاط كافية لتأكيد الحركة الحالية"
        )

    # ---------------------------------------------------------------
    # Final score
    # ---------------------------------------------------------------

    score = (
        presence_score * 0.45
        + activity_score * 0.35
        + concentration_score * 0.20
    )

    score = max(
        0.0,
        min(100.0, score),
    )

    # ---------------------------------------------------------------
    # Holder coverage
    # ---------------------------------------------------------------

    coverage = 0.0

    if holder_count > 0:
        coverage = min(
            100.0,
            (known_count / holder_count) * 100.0,
        )

    # ---------------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------------

    confidence = (
        45.0
        + min(25.0, known_count * 5.0)
        + min(20.0, coverage)
    )

    if activity_scores:
        confidence += 10.0

    confidence = min(
        100.0,
        confidence,
    )

    # ---------------------------------------------------------------
    # Reasons
    # ---------------------------------------------------------------

    if known_count >= 3:
        reasons.append(
            "وجود عدة محافظ Smart Money معروفة"
        )

    elif known_count == 1:
        reasons.append(
            "تم العثور على محفظة Smart Money معروفة"
        )

    reasons.append(
        "المطابقة تمت باستخدام عنوان المحفظة المالكة الفعلية"
    )

    return (
        round(score, 2),
        round(confidence, 2),
        reasons,
        warnings,
    )


# -------------------------------------------------------------------
# Main assessment
# -------------------------------------------------------------------

async def assess_smart_money(
    mint: str,
    top_holders: Optional[list[dict[str, Any]]] = None,
) -> SmartMoneyAssessment:
    """
    Build Smart Money intelligence for a token.

    The assessment combines:
    1. known Smart Money presence;
    2. holder coverage;
    3. recent wallet activity;
    4. amount held by matched wallets.

    It intentionally avoids unsupported BUY/SELL claims.
    """

    if not mint:
        return SmartMoneyAssessment(
            mint="",
            score=0.0,
            confidence=0.0,
            state="INVALID",
            warnings=[
                "عنوان التوكن غير صالح"
            ],
        )

    try:
        if top_holders is None:
            top_holders = (
                await get_token_largest_accounts(mint)
                or []
            )

        holder_count = min(
            len(top_holders),
            DEFAULT_TOP_HOLDERS,
        )

        known_signals = await collect_known_smart_money(
            mint,
            top_holders,
        )

        known_signals = await enrich_wallet_signals(
            known_signals
        )

        (
            score,
            confidence,
            reasons,
            warnings,
        ) = calculate_smart_money_score(
            known_signals,
            holder_count,
        )

        active_count = sum(
            1
            for signal in known_signals
            if (
                signal.activity is not None
                and signal.activity.signature_count > 0
            )
        )

        total_amount = sum(
            max(signal.amount, 0.0)
            for signal in known_signals
        )

        holder_coverage = 0.0

        if holder_count:
            holder_coverage = min(
                100.0,
                (
                    len(known_signals)
                    / holder_count
                ) * 100.0,
            )

        state = _state_from_score(
            score,
            len(known_signals),
        )

        if state == "STRONG":
            reasons.append(
                "إشارة Smart Money قوية نسبيًا"
            )

        elif state == "POSITIVE":
            reasons.append(
                "إشارة Smart Money إيجابية"
            )

        elif state == "NEUTRAL":
            reasons.append(
                "إشارة Smart Money محايدة"
            )

        elif state == "WEAK":
            warnings.append(
                "وجود Smart Money وحده غير كافٍ لاعتبار التوكن فرصة"
            )

        return SmartMoneyAssessment(
            mint=mint,
            score=score,
            confidence=confidence,
            state=state,
            known_wallet_count=len(
                known_signals
            ),
            active_known_wallet_count=active_count,
            total_known_holder_amount=total_amount,
            holder_coverage=round(
                holder_coverage,
                2,
            ),
            wallets=known_signals,
            reasons=list(
                dict.fromkeys(reasons)
            ),
            warnings=list(
                dict.fromkeys(warnings)
            ),
            metrics={
                "top_holder_count": holder_count,
                "known_smart_money_count": len(
                    known_signals
                ),
                "active_known_wallet_count": active_count,
                "total_known_holder_amount": total_amount,
                "holder_coverage_percent": round(
                    holder_coverage,
                    2,
                ),
                "buy_sell_classification": "UNSUPPORTED",
                "owner_resolution": "ENABLED",
            },
        )

    except Exception as exc:
        logger.exception(
            "Smart Money assessment failed for %s",
            mint[:12],
        )

        return SmartMoneyAssessment(
            mint=mint,
            score=0.0,
            confidence=0.0,
            state="ERROR",
            warnings=[
                "حدث خطأ أثناء تحليل Smart Money",
                str(exc),
            ],
        )


# -------------------------------------------------------------------
# Compatibility helpers
# -------------------------------------------------------------------

async def get_smart_money_score(
    mint: str,
) -> float:
    """Return only the Smart Money score."""

    assessment = await assess_smart_money(mint)

    return assessment.score


async def get_smart_money_summary(
    mint: str,
) -> dict[str, Any]:
    """Return a serializable Smart Money summary."""

    assessment = await assess_smart_money(mint)

    return {
        "mint": assessment.mint,
        "score": assessment.score,
        "confidence": assessment.confidence,
        "state": assessment.state,
        "known_wallet_count": (
            assessment.known_wallet_count
        ),
        "active_known_wallet_count": (
            assessment.active_known_wallet_count
        ),
        "total_known_holder_amount": (
            assessment.total_known_holder_amount
        ),
        "holder_coverage": (
            assessment.holder_coverage
        ),
        "reasons": assessment.reasons,
        "warnings": assessment.warnings,
        "metrics": assessment.metrics,
    }
