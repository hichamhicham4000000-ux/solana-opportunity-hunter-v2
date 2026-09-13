"""
Opportunity Discovery Engine.

Discovers Solana meme-token candidates from:
1. Pump.fun real-time token creation events.
2. DexScreener market candidates.

This module only discovers and ranks candidates.
It does NOT execute trades.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from data.cache import cache
from data.dexscreener import get_trending_tokens
from data.pumpfun import pump_stream

logger = logging.getLogger(__name__)


DISCOVERY_CACHE_TTL = 30
DEFAULT_MAX_CANDIDATES = 50


@dataclass
class Candidate:
    """A token discovered by the opportunity hunter."""

    mint: str
    source: str

    name: str = "Unknown"
    symbol: str = "???"

    price_usd: Optional[float] = None
    liquidity_usd: Optional[float] = None
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None

    price_change_5m: Optional[float] = None
    price_change_1h: Optional[float] = None
    price_change_6h: Optional[float] = None
    price_change_24h: Optional[float] = None

    txns_buys: int = 0
    txns_sells: int = 0
    maker_count: int = 0

    pair_url: Optional[str] = None
    dex_id: Optional[str] = None

    discovery_score: float = 0.0

    reasons: list[str] = field(
        default_factory=list
    )

    discovered_at: float = field(
        default_factory=time.time
    )

    raw: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
    )


def _float(value: Any) -> Optional[float]:
    """Safely convert a value to float."""
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    """Safely convert a value to int."""
    if value is None:
        return 0

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _mint_from_pump_event(
    event: dict[str, Any],
) -> Optional[str]:
    """Extract a mint address from a Pump.fun event."""

    for key in (
        "mint",
        "tokenAddress",
        "token_address",
        "address",
    ):
        value = event.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _candidate_from_dex(
    item: dict[str, Any],
) -> Optional[Candidate]:
    """Convert a DexScreener discovery result into a Candidate."""

    mint = item.get(
        "contract_address"
    )

    if not isinstance(mint, str) or not mint:
        return None

    return Candidate(
        mint=mint,
        source="dexscreener",

        name=item.get(
            "token_name",
            "Unknown",
        ),

        symbol=item.get(
            "token_symbol",
            "???",
        ),

        price_usd=_float(
            item.get("price_usd")
        ),

        liquidity_usd=_float(
            item.get("liquidity_usd")
        ),

        market_cap=_float(
            item.get("market_cap")
        ),

        volume_24h=_float(
            item.get("volume_24h")
        ),

        price_change_5m=_float(
            item.get("price_change_5m")
        ),

        price_change_1h=_float(
            item.get("price_change_1h")
        ),

        price_change_6h=_float(
            item.get("price_change_6h")
        ),

        price_change_24h=_float(
            item.get("price_change_24h")
        ),

        pair_url=item.get(
            "pair_url"
        ),

        raw=item,
    )


def _candidate_from_pump(
    event: dict[str, Any],
) -> Optional[Candidate]:
    """Convert a Pump.fun event into a Candidate."""

    mint = _mint_from_pump_event(
        event
    )

    if not mint:
        return None

    name = (
        event.get("name")
        or event.get("tokenName")
        or "Unknown"
    )

    symbol = (
        event.get("symbol")
        or event.get("tokenSymbol")
        or "???"
    )

    candidate = Candidate(
        mint=mint,
        source="pumpfun",

        name=str(name),

        symbol=str(symbol),

        price_usd=_float(
            event.get("price")
        ),

        liquidity_usd=_float(
            event.get("liquidity")
        ),

        market_cap=_float(
            event.get("marketCap")
        ),

        volume_24h=_float(
            event.get("volume")
        ),

        txns_buys=_int(
            event.get("buys")
        ),

        txns_sells=_int(
            event.get("sells")
        ),

        raw=event,
    )

    candidate.reasons.append(
        "توكن جديد تم اكتشافه من Pump.fun"
    )

    return candidate


def _score_candidate(
    candidate: Candidate,
) -> Candidate:
    """
    Score discovery quality only.

    This is NOT the final opportunity score.
    It decides which discovered tokens deserve deeper analysis.
    """

    score = 0.0
    reasons: list[str] = []

    # ---------------------------------------------------------
    # New-token signal
    # ---------------------------------------------------------

    if candidate.source == "pumpfun":
        score += 25
        reasons.append(
            "اكتشاف مبكر لتوكن جديد"
        )

    # ---------------------------------------------------------
    # Liquidity
    # ---------------------------------------------------------

    liquidity = (
        candidate.liquidity_usd or 0
    )

    if liquidity >= 100_000:
        score += 20
        reasons.append(
            "سيولة قوية"
        )

    elif liquidity >= 50_000:
        score += 15
        reasons.append(
            "سيولة جيدة"
        )

    elif liquidity >= 20_000:
        score += 10
        reasons.append(
            "سيولة مقبولة"
        )

    elif liquidity > 5_000:
        score += 5

    # ---------------------------------------------------------
    # Short-term momentum
    # ---------------------------------------------------------

    change_5m = (
        candidate.price_change_5m
    )

    change_1h = (
        candidate.price_change_1h
    )

    if change_5m is not None:

        if 1 <= change_5m <= 10:
            score += 15
            reasons.append(
                "زخم قصير المدى إيجابي"
            )

        elif change_5m > 10:
            score += 7
            reasons.append(
                "ارتفاع سريع يحتاج إلى الحذر"
            )

    if change_1h is not None:

        if 3 <= change_1h <= 20:
            score += 15
            reasons.append(
                "زخم ساعة إيجابي"
            )

        elif change_1h > 20:
            score += 5

    # ---------------------------------------------------------
    # Volume
    # ---------------------------------------------------------

    volume = (
        candidate.volume_24h or 0
    )

    if volume >= 1_000_000:
        score += 15
        reasons.append(
            "حجم تداول مرتفع"
        )

    elif volume >= 250_000:
        score += 10
        reasons.append(
            "حجم تداول جيد"
        )

    elif volume >= 50_000:
        score += 5

    # ---------------------------------------------------------
    # Buy pressure
    # ---------------------------------------------------------

    buys = candidate.txns_buys
    sells = candidate.txns_sells

    total = buys + sells

    if total > 0:

        buy_ratio = buys / total

        if 0.55 <= buy_ratio <= 0.70:
            score += 10
            reasons.append(
                "ضغط شراء متوازن"
            )

        elif 0.70 < buy_ratio <= 0.80:
            score += 7
            reasons.append(
                "ضغط شراء قوي"
            )

        elif buy_ratio > 0.80:
            score += 2
            reasons.append(
                "ضغط شراء مفرط — يحتاج تحقق"
            )

    # ---------------------------------------------------------
    # Market activity
    # ---------------------------------------------------------

    if candidate.maker_count >= 500:
        score += 10
        reasons.append(
            "عدد مرتفع من المتداولين"
        )

    elif candidate.maker_count >= 100:
        score += 5

    candidate.discovery_score = min(
        round(score, 1),
        100.0,
    )

    candidate.reasons = reasons

    return candidate


def _merge_candidate(
    existing: Candidate,
    incoming: Candidate,
) -> Candidate:
    """Merge two discoveries of the same token."""

    if existing.source != incoming.source:
        existing.source = (
            f"{existing.source}+"
            f"{incoming.source}"
        )

    for field_name in (
        "name",
        "symbol",
        "pair_url",
        "dex_id",
    ):
        current = getattr(
            existing,
            field_name,
        )

        incoming_value = getattr(
            incoming,
            field_name,
        )

        if (
            not current
            or current in (
                "Unknown",
                "???",
            )
        ):
            if incoming_value:
                setattr(
                    existing,
                    field_name,
                    incoming_value,
                )

    numeric_fields = (
        "price_usd",
        "liquidity_usd",
        "market_cap",
        "volume_24h",
        "price_change_5m",
        "price_change_1h",
        "price_change_6h",
        "price_change_24h",
    )

    for field_name in numeric_fields:
        current = getattr(
            existing,
            field_name,
        )

        incoming_value = getattr(
            incoming,
            field_name,
        )

        if current is None and incoming_value is not None:
            setattr(
                existing,
                field_name,
                incoming_value,
            )

    if incoming.txns_buys:
        existing.txns_buys = (
            max(
                existing.txns_buys,
                incoming.txns_buys,
            )
        )

    if incoming.txns_sells:
        existing.txns_sells = (
            max(
                existing.txns_sells,
                incoming.txns_sells,
            )
        )

    if incoming.maker_count:
        existing.maker_count = (
            max(
                existing.maker_count,
                incoming.maker_count,
            )
        )

    existing.reasons = list(
        dict.fromkeys(
            existing.reasons
            + incoming.reasons
        )
    )

    existing.raw.update(
        incoming.raw
    )

    return _score_candidate(
        existing
    )


class OpportunityDiscovery:
    """
    Candidate discovery coordinator.

    It can:
    - collect DexScreener candidates;
    - receive Pump.fun real-time discoveries;
    - deduplicate candidates;
    - rank them;
    - expose a candidate queue for deeper analysis.
    """

    def __init__(
        self,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ):
        self.max_candidates = max(
            1,
            max_candidates,
        )

        self._candidates: dict[
            str,
            Candidate,
        ] = {}

        self._lock = asyncio.Lock()

        self._started = False

    async def add_candidate(
        self,
        candidate: Candidate,
    ) -> Candidate:
        """Add or update a discovered candidate."""

        candidate = _score_candidate(
            candidate
        )

        async with self._lock:

            existing = self._candidates.get(
                candidate.mint
            )

            if existing:
                candidate = _merge_candidate(
                    existing,
                    candidate,
                )

            self._candidates[
                candidate.mint
            ] = candidate

            # Keep memory bounded.
            if (
                len(self._candidates)
                > self.max_candidates * 3
            ):
                self._trim_candidates()

        return candidate

    def _trim_candidates(self):
        """Remove low-priority stale candidates."""

        ranked = sorted(
            self._candidates.values(),
            key=lambda item: (
                item.discovery_score,
                item.discovered_at,
            ),
            reverse=True,
        )

        keep = ranked[
            : self.max_candidates * 2
        ]

        self._candidates = {
            item.mint: item
            for item in keep
        }

    async def handle_new_token(
        self,
        event: dict[str, Any],
    ):
        """Handle a Pump.fun new-token event."""

        candidate = _candidate_from_pump(
            event
        )

        if not candidate:
            logger.warning(
                "Pump.fun event did not contain a mint"
            )
            return

        candidate = await self.add_candidate(
            candidate
        )

        logger.info(
            "Discovered Pump.fun candidate: "
            f"{candidate.symbol} "
            f"{candidate.mint[:8]}... "
            f"score={candidate.discovery_score}"
        )

    async def handle_migration(
        self,
        event: dict[str, Any],
    ):
        """Handle a Pump.fun migration event."""

        candidate = _candidate_from_pump(
            event
        )

        if not candidate:
            return

        candidate.reasons.append(
            "توكن وصل إلى مرحلة Migration"
        )

        candidate.discovery_score = min(
            candidate.discovery_score + 10,
            100,
        )

        await self.add_candidate(
            candidate
        )

    async def discover_dexscreener(
        self,
    ) -> list[Candidate]:
        """Discover candidates from DexScreener."""

        try:
            items = await get_trending_tokens(
                limit=self.max_candidates
            )

        except Exception as e:
            logger.error(
                f"DexScreener discovery failed: {e}"
            )
            return []

        discovered = []

        for item in items:

            candidate = _candidate_from_dex(
                item
            )

            if not candidate:
                continue

            candidate = await self.add_candidate(
                candidate
            )

            discovered.append(
                candidate
            )

        return discovered

    async def discover_once(
        self,
    ) -> list[Candidate]:
        """
        Run one market-discovery cycle.

        Returns candidates sorted by discovery score.
        """

        cache_key = (
            "opportunity:discovery:v1"
        )

        cached = cache.get(
            cache_key
        )

        if cached is not None:
            return [
                Candidate(**item)
                for item in cached
            ]

        await self.discover_dexscreener()

        async with self._lock:

            ranked = sorted(
                self._candidates.values(),
                key=lambda item: (
                    item.discovery_score,
                    item.discovered_at,
                ),
                reverse=True,
            )

            selected = ranked[
                : self.max_candidates
            ]

        serializable = [
            self._serialize_candidate(
                candidate
            )
            for candidate in selected
        ]

        cache
