"""
Market Intelligence Engine.

Analyzes the market information currently available for a candidate.

Current evidence:
- Price changes across 5m / 1h / 6h / 24h
- Liquidity
- 24h volume
- Buy/sell transaction counts
- Maker count

This module does NOT invent candle data, order-book data, or historical
ticks that are not available.

It is an analytical layer, not a trading executor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from analysis.discovery import Candidate

logger = logging.getLogger(__name__)


@dataclass
class MarketIntelligence:
    """Structured market assessment."""

    mint: str

    market_score: float
    momentum_score: float
    volume_score: float
    pressure_score: float
    liquidity_score: float
    structure_score: float
    breakout_score: float

    market_state: str
    momentum_state: str
    volume_state: str
    pressure_state: str
    breakout_state: str

    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    metrics: dict[str, float] = field(default_factory=dict)


def _float(value: Optional[float]) -> float:
    if value is None:
        return 0.0

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:
    return max(minimum, min(value, maximum))


def _score_liquidity(
    liquidity_usd: Optional[float],
) -> float:
    liquidity = _float(liquidity_usd)

    if liquidity <= 0:
        return 0.0

    if liquidity < 5_000:
        return 10.0

    if liquidity < 20_000:
        return 35.0

    if liquidity < 50_000:
        return 60.0

    if liquidity < 100_000:
        return 80.0

    if liquidity < 500_000:
        return 90.0

    return 100.0


def _score_volume(
    volume_24h: Optional[float],
) -> float:
    volume = _float(volume_24h)

    if volume <= 0:
        return 0.0

    if volume < 25_000:
        return 15.0

    if volume < 50_000:
        return 30.0

    if volume < 250_000:
        return 55.0

    if volume < 1_000_000:
        return 80.0

    if volume < 5_000_000:
        return 90.0

    return 100.0


def _score_momentum(
    candidate: Candidate,
) -> tuple[float, str]:
    """
    Score momentum using multiple available time windows.

    Moderate, aligned momentum is preferred over extreme vertical moves.
    """

    change_5m = _float(candidate.price_change_5m)
    change_1h = _float(candidate.price_change_1h)
    change_6h = _float(candidate.price_change_6h)
    change_24h = _float(candidate.price_change_24h)

    score = 50.0

    # Short-term momentum.
    if 0 < change_5m <= 5:
        score += 10
    elif 5 < change_5m <= 12:
        score += 8
    elif 12 < change_5m <= 25:
        score += 3
    elif change_5m > 25:
        score -= 8
    elif change_5m < -15:
        score -= 12

    # One-hour momentum.
    if 2 <= change_1h <= 15:
        score += 15
    elif 15 < change_1h <= 30:
        score += 10
    elif 30 < change_1h <= 60:
        score += 3
    elif change_1h > 60:
        score -= 8
    elif change_1h < -20:
        score -= 15

    # Six-hour trend.
    if 0 < change_6h <= 30:
        score += 10
    elif 30 < change_6h <= 80:
        score += 6
    elif change_6h > 80:
        score -= 4
    elif change_6h < -30:
        score -= 12

    # 24-hour context.
    if 0 < change_24h <= 100:
        score += 8
    elif change_24h > 100:
        score -= 5
    elif change_24h < -40:
        score -= 10

    score = _clamp(score)

    if score >= 75:
        state = "STRONG"
    elif score >= 60:
        state = "POSITIVE"
    elif score >= 45:
        state = "NEUTRAL"
    elif score >= 30:
        state = "WEAK"
    else:
        state = "BEARISH"

    return round(score, 1), state


def _score_pressure(
    candidate: Candidate,
) -> tuple[float, str, float]:
    """
    Evaluate buy/sell pressure from transaction counts.

    This is transaction pressure, not proof of organic buying.
    """

    buys = max(0, int(candidate.txns_buys or 0))
    sells = max(0, int(candidate.txns_sells or 0))

    total = buys + sells

    if total <= 0:
        return 0.0, "UNKNOWN", 0.0

    buy_ratio = buys / total

    if 0.55 <= buy_ratio <= 0.70:
        score = 90.0
        state = "HEALTHY_BUY_PRESSURE"

    elif 0.70 < buy_ratio <= 0.80:
        score = 80.0
        state = "STRONG_BUY_PRESSURE"

    elif 0.45 <= buy_ratio < 0.55:
        score = 55.0
        state = "BALANCED"

    elif 0.35 <= buy_ratio < 0.45:
        score = 35.0
        state = "SELL_PRESSURE"

    elif buy_ratio > 0.80:
        score = 25.0
        state = "EXTREME_BUY_IMBALANCE"

    else:
        score = 15.0
        state = "HEAVY_SELL_PRESSURE"

    return round(score, 1), state, buy_ratio


def _score_structure(
    candidate: Candidate,
) -> float:
    """
    Estimate trend structure from available multi-timeframe changes.

    This is deliberately conservative because true market structure
    requires historical candles/ticks, which are not part of Candidate.
    """

    changes = [
        _float(candidate.price_change_5m),
        _float(candidate.price_change_1h),
        _float(candidate.price_change_6h),
        _float(candidate.price_change_24h),
    ]

    available = [
        value
        for value in changes
        if value != 0
    ]

    if not available:
        return 25.0

    positive = sum(
        1
        for value in available
        if value > 0
    )

    negative = sum(
        1
        for value in available
        if value < 0
    )

    if positive == len(available):
        return 90.0

    if positive > negative:
        return 70.0

    if positive == negative:
        return 50.0

    return 25.0


def _score_breakout(
    candidate: Candidate,
) -> tuple[float, str]:
    """
    Detect an early breakout-like condition from available percentage
    changes.

    This is NOT a confirmed technical breakout because no candle
    resistance/support history is currently available.
    """

    change_5m = _float(candidate.price_change_5m)
    change_1h = _float(candidate.price_change_1h)
    change_6h = _float(candidate.price_change_6h)

    score = 0.0

    if 3 <= change_5m <= 12:
        score += 30

    elif 12 < change_5m <= 25:
        score += 15

    elif change_5m > 25:
        score += 5

    if 5 <= change_1h <= 30:
        score += 35

    elif 30 < change_1h <= 60:
        score += 15

    elif change_1h > 60:
        score += 5

    if 0 <= change_6h <= 50:
        score += 20

    elif 50 < change_6h <= 100:
        score += 10

    if (
        change_5m > 0
        and change_1h > 0
        and change_6h > 0
    ):
        score += 15

    score = _clamp(score)

    if score >= 75:
        state = "POTENTIAL_BREAKOUT"
    elif score >= 50:
        state = "BREAKOUT_WATCH"
    elif score >= 25:
        state = "WEAK_BREAKOUT_SIGNAL"
    else:
        state = "NO_BREAKOUT_SIGNAL"

    return round(score, 1), state


def _market_state(
    market_score: float,
    momentum_score: float,
    liquidity_score: float,
) -> str:
    if (
        market_score >= 80
        and momentum_score >= 70
        and liquidity_score >= 70
    ):
        return "STRONG"

    if market_score >= 65:
        return "POSITIVE"

    if market_score >= 50:
        return "NEUTRAL"

    if market_score >= 35:
        return "WEAK"

    return "RISKY"


class MarketIntelligenceEngine:
    """
    Market intelligence calculator.

    No external API calls are performed here. The engine works from
    the Candidate data already collected by the discovery layer.
    """

    def analyze(
        self,
        candidate: Candidate,
    ) -> MarketIntelligence:

        liquidity_score = _score_liquidity(
            candidate.liquidity_usd
        )

        volume_score = _score_volume(
            candidate.volume_24h
        )

        momentum_score, momentum_state = _score_momentum(
            candidate
        )

        pressure_score, pressure_state, buy_ratio = (
            _score_pressure(candidate)
        )

        structure_score = _score_structure(
            candidate
        )

        breakout_score, breakout_state = _score_breakout(
            candidate
        )

        # Market score combines the available market evidence.
        market_score = (
            liquidity_score * 0.20
            + volume_score * 0.20
            + momentum_score * 0.25
            + pressure_score * 0.15
            + structure_score * 0.10
            + breakout_score * 0.10
        )

        market_score = round(
            _clamp(market_score),
            1,
        )

        reasons: list[str] = []
        warnings: list[str] = []

        # ---------------------------------------------------------
        # Positive signals
        # ---------------------------------------------------------

        if liquidity_score >= 80:
            reasons.append(
                "السيولة السوقية قوية نسبيًا"
            )

        if volume_score >= 80:
            reasons.append(
                "حجم التداول مرتفع"
            )

        if momentum_score >= 70:
            reasons.append(
                "الزخم إيجابي عبر أكثر من إطار زمني"
            )

        if pressure_score >= 80:
            reasons.append(
                "ضغط الشراء إيجابي"
            )

        if structure_score >= 70:
            reasons.append(
                "اتجاه زمني متناسق نسبيًا"
            )

        if breakout_score >= 75:
            reasons.append(
                "إشارة مبكرة لاحتمال توسع الحركة"
            )

        # ---------------------------------------------------------
        # Warnings
        # ---------------------------------------------------------

        if liquidity_score < 35:
            warnings.append(
                "السيولة منخفضة — خطر الانزلاق السعري مرتفع"
            )

        if volume_score < 30:
            warnings.append(
                "حجم التداول منخفض"
            )

        if momentum_score < 35:
            warnings.append(
                "الزخم ضعيف أو سلبي"
            )

        if pressure_score <= 25:
            warnings.append(
                "ضغط البيع أو عدم التوازن مرتفع"
            )

        if pressure_state == "EXTREME_BUY_IMBALANCE":
            warnings.append(
                "هيمنة شراء شديدة — قد تحتاج لفحص التلاعب"
            )

        if _float(candidate.price_change_5m) > 50:
            warnings.append(
                "ارتفاع عمودي خلال 5 دقائق"
            )

        if _float(candidate.price_change_1h) > 150:
            warnings.append(
                "ارتفاع حاد خلال ساعة"
            )

        # Important limitation:
        warnings.append(
            "تحليل Market Structure الحالي تقريبي لعدم توفر سجل شموع كامل"
        )

        metrics = {
            "liquidity_usd": _float(
                candidate.liquidity_usd
            ),
            "volume_24h_usd": _float(
                candidate.volume_24h
            ),
            "price_change_5m": _float(
                candidate.price_change_5m
            ),
            "price_change_1h": _float(
                candidate.price_change_1h
            ),
            "price_change_6h": _float(
                candidate.price_change_6h
            ),
            "price_change_24h": _float(
                candidate.price_change_24h
            ),
            "buy_transactions": float(
                max(0, int(candidate.txns_buys or 0))
            ),
            "sell_transactions": float(
                max(0, int(candidate.txns_sells or 0))
            ),
            "buy_ratio": round(
                buy_ratio,
                4,
            ),
            "maker_count": float(
                max(0, int(candidate.maker_count or 0))
            ),
        }

        result = MarketIntelligence(
            mint=candidate.mint,
            market_score=market_score,
            momentum_score=round(
                momentum_score,
                1,
            ),
            volume_score=round(
                volume_score,
                1,
            ),
            pressure_score=round(
                pressure_score,
                1,
            ),
            liquidity_score=round(
                liquidity_score,
                1,
            ),
            structure_score=round(
                structure_score,
                1,
            ),
            breakout_score=round(
                breakout_score,
                1,
            ),
            market_state=_market_state(
                market_score,
                momentum_score,
                liquidity_score,
            ),
            momentum_state=momentum_state,
            volume_state=(
                "HIGH"
                if volume_score >= 75
                else "MEDIUM"
                if volume_score >= 40
                else "LOW"
            ),
            pressure_state=pressure_state,
            breakout_state=breakout_state,
            reasons=list(
                dict.fromkeys(reasons)
            ),
            warnings=list(
                dict.fromkeys(warnings)
            ),
            metrics=metrics,
        )

        logger.debug(
            "Market intelligence %s: score=%.1f state=%s",
            candidate.mint[:8],
            result.market_score,
            result.market_state,
        )

        return result

    def analyze_many(
        self,
        candidates: list[Candidate],
    ) -> list[MarketIntelligence]:
        return [
            self.analyze(candidate)
            for candidate in candidates
        ]

    def strong_markets(
        self,
        candidates: list[Candidate],
        minimum_score: float = 70.0,
    ) -> list[MarketIntelligence]:

        results = self.analyze_many(candidates)

        return sorted(
            [
                result
                for result in results
                if result.market_score >= minimum_score
            ],
            key=lambda result: result.market_score,
            reverse=True,
        )


market_intelligence = MarketIntelligenceEngine()
