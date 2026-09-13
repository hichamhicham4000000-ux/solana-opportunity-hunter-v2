"""
Opportunity Hunter — market opportunity scoring layer.

This module is intentionally independent from Discord, trading execution,
and the existing RugScore security engine.

It converts available market signals into an opportunity score.
Security/rug analysis remains a separate gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class OpportunitySignal:
    """A single market signal contributing to an opportunity assessment."""

    name: str
    score: float
    weight: float
    reason: str
    data_available: bool = True


@dataclass
class OpportunityResult:
    """Result of the market opportunity assessment."""

    score: float
    label: str
    signals: list[OpportunitySignal] = field(default_factory=list)
    data_quality: float = 0.0
    reasons: list[str] = field(default_factory=list)


def _number(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamp a number to a bounded range."""
    return max(low, min(high, value))


def _signal(
    name: str,
    score: float,
    weight: float,
    reason: str,
    data_available: bool = True,
) -> OpportunitySignal:
    return OpportunitySignal(
        name=name,
        score=_bounded(score),
        weight=weight,
        reason=reason,
        data_available=data_available,
    )


def _price_momentum(market: dict[str, Any]) -> OpportunitySignal:
    """
    Estimate momentum from available price-change fields.

    This is deliberately conservative. It does not treat a large pump
    as automatically bullish.
    """
    change_1h = market.get("price_change_1h")
    change_6h = market.get("price_change_6h")
    change_24h = market.get("price_change_24h")

    available = [
        value
        for value in (change_1h, change_6h, change_24h)
        if value is not None
    ]

    if not available:
        return _signal(
            "price_momentum",
            0,
            0.20,
            "بيانات الزخم السعري غير متاحة",
            False,
        )

    h1 = _number(change_1h) if change_1h is not None else 0.0
    h6 = _number(change_6h) if change_6h is not None else h1
    h24 = _number(change_24h) if change_24h is not None else h6

    # Moderate positive momentum is preferred over an extreme vertical move.
    if 3 <= h1 <= 15:
        score = 90
        reason = f"زخم ساعة إيجابي ومعتدل ({h1:.1f}%)"
    elif 0 <= h1 < 3:
        score = 55
        reason = f"زخم ساعة ضعيف إلى محايد ({h1:.1f}%)"
    elif 15 < h1 <= 30:
        score = 70
        reason = f"زخم قوي لكن يحتاج مراقبة ({h1:.1f}%)"
    elif h1 > 30:
        score = 40
        reason = f"ارتفاع حاد خلال ساعة ({h1:.1f}%) — خطر مطاردة السعر"
    elif -5 <= h1 < 0:
        score = 35
        reason = f"زخم سلبي خفيف ({h1:.1f}%)"
    else:
        score = 15
        reason = f"زخم سلبي قوي ({h1:.1f}%)"

    # Confirm direction across longer windows.
    if h6 > 0 and h24 > 0:
        score += 5
    elif h6 < 0 and h24 < 0:
        score -= 10

    return _signal(
        "price_momentum",
        score,
        0.20,
        reason,
    )


def _volume_acceleration(market: dict[str, Any]) -> OpportunitySignal:
    """Evaluate whether recent volume is accelerating."""
    volume_1h = _number(market.get("volume_1h"))
    volume_6h = _number(market.get("volume_6h"))
    volume_24h = _number(market.get("volume_24h"))

    if volume_24h <= 0:
        return _signal(
            "volume_acceleration",
            0,
            0.20,
            "بيانات الحجم غير متاحة",
            False,
        )

    baseline_1h = volume_24h / 24.0
    ratio = volume_1h / baseline_1h if baseline_1h > 0 else 0.0

    if ratio >= 5:
        score = 90
        reason = f"تسارع قوي جدًا في الحجم ({ratio:.1f}x عن متوسط الساعة)"
    elif ratio >= 3:
        score = 80
        reason = f"تسارع قوي في الحجم ({ratio:.1f}x)"
    elif ratio >= 1.5:
        score = 65
        reason = f"الحجم أعلى من متوسطه ({ratio:.1f}x)"
    elif ratio >= 0.7:
        score = 45
        reason = f"الحجم قريب من معدله ({ratio:.1f}x)"
    else:
        score = 25
        reason = f"الحجم ضعيف ({ratio:.1f}x)"

    # If 6h data exists, use it as a secondary confirmation.
    if volume_6h > 0:
        hourly_6h = volume_6h / 6.0
        if volume_1h > hourly_6h * 1.5:
            score += 5

    return _signal(
        "volume_acceleration",
        score,
        0.20,
        reason,
    )


def _buy_pressure(market: dict[str, Any]) -> OpportunitySignal:
    """Evaluate buy/sell pressure from transaction counts."""
    buys = _number(market.get("txns_24h_buys"))
    sells = _number(market.get("txns_24h_sells"))
    total = buys + sells

    if total <= 0:
        return _signal(
            "buy_pressure",
            0,
            0.15,
            "بيانات الشراء والبيع غير متاحة",
            False,
        )

    buy_ratio = buys / total

    if 0.55 <= buy_ratio <= 0.70:
        score = 90
        reason = f"ضغط شراء صحي ({buy_ratio:.0%} من المعاملات)"
    elif 0.70 < buy_ratio <= 0.80:
        score = 75
        reason = f"ضغط شراء قوي ({buy_ratio:.0%})"
    elif 0.50 <= buy_ratio < 0.55:
        score = 60
        reason = f"ميل شرائي بسيط ({buy_ratio:.0%})"
    elif buy_ratio > 0.80:
        score = 45
        reason = f"شراء مفرط ({buy_ratio:.0%}) — احتمال حركة غير متوازنة"
    elif buy_ratio >= 0.40:
        score = 35
        reason = f"ضغط شراء ضعيف ({buy_ratio:.0%})"
    else:
        score = 15
        reason = f"ضغط بيع واضح ({buy_ratio:.0%} شراء)"

    return _signal(
        "buy_pressure",
        score,
        0.15,
        reason,
    )


def _liquidity_quality(market: dict[str, Any]) -> OpportunitySignal:
    """Evaluate whether liquidity is sufficient for a meaningful opportunity."""
    liquidity = _number(market.get("liquidity_usd"))
    market_cap = _number(market.get("market_cap"))

    if liquidity <= 0:
        return _signal(
            "liquidity_quality",
            0,
            0.15,
            "لا توجد بيانات سيولة قابلة للتحقق",
            False,
        )

    if liquidity >= 100_000:
        score = 90
        reason = f"سيولة قوية (${liquidity:,.0f})"
    elif liquidity >= 50_000:
        score = 80
        reason = f"سيولة جيدة (${liquidity:,.0f})"
    elif liquidity >= 20_000:
        score = 65
        reason = f"سيولة مقبولة (${liquidity:,.0f})"
    elif liquidity >= 5_000:
        score = 40
        reason = f"سيولة منخفضة (${liquidity:,.0f})"
    else:
        score = 15
        reason = f"سيولة شديدة الانخفاض (${liquidity:,.0f})"

    if market_cap > 0:
        ratio = market_cap / liquidity

        if ratio > 100:
            score -= 20
            reason += f"؛ MCap/Liquidity مرتفعة ({ratio:.0f}x)"
        elif ratio < 20:
            score += 5

    return _signal(
        "liquidity_quality",
        score,
        0.15,
        reason,
    )


def _market_activity(market: dict[str, Any]) -> OpportunitySignal:
    """Evaluate market activity using transactions and unique makers."""
    txns = _number(market.get("txns_24h_buys")) + _number(
        market.get("txns_24h_sells")
    )
    makers = _number(market.get("maker_count"))

    if txns <= 0 and makers <= 0:
        return _signal(
            "market_activity",
            0,
            0.10,
            "بيانات نشاط السوق غير متاحة",
            False,
        )

    score = 20

    if txns >= 10_000:
        score += 45
    elif txns >= 5_000:
        score += 35
    elif txns >= 1_000:
        score += 25
    elif txns >= 250:
        score += 15
    elif txns >= 50:
        score += 5

    if makers >= 1_000:
        score += 30
    elif makers >= 500:
        score += 25
    elif makers >= 100:
        score += 15
    elif makers >= 25:
        score += 5

    return _signal(
        "market_activity",
        _bounded(score),
        0.10,
        f"نشاط السوق: {int(txns):,} معاملات وقرابة {int(makers):,} متداول",
    )


def calculate_opportunity_score(
    market: Optional[dict[str, Any]],
) -> OpportunityResult:
    """
    Calculate a market opportunity score.

    This score measures opportunity characteristics only.
    It must NOT be interpreted as a safety score or profit guarantee.
    """
    market = market or {}

    signals = [
        _price_momentum(market),
        _volume_acceleration(market),
        _buy_pressure(market),
        _liquidity_quality(market),
        _market_activity(market),
    ]

    available_signals = [s for s in signals if s.data_available]
    total_weight = sum(s.weight for s in available_signals)

    if total_weight <= 0:
        return OpportunityResult(
            score=0.0,
            label="INSUFFICIENT DATA",
            signals=signals,
            data_quality=0.0,
            reasons=["لا توجد بيانات سوق كافية لحساب فرصة"],
        )

    weighted_score = sum(
        signal.score * signal.weight
        for signal in available_signals
    ) / total_weight

    data_quality = (len(available_signals) / len(signals)) * 100

    if weighted_score >= 80:
        label = "STRONG OPPORTUNITY"
    elif weighted_score >= 70:
        label = "OPPORTUNITY"
    elif weighted_score >= 55:
        label = "WATCH"
    elif weighted_score >= 40:
        label = "SPECULATIVE"
    else:
        label = "AVOID"

    reasons = [
        signal.reason
        for signal in signals
        if signal.data_available
    ]

    return OpportunityResult(
        score=round(_bounded(weighted_score), 1),
        label=label,
        signals=signals,
        data_quality=round(data_quality, 1),
        reasons=reasons,
    )


def result_to_dict(result: OpportunityResult) -> dict[str, Any]:
    """Convert an OpportunityResult into a serializable dictionary."""
    return {
        "opportunity_score": result.score,
        "opportunity_label": result.label,
        "data_quality": result.data_quality,
        "reasons": result.reasons,
        "signals": [
            {
                "name": signal.name,
                "score": signal.score,
                "weight": signal.weight,
                "reason": signal.reason,
                "data_available": signal.data_available,
            }
            for signal in result.signals
        ],
}
