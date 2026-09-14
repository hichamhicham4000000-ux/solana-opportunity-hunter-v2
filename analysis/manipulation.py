"""
Market Manipulation Detection Engine.

Detects suspicious market patterns using the evidence currently
available in Candidate.

Important:
- These are heuristic signals, not proof of manipulation.
- Wallet-level coordination requires deeper wallet data.
- No trades are executed by this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from analysis.discovery import Candidate
from analysis.market_intelligence import MarketIntelligence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass
class ManipulationSignal:
    """A single suspicious market signal."""

    name: str
    severity: str
    score: float
    description: str
    evidence: dict[str, float] = field(default_factory=dict)


@dataclass
class ManipulationAssessment:
    """Combined manipulation-risk assessment."""

    mint: str

    manipulation_score: float
    risk_level: str

    suspicious: bool
    severe: bool

    signals: list[ManipulationSignal] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    limitations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


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


def _add_signal(
    signals: list[ManipulationSignal],
    name: str,
    severity: str,
    score: float,
    description: str,
    evidence: Optional[dict[str, float]] = None,
) -> None:
    signals.append(
        ManipulationSignal(
            name=name,
            severity=severity,
            score=score,
            description=description,
            evidence=evidence or {},
        )
    )


# ---------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------


def _detect_extreme_price_acceleration(
    candidate: Candidate,
    signals: list[ManipulationSignal],
) -> None:
    change_5m = _float(candidate.price_change_5m)
    change_1h = _float(candidate.price_change_1h)

    if change_5m > 50:
        _add_signal(
            signals,
            name="EXTREME_5M_MOVE",
            severity="HIGH",
            score=25.0,
            description=(
                "ارتفاع سعري شديد خلال 5 دقائق؛ "
                "قد يشير إلى حركة مضاربية أو Pump سريع."
            ),
            evidence={
                "price_change_5m": change_5m,
            },
        )

    elif change_5m > 25:
        _add_signal(
            signals,
            name="FAST_5M_MOVE",
            severity="MEDIUM",
            score=12.0,
            description=(
                "ارتفاع سريع خلال 5 دقائق يستوجب التحقق."
            ),
            evidence={
                "price_change_5m": change_5m,
            },
        )

    if change_1h > 150:
        _add_signal(
            signals,
            name="EXTREME_1H_MOVE",
            severity="HIGH",
            score=25.0,
            description=(
                "ارتفاع شديد خلال ساعة؛ "
                "خطر مطاردة السعر أو Pump ممتد."
            ),
            evidence={
                "price_change_1h": change_1h,
            },
        )

    elif change_1h > 80:
        _add_signal(
            signals,
            name="FAST_1H_MOVE",
            severity="MEDIUM",
            score=12.0,
            description=(
                "ارتفاع قوي خلال ساعة يحتاج إلى تحقق إضافي."
            ),
            evidence={
                "price_change_1h": change_1h,
            },
        )


def _detect_extreme_buy_imbalance(
    candidate: Candidate,
    signals: list[ManipulationSignal],
) -> None:
    buys = max(0, int(candidate.txns_buys or 0))
    sells = max(0, int(candidate.txns_sells or 0))

    total = buys + sells

    if total < 20:
        return

    buy_ratio = buys / total

    if buy_ratio >= 0.95:
        _add_signal(
            signals,
            name="EXTREME_BUY_IMBALANCE",
            severity="HIGH",
            score=25.0,
            description=(
                "عمليات الشراء تهيمن بشكل استثنائي على التداول؛ "
                "يجب فحص المحافظ والصفقات للتأكد من أن النشاط عضوي."
            ),
            evidence={
                "buy_ratio": buy_ratio,
                "buy_transactions": float(buys),
                "sell_transactions": float(sells),
            },
        )

    elif buy_ratio >= 0.90:
        _add_signal(
            signals,
            name="HIGH_BUY_IMBALANCE",
            severity="MEDIUM",
            score=15.0,
            description=(
                "اختلال مرتفع في اتجاه الشراء."
            ),
            evidence={
                "buy_ratio": buy_ratio,
                "buy_transactions": float(buys),
                "sell_transactions": float(sells),
            },
        )


def _detect_extreme_sell_pressure(
    candidate: Candidate,
    signals: list[ManipulationSignal],
) -> None:
    buys = max(0, int(candidate.txns_buys or 0))
    sells = max(0, int(candidate.txns_sells or 0))

    total = buys + sells

    if total < 20:
        return

    sell_ratio = sells / total

    if sell_ratio >= 0.90:
        _add_signal(
            signals,
            name="EXTREME_SELL_PRESSURE",
            severity="HIGH",
            score=22.0,
            description=(
                "ضغط البيع مرتفع جدًا؛ "
                "قد يشير إلى تصريف أو انهيار في الطلب."
            ),
            evidence={
                "sell_ratio": sell_ratio,
                "buy_transactions": float(buys),
                "sell_transactions": float(sells),
            },
        )

    elif sell_ratio >= 0.75:
        _add_signal(
            signals,
            name="HIGH_SELL_PRESSURE",
            severity="MEDIUM",
            score=12.0,
            description=(
                "ضغط البيع مرتفع نسبيًا."
            ),
            evidence={
                "sell_ratio": sell_ratio,
                "buy_transactions": float(buys),
                "sell_transactions": float(sells),
            },
        )


def _detect_volume_price_mismatch(
    candidate: Candidate,
    signals: list[ManipulationSignal],
) -> None:
    volume = _float(candidate.volume_24h)
    change_24h = _float(candidate.price_change_24h)
    liquidity = _float(candidate.liquidity_usd)

    if volume <= 0 or liquidity <= 0:
        return

    # Very high volume relative to available liquidity can be legitimate,
    # but it is an important reason to inspect trade quality.
    volume_liquidity_ratio = volume / liquidity

    if volume_liquidity_ratio >= 100:
        _add_signal(
            signals,
            name="EXTREME_VOLUME_LIQUIDITY_RATIO",
            severity="HIGH",
            score=22.0,
            description=(
                "حجم التداول اليومي ضخم جدًا مقارنة بالسيولة؛ "
                "يجب التحقق من جودة الصفقات واحتمال تضخم الحجم."
            ),
            evidence={
                "volume_24h": volume,
                "liquidity_usd": liquidity,
                "volume_liquidity_ratio": volume_liquidity_ratio,
            },
        )

    elif volume_liquidity_ratio >= 50:
        _add_signal(
            signals,
            name="HIGH_VOLUME_LIQUIDITY_RATIO",
            severity="MEDIUM",
            score=12.0,
            description=(
                "حجم التداول مرتفع جدًا مقارنة بالسيولة."
            ),
            evidence={
                "volume_24h": volume,
                "liquidity_usd": liquidity,
                "volume_liquidity_ratio": volume_liquidity_ratio,
            },
        )

    # High volume with weak price response can also warrant investigation.
    if volume >= 1_000_000 and abs(change_24h) < 5:
        _add_signal(
            signals,
            name="HIGH_VOLUME_LOW_PRICE_RESPONSE",
            severity="MEDIUM",
            score=10.0,
            description=(
                "حجم تداول مرتفع مع تغير سعري محدود؛ "
                "يستحق فحص جودة الصفقات."
            ),
            evidence={
                "volume_24h": volume,
                "price_change_24h": change_24h,
            },
        )


def _detect_price_reversal_risk(
    candidate: Candidate,
    signals: list[ManipulationSignal],
) -> None:
    change_5m = _float(candidate.price_change_5m)
    change_1h = _float(candidate.price_change_1h)
    change_6h = _float(candidate.price_change_6h)

    # A very strong short-term move against a much weaker/negative
    # broader trend can indicate unstable price action.
    if (
        change_5m >= 20
        and change_1h >= 30
        and change_6h < 0
    ):
        _add_signal(
            signals,
            name="SHORT_TERM_SPIKE_AGAINST_TREND",
            severity="MEDIUM",
            score=14.0,
            description=(
                "ارتفاع قصير المدى قوي رغم اتجاه أضعف على الإطار الأطول؛ "
                "خطر انعكاس مرتفع."
            ),
            evidence={
                "price_change_5m": change_5m,
                "price_change_1h": change_1h,
                "price_change_6h": change_6h,
            },
        )


def _detect_low_liquidity_risk(
    candidate: Candidate,
    signals: list[ManipulationSignal],
) -> None:
    liquidity = _float(candidate.liquidity_usd)

    if liquidity <= 0:
        _add_signal(
            signals,
            name="MISSING_LIQUIDITY",
            severity="HIGH",
            score=30.0,
            description=(
                "لا توجد بيانات سيولة متاحة؛ "
                "لا يمكن تقييم قدرة السوق على امتصاص الصفقات."
            ),
        )

    elif liquidity < 5_000:
        _add_signal(
            signals,
            name="VERY_LOW_LIQUIDITY",
            severity="HIGH",
            score=25.0,
            description=(
                "السيولة منخفضة جدًا؛ "
                "احتمال الانزلاق السعري والتلاعب أعلى."
            ),
            evidence={
                "liquidity_usd": liquidity,
            },
        )

    elif liquidity < 20_000:
        _add_signal(
            signals,
            name="LOW_LIQUIDITY",
            severity="MEDIUM",
            score=12.0,
            description=(
                "السيولة محدودة نسبيًا."
            ),
            evidence={
                "liquidity_usd": liquidity,
            },
        )


def _detect_low_market_activity(
    candidate: Candidate,
    signals: list[ManipulationSignal],
) -> None:
    volume = _float(candidate.volume_24h)

    buys = max(0, int(candidate.txns_buys or 0))
    sells = max(0, int(candidate.txns_sells or 0))

    transactions = buys + sells

    if volume <= 0 and transactions == 0:
        _add_signal(
            signals,
            name="MISSING_MARKET_ACTIVITY",
            severity="MEDIUM",
            score=15.0,
            description=(
                "لا توجد بيانات كافية عن النشاط السوقي."
            ),
        )

    elif transactions > 0 and transactions < 20:
        _add_signal(
            signals,
            name="VERY_LOW_TRANSACTION_ACTIVITY",
            severity="LOW",
            score=5.0,
            description=(
                "عدد الصفقات المرصودة منخفض جدًا."
            ),
            evidence={
                "transactions": float(transactions),
            },
        )


def _detect_insufficient_wallet_evidence(
    candidate: Candidate,
    signals: list[ManipulationSignal],
) -> None:
    """
    Candidate currently does not contain wallet-level information.

    We do NOT convert this limitation into a manipulation accusation.
    We simply record that wallet coordination cannot yet be verified.
    """

    _ = candidate
    _ = signals


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------


class ManipulationEngine:
    """
    Conservative manipulation-risk detector.

    It identifies patterns that deserve deeper investigation.

    It does NOT claim to prove:
        - wash trading
        - coordinated wallets
        - insider trading
        - market-maker manipulation
        - pump-and-dump

    Those require transaction- and wallet-level evidence.
    """

    def analyze(
        self,
        candidate: Candidate,
        market: Optional[MarketIntelligence] = None,
    ) -> ManipulationAssessment:

        signals: list[ManipulationSignal] = []

        _detect_extreme_price_acceleration(
            candidate,
            signals,
        )

        _detect_extreme_buy_imbalance(
            candidate,
            signals,
        )

        _detect_extreme_sell_pressure(
            candidate,
            signals,
        )

        _detect_volume_price_mismatch(
            candidate,
            signals,
        )

        _detect_price_reversal_risk(
            candidate,
            signals,
        )

        _detect_low_liquidity_risk(
            candidate,
            signals,
        )

        _detect_low_market_activity(
            candidate,
            signals,
        )

        _detect_insufficient_wallet_evidence(
            candidate,
            signals,
        )

        # Market intelligence can provide additional context.
        if market is not None:

            if market.breakout_score >= 85:
                signals.append(
                    ManipulationSignal(
                        name="EXTENDED_BREAKOUT_RISK",
                        severity="MEDIUM",
                        score=8.0,
                        description=(
                            "الحركة تحمل خصائص توسع سعري قوي؛ "
                            "يجب التأكد من أن الاختراق مدعوم بتدفق عضوي."
                        ),
                        evidence={
                            "breakout_score": market.breakout_score,
                        },
                    )
                )

            if market.pressure_score <= 25:
                signals.append(
                    ManipulationSignal(
                        name="WEAK_BUY_PRESSURE",
                        severity="MEDIUM",
                        score=8.0,
                        description=(
                            "ضغط الشراء ضعيف وفق تحليل السوق."
                        ),
                        evidence={
                            "pressure_score": market.pressure_score,
                        },
                    )
                )

        # ---------------------------------------------------------
        # Aggregate risk
        # ---------------------------------------------------------

        # We cap the raw sum because multiple signals can describe
        # the same underlying event.
        raw_score = sum(
            signal.score
            for signal in signals
        )

        # Correlated signals receive diminishing influence.
        if len(signals) >= 3:
            raw_score *= 0.85

        if len(signals) >= 5:
            raw_score *= 0.80

        manipulation_score = round(
            _clamp(raw_score),
            1,
        )

        high_count = sum(
            1
            for signal in signals
            if signal.severity == "HIGH"
        )

        if manipulation_score >= 70 or high_count >= 3:
            risk_level = "CRITICAL"
            severe = True

        elif manipulation_score >= 50 or high_count >= 2:
            risk_level = "HIGH"
            severe = True

        elif manipulation_score >= 30 or high_count >= 1:
            risk_level = "MEDIUM"
            severe = False

        elif manipulation_score >= 15:
            risk_level = "LOW"
            severe = False

        else:
            risk_level = "MINIMAL"
            severe = False

        suspicious = manipulation_score >= 30

        # ---------------------------------------------------------
        # Reasons and warnings
        # ---------------------------------------------------------

        reasons: list[str] = []
        warnings: list[str] = []

        for signal in signals:
            if signal.severity in {"HIGH", "MEDIUM"}:
                reasons.append(
                    signal.description
                )

            elif signal.severity == "LOW":
                warnings.append(
                    signal.description
                )

        if severe:
            warnings.append(
                "مستوى الاشتباه مرتفع — لا ينبغي اعتبار الحركة السعرية "
                "دليلًا كافيًا على فرصة استثمارية."
            )

        elif suspicious:
            warnings.append(
                "تم اكتشاف إشارات تستوجب فحصًا أعمق للمحافظ والصفقات."
            )

        limitations = [
            "لا يمكن إثبات Wash Trading من بيانات السوق الإجمالية وحدها.",
            "لا يوجد في Candidate حاليًا سجل كامل لتدفقات المحافظ المنسقة.",
            "تأكيد التلاعب يحتاج إلى تحليل المحافظ والمعاملات على السلسلة.",
            "ارتفاع حجم التداول أو السعر لا يعني بحد ذاته وجود تلاعب.",
        ]

        result = ManipulationAssessment(
            mint=candidate.mint,
            manipulation_score=manipulation_score,
            risk_level=risk_level,
            suspicious=suspicious,
            severe=severe,
            signals=signals,
            reasons=list(
                dict.fromkeys(reasons)
            ),
            warnings=list(
                dict.fromkeys(warnings)
            ),
            limitations=limitations,
        )

        logger.debug(
            "Manipulation assessment %s: score=%.1f level=%s",
            candidate.mint[:8],
            result.manipulation_score,
            result.risk_level,
        )

        return result

    def analyze_many(
        self,
        candidates: list[Candidate],
        market_results: Optional[
            dict[str, MarketIntelligence]
        ] = None,
    ) -> list[ManipulationAssessment]:

        results: list[ManipulationAssessment] = []

        for candidate in candidates:
            market = None

            if market_results is not None:
                market = market_results.get(
                    candidate.mint
                )

            results.append(
                self.analyze(
                    candidate,
                    market,
                )
            )

        return results

    def high_risk(
        self,
        candidates: list[Candidate],
        market_results: Optional[
            dict[str, MarketIntelligence]
        ] = None,
    ) -> list[ManipulationAssessment]:

        results = self.analyze_many(
            candidates,
            market_results,
        )

        return sorted(
            [
                result
                for result in results
                if result.risk_level in {
                    "HIGH",
                    "CRITICAL",
                }
            ],
            key=lambda result: result.manipulation_score,
            reverse=True,
        )


manipulation_engine = ManipulationEngine()
