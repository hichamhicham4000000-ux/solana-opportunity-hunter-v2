"""
Security Gate.

Acts as the first defensive barrier after opportunity ranking.

The gate does not attempt to prove that a token is safe.
It identifies obvious risk conditions that should prevent a candidate
from immediately reaching deep analysis.

No trades are executed by this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from analysis.discovery import Candidate
from analysis.ranker import RankedOpportunity

logger = logging.getLogger(__name__)


@dataclass
class SecurityDecision:
    """Result of the initial security screening."""

    mint: str
    passed: bool
    status: str
    risk_score: float
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def should_deep_analyze(self) -> bool:
        return self.passed


def _safe_float(value: Optional[float]) -> float:
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


class SecurityGate:
    """
    Conservative pre-analysis security gate.

    Important:
    Passing this gate does NOT mean a token is safe.

    It only means that no obvious blocking condition was detected
    using the data currently available to the discovery/ranking layers.
    """

    def __init__(
        self,
        minimum_liquidity_usd: float = 5_000.0,
        maximum_5m_change: float = 100.0,
        maximum_1h_change: float = 300.0,
        maximum_risk_score: float = 60.0,
    ):
        self.minimum_liquidity_usd = minimum_liquidity_usd
        self.maximum_5m_change = maximum_5m_change
        self.maximum_1h_change = maximum_1h_change
        self.maximum_risk_score = maximum_risk_score

    def evaluate(
        self,
        candidate: Candidate,
        ranked: Optional[RankedOpportunity] = None,
    ) -> SecurityDecision:
        """
        Perform the initial security screen.

        The method intentionally uses only evidence available at this
        stage. Deeper contract, holder, developer, and wallet analysis
        belongs to later pipeline stages.
        """

        reasons: list[str] = []
        warnings: list[str] = []

        checks: dict[str, bool] = {}

        risk_score = 0.0

        liquidity = _safe_float(candidate.liquidity_usd)
        change_5m = _safe_float(candidate.price_change_5m)
        change_1h = _safe_float(candidate.price_change_1h)
        volume = _safe_float(candidate.volume_24h)

        buys = max(0, int(candidate.txns_buys or 0))
        sells = max(0, int(candidate.txns_sells or 0))
        transactions = buys + sells

        # ---------------------------------------------------------
        # 1. Liquidity check
        # ---------------------------------------------------------

        liquidity_ok = liquidity >= self.minimum_liquidity_usd
        checks["liquidity"] = liquidity_ok

        if not liquidity_ok:
            risk_score += 35
            reasons.append(
                "السيولة أقل من الحد الأدنى المحدد للحماية الأولية"
            )
        elif liquidity < 20_000:
            risk_score += 10
            warnings.append(
                "السيولة محدودة نسبيًا — يجب فحص الانزلاق السعري"
            )

        # ---------------------------------------------------------
        # 2. Extreme short-term price movement
        # ---------------------------------------------------------

        extreme_5m = change_5m > self.maximum_5m_change
        checks["extreme_5m_move"] = not extreme_5m

        if extreme_5m:
            risk_score += 30
            reasons.append(
                "حركة سعرية عمودية أو شديدة جدًا خلال 5 دقائق"
            )
        elif change_5m > 50:
            risk_score += 15
            warnings.append(
                "ارتفاع سريع جدًا خلال 5 دقائق"
            )

        # ---------------------------------------------------------
        # 3. Extreme hourly movement
        # ---------------------------------------------------------

        extreme_1h = change_1h > self.maximum_1h_change
        checks["extreme_1h_move"] = not extreme_1h

        if extreme_1h:
            risk_score += 25
            reasons.append(
                "حركة سعرية شديدة جدًا خلال ساعة"
            )
        elif change_1h > 150:
            risk_score += 10
            warnings.append(
                "ارتفاع حاد خلال ساعة — خطر مطاردة السعر"
            )

        # ---------------------------------------------------------
        # 4. Market data availability
        # ---------------------------------------------------------

        market_data_available = (
            liquidity > 0
            or volume > 0
            or candidate.price_usd is not None
        )

        checks["market_data_available"] = market_data_available

        if not market_data_available:
            risk_score += 30
            reasons.append(
                "بيانات السوق الأساسية غير متوفرة بما يكفي للفحص"
            )

        # ---------------------------------------------------------
        # 5. Trading activity
        # ---------------------------------------------------------

        activity_available = transactions > 0 or volume > 0
        checks["trading_activity_available"] = activity_available

        if not activity_available:
            risk_score += 15
            warnings.append(
                "لا توجد بيانات كافية عن نشاط التداول"
            )

        # ---------------------------------------------------------
        # 6. Buy/sell imbalance
        # ---------------------------------------------------------

        if transactions > 0:
            buy_ratio = buys / transactions

            if buy_ratio > 0.90:
                risk_score += 15
                warnings.append(
                    "هيمنة شراء غير طبيعية — يجب فحص احتمال التلاعب"
                )

            elif buy_ratio < 0.20:
                risk_score += 10
                warnings.append(
                    "ضغط بيع مرتفع"
                )

            checks["balanced_order_flow"] = 0.20 <= buy_ratio <= 0.90

        else:
            checks["balanced_order_flow"] = False

        # ---------------------------------------------------------
        # 7. Missing pair information
        # ---------------------------------------------------------

        pair_available = bool(candidate.pair_url)
        checks["pair_available"] = pair_available

        if not pair_available:
            risk_score += 5
            warnings.append(
                "رابط زوج التداول غير متوفر حاليًا"
            )

        # ---------------------------------------------------------
        # 8. Ranking warnings
        # ---------------------------------------------------------

        if ranked is not None:
            if ranked.rank_score < 35:
                risk_score += 10
                warnings.append(
                    "الأولوية السوقية منخفضة وفق طبقة الترتيب"
                )

            if ranked.warnings:
                warnings.extend(ranked.warnings[:3])

        # ---------------------------------------------------------
        # Final decision
        # ---------------------------------------------------------

        risk_score = round(_clamp(risk_score), 1)

        blocking_conditions = (
            not liquidity_ok
            or extreme_5m
            or extreme_1h
            or not market_data_available
        )

        if blocking_conditions:
            passed = False
            status = "FAIL"

        elif risk_score > self.maximum_risk_score:
            passed = False
            status = "FAIL"

        elif risk_score >= 40:
            passed = True
            status = "PASS_WITH_WARNINGS"

        else:
            passed = True
            status = "PASS"

        if passed:
            if status == "PASS":
                reasons.append(
                    "لم يتم اكتشاف شرط حظر أولي"
                )
            else:
                warnings.append(
                    "مرشح للتحليل العميق مع وجود إشارات تستوجب الحذر"
                )

        else:
            reasons.append(
                "تم حجب المرشح عن التحليل العميق الأولي"
            )

        result = SecurityDecision(
            mint=candidate.mint,
            passed=passed,
            status=status,
            risk_score=risk_score,
            reasons=list(dict.fromkeys(reasons)),
            warnings=list(dict.fromkeys(warnings)),
            checks=checks,
        )

        logger.debug(
            "Security gate: %s score=%.1f status=%s",
            candidate.mint[:8],
            risk_score,
            status,
        )

        return result

    def evaluate_many(
        self,
        opportunities: list[RankedOpportunity],
    ) -> list[tuple[RankedOpportunity, SecurityDecision]]:
        """
        Evaluate multiple ranked opportunities.

        Returns both the ranked opportunity and its security decision
        so downstream components can preserve the complete context.
        """

        results: list[
            tuple[RankedOpportunity, SecurityDecision]
        ] = []

        for opportunity in opportunities:
            decision = self.evaluate(
                opportunity.candidate,
                opportunity,
            )
            results.append((opportunity, decision))

        return results

    def approved(
        self,
        opportunities: list[RankedOpportunity],
    ) -> list[RankedOpportunity]:
        """
        Return only candidates that passed the initial security gate.
        """

        results = self.evaluate_many(opportunities)

        return [
            opportunity
            for opportunity, decision in results
            if decision.passed
        ]


security_gate = SecurityGate()
