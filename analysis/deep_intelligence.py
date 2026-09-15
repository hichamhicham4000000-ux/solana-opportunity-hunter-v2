"""Deep Intelligence and short-term trading decision engine.

This module keeps the existing analytical layers and adds a conservative
BUY / WAIT / AVOID decision for short-term Solana meme-coin analysis.
It never executes trades and never treats confidence as a profit guarantee.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from analysis.discovery import Candidate
from analysis.engine import analyze_token
from analysis.ranker import RankedOpportunity
from analysis.security_gate import SecurityDecision
from analysis.smart_money_intelligence import (
    SmartMoneyAssessment,
    assess_smart_money,
)
from analysis.market_intelligence import (
    MarketIntelligence,
    market_intelligence,
)
from analysis.manipulation import (
    ManipulationAssessment,
    manipulation_engine,
)

logger = logging.getLogger(__name__)

BUY = "BUY"
WAIT = "WAIT"
AVOID = "AVOID"

# Backward-compatible names used by older callers.
STRONG_OPPORTUNITY = BUY
OPPORTUNITY = BUY
WATCH = WAIT
SPECULATIVE = WAIT
INVALIDATED = AVOID


@dataclass
class DeepIntelligenceResult:
    mint: str
    token_name: str
    token_symbol: str

    security_score: float
    rank_score: float
    discovery_score: float
    data_quality: float

    smart_money_score: float
    smart_money_confidence: float
    smart_money_state: str

    market_score: float
    momentum_score: float
    volume_score: float
    pressure_score: float
    liquidity_score: float
    structure_score: float
    breakout_score: float
    market_state: str

    manipulation_score: float
    manipulation_risk_level: str
    manipulation_suspicious: bool
    manipulation_severe: bool

    opportunity_score: float
    confidence: float
    decision: str

    # Short-term trading layer.
    trade_action: str = WAIT
    trade_horizon: str = "SHORT_TERM"
    trade_setup: str = "NO_CLEAR_SETUP"

    risk_label: str = "UNKNOWN"
    risk_description: str = ""

    criteria: dict[str, dict[str, Any]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    raw_analysis: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.decision == BUY


def _number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(value, maximum))


def _criterion_details(analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name, result in (analysis.get("criteria") or {}).items():
        if hasattr(result, "score"):
            output[name] = {
                "name": getattr(result, "name", name),
                "score": _number(getattr(result, "score", 0)),
                "flags": list(getattr(result, "flags", []) or []),
                "details": dict(getattr(result, "details", {}) or {}),
                "estimated": bool(getattr(result, "estimated", False)),
            }
        elif isinstance(result, dict):
            output[name] = {
                "name": result.get("name", name),
                "score": _number(result.get("score", 0)),
                "flags": list(result.get("flags", []) or []),
                "details": dict(result.get("details", {}) or {}),
                "estimated": bool(result.get("estimated", False)),
            }
    return output


def _calculate_confidence(
    analysis: dict[str, Any],
    security_decision: SecurityDecision,
    smart_money: Optional[SmartMoneyAssessment] = None,
    market: Optional[MarketIntelligence] = None,
) -> float:
    data_quality = _number(analysis.get("data_quality"), 0.0)
    estimated = analysis.get("estimated_criteria") or []
    completeness = max(0, 8 - len(estimated)) / 8 * 100 if estimated else 100.0

    gate_quality = 100.0
    if getattr(security_decision, "status", "") == "PASS_WITH_WARNINGS":
        gate_quality = 75.0
    elif not security_decision.passed:
        gate_quality = 30.0

    smart_quality = _number(smart_money.confidence, 0.0) if smart_money else 0.0
    market_quality = 75.0 if market is not None else 0.0

    return round(_clamp(
        data_quality * 0.45
        + completeness * 0.20
        + gate_quality * 0.15
        + smart_quality * 0.10
        + market_quality * 0.10
    ), 1)


def _calculate_opportunity_score(
    analysis: dict[str, Any],
    ranked: RankedOpportunity,
    security_decision: SecurityDecision,
    smart_money: Optional[SmartMoneyAssessment] = None,
    market: Optional[MarketIntelligence] = None,
    manipulation: Optional[ManipulationAssessment] = None,
) -> float:
    """Existing tested composite score; manipulation is a separate penalty."""
    security_score = _number(analysis.get("total_score"), 0.0)
    rank_score = _number(ranked.rank_score, 0.0)
    discovery_score = _number(ranked.candidate.discovery_score, 0.0)
    data_quality = _number(analysis.get("data_quality"), 0.0)

    gate_score = 100.0
    if not security_decision.passed:
        gate_score = 0.0
    elif getattr(security_decision, "status", "") == "PASS_WITH_WARNINGS":
        gate_score = 65.0

    smart_score = _number(smart_money.score, 50.0) if smart_money else 50.0
    market_score = _number(market.market_score, 0.0) if market else 0.0

    score = (
        security_score * 0.25
        + rank_score * 0.15
        + discovery_score * 0.08
        + data_quality * 0.07
        + gate_score * 0.05
        + smart_score * 0.15
        + market_score * 0.25
    )

    if manipulation is not None:
        score -= min(20.0, manipulation.manipulation_score * 0.20)
        if manipulation.risk_level == "CRITICAL":
            score -= 5.0

    return round(_clamp(score), 1)


def _determine_trade_setup(
    market: Optional[MarketIntelligence],
    smart_money: Optional[SmartMoneyAssessment],
    manipulation: Optional[ManipulationAssessment],
) -> str:
    if market is None:
        return "NO_CLEAR_SETUP"
    if manipulation and manipulation.risk_level in {"HIGH", "CRITICAL"}:
        return "HIGH_RISK"
    if market.breakout_score >= 75 and market.momentum_score >= 70 and market.pressure_score >= 65:
        return "BREAKOUT_MOMENTUM"
    if market.momentum_score >= 75 and market.pressure_score >= 70:
        return "MOMENTUM"
    if smart_money and smart_money.state in {"STRONG", "POSITIVE"} and market.market_score >= 60:
        return "SMART_MONEY_FLOW"
    if market.pressure_score >= 65 and market.volume_score >= 60:
        return "BUY_PRESSURE"
    if market.market_score >= 60:
        return "POSITIVE_MARKET_STRUCTURE"
    return "NO_CLEAR_SETUP"


def _decision_from_scores(
    opportunity_score: float,
    security_score: float,
    confidence: float,
    security_decision: SecurityDecision,
    candidate: Candidate,
    manipulation: Optional[ManipulationAssessment] = None,
    market: Optional[MarketIntelligence] = None,
    smart_money: Optional[SmartMoneyAssessment] = None,
) -> str:
    """Conservative short-term BUY / WAIT / AVOID classifier."""
    if not security_decision.passed:
        return AVOID
    if confidence < 50 or security_score < 55:
        return WAIT

    if manipulation:
        if manipulation.risk_level == "CRITICAL":
            return AVOID
        if manipulation.risk_level == "HIGH" and opportunity_score < 82:
            return WAIT

    if market is None:
        return WAIT

    change_5m = _number(candidate.price_change_5m)
    change_1h = _number(candidate.price_change_1h)
    if change_5m > 50 or change_1h > 150:
        if opportunity_score < 85 or market.pressure_score < 75 or market.volume_score < 65:
            return WAIT

    if market.market_score < 55:
        return WAIT
    if market.momentum_score < 55:
        return WAIT
    if market.pressure_score < 55:
        return WAIT
    if market.volume_score < 50:
        return WAIT
    if market.liquidity_score < 50:
        return WAIT

    smart_support = bool(
        smart_money
        and smart_money.state in {"STRONG", "POSITIVE"}
        and smart_money.confidence >= 50
    )

    if opportunity_score >= 78:
        return BUY

    aligned = (
        market.momentum_score >= 70
        and market.pressure_score >= 65
        and market.volume_score >= 60
        and market.liquidity_score >= 60
    )

    if (
        opportunity_score >= 70
        and aligned
        and (smart_support or market.breakout_score >= 70)
        and confidence >= 60
        and security_score >= 65
    ):
        return BUY

    return WAIT


def _determine_trade_horizon(
    decision: str,
    market: Optional[MarketIntelligence],
) -> str:
    if decision != BUY:
        return "غير محدد — انتظار تأكيد"
    if market is None:
        return "قصير جدًا"
    if market.breakout_score >= 80 and market.momentum_score >= 75:
        return "15 دقيقة – ساعتان"
    if market.momentum_score >= 75 and market.pressure_score >= 70:
        return "30 دقيقة – 4 ساعات"
    if market.market_score >= 70:
        return "1 – 6 ساعات"
    return "30 دقيقة – 4 ساعات"


def _build_reasons(
    analysis: dict[str, Any],
    ranked: RankedOpportunity,
    opportunity_score: float,
    smart_money: Optional[SmartMoneyAssessment] = None,
    market: Optional[MarketIntelligence] = None,
    manipulation: Optional[ManipulationAssessment] = None,
) -> list[str]:
    reasons: list[str] = []
    security_score = _number(analysis.get("total_score"), 0.0)
    data_quality = _number(analysis.get("data_quality"), 0.0)

    if security_score >= 75:
        reasons.append("نتيجة السلامة والتحقق الأساسي قوية")
    elif security_score >= 60:
        reasons.append("نتيجة السلامة مقبولة مع الحاجة إلى متابعة")
    if ranked.rank_score >= 70:
        reasons.append("المرشح يحتل ترتيبًا مرتفعًا ضمن فرص التحليل")
    if ranked.candidate.discovery_score >= 70:
        reasons.append("إشارات الاكتشاف الأولية قوية نسبيًا")
    if data_quality >= 80:
        reasons.append("جودة البيانات مرتفعة")

    if market:
        if market.market_score >= 70:
            reasons.append("وضع السوق الحالي إيجابي نسبيًا")
        if market.momentum_score >= 70:
            reasons.append("الزخم قصير الأجل إيجابي")
        if market.pressure_score >= 80:
            reasons.append("ضغط الشراء قوي نسبيًا")
        elif market.pressure_score >= 65:
            reasons.append("ضغط الشراء يميل لصالح المشترين")
        if market.volume_score >= 70:
            reasons.append("جودة/نشاط الحجم يدعم الحركة")
        if market.liquidity_score >= 70:
            reasons.append("السيولة مناسبة نسبيًا للتداول القصير")
        if market.breakout_score >= 75:
            reasons.append("توجد إشارة مبكرة لاحتمال توسع الحركة")
        reasons.extend(market.reasons[:3])

    if smart_money:
        state_text = {
            "STRONG": "إشارة Smart Money قوية نسبيًا",
            "POSITIVE": "إشارة Smart Money إيجابية",
            "NEUTRAL": "إشارة Smart Money محايدة",
            "WEAK": "إشارة Smart Money ضعيفة",
            "NONE": "لا توجد محفظة Smart Money معروفة ضمن البيانات المتاحة",
        }.get(smart_money.state)
        if state_text:
            reasons.append(state_text)
        reasons.extend(smart_money.reasons[:3])

    if manipulation and manipulation.manipulation_score < 15:
        reasons.append("لا توجد إشارات تلاعب سوقي قوية ضمن البيانات المتاحة")
        reasons.extend(manipulation.reasons[:3])

    if opportunity_score >= 80:
        reasons.append("يوجد توافق جيد بين عدة طبقات مستقلة من الأدلة")

    reasons.extend(ranked.reasons[:4])
    return list(dict.fromkeys(reasons))


def _build_warnings(
    analysis: dict[str, Any],
    ranked: RankedOpportunity,
    security_decision: SecurityDecision,
    smart_money: Optional[SmartMoneyAssessment] = None,
    market: Optional[MarketIntelligence] = None,
    manipulation: Optional[ManipulationAssessment] = None,
) -> list[str]:
    warnings: list[str] = []
    warnings.extend(getattr(security_decision, "warnings", [])[:6])
    warnings.extend(getattr(ranked, "warnings", [])[:6])

    data_quality = _number(analysis.get("data_quality"), 0.0)
    if data_quality < 70:
        warnings.append("جودة البيانات أقل من المستوى المطلوب لثقة عالية")
    if analysis.get("estimated_criteria"):
        warnings.append("بعض معايير التحليل اعتمدت على بيانات تقديرية أو غير مكتملة")
    if market:
        warnings.extend(market.warnings[:5])
    if manipulation:
        warnings.extend(manipulation.warnings[:5])
        warnings.extend(getattr(manipulation, "limitations", [])[:2])
    if smart_money:
        warnings.extend(smart_money.warnings[:4])
        if smart_money.confidence < 50:
            warnings.append("ثقة Smart Money محدودة بسبب نقص بيانات النشاط")
        warnings.append("Smart Money إشارة مساعدة وليست ضمانًا للربح")

    warnings.append("قرار BUY هنا إشارة تحليلية قصيرة الأجل وليس ضمانًا للربح")
    return list(dict.fromkeys(warnings))


def _build_blockers(security_decision: SecurityDecision) -> list[str]:
    if security_decision.passed:
        return []
    return list(getattr(security_decision, "reasons", []) or [])


class DeepIntelligenceEngine:
    """Coordinates the existing analysis layers and the short-term decision."""

    async def analyze(
        self,
        ranked: RankedOpportunity,
        security_decision: SecurityDecision,
        force: bool = False,
    ) -> DeepIntelligenceResult:
        candidate = ranked.candidate

        if not security_decision.passed and not force:
            return self._blocked_result(candidate, ranked, security_decision)

        analysis = await analyze_token(candidate.mint)
        smart_money = await assess_smart_money(candidate.mint)
        market = market_intelligence.analyze(candidate)
        manipulation = manipulation_engine.analyze(candidate, market)

        security_score = _number(analysis.get("total_score"), 0.0)
        confidence = _calculate_confidence(
            analysis,
            security_decision,
            smart_money,
            market,
        )
        opportunity_score = _calculate_opportunity_score(
            analysis=analysis,
            ranked=ranked,
            security_decision=security_decision,
            smart_money=smart_money,
            market=market,
            manipulation=manipulation,
        )

        decision = _decision_from_scores(
            opportunity_score=opportunity_score,
            security_score=security_score,
            confidence=confidence,
            security_decision=security_decision,
            candidate=candidate,
            manipulation=manipulation,
            market=market,
            smart_money=smart_money,
        )

        setup = _determine_trade_setup(
            market,
            smart_money,
            manipulation,
        )
        horizon = _determine_trade_horizon(decision, market)
        criteria = _criterion_details(analysis)
        reasons = _build_reasons(
            analysis,
            ranked,
            opportunity_score,
            smart_money,
            market,
            manipulation,
        )
        warnings = _build_warnings(
            analysis,
            ranked,
            security_decision,
            smart_money,
            market,
            manipulation,
        )
        blockers = _build_blockers(security_decision)

        raw = dict(analysis)
        raw["short_term_decision"] = {
            "action": decision,
            "horizon": horizon,
            "setup": setup,
        }
        raw["smart_money"] = {
            "score": smart_money.score,
            "confidence": smart_money.confidence,
            "state": smart_money.state,
            "known_wallet_count": smart_money.known_wallet_count,
            "active_known_wallet_count": smart_money.active_known_wallet_count,
            "total_known_holder_amount": smart_money.total_known_holder_amount,
            "holder_coverage": smart_money.holder_coverage,
            "reasons": smart_money.reasons,
            "warnings": smart_money.warnings,
            "metrics": smart_money.metrics,
        }
        raw["market_intelligence"] = {
            "market_score": market.market_score,
            "momentum_score": market.momentum_score,
            "volume_score": market.volume_score,
            "pressure_score": market.pressure_score,
            "liquidity_score": market.liquidity_score,
            "structure_score": market.structure_score,
            "breakout_score": market.breakout_score,
            "market_state": market.market_state,
            "momentum_state": market.momentum_state,
            "volume_state": market.volume_state,
            "pressure_state": market.pressure_state,
            "breakout_state": market.breakout_state,
            "reasons": market.reasons,
            "warnings": market.warnings,
            "metrics": market.metrics,
        }
        raw["manipulation"] = {
            "score": manipulation.manipulation_score,
            "risk_level": manipulation.risk_level,
            "suspicious": manipulation.suspicious,
            "severe": manipulation.severe,
            "signals": [
                {
                    "name": signal.name,
                    "severity": signal.severity,
                    "score": signal.score,
                    "description": signal.description,
                    "evidence": signal.evidence,
                }
                for signal in manipulation.signals
            ],
            "reasons": manipulation.reasons,
            "warnings": manipulation.warnings,
            "limitations": manipulation.limitations,
        }

        return DeepIntelligenceResult(
            mint=candidate.mint,
            token_name=analysis.get("token_name", candidate.name),
            token_symbol=analysis.get("token_symbol", candidate.symbol),
            security_score=round(security_score, 1),
            rank_score=round(ranked.rank_score, 1),
            discovery_score=round(candidate.discovery_score, 1),
            data_quality=round(_number(analysis.get("data_quality"), 0.0), 1),
            smart_money_score=round(smart_money.score, 1),
            smart_money_confidence=round(smart_money.confidence, 1),
            smart_money_state=smart_money.state,
            market_score=round(market.market_score, 1),
            momentum_score=round(market.momentum_score, 1),
            volume_score=round(market.volume_score, 1),
            pressure_score=round(market.pressure_score, 1),
            liquidity_score=round(market.liquidity_score, 1),
            structure_score=round(market.structure_score, 1),
            breakout_score=round(market.breakout_score, 1),
            market_state=market.market_state,
            manipulation_score=round(manipulation.manipulation_score, 1),
            manipulation_risk_level=manipulation.risk_level,
            manipulation_suspicious=manipulation.suspicious,
            manipulation_severe=manipulation.severe,
            opportunity_score=opportunity_score,
            confidence=confidence,
            decision=decision,
            trade_action=decision,
            trade_horizon=horizon,
            trade_setup=setup,
            risk_label=str(analysis.get("risk_label", "UNKNOWN")),
            risk_description=str(analysis.get("risk_desc", "")),
            criteria=criteria,
            reasons=reasons,
            warnings=warnings,
            blockers=blockers,
            raw_analysis=raw,
        )

    async def analyze_candidate(
        self,
        candidate: Candidate,
        ranked: RankedOpportunity,
        security_decision: SecurityDecision,
    ) -> DeepIntelligenceResult:
        if ranked.candidate.mint != candidate.mint:
            raise ValueError("Candidate and RankedOpportunity mint addresses do not match")
        return await self.analyze(
            ranked=ranked,
            security_decision=security_decision,
        )

    async def analyze_many(
        self,
        opportunities: list[tuple[RankedOpportunity, SecurityDecision]],
        limit: Optional[int] = None,
    ) -> list[DeepIntelligenceResult]:
        selected = opportunities if limit is None else opportunities[:max(0, limit)]
        results: list[DeepIntelligenceResult] = []

        for ranked, security_decision in selected:
            if not security_decision.passed:
                results.append(
                    self._blocked_result(
                        ranked.candidate,
                        ranked,
                        security_decision,
                    )
                )
                continue
            try:
                results.append(
                    await self.analyze(
                        ranked=ranked,
                        security_decision=security_decision,
                    )
                )
            except Exception as exc:
                logger.exception(
                    "Deep analysis failed for %s: %s",
                    ranked.candidate.mint,
                    exc,
                )
        return results

    def actionable(
        self,
        results: list[DeepIntelligenceResult],
    ) -> list[DeepIntelligenceResult]:
        return [result for result in results if result.is_actionable]

    @staticmethod
    def _blocked_result(
        candidate: Candidate,
        ranked: RankedOpportunity,
        security_decision: SecurityDecision,
    ) -> DeepIntelligenceResult:
        return DeepIntelligenceResult(
            mint=candidate.mint,
            token_name=candidate.name,
            token_symbol=candidate.symbol,
            security_score=0.0,
            rank_score=round(ranked.rank_score, 1),
            discovery_score=round(candidate.discovery_score, 1),
            data_quality=0.0,
            smart_money_score=0.0,
            smart_money_confidence=0.0,
            smart_money_state="NOT_ANALYZED",
            market_score=0.0,
            momentum_score=0.0,
            volume_score=0.0,
            pressure_score=0.0,
            liquidity_score=0.0,
            structure_score=0.0,
            breakout_score=0.0,
            market_state="NOT_ANALYZED",
            manipulation_score=0.0,
            manipulation_risk_level="NOT_ANALYZED",
            manipulation_suspicious=False,
            manipulation_severe=False,
            opportunity_score=0.0,
            confidence=0.0,
            decision=AVOID,
            trade_action=AVOID,
            trade_horizon="غير متاح",
            trade_setup="SECURITY_BLOCKED",
            risk_label="BLOCKED",
            risk_description=(
                "تم حجب المرشح قبل التحليل العميق بسبب فشل بوابة الأمان الأولية."
            ),
            criteria={},
            reasons=[],
            warnings=list(getattr(security_decision, "warnings", []) or []),
            blockers=list(getattr(security_decision, "reasons", []) or []),
            raw_analysis={
                "short_term_decision": {
                    "action": AVOID,
                    "horizon": "غير متاح",
                    "setup": "SECURITY_BLOCKED",
                }
            },
        )


deep_intelligence = DeepIntelligenceEngine()
