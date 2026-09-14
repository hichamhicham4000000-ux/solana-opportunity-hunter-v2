"""
Deep Intelligence Engine.

Combines:
    Discovery
        -> Ranking
        -> Security Gate
        -> Full 8-criteria analysis
        -> Smart Money Intelligence
        -> Opportunity decision

This module is an analytical decision layer.

It does NOT:
- execute trades
- hold funds
- guarantee profits
- treat confidence as probability of profit
- treat Smart Money presence as proof of future profit

The existing analysis.engine remains responsible for the actual
on-chain/security criteria analysis.
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Decision labels
# ---------------------------------------------------------------------

STRONG_OPPORTUNITY = "STRONG OPPORTUNITY"
OPPORTUNITY = "OPPORTUNITY"
WATCH = "WATCH"
SPECULATIVE = "SPECULATIVE"
AVOID = "AVOID"
INVALIDATED = "INVALIDATED"


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class DeepIntelligenceResult:
    """
    Final analytical result for a discovered candidate.

    opportunity_score:
        Composite opportunity score from the available evidence.

    confidence:
        Confidence in the analysis quality/data completeness.
        It is NOT a probability of profit.

    smart_money_score:
        Supporting Smart Money signal from known-wallet presence
        and observed wallet activity.

    smart_money_confidence:
        Confidence in the Smart Money data quality.
    """

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

    opportunity_score: float
    confidence: float

    decision: str

    risk_label: str
    risk_description: str

    criteria: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )

    reasons: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    blockers: list[str] = field(
        default_factory=list
    )

    raw_analysis: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def is_actionable(self) -> bool:
        return self.decision in {
            STRONG_OPPORTUNITY,
            OPPORTUNITY,
        }


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _number(
    value: Any,
    default: float = 0.0,
) -> float:
    """Safely convert a value to float."""

    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:
    """Clamp a numeric value to a bounded range."""

    return max(
        minimum,
        min(value, maximum),
    )


def _criterion_score(
    analysis: dict[str, Any],
    criterion_name: str,
) -> Optional[float]:
    """
    Extract a criterion score from the existing engine result.

    The existing engine returns CriterionResult objects inside
    analysis["criteria"].
    """

    criteria = analysis.get("criteria") or {}

    result = criteria.get(criterion_name)

    if result is None:
        return None

    if hasattr(result, "score"):
        return _number(
            result.score,
            0.0,
        )

    if isinstance(result, dict):
        return _number(
            result.get("score"),
            0.0,
        )

    return None


def _criterion_details(
    analysis: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Convert CriterionResult objects into a safe serializable structure.
    """

    output: dict[str, dict[str, Any]] = {}

    criteria = analysis.get("criteria") or {}

    for name, result in criteria.items():

        if hasattr(result, "score"):
            output[name] = {
                "name": getattr(
                    result,
                    "name",
                    name,
                ),
                "score": _number(
                    getattr(
                        result,
                        "score",
                        0,
                    )
                ),
                "flags": list(
                    getattr(
                        result,
                        "flags",
                        [],
                    )
                    or []
                ),
                "details": dict(
                    getattr(
                        result,
                        "details",
                        {},
                    )
                    or {}
                ),
                "estimated": bool(
                    getattr(
                        result,
                        "estimated",
                        False,
                    )
                ),
            }

        elif isinstance(result, dict):
            output[name] = {
                "name": result.get(
                    "name",
                    name,
                ),
                "score": _number(
                    result.get(
                        "score",
                        0,
                    )
                ),
                "flags": list(
                    result.get(
                        "flags",
                        [],
                    )
                    or []
                ),
                "details": dict(
                    result.get(
                        "details",
                        {},
                    )
                    or {}
                ),
                "estimated": bool(
                    result.get(
                        "estimated",
                        False,
                    )
                ),
            }

    return output


def _extract_flags(
    analysis: dict[str, Any],
) -> list[str]:
    """Extract all analysis flags."""

    flags = analysis.get(
        "all_flags"
    ) or []

    return [
        str(flag)
        for flag in flags
        if flag
    ]


# ---------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------

def _calculate_confidence(
    analysis: dict[str, Any],
    security_decision: SecurityDecision,
    smart_money: Optional[
        SmartMoneyAssessment
    ] = None,
) -> float:
    """
    Estimate analytical confidence.

    This represents evidence/data quality,
    not expected return.

    Smart Money data quality is included as a supporting
    confidence component.
    """

    data_quality = _number(
        analysis.get(
            "data_quality"
        ),
        0.0,
    )

    estimated = (
        analysis.get(
            "estimated_criteria"
        )
        or []
    )

    criterion_count = 8

    if estimated:
        completeness = (
            max(
                0,
                criterion_count
                - len(estimated),
            )
            / criterion_count
            * 100
        )
    else:
        completeness = 100.0

    gate_quality = 100.0

    if (
        security_decision.status
        == "PASS_WITH_WARNINGS"
    ):
        gate_quality = 75.0

    elif not security_decision.passed:
        gate_quality = 30.0

    smart_money_quality = 50.0

    if smart_money is not None:
        smart_money_quality = _number(
            smart_money.confidence,
            0.0,
        )

    confidence = (
        data_quality * 0.50
        + completeness * 0.20
        + gate_quality * 0.15
        + smart_money_quality * 0.15
    )

    return round(
        _clamp(confidence),
        1,
    )


# ---------------------------------------------------------------------
# Opportunity score
# ---------------------------------------------------------------------

def _calculate_opportunity_score(
    analysis: dict[str, Any],
    ranked: RankedOpportunity,
    security_decision: SecurityDecision,
    smart_money: Optional[
        SmartMoneyAssessment
    ] = None,
) -> float:
    """
    Calculate the opportunity score.

    Important:
    This is NOT a price prediction.

    It combines:
        - deep safety/quality analysis
        - market ranking
        - discovery strength
        - data quality
        - security gate result
        - Smart Money intelligence

    Smart Money contributes 15%.

    A failed security gate always receives zero gate weight.
    Smart Money cannot override a failed security gate.
    """

    security_score = _number(
        analysis.get(
            "total_score"
        ),
        0.0,
    )

    rank_score = _number(
        ranked.rank_score,
        0.0,
    )

    discovery_score = _number(
        ranked.candidate.discovery_score,
        0.0,
    )

    data_quality = _number(
        analysis.get(
            "data_quality"
        ),
        0.0,
    )

    gate_score = 100.0

    if not security_decision.passed:
        gate_score = 0.0

    elif (
        security_decision.status
        == "PASS_WITH_WARNINGS"
    ):
        gate_score = 65.0

    # No Smart Money evidence is treated as neutral,
    # not as a positive signal.
    smart_money_score = 50.0

    if smart_money is not None:
        smart_money_score = _number(
            smart_money.score,
            0.0,
        )

    # Final weights:
    #
    # Security       35%
    # Ranking        20%
    # Discovery      10%
    # Data Quality   10%
    # Gate           10%
    # Smart Money    15%
    #
    # Total          100%
    score = (
        security_score * 0.35
        + rank_score * 0.20
        + discovery_score * 0.10
        + data_quality * 0.10
        + gate_score * 0.10
        + smart_money_score * 0.15
    )

    return round(
        _clamp(score),
        1,
    )


# ---------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------

def _decision_from_scores(
    opportunity_score: float,
    security_score: float,
    confidence: float,
    security_decision: SecurityDecision,
    candidate: Candidate,
) -> str:
    """
    Convert evidence into an analytical decision.

    Conservative thresholds intentionally prevent the system from
    calling weak or poorly verified tokens opportunities.
    """

    if not security_decision.passed:
        return AVOID

    if confidence < 45:
        return WATCH

    if security_score < 40:
        return SPECULATIVE

    # Extremely vertical moves are not automatically opportunities.
    change_5m = _number(
        candidate.price_change_5m
    )

    change_1h = _number(
        candidate.price_change_1h
    )

    if (
        change_5m > 50
        or change_1h > 150
    ):
        if opportunity_score < 80:
            return WATCH

    if (
        opportunity_score >= 82
        and security_score >= 75
    ):
        return STRONG_OPPORTUNITY

    if (
        opportunity_score >= 68
        and security_score >= 60
    ):
        return OPPORTUNITY

    if opportunity_score >= 50:
        return WATCH

    return SPECULATIVE


# ---------------------------------------------------------------------
# Reasons
# ---------------------------------------------------------------------

def _build_reasons(
    analysis: dict[str, Any],
    ranked: RankedOpportunity,
    opportunity_score: float,
    smart_money: Optional[
        SmartMoneyAssessment
    ] = None,
) -> list[str]:
    """Build human-readable reasons supporting the result."""

    reasons: list[str] = []

    security_score = _number(
        analysis.get(
            "total_score"
        ),
        0.0,
    )

    data_quality = _number(
        analysis.get(
            "data_quality"
        ),
        0.0,
    )

    if security_score >= 75:
        reasons.append(
            "نتيجة السلامة والتحقق الأساسي قوية"
        )

    elif security_score >= 60:
        reasons.append(
            "نتيجة السلامة مقبولة مع الحاجة إلى متابعة"
        )

    if ranked.rank_score >= 70:
        reasons.append(
            "المرشح يحتل ترتيبًا مرتفعًا ضمن فرص الاكتشاف"
        )

    if ranked.candidate.discovery_score >= 70:
        reasons.append(
            "إشارات الاكتشاف الأولية قوية نسبيًا"
        )

    if data_quality >= 80:
        reasons.append(
            "جودة البيانات مرتفعة"
        )

    if opportunity_score >= 80:
        reasons.append(
            "توافق جيد بين السلامة والترتيب وجودة البيانات"
        )

    if smart_money is not None:

        if smart_money.state == "STRONG":
            reasons.append(
                "إشارة Smart Money قوية نسبيًا"
            )

        elif smart_money.state == "POSITIVE":
            reasons.append(
                "إشارة Smart Money إيجابية"
            )

        elif smart_money.state == "NEUTRAL":
            reasons.append(
                "إشارة Smart Money محايدة"
            )

        elif smart_money.state == "WEAK":
            reasons.append(
                "إشارة Smart Money ضعيفة"
            )

        elif smart_money.state == "NONE":
            reasons.append(
                "لا توجد محفظة Smart Money معروفة ضمن البيانات المتاحة"
            )

    # Preserve the strongest reasons generated by
    # the ranking layer.
    reasons.extend(
        ranked.reasons[:4]
    )

    if smart_money is not None:
        reasons.extend(
            smart_money.reasons[:3]
        )

    return list(
        dict.fromkeys(reasons)
    )


# ---------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------

def _build_warnings(
    analysis: dict[str, Any],
    ranked: RankedOpportunity,
    security_decision: SecurityDecision,
    smart_money: Optional[
        SmartMoneyAssessment
    ] = None,
) -> list[str]:
    """Build human-readable warnings."""

    warnings: list[str] = []

    warnings.extend(
        security_decision.warnings[:6]
    )

    warnings.extend(
        ranked.warnings[:6]
    )

    data_quality = _number(
        analysis.get(
            "data_quality"
        ),
        0.0,
    )

    if data_quality < 70:
        warnings.append(
            "جودة البيانات أقل من المستوى المطلوب لثقة عالية"
        )

    estimated = (
        analysis.get(
            "estimated_criteria"
        )
        or []
    )

    if estimated:
        warnings.append(
            "بعض معايير التحليل اعتمدت على بيانات تقديرية أو غير مكتملة"
        )

    if smart_money is not None:

        warnings.extend(
            smart_money.warnings[:4]
        )

        if smart_money.confidence < 50:
            warnings.append(
                "ثقة Smart Money محدودة بسبب نقص بيانات النشاط"
            )

        warnings.append(
            "Smart Money إشارة مساعدة وليست ضمانًا للربح"
        )

    return list(
        dict.fromkeys(warnings)
    )


# ---------------------------------------------------------------------
# Blockers
# ---------------------------------------------------------------------

def _build_blockers(
    security_decision: SecurityDecision,
) -> list[str]:
    """Return security blockers."""

    if security_decision.passed:
        return []

    return list(
        dict.fromkeys(
            security_decision.reasons
        )
    )


# ---------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------

class DeepIntelligenceEngine:
    """
    Coordinates ranked opportunities with the existing full analysis
    engine and Smart Money Intelligence.

    The final decision remains conservative:
    security gate first, then evidence aggregation.
    """

    async def analyze(
        self,
        ranked: RankedOpportunity,
        security_decision: SecurityDecision,
        force: bool = False,
    ) -> DeepIntelligenceResult:

        candidate = ranked.candidate

        # A failed security gate should normally never reach
        # deep analysis.
        #
        # The force option exists for diagnostics/testing.
        if (
            not security_decision.passed
            and not force
        ):
            return self._blocked_result(
                candidate,
                ranked,
                security_decision,
            )

        logger.info(
            "Starting deep intelligence for %s (%s)",
            candidate.symbol,
            candidate.mint[:8],
        )

        # -------------------------------------------------------------
        # Existing deep analysis
        # -------------------------------------------------------------

        analysis = await analyze_token(
            candidate.mint
        )

        # -------------------------------------------------------------
        # Smart Money Intelligence
        # -------------------------------------------------------------

        smart_money = await assess_smart_money(
            candidate.mint
        )

        security_score = _number(
            analysis.get(
                "total_score"
            ),
            0.0,
        )

        confidence = _calculate_confidence(
            analysis,
            security_decision,
            smart_money,
        )

        opportunity_score = (
            _calculate_opportunity_score(
                analysis,
                ranked,
                security_decision,
                smart_money,
            )
        )

        decision = _decision_from_scores(
            opportunity_score=opportunity_score,
            security_score=security_score,
            confidence=confidence,
            security_decision=security_decision,
            candidate=candidate,
        )

        criteria = _criterion_details(
            analysis
        )

        reasons = _build_reasons(
            analysis,
            ranked,
            opportunity_score,
            smart_money,
        )

        warnings = _build_warnings(
            analysis,
            ranked,
            security_decision,
            smart_money,
        )

        blockers = _build_blockers(
            security_decision
        )

        # -------------------------------------------------------------
        # Result
        # -------------------------------------------------------------

        result = DeepIntelligenceResult(
            mint=candidate.mint,

            token_name=analysis.get(
                "token_name",
                candidate.name,
            ),

            token_symbol=analysis.get(
                "token_symbol",
                candidate.symbol,
            ),

            security_score=round(
                security_score,
                1,
            ),

            rank_score=round(
                ranked.rank_score,
                1,
            ),

            discovery_score=round(
                candidate.discovery_score,
                1,
            ),

            data_quality=round(
                _number(
                    analysis.get(
                        "data_quality"
                    ),
                    0.0,
                ),
                1,
            ),

            smart_money_score=round(
                smart_money.score,
                1,
            ),

            smart_money_confidence=round(
                smart_money.confidence,
                1,
            ),

            smart_money_state=(
                smart_money.state
            ),

            opportunity_score=(
                opportunity_score
            ),

            confidence=confidence,

            decision=decision,

            risk_label=str(
                analysis.get(
                    "risk_label",
                    "UNKNOWN",
                )
            ),

            risk_description=str(
                analysis.get(
                    "risk_desc",
                    "",
                )
            ),

            criteria=criteria,

            reasons=reasons,

            warnings=warnings,

            blockers=blockers,

            raw_analysis={
                **analysis,

                "smart_money": {
                    "score": (
                        smart_money.score
                    ),

                    "confidence": (
                        smart_money.confidence
                    ),

                    "state": (
                        smart_money.state
                    ),

                    "known_wallet_count": (
                        smart_money.known_wallet_count
                    ),

                    "active_known_wallet_count": (
                        smart_money.active_known_wallet_count
                    ),

                    "total_known_holder_amount": (
                        smart_money.total_known_holder_amount
                    ),

                    "holder_coverage": (
                        smart_money.holder_coverage
                    ),

                    "reasons": (
                        smart_money.reasons
                    ),

                    "warnings": (
                        smart_money.warnings
                    ),

                    "metrics": (
                        smart_money.metrics
                    ),
                },
            },
        )

        logger.info(
            "Deep intelligence completed: %s %s "
            "opportunity=%.1f confidence=%.1f "
            "smart_money=%.1f",
            result.token_symbol,
            result.decision,
            result.opportunity_score,
            result.confidence,
            result.smart_money_score,
        )

        return result

    # -----------------------------------------------------------------
    # Candidate wrapper
    # -----------------------------------------------------------------

    async def analyze_candidate(
        self,
        candidate: Candidate,
        ranked: RankedOpportunity,
        security_decision: SecurityDecision,
    ) -> DeepIntelligenceResult:
        """
        Convenience wrapper for callers that already have a candidate.
        """

        if (
            ranked.candidate.mint
            != candidate.mint
        ):
            raise ValueError(
                "Candidate and RankedOpportunity mint addresses do not match"
            )

        if (
            security_decision.mint
            != candidate.mint
        ):
            raise ValueError(
                "Candidate and SecurityDecision mint addresses do not match"
            )

        return await self.analyze(
            ranked=ranked,
            security_decision=security_decision,
        )

    # -----------------------------------------------------------------
    # Batch analysis
    # -----------------------------------------------------------------

    async def analyze_many(
        self,
        opportunities: list[
            tuple[
                RankedOpportunity,
                SecurityDecision,
            ]
        ],
        limit: Optional[int] = None,
    ) -> list[DeepIntelligenceResult]:
        """
        Analyze multiple candidates sequentially.

        Sequential execution is intentional here:
        the existing full engine already performs parallel API calls,
        while unrestricted outer concurrency could create unnecessary
        API pressure.
        """

        selected = opportunities

        if limit is not None:
            selected = opportunities[
                : max(0, limit)
            ]

        results: list[
            DeepIntelligenceResult
        ] = []

        for (
            ranked,
            security_decision,
        ) in selected:

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

                result = await self.analyze(
                    ranked=ranked,
                    security_decision=security_decision,
                )

                results.append(
                    result
                )

            except Exception as exc:

                logger.exception(
                    "Deep analysis failed for %s: %s",
                    ranked.candidate.mint,
                    exc,
                )

        return results

    # -----------------------------------------------------------------
    # Actionable filtering
    # -----------------------------------------------------------------

    def actionable(
        self,
        results: list[
            DeepIntelligenceResult
        ],
    ) -> list[
        DeepIntelligenceResult
    ]:
        """
        Return only analytical opportunity candidates.

        These results still require continuous monitoring and
        historical calibration before any future trading decision.
        """

        return [
            result
            for result in results
            if result.is_actionable
        ]

    # -----------------------------------------------------------------
    # Blocked result
    # -----------------------------------------------------------------

    @staticmethod
    def _blocked_result(
        candidate: Candidate,
        ranked: RankedOpportunity,
        security_decision: SecurityDecision,
    ) -> DeepIntelligenceResult:
        """
        Build a result for candidates blocked by the security gate.

        Smart Money is deliberately not analyzed for blocked tokens.
        """

        return DeepIntelligenceResult(
            mint=candidate.mint,

            token_name=candidate.name,

            token_symbol=candidate.symbol,

            security_score=0.0,

            rank_score=round(
                ranked.rank_score,
                1,
            ),

            discovery_score=round(
                candidate.discovery_score,
                1,
            ),

            data_quality=0.0,

            smart_money_score=0.0,

            smart_money_confidence=0.0,

            smart_money_state="NOT_ANALYZED",

            opportunity_score=0.0,

            confidence=100.0,

            decision=AVOID,

            risk_label="BLOCKED",

            risk_description=(
                "تم حجب المرشح قبل التحليل العميق "
                "بسبب فشل بوابة الأمان الأولية."
            ),

            criteria={},

            reasons=[],

            warnings=security_decision.warnings,

            blockers=security_decision.reasons,

            raw_analysis={},
        )


deep_intelligence = DeepIntelligenceEngine()
