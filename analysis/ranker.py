"""
Opportunity Ranking Engine.

Ranks discovered Solana meme-token candidates before deep analysis.

Important:
- This is a ranking layer, not a trading engine.
- It does not execute transactions.
- A high ranking is NOT a guarantee of profit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable, Optional

from analysis.discovery import Candidate

logger = logging.getLogger(__name__)


@dataclass
class RankedOpportunity:
    """A candidate enriched with ranking information."""

    candidate: Candidate
    rank_score: float = 0.0
    priority: str = "LOW"
    reasons: list[str] | None = None
    warnings: list[str] | None = None

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = []
        if self.warnings is None:
            self.warnings = []


def _safe_float(value: Optional[float]) -> float:
    if value is None:
        return 0.0

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(value, maximum))


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0

    return _clamp((value - low) / (high - low) * 100.0)


def _freshness_score(discovered_at: float) -> float:
    """
    Prefer recently discovered candidates.

    Freshness decays gradually rather than eliminating older candidates
    immediately.
    """

    age_seconds = max(0.0, time.time() - discovered_at)

    if age_seconds <= 60:
        return 100.0

    if age_seconds <= 5 * 60:
        return 90.0

    if age_seconds <= 15 * 60:
        return 75.0

    if age_seconds <= 30 * 60:
        return 60.0

    if age_seconds <= 60 * 60:
        return 40.0

    return 20.0


def _liquidity_score(liquidity_usd: Optional[float]) -> float:
    liquidity = _safe_float(liquidity_usd)

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

    return 100.0


def _volume_score(volume_24h: Optional[float]) -> float:
    volume = _safe_float(volume_24h)

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

    return 100.0


def _momentum_score(candidate: Candidate) -> float:
    """
    Evaluate momentum using several time windows.

    Moderate positive momentum is preferred over extreme vertical moves.
    """

    change_5m = _safe_float(candidate.price_change_5m)
    change_1h = _safe_float(candidate.price_change_1h)
    change_6h = _safe_float(candidate.price_change_6h)

    score = 0.0

    # 5-minute momentum.
    if 0 < change_5m <= 5:
        score += 25
    elif 5 < change_5m <= 12:
        score += 20
    elif 12 < change_5m <= 25:
        score += 10
    elif change_5m > 25:
        score += 3
    elif change_5m < -15:
        score -= 10

    # 1-hour momentum.
    if 2 <= change_1h <= 15:
        score += 30
    elif 15 < change_1h <= 30:
        score += 20
    elif 30 < change_1h <= 60:
        score += 8
    elif change_1h > 60:
        score += 2
    elif change_1h < -20:
        score -= 10

    # 6-hour trend.
    if 0 < change_6h <= 30:
        score += 25
    elif 30 < change_6h <= 80:
        score += 15
    elif change_6h > 80:
        score += 5
    elif change_6h < -30:
        score -= 10

    return _clamp(score, 0.0, 100.0)


def _buy_pressure_score(candidate: Candidate) -> float:
    buys = max(0, int(candidate.txns_buys or 0))
    sells = max(0, int(candidate.txns_sells or 0))

    total = buys + sells

    if total <= 0:
        return 0.0

    buy_ratio = buys / total

    if 0.55 <= buy_ratio <= 0.70:
        return 100.0

    if 0.70 < buy_ratio <= 0.80:
        return 85.0

    if 0.45 <= buy_ratio < 0.55:
        return 60.0

    if 0.35 <= buy_ratio < 0.45:
        return 35.0

    if buy_ratio > 0.80:
        return 25.0

    return 15.0


def _market_activity_score(candidate: Candidate) -> float:
    """
    Estimate market activity from volume and transaction counts.

    This is intentionally not treated as proof of organic activity.
    The manipulation engine will perform deeper validation later.
    """

    volume_score = _volume_score(candidate.volume_24h)

    buys = max(0, int(candidate.txns_buys or 0))
    sells = max(0, int(candidate.txns_sells or 0))
    transactions = buys + sells

    if transactions >= 2_000:
        transaction_score = 100.0
    elif transactions >= 1_000:
        transaction_score = 85.0
    elif transactions >= 500:
        transaction_score = 70.0
    elif transactions >= 100:
        transaction_score = 50.0
    elif transactions > 0:
        transaction_score = 25.0
    else:
        transaction_score = 0.0

    return (volume_score * 0.6) + (transaction_score * 0.4)


def _discovery_source_score(candidate: Candidate) -> float:
    source = (candidate.source or "").lower()

    if "pumpfun" in source and "dexscreener" in source:
        return 100.0

    if "pumpfun" in source:
        return 90.0

    if "dexscreener" in source:
        return 70.0

    return 40.0


def _build_reasons(
    candidate: Candidate,
    scores: dict[str, float],
) -> list[str]:
    reasons: list[str] = []

    if scores["freshness"] >= 90:
        reasons.append("اكتشاف حديث جدًا")

    if scores["liquidity"] >= 80:
        reasons.append("سيولة قوية نسبيًا")

    if scores["volume"] >= 80:
        reasons.append("نشاط تداول مرتفع")

    if scores["momentum"] >= 70:
        reasons.append("زخم متعدد الأطر الزمنية")

    if scores["pressure"] >= 80:
        reasons.append("ضغط شراء إيجابي")

    if scores["activity"] >= 75:
        reasons.append("نشاط سوق مرتفع")

    if scores["source"] >= 90:
        reasons.append("تم تأكيد الاكتشاف من أكثر من مصدر")

    if candidate.reasons:
        reasons.extend(candidate.reasons[:3])

    return list(dict.fromkeys(reasons))


def _build_warnings(
    candidate: Candidate,
    scores: dict[str, float],
) -> list[str]:
    warnings: list[str] = []

    liquidity = _safe_float(candidate.liquidity_usd)
    change_5m = _safe_float(candidate.price_change_5m)
    change_1h = _safe_float(candidate.price_change_1h)

    if liquidity < 20_000:
        warnings.append("السيولة منخفضة — خطر الانزلاق السعري مرتفع")

    if change_5m > 25:
        warnings.append("ارتفاع سريع جدًا خلال 5 دقائق")

    if change_1h > 60:
        warnings.append("ارتفاع حاد خلال ساعة — احتمال مطاردة السعر")

    if scores["pressure"] <= 25:
        warnings.append("ضغط الشراء غير مقنع")

    if scores["volume"] < 30:
        warnings.append("حجم التداول منخفض")

    if not candidate.pair_url:
        warnings.append("لا يوجد رابط زوج تداول متاح حاليًا")

    return list(dict.fromkeys(warnings))


def _priority(score: float) -> str:
    if score >= 80:
        return "VERY_HIGH"

    if score >= 65:
        return "HIGH"

    if score >= 50:
        return "MEDIUM"

    if score >= 35:
        return "LOW"

    return "VERY_LOW"


class OpportunityRanker:
    """
    Ranks discovered candidates for downstream deep analysis.

    The ranker is deliberately conservative:
    it identifies candidates worth investigating; it does not declare
    them safe or profitable.
    """

    def __init__(self, minimum_score: float = 35.0):
        self.minimum_score = _clamp(minimum_score)

    def rank(self, candidate: Candidate) -> RankedOpportunity:
        freshness = _freshness_score(candidate.discovered_at)
        liquidity = _liquidity_score(candidate.liquidity_usd)
        volume = _volume_score(candidate.volume_24h)
        momentum = _momentum_score(candidate)
        pressure = _buy_pressure_score(candidate)
        activity = _market_activity_score(candidate)
        source = _discovery_source_score(candidate)

        scores = {
            "freshness": freshness,
            "liquidity": liquidity,
            "volume": volume,
            "momentum": momentum,
            "pressure": pressure,
            "activity": activity,
            "source": source,
        }

        # Ranking weights.
        #
        # Freshness matters for opportunity discovery, but liquidity,
        # market activity and momentum receive substantial weight.
        score = (
            freshness * 0.15
            + liquidity * 0.20
            + volume * 0.15
            + momentum * 0.20
            + pressure * 0.10
            + activity * 0.10
            + source * 0.10
        )

        score = round(_clamp(score), 1)

        reasons = _build_reasons(candidate, scores)
        warnings = _build_warnings(candidate, scores)

        result = RankedOpportunity(
            candidate=candidate,
            rank_score=score,
            priority=_priority(score),
            reasons=reasons,
            warnings=warnings,
        )

        logger.debug(
            "Ranked candidate %s score=%.1f priority=%s",
            candidate.mint[:8],
            score,
            result.priority,
        )

        return result

    def rank_many(
        self,
        candidates: Iterable[Candidate],
        limit: Optional[int] = None,
    ) -> list[RankedOpportunity]:
        ranked = [self.rank(candidate) for candidate in candidates]

        ranked.sort(
            key=lambda item: (
                item.rank_score,
                item.candidate.discovery_score,
                item.candidate.discovered_at,
            ),
            reverse=True,
        )

        # Candidates below the minimum ranking threshold are still allowed
        # to exist, but they are not prioritized for deep analysis.
        ranked = [
            item
            for item in ranked
            if item.rank_score >= self.minimum_score
        ]

        if limit is not None:
            ranked = ranked[: max(0, limit)]

        return ranked

    def top(
        self,
        candidates: Iterable[Candidate],
        limit: int = 10,
    ) -> list[RankedOpportunity]:
        return self.rank_many(candidates, limit=limit)


opportunity_ranker = OpportunityRanker()
