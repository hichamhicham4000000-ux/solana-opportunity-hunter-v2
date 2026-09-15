"""
Opportunity Hunter Pipeline.

Coordinates the existing analytical layers:

Discovery
    ↓
Ranking
    ↓
Security Gate
    ↓
Deep Intelligence
    ↓
Final Opportunity Decision

This module does not execute trades.
It only coordinates analysis and returns structured results.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from analysis.discovery import (
    Candidate,
    OpportunityDiscovery,
    opportunity_discovery,
)

from analysis.ranker import (
    RankedOpportunity,
    OpportunityRanker,
    opportunity_ranker,
)

from analysis.security_gate import (
    SecurityDecision,
    SecurityGate,
    security_gate,
)

from analysis.deep_intelligence import (
    DeepIntelligenceResult,
    DeepIntelligenceEngine,
    deep_intelligence,
)

from analysis.engine import analyze_token


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------


@dataclass
class OpportunityPipelineResult:
    """Complete result of one Opportunity Hunter cycle."""

    candidates: list[Candidate] = field(
        default_factory=list
    )

    ranked: list[RankedOpportunity] = field(
        default_factory=list
    )

    security_results: list[
        tuple[
            RankedOpportunity,
            SecurityDecision,
        ]
    ] = field(
        default_factory=list
    )

    deep_results: list[
        DeepIntelligenceResult
    ] = field(
        default_factory=list
    )

    actionable: list[
        DeepIntelligenceResult
    ] = field(
        default_factory=list
    )

    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


class OpportunityPipeline:
    """
    Main coordinator for the Opportunity Hunter.

    Each analytical layer remains independent so individual
    components can be improved without rebuilding the system.
    """

    def __init__(
        self,
        discovery: Optional[
            OpportunityDiscovery
        ] = None,
        ranker: Optional[
            OpportunityRanker
        ] = None,
        gate: Optional[
            SecurityGate
        ] = None,
        deep_engine: Optional[
            DeepIntelligenceEngine
        ] = None,
    ) -> None:

        self.discovery = (
            discovery
            or opportunity_discovery
        )

        self.ranker = (
            ranker
            or opportunity_ranker
        )

        self.security_gate = (
            gate
            or security_gate
        )

        self.deep_engine = (
            deep_engine
            or deep_intelligence
        )

    # -----------------------------------------------------------------
    # Discovery
    # -----------------------------------------------------------------

    async def discover(
        self,
        limit: Optional[int] = None,
    ) -> list[Candidate]:

        candidates = (
            await self.discovery.discover_once()
        )

        if limit is not None:
            candidates = candidates[
                : max(0, limit)
            ]

        logger.info(
            "Opportunity pipeline discovered %s candidates",
            len(candidates),
        )

        return candidates

    # -----------------------------------------------------------------
    # Ranking
    # -----------------------------------------------------------------

    def rank(
        self,
        candidates: list[Candidate],
        limit: Optional[int] = None,
    ) -> list[RankedOpportunity]:

        ranked = self.ranker.rank_many(
            candidates,
            limit=limit,
        )

        logger.info(
            "Opportunity pipeline ranked %s candidates",
            len(ranked),
        )

        return ranked

    # -----------------------------------------------------------------
    # Security
    # -----------------------------------------------------------------

    def security_screen(
        self,
        ranked: list[RankedOpportunity],
    ) -> list[
        tuple[
            RankedOpportunity,
            SecurityDecision,
        ]
    ]:

        results = (
            self.security_gate.evaluate_many(
                ranked
            )
        )

        passed = sum(
            1
            for _ranked, decision
            in results
            if decision.passed
        )

        logger.info(
            "Security gate: %s/%s candidates passed",
            passed,
            len(results),
        )

        return results

    # -----------------------------------------------------------------
    # Deep Intelligence
    # -----------------------------------------------------------------

    async def deep_analyze(
        self,
        security_results: list[
            tuple[
                RankedOpportunity,
                SecurityDecision,
            ]
        ],
        limit: Optional[int] = None,
    ) -> list[DeepIntelligenceResult]:

        approved = [
            (
                ranked,
                decision,
            )
            for ranked, decision
            in security_results
            if decision.passed
        ]

        if limit is not None:
            approved = approved[
                : max(0, limit)
            ]

        logger.info(
            "Deep Intelligence queue: %s candidates",
            len(approved),
        )

        if not approved:
            return []

        return await self.deep_engine.analyze_many(
            approved
        )

    # -----------------------------------------------------------------
    # Full Opportunity Hunt
    # -----------------------------------------------------------------

    async def hunt(
        self,
        discovery_limit: Optional[int] = None,
        ranking_limit: Optional[int] = None,
        deep_limit: Optional[int] = None,
    ) -> OpportunityPipelineResult:

        started = time.monotonic()

        # -------------------------------------------------------------
        # 1. Discovery
        # -------------------------------------------------------------

        candidates = await self.discover(
            limit=discovery_limit
        )

        if not candidates:
            elapsed = (
                time.monotonic()
                - started
            )

            return OpportunityPipelineResult(
                candidates=[],
                ranked=[],
                security_results=[],
                deep_results=[],
                actionable=[],
                elapsed_seconds=round(
                    elapsed,
                    3,
                ),
            )

        # -------------------------------------------------------------
        # 2. Ranking
        # -------------------------------------------------------------

        ranked = self.rank(
            candidates,
            limit=ranking_limit,
        )

        if not ranked:
            elapsed = (
                time.monotonic()
                - started
            )

            return OpportunityPipelineResult(
                candidates=candidates,
                ranked=[],
                security_results=[],
                deep_results=[],
                actionable=[],
                elapsed_seconds=round(
                    elapsed,
                    3,
                ),
            )

        # -------------------------------------------------------------
        # 3. Security Gate
        # -------------------------------------------------------------

        security_results = (
            self.security_screen(
                ranked
            )
        )

        # -------------------------------------------------------------
        # 4. Deep Intelligence
        # -------------------------------------------------------------

        deep_results = (
            await self.deep_analyze(
                security_results,
                limit=deep_limit,
            )
        )

        # -------------------------------------------------------------
        # 5. Actionable Opportunities
        # -------------------------------------------------------------

        actionable = (
            self.deep_engine.actionable(
                deep_results
            )
        )

        elapsed = (
            time.monotonic()
            - started
        )

        result = OpportunityPipelineResult(
            candidates=candidates,
            ranked=ranked,
            security_results=security_results,
            deep_results=deep_results,
            actionable=actionable,
            elapsed_seconds=round(
                elapsed,
                3,
            ),
        )

        logger.info(
            "Opportunity hunt completed: "
            "discovered=%s ranked=%s deep=%s actionable=%s "
            "elapsed=%.2fs",
            len(result.candidates),
            len(result.ranked),
            len(result.deep_results),
            len(result.actionable),
            result.elapsed_seconds,
        )

        return result

    # -----------------------------------------------------------------
    # Single-token analytical pipeline
    # -----------------------------------------------------------------

    async def analyze_mint(
        self,
        mint: str,
    ) -> DeepIntelligenceResult:
        """
        Run the complete analytical pipeline for one known Mint.

        Used by the Discord /scan command.

        Important:
        The Candidate receives the complete market snapshot returned
        by analyze_token(), including short-term momentum and
        buy/sell activity.
        """

        mint = mint.strip()

        if not mint:
            raise ValueError(
                "Mint address cannot be empty."
            )

        # -------------------------------------------------------------
        # Existing token analysis
        # -------------------------------------------------------------

        analysis = await analyze_token(
            mint
        )

        # -------------------------------------------------------------
        # Pair information
        # -------------------------------------------------------------

        pair_address = analysis.get(
            "pair_address"
        )

        dex_id = analysis.get(
            "dex_id"
        )

        pair_url = analysis.get(
            "pair_url"
        )

        if not pair_url and pair_address:
            pair_url = (
                "https://dexscreener.com/"
                f"solana/{pair_address}"
            )

        # -------------------------------------------------------------
        # Candidate construction
        #
        # IMPORTANT:
        # These short-term market fields were previously omitted.
        # They are now explicitly transferred into Candidate.
        # -------------------------------------------------------------

        candidate = Candidate(
            mint=mint,

            source="manual_scan",

            name=analysis.get(
                "token_name",
                "Unknown",
            ),

            symbol=analysis.get(
                "token_symbol",
                "???",
            ),

            # Basic market data
            price_usd=analysis.get(
                "price_usd"
            ),

            liquidity_usd=analysis.get(
                "liquidity_usd"
            ),

            market_cap=analysis.get(
                "market_cap"
            ),

            volume_24h=analysis.get(
                "volume_24h"
            ),

            # ---------------------------------------------------------
            # Short-term price momentum
            # ---------------------------------------------------------

            price_change_5m=analysis.get(
                "price_change_5m"
            ),

            price_change_1h=analysis.get(
                "price_change_1h"
            ),

            price_change_6h=analysis.get(
                "price_change_6h"
            ),

            price_change_24h=analysis.get(
                "price_change_24h"
            ),

            # ---------------------------------------------------------
            # Buy / Sell activity
            # ---------------------------------------------------------

            txns_buys=(
                analysis.get(
                    "txns_24h_buys"
                )
                or 0
            ),

            txns_sells=(
                analysis.get(
                    "txns_24h_sells"
                )
                or 0
            ),

            maker_count=(
                analysis.get(
                    "maker_count"
                )
                or 0
            ),

            # ---------------------------------------------------------
            # Pair
            # ---------------------------------------------------------

            pair_url=pair_url,

            dex_id=dex_id,

            # Manual scans are not discovery events.
            # Keep a neutral discovery score.
            discovery_score=50.0,

            reasons=[
                "تحليل يدوي لعنوان Mint",
            ],
        )

        # -------------------------------------------------------------
        # Rank
        # -------------------------------------------------------------

        ranked = self.ranker.rank(
            candidate
        )

        # -------------------------------------------------------------
        # Security Gate
        # -------------------------------------------------------------

        security_decision = (
            self.security_gate.evaluate(
                candidate,
                ranked,
            )
        )

        # -------------------------------------------------------------
        # Deep Intelligence
        # -------------------------------------------------------------

        result = await self.deep_engine.analyze(
            ranked=ranked,
            security_decision=security_decision,
        )

        return result

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def best(
        results: list[
            DeepIntelligenceResult
        ],
        limit: int = 10,
    ) -> list[
        DeepIntelligenceResult
    ]:
        """
        Return the strongest analytical opportunities.

        Sort order:
            1. Opportunity score
            2. Confidence
            3. Security score
        """

        return sorted(
            results,
            key=lambda item: (
                item.opportunity_score,
                item.confidence,
                item.security_score,
            ),
            reverse=True,
        )[
            : max(0, limit)
        ]


# ---------------------------------------------------------------------
# Global pipeline instance
# ---------------------------------------------------------------------

opportunity_pipeline = OpportunityPipeline()
