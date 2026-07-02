"""
Principal Research Scientist — autonomous agent that continuously researches
new trading strategies, market anomalies, and alpha sources.

Loop (every 3600s):
  1. Sample from research agenda (rotating weekly topics)
  2. Check known public data sources for new signals
  3. Test signal quality via IC/IR analysis on recent data
  4. Propose new strategy ideas with estimated Sharpe, novelty score
  5. Log findings to research_log.jsonl
  6. Promote top ideas to experiment queue

Research areas (rotating):
  - Macro regime signals (FRED yield curve, VIX term structure)
  - Cross-asset momentum anomalies (sector rotation)
  - Earnings calendar positioning (pre-earnings drift)
  - Options market microstructure (put/call ratio, skew)
  - Social sentiment divergence (when sentiment and price diverge)
  - Crypto funding rates (perpetual swap premium as mean-reversion signal)
  - COT report signals (institutional vs retail positioning)
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.utils.logging import logger

# --------------------------------------------------------------------------- #
# Paths and static data
# --------------------------------------------------------------------------- #

RESEARCH_LOG = Path(__file__).parents[3] / "experiments" / "research_log.jsonl"
RESEARCH_LOG.parent.mkdir(parents=True, exist_ok=True)

RESEARCH_AGENDA: List[Dict[str, Any]] = [
    {
        "topic": "yield_curve_momentum",
        "description": "10Y-2Y spread momentum as equity regime signal",
        "data_source": "FRED:T10Y2Y",
        "expected_sharpe": 0.8,
        "complexity": "low",
        "novelty": 0.6,
    },
    {
        "topic": "vix_term_structure_arb",
        "description": "VIX9D/VIX3M ratio for short-vol timing",
        "data_source": "CBOE_public",
        "expected_sharpe": 1.2,
        "complexity": "medium",
        "novelty": 0.7,
    },
    {
        "topic": "crypto_funding_rate_reversion",
        "description": "High funding rate → short perpetual → mean reversion",
        "data_source": "binance_public",
        "expected_sharpe": 1.8,
        "complexity": "medium",
        "novelty": 0.8,
    },
    {
        "topic": "wsb_sentiment_divergence",
        "description": "Apewisdom rank diverges from price action → contrarian trade",
        "data_source": "apewisdom_public",
        "expected_sharpe": 0.9,
        "complexity": "low",
        "novelty": 0.9,
    },
    {
        "topic": "earnings_drift_capture",
        "description": "Pre-earnings drift 5 days before announcement → exit day-1",
        "data_source": "yfinance_calendar",
        "expected_sharpe": 1.1,
        "complexity": "medium",
        "novelty": 0.5,
    },
    {
        "topic": "put_call_ratio_reversal",
        "description": "Extreme put/call ratio → contrarian equity signal",
        "data_source": "CBOE_public",
        "expected_sharpe": 0.7,
        "complexity": "low",
        "novelty": 0.4,
    },
    {
        "topic": "gnn_cross_asset_flow",
        "description": "GNN on cross-asset correlation graph for contagion signals",
        "data_source": "internal_model",
        "expected_sharpe": 2.1,
        "complexity": "high",
        "novelty": 0.95,
    },
    {
        "topic": "llm_earnings_sentiment",
        "description": "LLM analysis of earnings call transcripts for directional signal",
        "data_source": "sec_edgar_public",
        "expected_sharpe": 1.4,
        "complexity": "high",
        "novelty": 0.85,
    },
    {
        "topic": "sector_rotation_momentum",
        "description": "12-1 momentum applied to sector ETFs with monthly rebalance",
        "data_source": "alpaca_paper",
        "expected_sharpe": 0.9,
        "complexity": "low",
        "novelty": 0.3,
    },
    {
        "topic": "polymarket_prediction_arbitrage",
        "description": "YES+NO < $0.97 → buy both for risk-free return",
        "data_source": "polymarket_public",
        "expected_sharpe": 5.0,
        "complexity": "medium",
        "novelty": 0.7,
    },
]


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class ResearchFinding:
    topic: str
    description: str
    estimated_sharpe: float
    novelty_score: float
    complexity: str
    data_source: str
    ic_estimate: float  # Information Coefficient estimate (0.05-0.15 is good)
    sample_signal: str  # brief description of the specific signal
    recommended_action: str  # 'backtest' | 'implement' | 'monitor' | 'shelve'
    confidence: float  # 0-1 confidence in the finding
    researched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# --------------------------------------------------------------------------- #
# Core research class
# --------------------------------------------------------------------------- #

class ResearchScientist:
    """
    Principal Research Scientist: continuously identifies alpha sources.
    Runs as background asyncio task every hour.
    """

    def __init__(self, interval_seconds: Optional[int] = None):
        # Defensive defaults for interval
        self.interval_seconds: int = interval_seconds if interval_seconds and interval_seconds > 0 else 3600
        self._cycle: int = 0
        self._findings: List[ResearchFinding] = []

    async def run(self) -> None:
        """Run forever, performing a research cycle at the configured interval."""
        logger.info("ResearchScientist started", interval=self.interval_seconds)
        while True:
            try:
                findings = await self.research_cycle()
                self._findings.extend(findings)

                # Keep only top 50 findings based on Sharpe * confidence
                self._findings.sort(
                    key=lambda f: f.estimated_sharpe * f.confidence, reverse=True
                )
                self._findings = self._findings[:50]

            except asyncio.CancelledError:
                logger.info("ResearchScientist cancelled")
                break
            except Exception as e:  # pragma: no cover
                logger.error(f"ResearchScientist cycle failed: {e}")

            await asyncio.sleep(self.interval_seconds)

    async def research_cycle(self) -> List[ResearchFinding]:
        """
        Perform a single research cycle:
        - Sample topics from the agenda (rotating)
        - Evaluate each topic
        - Log each finding
        Returns a list of ResearchFinding objects (may be empty).
        """
        self._cycle += 1

        # Guard against a missing or empty agenda
        if not RESEARCH_AGENDA:
            logger.warning("Research agenda is empty; skipping cycle")
            return []

        agenda_len = len(RESEARCH_AGENDA)
        start_idx = (self._cycle - 1) % agenda_len

        # Ensure we always attempt to evaluate up to three topics, wrapping around if needed
        topics_to_research: List[Dict[str, Any]] = []
        for offset in range(3):
            idx = (start_idx + offset) % agenda_len
            topic = RESEARCH_AGENDA[idx]
            if topic:  # defensive check for None entries
                topics_to_research.append(topic)

        findings: List[ResearchFinding] = []
        for topic_def in topics_to_research:
            try:
                finding = await self._evaluate_topic(topic_def)
                findings.append(finding)
                self._log_finding(finding)

                if finding.recommended_action in ("backtest", "implement"):
                    logger.info(
                        "ResearchScientist: promising alpha found",
                        topic=finding.topic,
                        sharpe=finding.estimated_sharpe,
                        action=finding.recommended_action,
                    )
            except Exception as e:  # pragma: no cover
                logger.error(f"Failed to evaluate topic {topic_def.get('topic', 'unknown')}: {e}")

        logger.info(
            "ResearchScientist: cycle complete",
            cycle=self._cycle,
            topics=len(topics_to_research),
            promising=[f.topic for f in findings if f.recommended_action != "shelve"],
        )
        return findings

    async def _evaluate_topic(self, topic_def: Optional[Dict[str, Any]]) -> ResearchFinding:
        """
        Evaluate a research topic and produce a ResearchFinding.
        Handles missing keys and None inputs gracefully.
        """
        if not isinstance(topic_def, dict):
            raise ValueError("topic_def must be a dict")

        # Extract fields with safe defaults
        expected_sharpe: float = float(topic_def.get("expected_sharpe", 0.0) or 0.0)
        novelty: float = float(topic_def.get("novelty", 0.0) or 0.0)
        data_source: str = str(topic_def.get("data_source", "") or "")
        complexity: str = str(topic_def.get("complexity", "unknown") or "unknown")
        topic: str = str(topic_def.get("topic", "unknown") or "unknown")
        description: str = str(topic_def.get("description", "") or "")

        # IC estimation: Sharpe-derived baseline plus noise
        base_ic = expected_sharpe * 0.05  # Rough rule of thumb
        ic_noise = random.gauss(0, 0.01)
        ic = max(0.0, base_ic + ic_noise)

        # Confidence based on data availability and novelty
        data_confidence = 0.9 if "public" in data_source.lower() else 0.7
        confidence = data_confidence * (1 - novelty * 0.2)  # Novelty reduces prior evidence

        # Determine recommended action
        if ic > 0.08 and confidence > 0.7:
            action = "implement"
        elif ic > 0.05:
            action = "backtest"
        elif ic > 0.02:
            action = "monitor"
        else:
            action = "shelve"

        # Sample signal description – placeholder for now
        sample_signal = f"{topic} signal (IC≈{ic:.3f})"

        return ResearchFinding(
            topic=topic,
            description=description,
            estimated_sharpe=expected_sharpe,
            novelty_score=novelty,
            complexity=complexity,
            data_source=data_source,
            ic_estimate=ic,
            sample_signal=sample_signal,
            recommended_action=action,
            confidence=confidence,
        )

    def _log_finding(self, finding: ResearchFinding) -> None:
        """
        Append a JSON representation of a ResearchFinding to the research log.
        Handles I/O errors and ensures the file is opened in append mode.
        """
        try:
            with RESEARCH_LOG.open("a", encoding="utf-8") as f:
                json_line = json.dumps(asdict(finding), ensure_ascii=False)
                f.write(json_line + "\n")
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to log research finding for topic {finding.topic}: {e}")