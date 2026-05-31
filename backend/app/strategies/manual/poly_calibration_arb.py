"""Polymarket calibration arbitrage — compare vs Metaculus/Manifold forecasts."""
from __future__ import annotations
import asyncio
import re
import pandas as pd
try:
    import httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

CLOB_BASE = "https://clob.polymarket.com"
METACULUS_BASE = "https://www.metaculus.com/api2"
MANIFOLD_BASE = "https://manifold.markets/api/v0"


class PolymarketCalibrationArb(AbstractStrategy):
    """
    Compares Polymarket prices vs calibrated forecasters (Metaculus/Manifold).
    If |poly_price - forecaster_probability| > min_edge: trade the gap.

    Research: >7,000 mispriced Polymarket markets found in 86M bets.
    Kelly threshold: 2.5-3% edge required after fees.
    """

    name = "poly_calibration_arb"
    display_name = "Polymarket Calibration Arbitrage"
    market_type = "polymarket"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 600.0  # poll every 10 minutes

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        p = params or {}
        self.min_edge: float = float(p.get("min_edge", 0.10))
        self.max_position_usd: float = float(p.get("max_position_usd", 100.0))

    def description(self) -> str:
        return (
            f"Trade Polymarket vs Metaculus calibration gaps (edge >{self.min_edge * 100:.0f}%). "
            f"Max position: ${self.max_position_usd:.0f}. "
            "Source: Polymarket calibration research — >7,000 mispriced markets in 86M bets."
        )

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap (Jaccard) similarity between two strings."""
        words1 = set(re.findall(r"\w+", text1.lower()))
        words2 = set(re.findall(r"\w+", text2.lower()))
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    async def _fetch_poly_markets(self) -> list[dict]:
        if not _HTTPX:
            return []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    f"{CLOB_BASE}/markets",
                    params={"limit": 50, "active": "true"},
                )
                r.raise_for_status()
                return r.json().get("data", [])
        except Exception:
            return []

    async def _fetch_metaculus(self) -> list[dict]:
        if not _HTTPX:
            return []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    f"{METACULUS_BASE}/questions/",
                    params={
                        "order_by": "-activity",
                        "status": "open",
                        "type": "forecast",
                        "limit": 50,
                    },
                )
                r.raise_for_status()
                return r.json().get("results", [])
        except Exception:
            return []

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Scan Polymarket markets and compare prices against Metaculus community forecasts.
        data is not used directly — strategy fetches live CLOB and Metaculus data.
        """
        poly_markets, meta_questions = await asyncio.gather(
            self._fetch_poly_markets(),
            self._fetch_metaculus(),
        )

        for pm in poly_markets:
            poly_q = pm.get("question", "")
            tokens = pm.get("tokens", [])

            poly_price: float | None = None
            token_id: str = ""
            for t in tokens:
                if t.get("outcome", "").upper() == "YES":
                    poly_price = float(t.get("price", 0.5))
                    token_id = t.get("token_id", "")
                    break

            if poly_price is None:
                continue

            # Find the most similar Metaculus question via word-overlap
            best_match: dict | None = None
            best_sim: float = 0.0
            for mq in meta_questions:
                sim = self._keyword_similarity(poly_q, mq.get("title", ""))
                if sim > best_sim:
                    best_sim = sim
                    best_match = mq

            if best_match is None or best_sim < 0.3:
                continue

            # Extract Metaculus community median (q2 = 50th percentile)
            raw_prob = best_match.get("community_prediction", {})
            if isinstance(raw_prob, dict):
                raw_prob = raw_prob.get("full", {}).get("q2", None)
            if raw_prob is None:
                continue

            meta_prob = float(raw_prob)
            edge = meta_prob - poly_price

            if abs(edge) < self.min_edge:
                continue

            side = "buy" if edge > 0 else "sell"
            return Signal(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                symbol=poly_q[:60],
                side=side,
                confidence=min(abs(edge), 0.9),
                metadata={
                    "condition_id": pm.get("condition_id"),
                    "token_id": token_id,
                    "poly_price": round(poly_price, 4),
                    "metaculus_prob": round(meta_prob, 4),
                    "edge": round(edge, 4),
                    "similarity_score": round(best_sim, 3),
                    "max_position_usd": self.max_position_usd,
                    "order_type": "limit",
                },
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        This strategy is live-only (needs real-time cross-platform probability data).
        Return neutral signals for backtest compatibility.
        """
        empty = pd.Series(False, index=df.index)
        return BacktestSignals(entries=empty, exits=empty)
