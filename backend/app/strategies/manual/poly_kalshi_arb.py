"""Kalshi ↔ Polymarket cross-platform arbitrage strategy."""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone, timedelta

import pandas as pd

try:
    import httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

KALSHI_BASE = "https://trading-api.kalshi.com/trade-api/v2"
CLOB_BASE = "https://clob.polymarket.com"


def _normalize(text: str) -> set[str]:
    """Lowercase, strip punctuation, return word set."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return set(text.split())


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two word sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


class PolyKalshiArb(AbstractStrategy):
    """
    Cross-platform arbitrage between Kalshi and Polymarket.

    Finds matched event markets on both platforms and signals
    when the price discrepancy (after fees) exceeds MIN_EDGE.

    NOTE: Kalshi side requires manual execution — this strategy
    signals the opportunity but only auto-executes on Polymarket.
    """

    name = "poly_kalshi_arb"
    display_name = "Kalshi↔Polymarket Cross-Platform Arb"
    market_type = "polymarket"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 300.0

    # Configurable parameters with defaults
    MIN_EDGE: float = 0.03          # 3% net edge required after fees
    TAKER_FEE_EACH: float = 0.002   # 0.2% taker fee per side (Polymarket CLOB v2)
    MATCH_THRESHOLD: float = 0.5    # Jaccard similarity threshold
    MAX_DAYS_TO_CLOSE: int = 30     # Only match markets closing within 30 days

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        p = params or {}
        self.MIN_EDGE = float(p.get("min_edge", self.MIN_EDGE))
        self.TAKER_FEE_EACH = float(p.get("taker_fee_each", self.TAKER_FEE_EACH))
        self.MATCH_THRESHOLD = float(p.get("match_threshold", self.MATCH_THRESHOLD))
        self.MAX_DAYS_TO_CLOSE = int(p.get("max_days_to_close", self.MAX_DAYS_TO_CLOSE))

    def description(self) -> str:
        return (
            f"Cross-platform arbitrage between Kalshi and Polymarket. "
            f"Signals when net edge > {self.MIN_EDGE * 100:.1f}% after "
            f"{self.TAKER_FEE_EACH * 100:.1f}% taker fees per side. "
            "Kalshi execution is manual; Polymarket side executes automatically."
        )

    async def _fetch_kalshi_markets(self) -> list[dict]:
        """Fetch open Kalshi markets. Prices are in cents (0-100)."""
        if not _HTTPX:
            return []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    f"{KALSHI_BASE}/markets",
                    params={"limit": 100, "status": "open"},
                )
                r.raise_for_status()
                return r.json().get("markets", [])
        except Exception:
            return []

    async def _fetch_poly_markets(self) -> list[dict]:
        """Fetch active Polymarket CLOB markets. Prices are already 0-1."""
        if not _HTTPX:
            return []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    f"{CLOB_BASE}/markets",
                    params={"limit": 100, "active": "true"},
                )
                r.raise_for_status()
                return r.json().get("data", [])
        except Exception:
            return []

    def _days_until_close(self, close_time_str: str) -> float:
        """Return days until market closes. Returns 9999 on parse error."""
        try:
            close_dt = datetime.fromisoformat(
                close_time_str.replace("Z", "+00:00")
            )
            now = datetime.now(timezone.utc)
            return (close_dt - now).total_seconds() / 86400.0
        except Exception:
            return 9999.0

    def _match_markets(
        self, kalshi_markets: list[dict], poly_markets: list[dict]
    ) -> list[tuple[dict, dict, float]]:
        """
        Find matched (kalshi, poly, similarity) pairs using Jaccard similarity
        on normalized market titles.

        Only returns matches where:
        - Jaccard similarity >= MATCH_THRESHOLD
        - Both markets close within MAX_DAYS_TO_CLOSE days
        """
        matches: list[tuple[dict, dict, float]] = []
        now = datetime.now(timezone.utc)
        cutoff_dt = now + timedelta(days=self.MAX_DAYS_TO_CLOSE)

        for km in kalshi_markets:
            k_title = km.get("title", "") or km.get("ticker", "")
            k_close = km.get("close_time", "") or km.get("expiration_time", "")
            if not k_title:
                continue
            k_days = self._days_until_close(k_close) if k_close else 9999.0
            if k_days <= 0 or k_days > self.MAX_DAYS_TO_CLOSE:
                continue

            k_words = _normalize(k_title)

            for pm in poly_markets:
                p_title = pm.get("question", "") or pm.get("title", "")
                p_end = pm.get("end_date_iso", "") or pm.get("end_date", "")
                if not p_title:
                    continue
                p_days = self._days_until_close(p_end) if p_end else 9999.0
                if p_days <= 0 or p_days > self.MAX_DAYS_TO_CLOSE:
                    continue

                p_words = _normalize(p_title)
                sim = _jaccard(k_words, p_words)
                if sim >= self.MATCH_THRESHOLD:
                    matches.append((km, pm, sim))

        return matches

    def _find_arb(
        self, kalshi_market: dict, poly_market: dict
    ) -> dict | None:
        """
        Compute arbitrage edge between matched Kalshi and Polymarket markets.

        Returns arb opportunity dict if net_edge >= MIN_EDGE, else None.
        Kalshi prices are in cents (0-100); divide by 100 for probability.
        Polymarket prices are already 0-1.
        """
        # Kalshi mid price (cents → probability)
        yes_bid = kalshi_market.get("yes_bid")
        yes_ask = kalshi_market.get("yes_ask")
        if yes_bid is None or yes_ask is None:
            return None
        kalshi_mid = (float(yes_bid) + float(yes_ask)) / 2.0 / 100.0

        # Polymarket YES token price
        tokens = poly_market.get("tokens", [])
        poly_yes_price: float | None = None
        poly_token_id: str | None = None
        for token in tokens:
            if token.get("outcome", "").upper() == "YES":
                poly_yes_price = float(token.get("price", 0))
                poly_token_id = token.get("token_id")
                break

        if poly_yes_price is None:
            return None

        gross_edge = abs(kalshi_mid - poly_yes_price)
        net_edge = gross_edge - 2.0 * self.TAKER_FEE_EACH

        if net_edge < self.MIN_EDGE:
            return None

        # Determine direction: buy whichever side is cheaper
        if kalshi_mid < poly_yes_price:
            direction = "buy_kalshi"
        else:
            direction = "buy_poly"

        return {
            "kalshi_market_id": kalshi_market.get("id", ""),
            "kalshi_ticker": kalshi_market.get("ticker", ""),
            "kalshi_price": round(kalshi_mid, 4),
            "poly_market_id": poly_market.get("condition_id", ""),
            "poly_token_id": poly_token_id,
            "poly_price": round(poly_yes_price, 4),
            "gross_edge": round(gross_edge, 4),
            "net_edge_pct": round(net_edge, 4),
            "direction": direction,
            "note": "kalshi_side_requires_manual_execution",
        }

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Scan Kalshi and Polymarket for matched markets with arb opportunities.
        Returns Signal for the best (highest net_edge) opportunity found.
        data argument is not used — strategy fetches live CLOB data.
        """
        if not _HTTPX:
            return None

        # Fetch both markets concurrently
        kalshi_markets, poly_markets = await asyncio.gather(
            self._fetch_kalshi_markets(),
            self._fetch_poly_markets(),
        )

        if not kalshi_markets or not poly_markets:
            return None

        matched_pairs = self._match_markets(kalshi_markets, poly_markets)

        # Find all arb opportunities
        opportunities: list[dict] = []
        for km, pm, sim in matched_pairs:
            arb = self._find_arb(km, pm)
            if arb is not None:
                arb["match_similarity"] = round(sim, 4)
                arb["kalshi_title"] = km.get("title", km.get("ticker", ""))
                arb["poly_question"] = pm.get("question", "")
                opportunities.append(arb)

        if not opportunities:
            return None

        # Return Signal for best opportunity (highest net_edge)
        best = max(opportunities, key=lambda x: x["net_edge_pct"])
        net_edge = best["net_edge_pct"]
        poly_question = best.get("poly_question") or best.get("kalshi_title") or "POLY_KALSHI_ARB"

        return Signal(
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            symbol=poly_question,
            side="buy",
            confidence=min(net_edge / 0.10, 0.9),
            metadata=best,
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Live-only strategy — no historical cross-platform price data available.
        Returns all-false signals.
        """
        false_series = pd.Series(False, index=df.index)
        return BacktestSignals(
            entries=false_series,
            exits=false_series,
        )
