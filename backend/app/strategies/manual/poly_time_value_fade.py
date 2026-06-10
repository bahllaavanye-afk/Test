"""
Polymarket Time Value Fade.

Harvest time value on near-certain Polymarket outcomes.
Markets priced > 95% YES or < 5% YES haven't resolved yet.
Buy the near-certain side and harvest remaining time premium.
Only trade markets with > $10k open interest.
Uses Polymarket Gamma API (public, no auth): https://gamma-api.polymarket.com/markets
"""
from __future__ import annotations

import pandas as pd

try:
    import httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

GAMMA_BASE = "https://gamma-api.polymarket.com"
MIN_OPEN_INTEREST = 10_000  # $10k minimum open interest


class PolyTimeValueFadeStrategy(AbstractStrategy):
    """
    Harvest time value on near-certain Polymarket outcomes.
    Markets priced > 95% YES or < 5% YES haven't resolved yet.
    Buy the near-certain side and harvest remaining time premium.
    Only trade markets with > $10k open interest.
    Uses Polymarket Gamma API (public, no auth): https://gamma-api.polymarket.com/markets
    """

    name = "poly_time_value_fade"
    display_name = "Polymarket Time Value Fade"
    market_type = "polymarket"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 60.0  # poll every 60 seconds

    DEFAULT_PARAMS = {
        "yes_threshold": 0.95,   # buy YES when price > 95%
        "no_threshold": 0.05,    # buy NO when price < 5% (YES)
        "min_open_interest": MIN_OPEN_INTEREST,
        "max_position_usd": 300,
        "kelly_fraction": 0.20,
    }

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        self.yes_threshold: float = float(p["yes_threshold"])
        self.no_threshold: float = float(p["no_threshold"])
        self.min_open_interest: float = float(p["min_open_interest"])
        self.max_position_usd: float = float(p["max_position_usd"])
        self.kelly_fraction: float = float(p["kelly_fraction"])

    def description(self) -> str:
        return (
            f"Buy near-certain (YES>{self.yes_threshold:.0%} or YES<{self.no_threshold:.0%}) "
            f"Polymarket contracts to harvest time premium. "
            f"Min open interest: ${self.min_open_interest:,.0f}. "
            "Source: Gamma API."
        )

    async def _fetch_markets(self) -> list[dict]:
        if not _HTTPX:
            return []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    f"{GAMMA_BASE}/markets",
                    params={"active": "true", "closed": "false", "limit": 200},
                )
                r.raise_for_status()
                data = r.json()
                # Gamma API returns list directly or wrapped
                return data if isinstance(data, list) else data.get("markets", [])
        except Exception:
            return []

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Scan Gamma API for near-certain markets with adequate liquidity.
        data is not used directly — strategy fetches live Gamma data.
        """
        markets = await self._fetch_markets()

        for market in markets:
            # Filter by open interest
            open_interest = float(market.get("volumeNum", market.get("volume", 0)) or 0)
            if open_interest < self.min_open_interest:
                continue

            tokens = market.get("tokens", [])
            if not tokens:
                # Some Gamma responses embed yes/no directly
                yes_price = float(market.get("bestAsk", market.get("lastTradePrice", 0.5)) or 0.5)
            else:
                yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), None)
                if not yes_token:
                    continue
                yes_price = float(yes_token.get("price", 0.5) or 0.5)

            # Only trade near-certain outcomes
            near_yes = yes_price >= self.yes_threshold
            near_no = yes_price <= self.no_threshold

            if not (near_yes or near_no):
                continue

            if near_yes:
                side = "buy"
                trade_price = yes_price
                outcome_label = "YES"
                # Expected profit: (1.0 - price) / price
                expected_return = (1.0 - yes_price) / yes_price if yes_price < 1.0 else 0.0
            else:
                side = "buy"
                trade_price = 1.0 - yes_price  # NO price
                outcome_label = "NO"
                expected_return = (1.0 - trade_price) / trade_price if trade_price < 1.0 else 0.0

            # Confidence scales with distance from 50%
            certainty = abs(yes_price - 0.5) * 2  # 0→1 as price moves 50%→100%
            confidence = min(0.95, 0.70 + certainty * 0.25)

            return Signal(
                symbol=market.get("question", market.get("slug", "POLY_MARKET")),
                side=side,
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "market_id": market.get("id", market.get("conditionId", "")),
                    "outcome": outcome_label,
                    "yes_price": round(yes_price, 4),
                    "trade_price": round(trade_price, 4),
                    "open_interest": round(open_interest, 2),
                    "expected_return_pct": round(expected_return * 100, 2),
                    "arb_type": "time_value_fade",
                    "order_type": "limit",
                },
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Proxy backtest: enter when close (YES price proxy) is near-certain.
        In real use this strategy is live-only (open interest check requires API).
        Uses shift(1) to prevent lookahead bias.
        """
        false_series = pd.Series(False, index=df.index)
        if "close" not in df.columns or len(df) < 2:
            return BacktestSignals(entries=false_series, exits=false_series)

        close = df["close"].astype(float).shift(1)
        entries = ((close >= self.yes_threshold) | (close <= self.no_threshold)).fillna(False)
        exits = ((close >= 0.99) | (close <= 0.01)).fillna(False)

        return BacktestSignals(
            entries=entries.astype(bool),
            exits=exits.astype(bool),
        )
