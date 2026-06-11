"""
Polymarket Binary Arbitrage.

If YES_price + NO_price < $0.97 (accounting for fees), buy both sides.
At resolution, one side pays $1.00 — guaranteed profit.
"""
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class PolyBinaryArbStrategy(AbstractStrategy):
    name = "poly_binary_arb"
    display_name = "Polymarket Binary Arbitrage"
    market_type = "polymarket"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 5.0   # poll every 5 seconds

    DEFAULT_PARAMS = {
        "min_edge_pct": 3.0,
        "max_position_usd": 500,
        "kelly_fraction": 0.25,
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        effective = {**self.DEFAULT_PARAMS, **(params or {})}
        self.min_edge_pct = effective["min_edge_pct"]
        self.max_position_usd = effective["max_position_usd"]
        self.kelly_fraction = effective["kelly_fraction"]
        # max_sum derived from min_edge_pct: if edge >= 3%, YES+NO <= 0.97
        self.max_sum = 1.0 - self.min_edge_pct / 100.0
        self.min_liquidity = params.get("min_liquidity", 100) if params else 100  # $100 minimum depth

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        # data has 'yes_price', 'no_price', 'yes_liquidity', 'no_liquidity'
        if "yes_price" not in data.columns:
            return None

        yes = data["yes_price"].iloc[-1]
        no = data["no_price"].iloc[-1]
        yes_liq = data.get("yes_liquidity", pd.Series([1e9])).iloc[-1]
        no_liq = data.get("no_liquidity", pd.Series([1e9])).iloc[-1]

        price_sum = yes + no
        min_liq = min(yes_liq, no_liq)

        if price_sum < self.max_sum and min_liq >= self.min_liquidity:
            profit_pct = (1.0 - price_sum) / price_sum
            confidence = min(0.99, 0.80 + profit_pct * 2)  # higher spread = higher confidence
            return Signal(
                symbol=symbol,
                side="buy",   # buy BOTH yes and no
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "yes_price": round(float(yes), 4),
                    "no_price": round(float(no), 4),
                    "price_sum": round(float(price_sum), 4),
                    "expected_profit_pct": round(profit_pct * 100, 2),
                    "arb_type": "binary_both_sides",
                },
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if "yes_price" not in df.columns:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        price_sum = (df["yes_price"] + df["no_price"]).shift(1)
        entries = price_sum < self.max_sum
        exits = price_sum >= 0.99  # near resolution
        return BacktestSignals(entries=entries.fillna(False), exits=exits.fillna(False))
