"""
Polymarket Binary Arbitrage.

If YES_price + NO_price < $0.97 (accounting for fees), buy both sides.
At resolution, one side pays $1.00 — guaranteed profit.
"""
import logging
import time
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


logger = logging.getLogger(__name__)


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

        # Monitoring metrics
        self._signal_count: int = 0
        self._cumulative_expected_profit: float = 0.0

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        start_time = time.perf_counter()

        # data has 'yes_price', 'no_price', 'yes_liquidity', 'no_liquidity'
        if "yes_price" not in data.columns:
            exec_time_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                "PolyBinaryArb analyze skipped: missing yes_price column",
                extra={"execution_time_ms": exec_time_ms, "symbol": symbol},
            )
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

            signal = Signal(
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

            # Update monitoring metrics
            self._signal_count += 1
            self._cumulative_expected_profit += profit_pct

            exec_time_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "PolyBinaryArb signal generated",
                extra={
                    "signal_count": self._signal_count,
                    "execution_time_ms": exec_time_ms,
                    "expected_profit_pct": round(profit_pct * 100, 4),
                    "symbol": symbol,
                },
            )
            return signal

        exec_time_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(
            "PolyBinaryArb analyze completed: no arbitrage opportunity",
            extra={"execution_time_ms": exec_time_ms, "symbol": symbol},
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