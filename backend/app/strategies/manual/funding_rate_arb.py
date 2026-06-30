import logging
import time
import numpy as np
import pandas as pd
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

logger = logging.getLogger(__name__)

class FundingRateArbStrategy(AbstractStrategy):
    """
    Funding rate mean-reversion arbitrage on perpetual futures.

    In production, uses Binance FAPI /fapi/v1/fundingRate endpoint (8-hour
    settlement periods). In backtest mode, proxies funding imbalance via
    3-bar price momentum on daily OHLCV bars.
    """

    name = "funding_rate_arb"
    display_name = "Funding Rate Arbitrage"
    market_type = "crypto"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 28800.0   # 8 hours (funding settlement cadence)
    confidence_threshold = 0.60

    # Z-score thresholds
    ENTRY_Z = 2.0      # Enter when |z| > 2.0
    EXIT_Z  = 0.5      # Exit when |z| < 0.5

    # Lookback for rolling mean/std
    LOOKBACK = 90

    # Momentum proxy threshold (%) on 3-bar return
    PROXY_THRESHOLD = 0.04  # 4 %

    DEFAULT_PARAMS = {
        "min_funding_rate": 0.01,
        "hedge_ratio": 1.0,
        "max_position_usd": 5000,
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        effective = {**self.DEFAULT_PARAMS, **(params or {})}
        self.min_funding_rate = effective["min_funding_rate"]
        self.hedge_ratio = effective["hedge_ratio"]
        self.max_position_usd = effective["max_position_usd"]

    def description(self) -> str:
        return (
            "Funding Rate Arbitrage (manual) — "
            "trades perpetual futures mean reversion via funding rate z-score. "
            "Long when shorts are crowded (z < -2), short when longs are crowded (z > +2). "
            "Backtest uses 3-bar momentum proxy on daily OHLCV."
        )

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Production signal requires Binance FAPI /fapi/v1/fundingRate endpoint.
        Not accessible in sandbox — return None gracefully.
        """
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Vectorized backtest using daily OHLCV with a 3-bar momentum proxy for
        funding rate imbalance.

        Steps:
          1. Compute 3-bar return = close / close.shift(3) - 1
          2. Compute rolling 90-bar z-score of that return
          3. shift(1) all indicators to prevent lookahead bias
          4. Long entry: proxy_z < -ENTRY_Z  (over-sold / shorts crowded)
             Short entry: proxy_z > +ENTRY_Z (over-bought / longs crowded)
          5. Exit long:  proxy_z > -EXIT_Z
             Exit short: proxy_z < +EXIT_Z
        """
        start_time = time.time()

        min_bars = self.LOOKBACK + 5
        false_series = pd.Series(False, index=df.index, dtype=bool)

        if "close" not in df.columns or len(df) < min_bars:
            elapsed = time.time() - start_time
            logger.info(
                "FundingRateArb backtest skipped due to insufficient data",
                extra={
                    "strategy": self.name,
                    "signal_counts": {
                        "entries": 0,
                        "exits": 0,
                        "short_entries": 0,
                        "short_exits": 0,
                    },
                    "execution_time_seconds": elapsed,
                    "pnl": None,
                },
            )
            return BacktestSignals(
                entries=false_series,
                exits=false_series,
                short_entries=false_series,
                short_exits=false_series,
            )

        close = df["close"].astype(float)

        # 3-bar return as funding rate proxy
        mom3 = close / close.shift(3) - 1.0

        # Rolling z-score of the 3-bar return
        roll_mean = mom3.rolling(self.LOOKBACK, min_periods=self.LOOKBACK // 2).mean()
        roll_std = mom3.rolling(self.LOOKBACK, min_periods=self.LOOKBACK // 2).std()
        proxy_z = (mom3 - roll_mean) / roll_std.clip(lower=1e-8)

        # Shift(1) — no lookahead
        proxy_z_lag = proxy_z.shift(1)

        # Long: shorts crowded (z very negative → expect squeeze → collect funding)
        entries = (proxy_z_lag < -self.ENTRY_Z).fillna(False)
        exits = (proxy_z_lag > -self.EXIT_Z).fillna(False)

        # Short: longs crowded (z very positive → expect flush)
        short_entries = (proxy_z_lag > self.ENTRY_Z).fillna(False)
        short_exits = (proxy_z_lag < self.EXIT_Z).fillna(False)

        # Compute signal counts for logging
        signal_counts = {
            "entries": int(entries.sum()),
            "exits": int(exits.sum()),
            "short_entries": int(short_entries.sum()),
            "short_exits": int(short_exits.sum()),
        }

        elapsed = time.time() - start_time

        # P&L is not computed within this method; set to None for structured logging
        logger.info(
            "FundingRateArb backtest signals generated",
            extra={
                "strategy": self.name,
                "signal_counts": signal_counts,
                "execution_time_seconds": elapsed,
                "pnl": None,
            },
        )

        return BacktestSignals(
            entries=entries.astype(bool),
            exits=exits.astype(bool),
            short_entries=short_entries.astype(bool),
            short_exits=short_exits.astype(bool),
        )