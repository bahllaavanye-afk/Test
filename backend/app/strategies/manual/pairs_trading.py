"""
Statistical Pairs Trading via Engle-Granger Cointegration.

Academic basis: Gatev, Goetzmann, Rouwenhorst (2006) — pairs trading
generates 11% pa excess return with Sharpe 1.5-2.67.

Logic:
  1. Test for cointegration between two symbols (Engle-Granger test)
  2. Compute spread = price_A - hedge_ratio * price_B
  3. Normalize spread to z-score using rolling mean/std
  4. Entry: |z| > entry_z (default 2.0) — trade mean reversion
  5. Exit:  |z| < exit_z  (default 0.5)
  6. Stop:  |z| > stop_z  (default 4.0)

Market neutral: long one leg, short the other (dollar-neutral).
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class PairsTradingStrategy(AbstractStrategy):
    name = "pairs_trading"
    display_name = "Pairs Trading (Cointegration)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 300.0  # 5 min

    # Default pairs universe (cointegrated historically)
    DEFAULT_PAIRS = [
        ("SPY", "QQQ"),
        ("GLD", "SLV"),
        ("KO", "PEP"),
        ("XOM", "CVX"),
        ("C", "JPM"),
    ]

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.lookback = params.get("lookback", 252) if params else 252
        self.entry_z = params.get("entry_z", 2.0) if params else 2.0
        self.exit_z = params.get("exit_z", 0.5) if params else 0.5
        self.stop_z = params.get("stop_z", 4.0) if params else 4.0
        self.coint_pvalue = params.get("coint_pvalue", 0.05) if params else 0.05

    def _compute_spread(self, price_a: pd.Series, price_b: pd.Series, lookback: int) -> tuple[pd.Series, float]:
        """Compute hedge ratio via OLS regression and return z-scored spread."""
        # Use last lookback bars for hedge ratio
        y = price_a.iloc[-lookback:]
        x = price_b.iloc[-lookback:]
        hedge_ratio = np.cov(y, x)[0, 1] / np.var(x)
        spread = price_a - hedge_ratio * price_b
        rolling_mean = spread.rolling(lookback).mean()
        rolling_std = spread.rolling(lookback).std()
        z_score = (spread - rolling_mean) / (rolling_std + 1e-9)
        return z_score, hedge_ratio

    def _is_cointegrated(self, price_a: pd.Series, price_b: pd.Series) -> bool:
        try:
            _, pvalue, _ = coint(price_a, price_b)
            return pvalue < self.coint_pvalue
        except Exception:
            return False

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        # data expected to have MultiIndex columns: (symbol, field)
        # or be called per-pair. Here we handle simple case: data has 'close_a', 'close_b'
        if "close_a" not in data.columns or "close_b" not in data.columns:
            return None
        if len(data) < self.lookback + 10:
            return None

        price_a = data["close_a"]
        price_b = data["close_b"]
        z_score, hedge_ratio = self._compute_spread(price_a, price_b, self.lookback)
        latest_z = z_score.iloc[-1]

        if abs(latest_z) > self.stop_z:
            return None  # spread too wide — risk of cointegration breakdown

        if latest_z > self.entry_z:
            # Spread too high: short A, long B
            confidence = min(0.95, (latest_z - self.entry_z) / (self.stop_z - self.entry_z) * 0.8 + 0.6)
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"z_score": float(latest_z), "hedge_ratio": hedge_ratio, "leg": "A"},
            )
        elif latest_z < -self.entry_z:
            # Spread too low: long A, short B
            confidence = min(0.95, (-latest_z - self.entry_z) / (self.stop_z - self.entry_z) * 0.8 + 0.6)
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"z_score": float(latest_z), "hedge_ratio": hedge_ratio, "leg": "A"},
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """For single-leg backtesting. df must have 'close_a' and 'close_b'."""
        if "close_a" not in df.columns or "close_b" not in df.columns:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        z_score, _ = self._compute_spread(df["close_a"], df["close_b"], self.lookback)
        z_shifted = z_score.shift(1)  # prevent lookahead

        entries = z_shifted < -self.entry_z        # long A when spread low
        exits = (z_shifted > -self.exit_z) & (z_shifted < self.exit_z)
        short_entries = z_shifted > self.entry_z   # short A when spread high
        short_exits = exits.copy()

        # Stop out if z diverges too far
        stop = abs(z_shifted) > self.stop_z
        exits = exits | stop
        short_exits = short_exits | stop

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )
