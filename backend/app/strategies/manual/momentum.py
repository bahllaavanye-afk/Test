"""
Momentum Strategy — Jegadeesh & Titman (1993).

Buy top-decile performers (12-month return excluding last month).
Historically: 17.9% annualized excess return, Sharpe 0.68-1.0.

For single-symbol mode: signal when symbol's 12-1 momentum is strongly positive
and trending up compared to prior period.
"""
import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class MomentumStrategy(AbstractStrategy):
    name = "momentum"
    display_name = "Momentum (Jegadeesh-Titman)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0  # hourly check

    DEFAULT_PARAMS = {
        "formation_period": 252,   # ~12 months trading days
        "skip_period": 21,         # skip last 1 month (avoid short-term reversal)
        "holding_period": 21,      # hold for ~1 month
        "momentum_threshold": 0.08,  # minimum momentum score to generate signal
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        effective = {**self.DEFAULT_PARAMS, **(params or {})}
        self.formation_period = effective["formation_period"]
        self.skip_period = effective["skip_period"]
        self.holding_period = effective["holding_period"]
        self.momentum_threshold = effective["momentum_threshold"]
        self.min_bars = self.formation_period + self.skip_period + 10

    def _compute_momentum_score(self, close: pd.Series) -> float:
        """12-1 momentum: return over formation period excluding last skip_period."""
        if len(close) < self.min_bars:
            return 0.0
        past = close.iloc[-(self.formation_period + self.skip_period)]
        current = close.iloc[-self.skip_period]
        return float((current - past) / past) if past > 0 else 0.0

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if "close" not in data.columns or len(data) < self.min_bars:
            return None

        close = data["close"]
        mom_score = self._compute_momentum_score(close)
        vol = close.pct_change().rolling(21).std().iloc[-1]

        if vol == 0 or np.isnan(vol):
            return None

        # Risk-adjusted momentum (like Sharpe)
        risk_adj = mom_score / (vol * np.sqrt(252) + 1e-9)
        confidence = min(0.90, max(0.50, 0.50 + risk_adj * 0.15))

        if mom_score > self.momentum_threshold:   # > threshold momentum → buy signal
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"momentum_score": round(mom_score, 4), "risk_adj": round(risk_adj, 4)},
            )
        elif mom_score < -self.momentum_threshold:  # strong negative momentum → sell short
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"momentum_score": round(mom_score, 4), "risk_adj": round(risk_adj, 4)},
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        close = df["close"]
        # 12-1 momentum (shifted to prevent lookahead)
        mom = (close.shift(self.skip_period) / close.shift(self.formation_period + self.skip_period) - 1).shift(1)
        entries = mom > self.momentum_threshold
        exits = mom < 0.0
        short_entries = mom < -self.momentum_threshold
        short_exits = mom > 0.0
        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )
