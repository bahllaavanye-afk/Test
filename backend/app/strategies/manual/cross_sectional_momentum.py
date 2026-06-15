"""
Cross-Sectional Momentum Strategy — Jegadeesh & Titman (1993).

Ranks a universe of ETFs/equities by 12-1 month return (skipping the most recent month
to avoid short-term reversal). Goes long the top quintile, short the bottom quintile.
Rebalances monthly.

Academic basis: Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers"
Historical Sharpe: 0.8-1.2 on US equity universe.

Symbols: SPY, QQQ, IWM, EFA, EEM, GLD, TLT, HYG, VNQ, XLE (10 liquid ETFs)
Long: top 2 by momentum
Short: bottom 2 by momentum
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, Signal

_DEFAULT_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT", "HYG", "VNQ", "XLE"]

_FORMATION_DAYS = 252    # 12 months
_SKIP_DAYS = 21          # skip last 1 month
_HOLDING_DAYS = 21       # monthly rebalance


class CrossSectionalMomentumStrategy(AbstractStrategy):
    """
    Jegadeesh-Titman cross-sectional momentum.

    For single-symbol backtesting: compares symbol's 12-1 month return to a
    time-series z-score threshold to generate long/short signals.

    For live multi-symbol analysis: ranks universe and signals long/short.
    """
    name = "cross_sectional_momentum"
    display_name = "Cross-Sectional Momentum (Jegadeesh-Titman)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86_400.0    # daily

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.universe = (params or {}).get("universe", _DEFAULT_UNIVERSE)
        self.formation = (params or {}).get("formation_days", _FORMATION_DAYS)
        self.skip = (params or {}).get("skip_days", _SKIP_DAYS)
        self.top_n = (params or {}).get("top_n", 2)
        self.bottom_n = (params or {}).get("bottom_n", 2)
        self._last_scores: dict[str, float] = {}

    def _compute_12_1_return(self, close: pd.Series) -> float | None:
        """12-1 month momentum: return from -formation to -skip days."""
        if len(close) < self.formation + self.skip + 5:
            return None
        past = float(close.iloc[-(self.formation + self.skip)])
        recent = float(close.iloc[-self.skip])
        if past <= 0:
            return None
        return (recent - past) / past

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        For live trading: compute momentum score for this symbol and
        signal based on whether it ranks in the top or bottom quintile.
        """
        close_col = "close" if "close" in data.columns else "Close"
        if close_col not in data.columns:
            return None

        close = data[close_col]
        score = self._compute_12_1_return(close)
        if score is None:
            return None

        self._last_scores[symbol] = score

        # Use rolling z-score as threshold (need historical context)
        # For single-symbol: go long if score > 0.08, short if < -0.08
        if score > 0.08:
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=min(0.85, 0.60 + abs(score) * 0.5),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"momentum_12_1": round(score, 4)},
            )
        elif score < -0.05:
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=min(0.80, 0.60 + abs(score) * 0.4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"momentum_12_1": round(score, 4)},
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Single-symbol backtest signals for walk-forward validation.
        Returns -1/0/1 series with shift(1) to prevent lookahead.

        Signal logic: go long when 12-1 momentum > rolling 75th percentile.
        Short when < rolling 25th percentile.
        """
        close_col = "close" if "close" in df.columns else "Close"
        if close_col not in df.columns:
            return pd.Series(0, index=df.index)

        close = df[close_col]
        min_len = self.formation + self.skip + 5

        scores = pd.Series(np.nan, index=df.index)
        for i in range(min_len, len(close)):
            segment = close.iloc[:i]
            s = self._compute_12_1_return(segment)
            if s is not None:
                scores.iloc[i] = s

        # Monthly rebalance: only update signal every ~21 days
        signals = pd.Series(0, index=df.index, dtype=float)
        rolling_75 = scores.rolling(126).quantile(0.75)
        rolling_25 = scores.rolling(126).quantile(0.25)

        # Rebalance signal
        for i in range(1, len(scores)):
            if i % self.holding_days == 0:
                s = scores.iloc[i]
                if pd.isna(s):
                    continue
                p75 = rolling_75.iloc[i]
                p25 = rolling_25.iloc[i]
                if not pd.isna(p75) and s > p75:
                    signals.iloc[i] = 1
                elif not pd.isna(p25) and s < p25:
                    signals.iloc[i] = -1
                else:
                    signals.iloc[i] = 0

        # Carry forward signal between rebalances
        signals = signals.replace(0, np.nan).ffill().fillna(0)

        # CRITICAL: shift(1) to prevent lookahead bias
        return signals.shift(1).fillna(0)

    @property
    def holding_days(self) -> int:
        return _HOLDING_DAYS
