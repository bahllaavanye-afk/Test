"""
Realized Volatility Asymmetry Strategy

Academic basis: Barndorff-Nielsen, Kinnebrock, Shephard (2010) "Measuring Downside
Risk — Realised Semivariance" and Feunou, Jahan-Parvar, Tedongap (2017) "Which
parametric model for conditional skewness?"

Upside realized semivariance = std(positive daily log-returns).
Downside realized semivariance = std(negative daily log-returns).
Ratio = upside_vol / downside_vol.

When ratio > 1.15: price making larger upward moves than downward → positive skew
  → long signal (momentum regime)
When ratio < 0.87: downside moves dominate → negative skew
  → short/avoid signal (crash risk)

Feunou et al. show the ratio predicts next-period equity index returns with
IC ≈ 0.04–0.06 at monthly horizon, Sharpe ~1.3 when combined with momentum.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class RealizedVolAsymmetryParams(BaseModel):
    """Configuration parameters for the Realized Volatility Asymmetry strategy.

    Attributes
    ----------
    lookback : int
        Number of days used to compute the semivariance window.
    long_threshold : float
        Upside/downside ratio above which a long (buy) signal is generated.
    short_threshold : float
        Ratio below which a short (sell) signal is generated.
    min_bars : int
        Minimum number of bars required for a valid analysis.
    confidence_threshold : float
        Minimum confidence score required to emit a signal.
    """

    lookback: int = Field(
        default=60,
        ge=1,
        description="Number of days for the semivariance rolling window.",
        example=60,
    )
    long_threshold: float = Field(
        default=1.15,
        gt=0,
        description="Upside/downside ratio threshold to trigger a long signal.",
        example=1.15,
    )
    short_threshold: float = Field(
        default=0.87,
        gt=0,
        description="Upside/downside ratio threshold to trigger a short signal.",
        example=0.87,
    )
    min_bars: int = Field(
        default=30,
        ge=1,
        description="Minimum number of bars required for analysis.",
        example=30,
    )
    confidence_threshold: float = Field(
        default=0.65,
        ge=0,
        le=1,
        description="Minimum confidence required to emit a signal.",
        example=0.65,
    )

    @validator("short_threshold")
    def short_must_be_less_than_long(cls, v, values):
        long_thr = values.get("long_threshold")
        if long_thr is not None and v >= long_thr:
            raise ValueError("short_threshold must be less than long_threshold")
        return v


class RealizedVolAsymmetryStrategy(AbstractStrategy):
    name = "realized_vol_asymmetry"
    display_name = "Realized Volatility Asymmetry (Semivariance)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0

    # Default parameters; can be overridden via `params` attribute if needed.
    params = RealizedVolAsymmetryParams()

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < self.params.min_bars or "close" not in data.columns:
            return None

        close = data["close"].astype(float).tail(self.params.lookback)
        log_rets = np.log(close).diff().dropna()
        if len(log_rets) < self.params.min_bars:
            return None

        pos_rets = log_rets[log_rets > 0]
        neg_rets = log_rets[log_rets < 0]
        if len(pos_rets) < 5 or len(neg_rets) < 5:
            return None

        upside_vol = float(pos_rets.std())
        downside_vol = float(neg_rets.abs().std())
        if downside_vol < 1e-9:
            return None

        ratio = upside_vol / downside_vol

        if ratio >= self.params.long_threshold:
            side = "buy"
            conf = min(
                0.63 + (ratio - self.params.long_threshold) * 1.5,
                0.92,
            )
        elif ratio <= self.params.short_threshold:
            side = "sell"
            conf = min(
                0.63 + (self.params.short_threshold - ratio) * 1.5,
                0.92,
            )
        else:
            return None

        if conf < self.params.confidence_threshold:
            return None

        spot = float(data["close"].iloc[-1])
        return Signal(
            symbol=symbol,
            side=side,
            confidence=conf,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            target_price=spot,
            metadata={
                "upside_vol": round(upside_vol, 6),
                "downside_vol": round(downside_vol, 6),
                "ratio": round(ratio, 4),
                "n_pos": len(pos_rets),
                "n_neg": len(neg_rets),
                "lookback": self.params.lookback,
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if "close" not in df.columns or len(df) < self.params.lookback + 10:
            return BacktestSignals(
                entries=pd.Series(False, index=df.index),
                exits=pd.Series(False, index=df.index),
            )
        close = df["close"].astype(float)
        log_rets = np.log(close).diff()

        def _ratio(window: pd.Series) -> float:
            pos = window[window > 0]
            neg = window[window < 0]
            if len(pos) < 5 or len(neg) < 5:
                return 1.0
            dv = neg.abs().std()
            return 1.0 if dv < 1e-9 else pos.std() / dv

        ratio_series = log_rets.rolling(
            self.params.lookback, min_periods=self.params.min_bars
        ).apply(_ratio, raw=False)
        entries = (ratio_series.shift(1) >= self.params.long_threshold).fillna(False)
        short_entries = (
            ratio_series.shift(1) <= self.params.short_threshold
        ).fillna(False)
        exits = (ratio_series.shift(1) < 1.05).fillna(False)
        return BacktestSignals(
            entries=entries, exits=exits, short_entries=short_entries
        )