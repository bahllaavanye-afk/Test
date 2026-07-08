"""
Low Volatility Factor Strategy — Baker, Bradley, Wurgler (2011).

Buy low-volatility stocks (bottom 20% rolling 252-day std).
Historically achieves higher Sharpe than market with lower drawdown.

In single-symbol mode: score the symbol vs a universe, signal when it's
in the low-vol regime and trending up.
"""
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, root_validator

from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class LowVolatilityParams(BaseModel):
    """Configuration parameters for the Low Volatility strategy."""

    lookback_days: int = Field(
        ...,
        description="Rolling window in days for volatility calculation.",
        ge=1,
        example=252,
    )
    top_pct: float = Field(
        ...,
        description="Upper percentile threshold (0‑100) for selecting low‑volatility assets.",
        ge=0,
        le=100,
        example=30,
    )
    bottom_pct: float = Field(
        ...,
        description="Lower percentile threshold (0‑100) for reference; not used in current logic.",
        ge=0,
        le=100,
        example=20,
    )
    rebalance_freq: int = Field(
        ...,
        description="Rebalancing frequency expressed in trading days.",
        ge=1,
        example=21,
    )
    trend_ema: int = Field(
        50,
        description="Span for the EMA used as a trend filter.",
        ge=1,
        example=50,
    )

    @root_validator
    def check_percentiles(cls, values):
        top = values.get("top_pct")
        bottom = values.get("bottom_pct")
        if top is not None and bottom is not None and top <= bottom:
            raise ValueError("top_pct must be greater than bottom_pct")
        return values


class LowVolatilityStrategy(AbstractStrategy):
    name = "low_volatility"
    display_name = "Low Volatility Factor"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    DEFAULT_PARAMS = {
        "lookback_days": 252,
        "top_pct": 30,
        "bottom_pct": 20,
        "rebalance_freq": 21,
        "trend_ema": 50,
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        effective = {**self.DEFAULT_PARAMS, **(params or {})}
        # Validate and coerce parameters via Pydantic model
        validated_params = LowVolatilityParams(**effective)

        self.vol_period = validated_params.lookback_days
        self.vol_percentile = validated_params.top_pct
        self.rebalance_freq = validated_params.rebalance_freq
        self.trend_ema = validated_params.trend_ema
        # bottom_pct is retained for possible future extensions
        self.bottom_pct = validated_params.bottom_pct

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < self.vol_period + 10:
            return None

        close = data["close"]
        daily_returns = close.pct_change()
        rolling_vol = daily_returns.rolling(self.vol_period).std() * np.sqrt(252)
        current_vol = rolling_vol.iloc[-1]
        ema = close.ewm(span=self.trend_ema).mean().iloc[-1]
        price = close.iloc[-1]

        # Historical vol distribution for ranking
        historical_vols = rolling_vol.dropna()
        if len(historical_vols) < 10:
            return None
        percentile_rank = (historical_vols < current_vol).mean() * 100

        if percentile_rank <= self.vol_percentile and price > ema:
            confidence = min(
                0.80,
                0.55 + (self.vol_percentile - percentile_rank) / self.vol_percentile * 0.3,
            )
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "annualized_vol": round(float(current_vol), 4),
                    "vol_percentile": round(percentile_rank, 1),
                },
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        close = df["close"]
        daily_ret = close.pct_change()
        rolling_vol = daily_ret.rolling(self.vol_period).std().shift(1) * np.sqrt(252)
        ema = close.ewm(span=self.trend_ema).mean().shift(1)
        close_s = close.shift(1)

        # Low vol = below configured percentile of own rolling vol history
        expanding_pct = rolling_vol.expanding().rank(pct=True) * 100
        entries = (expanding_pct <= self.vol_percentile) & (close_s > ema)
        exits = (expanding_pct > 50) | (close_s < ema)
        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
        )