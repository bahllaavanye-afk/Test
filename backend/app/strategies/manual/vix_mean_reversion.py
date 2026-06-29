"""
VIX Spike Mean Reversion Strategy.

Rationale: VIX historically mean-reverts after spikes. When fear (VIX > 30)
is at an extreme, equity prices are oversold and offer mean-reversion longs.
When VIX < 15 (complacency), markets may be overvalued — reduce exposure.

Since Alpaca doesn't carry VIX directly, we proxy VIX using VXX (iPath S&P 500
VIX Short-Term Futures ETN) or VIXY (ProShares VIX Short-Term Futures ETF).
High RSI(5) on VXX ≈ elevated VIX ≈ fear spike → BUY SPY.
Low RSI(5) on VXX ≈ suppressed VIX ≈ complacency → SELL/HEDGE SPY.

Sharpe target: 0.9–1.4
"""

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator

from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class VIXMeanReversionParams(BaseModel):
    """Configuration parameters for the VIX Mean Reversion strategy."""

    fear_rsi_threshold: float = Field(
        default=70,
        ge=0,
        le=100,
        description="RSI threshold above which a fear spike is detected, triggering a BUY signal.",
        example=70,
    )
    complacency_rsi_threshold: float = Field(
        default=30,
        ge=0,
        le=100,
        description="RSI threshold below which market complacency is detected, triggering a SELL signal.",
        example=30,
    )
    rsi_period: int = Field(
        default=5,
        ge=1,
        description="Look‑back period for the RSI calculation (must be >= 1).",
        example=5,
    )
    target_symbol: str = Field(
        default="SPY",
        description="Ticker symbol to trade when a signal is generated.",
        example="SPY",
    )

    @validator("fear_rsi_threshold")
    def fear_above_complacency(cls, v, values):
        """Ensure fear threshold is higher than complacency threshold."""
        comp = values.get("complacency_rsi_threshold")
        if comp is not None and v <= comp:
            raise ValueError("fear_rsi_threshold must be greater than complacency_rsi_threshold")
        return v


class VIXMeanReversionStrategy(AbstractStrategy):
    name = "vix_mean_reversion"
    display_name = "VIX Mean Reversion (VXX Proxy)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0  # Hourly check

    # Signal thresholds on VXX RSI(5)
    FEAR_RSI_THRESHOLD = 70      # VXX RSI(5) > 70 → fear spike → BUY SPY
    COMPLACENCY_RSI_THRESHOLD = 30  # VXX RSI(5) < 30 → complacency → SELL SPY
    RSI_PERIOD = 5
    TARGET_SYMBOL = "SPY"         # Trade SPY on VIX signals

    def __init__(self, params: dict | VIXMeanReversionParams | None = None):
        super().__init__(params)
        # Validate and normalize parameters using the Pydantic schema
        if isinstance(params, VIXMeanReversionParams):
            validated = params
        else:
            validated = VIXMeanReversionParams.parse_obj(params or {})

        self.fear_rsi = float(validated.fear_rsi_threshold)
        self.complacency_rsi = float(validated.complacency_rsi_threshold)
        self.rsi_period = int(validated.rsi_period)
        self.target_symbol = validated.target_symbol

    def _compute_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Wilder RSI (same as TradingView default)."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))
        # Use EWM with alpha = 1/period for Wilder smoothing
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        data may contain:
        - 'close': price of the VXX/VIXY proxy (or SPY if running on SPY directly)
        - 'vxx_close' / 'vixy_close': VXX price column if available
        - 'vix_level': raw VIX value if available (overrides RSI logic)

        The strategy is meant to run on SPY/QQQ but needs VXX data as input.
        When data['close'] is passed for the VXX proxy, it generates SPY signals.
        """
        if "close" not in data.columns or len(data) < self.rsi_period + 2:
            return None

        close = data["close"]

        # Prefer explicit vix_level column if available
        if "vix_level" in data.columns:
            vix_val = float(data["vix_level"].iloc[-1])
            if pd.isna(vix_val):
                return None
            if vix_val > 30:
                confidence = min(0.90, 0.60 + (vix_val - 30) / 30.0 * 0.30)
                return Signal(
                    symbol=self.target_symbol,
                    side="buy",
                    confidence=round(confidence, 4),
                    strategy_name=self.name,
                    strategy_type=self.strategy_type,
                    risk_bucket=self.risk_bucket,
                    metadata={"vix_level": vix_val, "signal_source": "vix_direct"},
                )
            elif vix_val < 15:
                confidence = min(0.80, 0.60 + (15 - vix_val) / 15.0 * 0.20)
                return Signal(
                    symbol=self.target_symbol,
                    side="sell",
                    confidence=round(confidence, 4),
                    strategy_name=self.name,
                    strategy_type=self.strategy_type,
                    risk_bucket=self.risk_bucket,
                    metadata={"vix_level": vix_val, "signal_source": "vix_direct"},
                )
            return None

        # Use VXX RSI proxy
        rsi = self._compute_rsi(close, self.rsi_period)
        current_rsi = float(rsi.iloc[-1])

        if pd.isna(current_rsi):
            return None

        current_close = float(close.iloc[-1])

        if current_rsi > self.fear_rsi:
            # Fear spike — mean reversion long on SPY
            # Higher RSI = more extreme fear = higher confidence
            confidence = min(0.90, 0.60 + (current_rsi - self.fear_rsi) / (100 - self.fear_rsi) * 0.30)
            return Signal(
                symbol=self.target_symbol,
                side="buy",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "vxx_rsi": round(current_rsi, 2),
                    "vxx_close": round(current_close, 4),
                    "signal_source": "vxx_rsi_fear_spike",
                    "interpretation": "VIX proxy elevated — fear spike, expect equity mean reversion",
                },
            )

        elif current_rsi < self.complacency_rsi:
            # Complacency — reduce SPY exposure
            confidence = min(0.80, 0.60 + (self.complacency_rsi - current_rsi) / self.complacency_rsi * 0.20)
            return Signal(
                symbol=self.target_symbol,
                side="sell",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "vxx_rsi": round(current_rsi, 2),
                    "vxx_close": round(current_close, 4),
                    "signal_source": "vxx_rsi_complacency",
                    "interpretation": "VIX proxy suppressed — complacency, reduce equity exposure",
                },
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Vectorized backtest signals.
        Expects df['close'] to be VXX or VIXY price data.
        Alternatively uses df['vix_level'] if present.
        """
        if "vix_level" in df.columns:
            vix = df["vix_level"].ffill()
            entries = (vix > 30).shift(1).fillna(False)
            exits = (vix <= 20).shift(1).fillna(False)
            short_entries = (vix < 15).shift(1).fillna(False)
            short_exits = (vix >= 20).shift(1).fillna(False)
        else:
            close = df["close"]
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta.clip(upper=0))
            avg_gain = gain.ewm(alpha=1.0 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1.0 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100.0 - (100.0 / (1.0 + rs))

            # Shift 1 to prevent lookahead bias
            entries = (rsi > self.fear_rsi).shift(1).fillna(False)
            exits = (rsi < 50).shift(1).fillna(False)   # exit when fear normalises
            short_entries = (rsi < self.complacency_rsi).shift(1).fillna(False)
            short_exits = (rsi > 50).shift(1).fillna(False)

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )