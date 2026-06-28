"""
Put/Call Ratio Contrarian Strategy.

When the equity put/call ratio spikes to extreme fear levels (> 1.2),
the market is likely overly pessimistic → contrarian long SPY.
When ratio drops to extreme greed levels (< 0.5), go short.

Data source: CBOE daily equity put/call ratio (free, public).
Falls back to a synthetic proxy using VIX level when P/C data unavailable.
"""
import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class PutCallRatioContrarianStrategy(AbstractStrategy):
    name = "put_call_ratio_contrarian"
    display_name = "Put/Call Ratio Contrarian"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0  # daily signal

    FEAR_THRESHOLD = 1.2    # PCR > 1.2 → extreme fear → go long
    GREED_THRESHOLD = 0.5   # PCR < 0.5 → extreme greed → go short
    SMOOTHING = 5           # 5-day MA to reduce noise

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.fear_threshold = p.get("fear_threshold", self.FEAR_THRESHOLD)
        self.greed_threshold = p.get("greed_threshold", self.GREED_THRESHOLD)
        self.smoothing = p.get("smoothing", self.SMOOTHING)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Expects 'put_call_ratio' column (or 'vix_close' as fallback).
        For backtesting / live: data must include daily PCR.
        """
        if "close" not in data.columns or len(data) < self.smoothing + 2:
            return None

        if "put_call_ratio" in data.columns:
            pcr_series = data["put_call_ratio"].dropna()
        elif "vix_close" in data.columns:
            # Proxy: normalize VIX to PCR-like scale
            vix = data["vix_close"].dropna()
            # Map VIX 10-40 → PCR 0.4-1.4 linearly
            pcr_series = 0.4 + (vix - 10) / 30.0
        else:
            # Fall back: use realized vol proxy
            ret = data["close"].pct_change()
            rv = ret.rolling(10).std() * np.sqrt(252) * 100
            pcr_series = 0.4 + (rv.clip(10, 40) - 10) / 30.0

        if len(pcr_series) < self.smoothing + 1:
            return None

        pcr_smooth = pcr_series.ewm(span=self.smoothing, adjust=False).mean()  # MUTATION: use EMA for quicker response to regime shifts
        current_pcr = float(pcr_smooth.iloc[-1])

        if np.isnan(current_pcr):
            return None

        if current_pcr > self.fear_threshold:
            confidence = min(0.85, 0.60 + (current_pcr - self.fear_threshold) * 0.5)
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"pcr": round(current_pcr, 3), "signal": "extreme_fear"},
            )
        elif current_pcr < self.greed_threshold:
            confidence = min(0.85, 0.60 + (self.greed_threshold - current_pcr) * 0.8)
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"pcr": round(current_pcr, 3), "signal": "extreme_greed"},
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if "put_call_ratio" in df.columns:
            pcr = df["put_call_ratio"]
        elif "vix_close" in df.columns:
            vix = df["vix_close"]
            pcr = 0.4 + (vix.clip(10, 40) - 10) / 30.0
        else:
            ret = df["close"].pct_change()
            rv = ret.rolling(10).std() * np.sqrt(252) * 100
            pcr = 0.4 + (rv.clip(10, 40) - 10) / 30.0

        pcr_smooth = pcr.rolling(self.smoothing).mean().shift(1)  # shift to avoid lookahead

        en
