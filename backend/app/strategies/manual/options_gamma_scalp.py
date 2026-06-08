"""
Options Gamma Scalping Strategy.

When IV Rank > 50 and within 2 days of expiry, options are cheap (high gamma)
and IV is elevated. Delta-hedge every 30 minutes to scalp gamma.

Strategy mechanics:
  - Buy straddle (long gamma) when IV rank is high near expiry
  - Delta-hedge the position every 30 minutes using the underlying
  - Profit comes from realized volatility exceeding implied

IV Rank = (current IV - 52-week low IV) / (52-week high IV - 52-week low IV)

Falls back to realized vol proxy when options data is unavailable.
"""
import numpy as np
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class OptionsGammaScalpStrategy(AbstractStrategy):
    name = "options_gamma_scalp"
    display_name = "Options Gamma Scalping (High IV Rank)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 1800.0   # 30-minute delta hedging

    IV_RANK_THRESHOLD = 50.0   # IV rank > 50 → elevated premium
    DAYS_TO_EXPIRY_MAX = 2     # within 2 days of expiry
    RV_IV_RATIO_MIN = 0.80     # realized/implied vol ratio (expect RV > IV)
    LOOKBACK_252 = 252         # 1-year lookback for IV rank

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.iv_rank_threshold = p.get("iv_rank_threshold", self.IV_RANK_THRESHOLD)
        self.dte_max = p.get("days_to_expiry_max", self.DAYS_TO_EXPIRY_MAX)
        self.rv_iv_ratio_min = p.get("rv_iv_ratio_min", self.RV_IV_RATIO_MIN)

    def _iv_rank(self, iv_series: pd.Series) -> float:
        """Compute IV rank over 52-week window."""
        if len(iv_series) < self.LOOKBACK_252:
            window = iv_series
        else:
            window = iv_series.iloc[-self.LOOKBACK_252:]
        low = float(window.min())
        high = float(window.max())
        current = float(iv_series.iloc[-1])
        if high <= low:
            return 50.0
        return (current - low) / (high - low) * 100.0

    def _realized_vol(self, close: pd.Series, window: int = 20) -> float:
        """Annualized realized volatility over window days."""
        if len(close) < window + 1:
            return 0.0
        ret = close.pct_change().dropna()
        return float(ret.rolling(window).std().iloc[-1] * np.sqrt(252))

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if "close" not in data.columns or len(data) < 30:
            return None

        close = data["close"]

        # IV rank computation
        if "iv" in data.columns:
            iv_series = data["iv"].dropna()
            iv_rank = self._iv_rank(iv_series)
            current_iv = float(iv_series.iloc[-1])
        else:
            # Proxy: use rolling 20-day realized vol as IV estimate
            rv20 = close.pct_change().rolling(20).std() * np.sqrt(252)
            iv_series = rv20.dropna()
            if len(iv_series) < 5:
                return None
            iv_rank = self._iv_rank(iv_series)
            current_iv = float(iv_series.iloc[-1])

        # Days to expiry check
        if "days_to_expiry" in data.columns:
            dte = int(data["days_to_expiry"].iloc[-1])
        else:
            # Infer from trading calendar: assume monthly expiry cycles
            # Use a proxy: enter in last 2 days of each month
            from datetime import datetime
            import calendar
            if hasattr(data.index[-1], 'month'):
                last_day = calendar.monthrange(
                    data.index[-1].year, data.index[-1].month
                )[1]
                day_of_month = data.index[-1].day
                dte = max(0, last_day - day_of_month)
            else:
                dte = 1  # assume near expiry if unknown

        # Realized/Implied vol ratio
        rv = self._realized_vol(close, 10)
        rv_iv_ratio = rv / current_iv if current_iv > 1e-6 else 0.0

        if iv_rank > self.iv_rank_threshold and dte <= self.dte_max:
            # High IV rank near expiry → buy gamma (long straddle)
            confidence = min(0.80, 0.55 + iv_rank / 200 + rv_iv_ratio * 0.1)
            return Signal(
                symbol=symbol,
                side="buy",    # buy the straddle / long gamma
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "iv_rank": round(iv_rank, 1),
                    "days_to_expiry": dte,
                    "implied_vol": round(current_iv * 100, 2),
                    "realized_vol": round(rv * 100, 2),
                    "rv_iv_ratio": round(rv_iv_ratio, 3),
                },
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        close = df["close"]
        ret = close.pct_change()

        # IV proxy: 20-day realized vol
        if "iv" in df.columns:
            iv = df["iv"]
        else:
            iv = ret.rolling(20).std() * np.sqrt(252)

        iv_252low = iv.rolling(self.LOOKBACK_252, min_periods=30).min()
        iv_252high = iv.rolling(self.LOOKBACK_252, min_periods=30).max()
        iv_rank = ((iv - iv_252low) / (iv_252high - iv_252low + 1e-10) * 100).shift(1)

        # Realized vol ratio
        rv10 = ret.rolling(10).std() * np.sqrt(252)
        rv_iv = (rv10 / iv.replace(0, np.nan)).shift(1)

        # Entry: high IV rank (options expensive, gamma cheap relative to IV)
        entries = (iv_rank > self.iv_rank_threshold) & (rv_iv > self.rv_iv_ratio_min)
        exits = iv_rank < 30.0
        # Gamma scalp is direction-neutral — use both entries/short_entries
        short_entries = (iv_rank > self.iv_rank_threshold) & (rv_iv > self.rv_iv_ratio_min)
        short_exits = iv_rank < 30.0

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )
