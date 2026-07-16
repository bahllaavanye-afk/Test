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
        """Compute IV rank over 52‑week window."""
        if iv_series is None or iv_series.empty:
            return 50.0
        if len(iv_series) < self.LOOKBACK_252:
            window = iv_series
        else:
            window = iv_series.iloc[-self.LOOKBACK_252:]
        low = float(window.min())
        high = float(window.max())
        if high <= low:
            return 50.0
        current = float(iv_series.iloc[-1])
        return (current - low) / (high - low) * 100.0

    def _realized_vol(self, close: pd.Series, window: int = 20) -> float:
        """Annualized realized volatility over `window` days."""
        if close is None or close.empty or len(close) < window + 1:
            return 0.0
        ret = close.pct_change().dropna()
        return float(ret.rolling(window).std().iloc[-1] * np.sqrt(252))

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        # Defensive checks for None / empty inputs
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            return None
        if "close" not in data.columns or data["close"].empty:
            return None
        if len(data) < 30:
            return None

        close = data["close"]

        # IV rank computation
        if "iv" in data.columns:
            iv_series = data["iv"].dropna()
            if iv_series.empty:
                iv_rank = 50.0
                current_iv = 0.0
            else:
                iv_rank = self._iv_rank(iv_series)
                current_iv = float(iv_series.iloc[-1])
        else:
            # Proxy: use rolling 20‑day realized vol as IV estimate
            rv20 = close.pct_change().rolling(20).std() * np.sqrt(252)
            iv_series = rv20.dropna()
            if iv_series.empty or len(iv_series) < 5:
                return None
            iv_rank = self._iv_rank(iv_series)
            current_iv = float(iv_series.iloc[-1])

        # Days to expiry check
        if "days_to_expiry" in data.columns:
            try:
                dte = int(data["days_to_expiry"].iloc[-1])
            except (ValueError, TypeError):
                dte = 1
        else:
            # Infer from trading calendar: assume monthly expiry cycles
            from datetime import datetime
            import calendar
            last_index = data.index[-1] if hasattr(data, "index") else None
            if hasattr(last_index, "month") and hasattr(last_index, "year") and hasattr(last_index, "day"):
                last_day = calendar.monthrange(last_index.year, last_index.month)[1]
                day_of_month = last_index.day
                dte = max(0, last_day - day_of_month)
            else:
                dte = 1  # assume near expiry if unknown

        # Realized/Implied vol ratio
        rv = self._realized_vol(close, 10)
        rv_iv_ratio = rv / current_iv if current_iv > 1e-6 else 0.0

        if iv_rank > self.iv_rank_threshold and dte <= self.dte_max:
            confidence = min(0.80, 0.55 + iv_rank / 200 + rv_iv_ratio * 0.1)
            return Signal(
                symbol=symbol,
                side="buy",
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
        # Defensive guard for None / empty inputs
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or "close" not in df.columns:
            empty_series = pd.Series([], dtype=bool)
            return BacktestSignals(
                entries=empty_series,
                exits=empty_series,
                short_entries=empty_series,
                short_exits=empty_series,
            )

        close = df["close"]
        ret = close.pct_change()

        # IV proxy: 20‑day realized vol if not present
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

        # Entry / exit logic
        entries = (iv_rank > self.iv_rank_threshold) & (rv_iv > self.rv_iv_ratio_min)
        exits = iv_rank < 30.0
        short_entries = entries  # gamma scalp is direction‑neutral
        short_exits = exits

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )