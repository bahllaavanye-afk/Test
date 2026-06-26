import logging
import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

logger = logging.getLogger(__name__)


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
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        effective = {**self.DEFAULT_PARAMS, **(params or {})}
        self.vol_period = effective["lookback_days"]
        self.vol_percentile = effective["top_pct"]
        self.rebalance_freq = effective["rebalance_freq"]
        self.trend_ema = params.get("trend_ema", 50) if params else 50

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        # Edge‑case handling for None / empty inputs
        if data is None or not isinstance(data, pd.DataFrame):
            logger.debug("Analyze called with invalid data: %s", type(data))
            return None
        if "close" not in data.columns:
            logger.debug("Analyze missing 'close' column")
            return None
        if data.empty:
            logger.debug("Analyze received empty DataFrame")
            return None
        if len(data) < self.vol_period + 10:
            logger.debug(
                "Insufficient rows: %d < required %d", len(data), self.vol_period + 10
            )
            return None

        close = data["close"]
        if close.empty:
            logger.debug("Close series is empty")
            return None

        daily_returns = close.pct_change()
        rolling_vol = daily_returns.rolling(self.vol_period).std() * np.sqrt(252)

        # Guard against all‑NaN rolling volatility
        if rolling_vol.isna().all():
            logger.debug("Rolling volatility series is all NaN")
            return None

        current_vol = rolling_vol.iloc[-1]
        if pd.isna(current_vol):
            logger.debug("Current volatility is NaN")
            return None

        ema_series = close.ewm(span=self.trend_ema).mean()
        ema50 = ema_series.iloc[-1]
        if pd.isna(ema50):
            logger.debug("EMA value is NaN")
            return None

        price = close.iloc[-1]
        if pd.isna(price):
            logger.debug("Latest price is NaN")
            return None

        # Historical vol distribution for ranking
        historical_vols = rolling_vol.dropna()
        if historical_vols.empty or len(historical_vols) < 10:
            logger.debug(
                "Insufficient historical volatility data: %d entries",
                len(historical_vols),
            )
            return None

        percentile_rank = (historical_vols < current_vol).mean() * 100

        if percentile_rank <= self.vol_percentile and price > ema50:
            confidence = min(
                0.80,
                0.55
                + (self.vol_percentile - percentile_rank) / self.vol_percentile * 0.3,
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
        # Edge‑case handling for None / empty inputs
        if df is None or not isinstance(df, pd.DataFrame):
            logger.debug("Backtest called with invalid DataFrame: %s", type(df))
            empty_series = pd.Series(dtype=bool)
            return BacktestSignals(entries=empty_series, exits=empty_series)

        if "close" not in df.columns:
            logger.debug("Backtest DataFrame missing 'close' column")
            empty_series = pd.Series(dtype=bool)
            return BacktestSignals(entries=empty_series, exits=empty_series)

        if df.empty:
            logger.debug("Backtest received empty DataFrame")
            empty_series = pd.Series(dtype=bool)
            return BacktestSignals(entries=empty_series, exits=empty_series)

        close = df["close"]
        if close.empty:
            logger.debug("Close series in backtest is empty")
            empty_series = pd.Series(dtype=bool)
            return BacktestSignals(entries=empty_series, exits=empty_series)

        daily_ret = close.pct_change()
        rolling_vol = daily_ret.rolling(self.vol_period).std().shift(1) * np.sqrt(252)
        ema = close.ewm(span=self.trend_ema).mean().shift(1)
        close_s = close.shift(1)

        # Low vol = below configured percentile of its own rolling vol history
        expanding_pct = rolling_vol.expanding().rank(pct=True) * 100

        entries = (expanding_pct <= self.vol_percentile) & (close_s > ema)
        exits = (expanding_pct > 50) | (close_s < ema)

        # Ensure boolean Series align with the original index and handle NaNs
        entries = entries.fillna(False)
        exits = exits.fillna(False)

        return BacktestSignals(entries=entries, exits=exits)