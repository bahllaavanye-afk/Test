"""
ETF Statistical Arbitrage — SPY vs IVV vs VOO (same index, different ETFs).

When spread between any two ETFs tracking the same index exceeds a configurable
threshold, buy the cheaper ETF and sell the dearer one.

These three ETFs all track the S&P 500, so their prices should co‑move.
Small mispricings are arbitraged away quickly by authorized participants,
making this a near‑risk‑free short‑duration play when spreads widen.
"""

import numpy as np
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class StatArbETFStrategy(AbstractStrategy):
    """Statistical arbitrage strategy for SPY / IVV / VOO."""

    name = "stat_arb_etf"
    display_name = "ETF Stat Arb (SPY/IVV/VOO)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 60.0

    # Default configurable parameters
    DEFAULT_SPREAD_THRESHOLD_BPS = 2.0          # 2 bps = 0.0002
    DEFAULT_ENTRY_Z_THRESHOLD = 2.0
    DEFAULT_EXIT_Z_THRESHOLD = 0.5
    DEFAULT_CONFIRMATION_WINDOW = 3

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        cfg = params or {}
        self.spread_threshold = cfg.get(
            "spread_threshold_bps", self.DEFAULT_SPREAD_THRESHOLD_BPS
        ) / 10_000.0
        self.entry_z_threshold = cfg.get(
            "entry_z_threshold", self.DEFAULT_ENTRY_Z_THRESHOLD
        )
        self.exit_z_threshold = cfg.get(
            "exit_z_threshold", self.DEFAULT_EXIT_Z_THRESHOLD
        )
        self.confirmation_window = cfg.get(
            "confirmation_window", self.DEFAULT_CONFIRMATION_WINDOW
        )

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Generate a trading signal.

        Expected columns:
            - close (primary symbol)
            - close_ivv / close_voo (optional secondary ETFs)
            - volume / volume_ivv / volume_voo (optional for confirmation)

        Returns:
            Signal instance when entry criteria are met, otherwise None.
        """
        if "close" not in data.columns or len(data) < 20:
            return None

        close = data["close"]

        # ------------------------------------------------------------------
        # Single‑symbol fallback – mean‑reversion of price vs its 20‑day MA
        # ------------------------------------------------------------------
        if "close_ivv" not in data.columns and "close_voo" not in data.columns:
            ma = close.rolling(20).mean()
            std = close.rolling(20).std().replace(0, np.nan)
            if std.isna().all():
                return None

            z = ((close - ma) / std).iloc[-1]
            if abs(z) < self.entry_z_threshold:
                return None

            side = "sell" if z > 0 else "buy"
            confidence = min(0.85, 0.60 + abs(z) * 0.05)
            return Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"z_score": round(float(z), 4)},
            )

        # ------------------------------------------------------------------
        # Multi‑symbol path – compare primary symbol with a secondary ETF
        # ------------------------------------------------------------------
        # Choose the first available secondary ETF
        if "close_ivv" in data.columns:
            secondary = data["close_ivv"]
            secondary_name = "IVV"
            volume_col = "volume_ivv"
        else:
            secondary = data["close_voo"]
            secondary_name = "VOO"
            volume_col = "volume_voo"

        # Price ratio and its rolling statistics
        ratio = close / secondary
        ratio_mean = ratio.rolling(20).mean()
        ratio_std = ratio.rolling(20).std().replace(0, np.nan)

        if ratio_std.isna().all():
            return None

        # Current z‑score of the ratio
        z = (ratio.iloc[-1] - ratio_mean.iloc[-1]) / ratio_std.iloc[-1]
        spread = ratio.iloc[-1] - 1.0  # deviation from parity

        # ------------------------------------------------------------------
        # Entry filters
        # ------------------------------------------------------------------
        if abs(z) < self.entry_z_threshold or abs(spread) < self.spread_threshold:
            return None

        # Confirmation: require consistent signal direction over the last N bars
        if not self._confirm_signal(ratio, ratio_mean, self.confirmation_window):
            return None

        # Optional volume confirmation if volume data is present
        if volume_col in data.columns:
            vol_primary = data["volume"]
            vol_secondary = data[volume_col]
            # Ensure the cheaper side has at least 20 % higher average volume
            avg_vol_primary = vol_primary.tail(20).mean()
            avg_vol_secondary = vol_secondary.tail(20).mean()
            if spread > 0 and avg_vol_primary < 1.2 * avg_vol_secondary:
                return None
            if spread < 0 and avg_vol_secondary < 1.2 * avg_vol_primary:
                return None

        side = "sell" if z > 0 else "buy"
        confidence = min(0.90, 0.60 + abs(z) * 0.05)

        return Signal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "z_score": round(float(z), 4),
                "spread_bps": round(float(spread) * 10_000, 2),
                "pair": f"{symbol}/{secondary_name}",
            },
        )

    def _confirm_signal(
        self, ratio: pd.Series, ratio_mean: pd.Series, window: int
    ) -> bool:
        """
        Confirm that the sign of (ratio - ratio_mean) has been consistent for
        the last `window` periods and that the magnitude exceeds the entry
        threshold.

        Returns True if confirmation passes, otherwise False.
        """
        if len(ratio) < window:
            return False

        recent = ratio.iloc[-window:] - ratio_mean.iloc[-window:]
        signs = np.sign(recent.dropna())
        if signs.empty:
            return False
        # All signs must be identical (all positive or all negative)
        if not np.all(signs == signs.iloc[0]):
            return False
        # Magnitude check: average absolute deviation must exceed spread threshold
        avg_dev = recent.abs().mean()
        return avg_dev >= self.spread_threshold

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Generate back‑test signals using rolling z‑score of price vs 20‑day MA.
        Entry/exit thresholds are derived from configurable parameters.
        """
        close = df["close"]
        ma = close.rolling(20).mean()
        std = close.rolling(20).std().replace(0, np.nan)

        # Z‑score shifted by one bar to avoid look‑ahead bias
        z = ((close - ma) / std).shift(1)

        # Long side
        entries = z < -self.entry_z_threshold
        exits = z > -self.exit_z_threshold

        # Short side
        short_entries = z > self.entry_z_threshold
        short_exits = z < self.exit_z_threshold

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )