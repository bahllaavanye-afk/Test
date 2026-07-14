"""
ETF Statistical Arbitrage — SPY vs IVV vs VOO (same index, different ETFs).

When spread between any two ETFs tracking the same index exceeds a configurable threshold,
buy the cheaper ETF and sell the dearer one.

These three ETFs all track the S&P 500, so their prices should co‑move.
Small mispricings are arbitraged away quickly by authorized participants,
making this a near‑risk‑free short‑duration play when spreads widen.
"""
import numpy as np
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class StatArbETFStrategy(AbstractStrategy):
    name = "stat_arb_etf"
    display_name = "ETF Stat Arb (SPY/IVV/VOO)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 60.0

    # Default spread threshold in basis points (2 bps = 0.0002)
    SPREAD_THRESHOLD_BPS = 2.0
    # Confirmation: number of consecutive periods the condition must hold
    CONFIRMATION_PERIOD = 3
    # Z‑score entry/exit thresholds used for tighter filtering
    Z_ENTRY_THRESHOLD = 2.0
    Z_EXIT_THRESHOLD = 0.3

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.spread_threshold = (params or {}).get(
            "spread_threshold_bps", self.SPREAD_THRESHOLD_BPS
        ) / 10_000.0
        self.confirmation_period = (params or {}).get(
            "confirmation_period", self.CONFIRMATION_PERIOD
        )
        self.z_entry_threshold = (params or {}).get(
            "z_entry_threshold", self.Z_ENTRY_THRESHOLD
        )
        self.z_exit_threshold = (params or {}).get(
            "z_exit_threshold", self.Z_EXIT_THRESHOLD
        )

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Analyze the latest market data and emit a signal if a statistically significant
        spread is detected.

        Expected columns:
            - close            : primary price series (e.g., SPY)
            - close_ivv / close_voo : secondary ETF prices (optional)
            - volume (optional) : trading volume for optional confirmation
        """
        if "close" not in data.columns or len(data) < 30:
            return None

        close = data["close"]

        # Optional volume filter – require current volume to be above the recent median
        def volume_okay() -> bool:
            if "volume" not in data.columns:
                return True
            recent_vol = data["volume"].rolling(20).median()
            return float(data["volume"].iloc[-1]) >= float(recent_vol.iloc[-1])

        if not volume_okay():
            return None

        # ----------------------------------------------------------------------
        # Single‑symbol fallback: mean‑reversion of price vs its 20‑day MA
        # ----------------------------------------------------------------------
        if "close_ivv" not in data.columns and "close_voo" not in data.columns:
            ma = close.rolling(20).mean()
            std = close.rolling(20).std().replace(0, np.nan)
            z_series = ((close - ma) / std).shift(1)

            # Require a sustained extreme deviation
            if len(z_series) < self.confirmation_period:
                return None
            recent_z = z_series.iloc[-self.confirmation_period :]
            if not (abs(recent_z) > self.z_entry_threshold).all():
                return None

            spread = (close.iloc[-1] - ma.iloc[-1]) / ma.iloc[-1]
            if abs(spread) < self.spread_threshold:
                return None

            side = "sell" if spread > 0 else "buy"
            confidence = min(0.90, 0.60 + min(abs(spread) * 500, 0.30))
            return Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "z_score": round(float(z_series.iloc[-1]), 4),
                    "spread_pct": round(float(spread) * 100, 4),
                },
            )

        # ----------------------------------------------------------------------
        # Multi‑symbol path: ratio between primary ETF and a secondary ETF
        # ----------------------------------------------------------------------
        if "close_ivv" in data.columns:
            secondary = data["close_ivv"]
            secondary_name = "IVV"
        else:
            secondary = data["close_voo"]
            secondary_name = "VOO"

        ratio = close / secondary
        ratio_mean = ratio.rolling(20).mean()
        ratio_std = ratio.rolling(20).std().replace(0, np.nan)

        # Z‑score series for the ratio
        z_series = (ratio - ratio_mean) / ratio_std
        z_series = z_series.shift(1)

        # Confirmation: recent periods must all exceed entry threshold
        if len(z_series) < self.confirmation_period:
            return None
        recent_z = z_series.iloc[-self.confirmation_period :]
        if not (abs(recent_z) > self.z_entry_threshold).all():
            return None

        # Current spread (deviation from parity)
        spread = ratio.iloc[-1] - 1.0
        if abs(spread) < self.spread_threshold:
            return None

        side = "sell" if recent_z.iloc[-1] > 0 else "buy"
        confidence = min(0.90, 0.60 + min(abs(recent_z.iloc[-1]) * 0.1, 0.30))

        return Signal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "z_score": round(float(recent_z.iloc[-1]), 4),
                "spread_bps": round(float(spread) * 10_000, 2),
                "paired_etf": secondary_name,
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Generate back‑test‑ready entry/exit masks using a tighter Z‑score based
        mean‑reversion logic. The signals are shifted by one period to avoid look‑ahead.
        """
        close = df["close"]
        ma = close.rolling(20).mean()
        std = close.rolling(20).std().replace(0, np.nan)
        z = ((close - ma) / std).shift(1)

        # Long side: extreme negative deviation → entry, exit near zero
        entries = z < -self.z_entry_threshold
        exits = abs(z) < self.z_exit_threshold

        # Short side: extreme positive deviation → entry, exit near zero
        short_entries = z > self.z_entry_threshold
        short_exits = abs(z) < self.z_exit_threshold

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )