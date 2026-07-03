"""
ETF Statistical Arbitrage — SPY vs IVV vs VOO (same index, different ETFs).

When spread between any two ETFs tracking the same index exceeds 2bps,
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

    # Spread threshold in basis points (2bps = 0.0002)
    SPREAD_THRESHOLD_BPS = 2.0

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.spread_threshold = (params or {}).get(
            "spread_threshold_bps", self.SPREAD_THRESHOLD_BPS
        ) / 10_000.0

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Analyze price data for ETF statistical arbitrage.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain at least a 'close' column. May also contain
            'close_ivv' and/or 'close_voo' for multi‑symbol calculations.
        symbol : str
            The primary ticker symbol for the generated Signal.

        Returns
        -------
        Signal | None
            Returns a Signal when a spread exceeds the configured threshold,
            otherwise None.
        """
        # Guard against None / empty input
        if data is None or data.empty:
            return None

        # Require a 'close' column and a minimum amount of history for rolling stats
        if "close" not in data.columns or len(data) < 20:
            return None

        close = data["close"]

        # Multi‑symbol: compare two ETFs via ratio
        if "close_ivv" in data.columns or "close_voo" in data.columns:
            # Prefer IVV if both are present
            if "close_ivv" in data.columns:
                other = data["close_ivv"]
            else:
                other = data["close_voo"]

            # Ensure the other series is valid and has sufficient length
            if other is None or other.empty or len(other) < 20:
                return None

            ratio = close / other

            # Need enough points for rolling statistics
            if len(ratio) < 20:
                return None

            ratio_mean = ratio.rolling(20).mean()
            ratio_std = ratio.rolling(20).std()

            # Guard against NaN or near‑zero std deviation
            std_last = ratio_std.iloc[-1]
            if pd.isna(std_last) or std_last < 1e-10:
                return None

            z = (ratio.iloc[-1] - ratio_mean.iloc[-1]) / std_last
            spread = ratio.iloc[-1] - 1.0

            if pd.isna(spread) or abs(spread) < self.spread_threshold:
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
                metadata={
                    "z_score": round(float(z), 4),
                    "spread_bps": round(float(spread) * 10_000, 2),
                },
            )

        # Single‑symbol path: mean‑reversion of the series vs its MA
        ma = close.rolling(20).mean()
        # Avoid division by zero or NaN
        if pd.isna(ma.iloc[-1]) or ma.iloc[-1] == 0:
            return None

        spread = (close - ma) / ma
        spread_last = spread.iloc[-1]
        if pd.isna(spread_last) or abs(spread_last) < self.spread_threshold:
            return None

        side = "sell" if spread_last > 0 else "buy"
        confidence = min(
            0.80,
            0.60 + abs(float(spread_last)) * 100,
        )
        return Signal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={"spread_pct": round(float(spread_last) * 100, 4)},
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Generate back‑test signals based on a rolling z‑score of price vs 20‑day MA.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'close' column.

        Returns
        -------
        BacktestSignals
            Contains boolean Series for entry/exit signals.
        """
        # Defensive checks
        if df is None or df.empty or "close" not in df.columns or len(df) < 20:
            empty_series = pd.Series([False] * len(df) if df is not None else [], dtype=bool)
            return BacktestSignals(
                entries=empty_series,
                exits=empty_series,
                short_entries=empty_series,
                short_exits=empty_series,
            )

        close = df["close"]
        ma = close.rolling(20).mean()
        std = close.rolling(20).std().replace(0, np.nan)
        z = ((close - ma) / std).shift(1)

        entries = z < -2.0
        exits = z > -0.5
        short_entries = z > 2.0
        short_exits = z < 0.5

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )