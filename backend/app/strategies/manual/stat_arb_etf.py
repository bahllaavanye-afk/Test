"""
ETF Statistical Arbitrage — SPY vs IVV vs VOO (same index, different ETFs).

When spread between any two ETFs tracking the same index exceeds 2bps,
buy the cheaper ETF and sell the dearer one.

These three ETFs all track the S&P 500, so their prices should co-move.
Small mispricings are arbitraged away quickly by authorized participants,
making this a near-risk-free short-duration play when spreads widen.
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
        data must contain columns: 'close_spy', 'close_ivv', 'close_voo'
        (or fall back to 'close' for the primary symbol).
        For single-symbol mode, we use a rolling z-score on the price ratio.
        """
        if "close" not in data.columns or len(data) < 30:
            return None

        close = data["close"]

        # Multi-symbol: compare two ETFs via ratio
        if "close_ivv" in data.columns:
            ratio = close / data["close_ivv"]
        elif "close_voo" in data.columns:
            ratio = close / data["close_voo"]
        else:
            # Single-symbol: use mean-reversion of the series vs its MA
            ma = close.rolling(20).mean()
            spread = (close - ma) / ma
            if abs(spread.iloc[-1]) < self.spread_threshold:
                return None
            side = "sell" if spread.iloc[-1] > 0 else "buy"
            return Signal(
                symbol=symbol,
                side=side,
                confidence=min(0.80, 0.60 + abs(float(spread.iloc[-1])) * 100),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"spread_pct": round(float(spread.iloc[-1]) * 100, 4)},
            )

        # Multi-symbol path
        ratio_mean = ratio.rolling(20).mean()
        ratio_std = ratio.rolling(20).std()
        if ratio_std.iloc[-1] < 1e-10:
            return None
        z = (ratio.iloc[-1] - ratio_mean.iloc[-1]) / ratio_std.iloc[-1]
        spread = ratio.iloc[-1] - 1.0

        if abs(spread) < self.spread_threshold:
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
            metadata={"z_score": round(float(z), 4), "spread_bps": round(float(spread) * 10_000, 2)},
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        close = df["close"]
        # Rolling z-score of price vs 20-day MA, shifted to avoid lookahead
        ma = close.rolling(20).mean()
        std = close.rolling(20).std().replace(0, np.nan)
        z = ((close - ma) / std).shift(1)

        entries = z < -2.0          # price well below MA → buy (mean-revert up)
        exits = z > -0.5
        short_entries = z > 2.0
        short_exits = z < 0.5

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )
