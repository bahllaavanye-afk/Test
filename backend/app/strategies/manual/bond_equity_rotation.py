"""
Bond-Equity Tactical Rotation Strategy.

Rotates between TLT (long-duration Treasuries) and SPY (US equities)
based on the 30-day rolling correlation between equity returns and VIX changes.

Logic:
  - When correlation(SPY returns, ΔVIX) > 0 → fear regime → rotate to TLT
  - When correlation(SPY returns, ΔVIX) < 0 → risk-on regime → stay in SPY

This exploits the "flight to safety" dynamic: when stocks and volatility
move together, bonds offer diversification and alpha.
"""
import logging
import time
import numpy as np
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals

logger = logging.getLogger(__name__)


class BondEquityRotationStrategy(AbstractStrategy):
    name = "bond_equity_rotation"
    display_name = "Bond-Equity Tactical Rotation (TLT/SPY)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0  # daily signal, checked hourly

    CORR_WINDOW = 30      # days for rolling correlation
    CORR_THRESHOLD = 0.1  # correlation threshold

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self.corr_window = p.get("corr_window", self.CORR_WINDOW)
        self.corr_threshold = p.get("corr_threshold", self.CORR_THRESHOLD)
        # monitoring state
        self._signal_count = 0

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Expects data with columns: 'close', and optionally 'vix_close'.
        If vix_close is absent, falls back to VIX proxy from realized vol.
        """
        start_time = time.perf_counter()

        if "close" not in data.columns or len(data) < self.corr_window + 5:
            logger.info(
                "BondEquityRotation analyze skipped",
                extra={
                    "strategy": self.name,
                    "symbol": symbol,
                    "reason": "insufficient data",
                    "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                },
            )
            return None

        close = data["close"]
        ret = close.pct_change().dropna()

        if "vix_close" in data.columns:
            vix = data["vix_close"].dropna()
            dvix = vix.pct_change().dropna()
            # Align
            idx = ret.index.intersection(dvix.index)
            if len(idx) < self.corr_window:
                logger.info(
                    "BondEquityRotation analyze skipped",
                    extra={
                        "strategy": self.name,
                        "symbol": symbol,
                        "reason": "insufficient overlapping data",
                        "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                    },
                )
                return None
            corr = float(ret.loc[idx].rolling(self.corr_window).corr(dvix.loc[idx]).iloc[-1])
        else:
            # Proxy: use realized vol as VIX substitute
            rv = ret.rolling(5).std() * np.sqrt(252)
            drv = rv.pct_change().dropna()
            idx = ret.index.intersection(drv.index)
            if len(idx) < self.corr_window:
                logger.info(
                    "BondEquityRotation analyze skipped",
                    extra={
                        "strategy": self.name,
                        "symbol": symbol,
                        "reason": "insufficient overlapping proxy data",
                        "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                    },
                )
                return None
            corr = float(ret.loc[idx].rolling(self.corr_window).corr(drv.loc[idx]).iloc[-1])

        if np.isnan(corr):
            logger.info(
                "BondEquityRotation analyze skipped",
                extra={
                    "strategy": self.name,
                    "symbol": symbol,
                    "reason": "correlation NaN",
                    "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                },
            )
            return None

        signal = None
        if corr > self.corr_threshold:
            signal = Signal(
                symbol=symbol,
                side="sell",
                confidence=min(0.80, 0.60 + corr * 0.4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"regime": "risk_off", "corr_spy_vix": round(corr, 4)},
            )
        elif corr < -self.corr_threshold:
            signal = Signal(
                symbol=symbol,
                side="buy",
                confidence=min(0.80, 0.60 + abs(corr) * 0.4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={"regime": "risk_on", "corr_spy_vix": round(corr, 4)},
            )

        if signal:
            self._signal_count += 1

        logger.info(
            "BondEquityRotation analyze completed",
            extra={
                "strategy": self.name,
                "symbol": symbol,
                "signal_generated": bool(signal),
                "signal_count": self._signal_count,
                "corr_spy_vix": round(corr, 4),
                "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                # P&L is not available at analysis time; placeholder for downstream logging
                "pnl": None,
            },
        )
        return signal

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        start_time = time.perf_counter()

        close = df["close"]
        ret = close.pct_change()

        if "vix_close" in df.columns:
            dvix = df["vix_close"].pct_change()
        else:
            rv = ret.rolling(5).std()
            dvix = rv.pct_change()

        # Rolling correlation, shifted to prevent lookahead
        corr = ret.rolling(self.corr_window).corr(dvix).shift(1)

        # Risk-on: negative corr → buy equity
        entries = corr < -self.corr_threshold
        exits = corr > 0.0
        # Risk-off: positive corr → short equity (go to bonds)
        short_entries = corr > self.corr_threshold
        short_exits = corr < 0.0

        backtest = BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )

        logger.info(
            "BondEquityRotation backtest completed",
            extra={
                "strategy": self.name,
                "entries_count": int(backtest.entries.sum()),
                "exits_count": int(backtest.exits.sum()),
                "short_entries_count": int(backtest.short_entries.sum()),
                "short_exits_count": int(backtest.short_exits.sum()),
                "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                # P&L is computed downstream; placeholder for consistency
                "pnl": None,
            },
        )
        return backtest