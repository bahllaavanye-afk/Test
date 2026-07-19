"""
Bollinger Band Mean Reversion Strategy.
Enter when price touches lower/upper band; exit at middle band.
"""
import time
import logging
import pandas as pd
import app.ml.features.pandas_ta_compat as ta
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals

logger = logging.getLogger("strategies.mean_reversion")


class MeanReversionStrategy(AbstractStrategy):
    name = "mean_reversion"
    display_name = "Bollinger Band Mean Reversion"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 300.0

    DEFAULT_PARAMS = {
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        effective = {**self.DEFAULT_PARAMS, **(params or {})}
        self.bb_period = effective["bb_period"]
        self.bb_std = effective["bb_std"]
        self.rsi_period = effective["rsi_period"]
        self.rsi_oversold = effective["rsi_oversold"]
        self.rsi_overbought = effective["rsi_overbought"]

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        start_time = time.time()
        signal = None

        if "close" not in data.columns or len(data) < self.bb_period + 5:
            logger.info(
                "MeanReversion analyze skipped",
                extra={
                    "symbol": symbol,
                    "signal_count": 0,
                    "execution_time_ms": int((time.time() - start_time) * 1000),
                    "pnl": None,
                },
            )
            return None

        close = data["close"]
        bb = ta.bbands(close, length=self.bb_period, std=self.bb_std)
        rsi = ta.rsi(close, length=self.rsi_period)

        if bb is None or rsi is None:
            logger.info(
                "MeanReversion analyze missing indicators",
                extra={
                    "symbol": symbol,
                    "signal_count": 0,
                    "execution_time_ms": int((time.time() - start_time) * 1000),
                    "pnl": None,
                },
            )
            return None

        lower = bb[f"BBL_{self.bb_period}_{self.bb_std}"].iloc[-1]
        upper = bb[f"BBU_{self.bb_period}_{self.bb_std}"].iloc[-1]
        mid = bb[f"BBM_{self.bb_period}_{self.bb_std}"].iloc[-1]
        price = close.iloc[-1]
        rsi_val = rsi.iloc[-1]

        if price <= lower and rsi_val < self.rsi_oversold:
            pct_below = (lower - price) / lower
            confidence = min(0.88, 0.55 + pct_below * 5)
            signal = Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                target_price=mid,
                metadata={"rsi": round(rsi_val, 2), "bb_position": "lower"},
            )
        elif price >= upper and rsi_val > self.rsi_overbought:
            pct_above = (price - upper) / upper
            confidence = min(0.88, 0.55 + pct_above * 5)
            signal = Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                target_price=mid,
                metadata={"rsi": round(rsi_val, 2), "bb_position": "upper"},
            )

        exec_time_ms = int((time.time() - start_time) * 1000)
        signal_count = 1 if signal else 0
        logger.info(
            "MeanReversion analyze completed",
            extra={
                "symbol": symbol,
                "signal_count": signal_count,
                "execution_time_ms": exec_time_ms,
                "pnl": None,
                "side": signal.side if signal else None,
                "confidence": signal.confidence if signal else None,
            },
        )
        return signal

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        start_time = time.time()

        close = df["close"]
        bb = ta.bbands(close, length=self.bb_period, std=self.bb_std)
        rsi = ta.rsi(close, length=self.rsi_period)

        if bb is None or rsi is None:
            empty = pd.Series(False, index=df.index)
            logger.info(
                "MeanReversion backtest skipped",
                extra={
                    "signal_count": 0,
                    "execution_time_ms": int((time.time() - start_time) * 1000),
                    "pnl": None,
                },
            )
            return BacktestSignals(entries=empty, exits=empty)

        lower = bb[f"BBL_{self.bb_period}_{self.bb_std}"].shift(1)
        upper = bb[f"BBU_{self.bb_period}_{self.bb_std}"].shift(1)
        mid = bb[f"BBM_{self.bb_period}_{self.bb_std}"].shift(1)
        rsi_s = rsi.shift(1)
        close_s = close.shift(1)

        entries = (close_s <= lower) & (rsi_s < self.rsi_oversold)
        exits = close_s >= mid
        short_entries = (close_s >= upper) & (rsi_s > self.rsi_overbought)
        short_exits = close_s <= mid

        backtest = BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )

        exec_time_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "MeanReversion backtest completed",
            extra={
                "signal_count": int(backtest.entries.sum() + backtest.short_entries.sum()),
                "execution_time_ms": exec_time_ms,
                "pnl": None,
            },
        )
        return backtest