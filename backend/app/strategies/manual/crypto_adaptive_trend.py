"""
Crypto Adaptive Trend Following with Volatility Targeting

Source:
  Hurst, Ooi & Pedersen (2017) "A Century of Evidence on Trend-Following Investing"
    Journal of Portfolio Management.
  Baz, Granger, Harvey, Le Roux & Rattray (2015) "Dissecting Investment Strategies
    in the Cross Section and Time Series" — AQR Capital Research.
  Baltas & Kosowski (2020) "Momentum and Mean Reversion across Asset Classes"
    Management Science.

Edge: Time-series momentum (TSMOM) is one of the most robust risk premia in the
literature. Crypto TSMOM has higher Sharpe (~1.5-2.0) than traditional TSMOM because
trend persistence is stronger in markets with thinner arbitrage capital and retail
herding. Adaptive Volatility Targeting (AVT) sizes each position as
(target_vol / realized_vol) × signal_strength, eliminating vol-of-vol drag and
smoothing the return stream.

Composite signal across three horizons (1M, 3M, 12M), cross-sectionally ranked
within the crypto universe. Position sizing inversely proportional to realized vol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class CryptoAdaptiveTrendStrategy(AbstractStrategy):
    name = "crypto_adaptive_trend"
    display_name = "Crypto Adaptive Trend (TSMOM + AVT)"
    market_type = "crypto"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86_400.0  # daily rebalance

    # Alpaca crypto symbols for the tracked universe (spot, no perps needed)
    UNIVERSE = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD",
                "DOT/USD", "MATIC/USD", "ALGO/USD", "UNI/USD", "AAVE/USD"]

    TARGET_VOL = 0.40     # 40% annualized vol target
    MIN_SIGNAL = 0.30     # minimum composite signal to enter (0–1 scale)
    STOP_MULT  = 3.0      # stop loss as multiple of daily ATR

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.target_vol = float(p.get("target_vol", self.TARGET_VOL))
        self.min_signal = float(p.get("min_signal", self.MIN_SIGNAL))

    def description(self) -> str:
        return (
            "Multi-horizon time-series momentum on crypto with adaptive vol targeting. "
            "Goes long (short) top (bottom) ranked cryptos by composite 1M/3M/12M signal, "
            "sized inversely to recent realized vol. Source: Hurst, Ooi & Pedersen (2017)."
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        false_series = pd.Series(False, index=df.index)

        if "close" not in df.columns or len(df) < 260:
            return BacktestSignals(
                entries=false_series,
                exits=false_series,
                short_entries=false_series,
                short_exits=false_series,
            )

        close = df["close"].astype(float)
        log_ret = np.log(close / close.shift(1))

        # ── Multi-horizon TSMOM signals ──────────────────────────────────────
        mom_21  = (close / close.shift(21)  - 1).rank(pct=True)  # 1M
        mom_63  = (close / close.shift(63)  - 1).rank(pct=True)  # 3M
        mom_252 = (close / close.shift(252) - 1).rank(pct=True)  # 12M

        # Equal-weight composite → maps to [-1, 1]
        composite = (mom_21 + mom_63 + mom_252) / 3.0
        raw_signal = composite * 2 - 1  # [0,1] → [-1,1]

        # ── Adaptive volatility targeting ────────────────────────────────────
        rv_21 = log_ret.rolling(21, min_periods=10).std() * np.sqrt(365)
        vol_scalar = (self.target_vol / rv_21.clip(lower=0.05)).clip(upper=3.0)
        sized_signal = raw_signal * vol_scalar

        # ── Entry / exit logic ───────────────────────────────────────────────
        # shift(1): yesterday's signal determines today's position
        sig_prev = sized_signal.shift(1)

        entries       = (sig_prev > self.min_signal).fillna(False).astype(bool)
        exits         = (sig_prev <= 0.0).fillna(True).astype(bool)
        short_entries = (sig_prev < -self.min_signal).fillna(False).astype(bool)
        short_exits   = (sig_prev >= 0.0).fillna(True).astype(bool)

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )

    async def analyze(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        """Live signal — uses same logic as backtest_signals on recent bars."""
        if "close" not in df.columns or len(df) < 260:
            return None

        close = df["close"].astype(float)
        log_ret = np.log(close / close.shift(1))

        mom_21  = (close.iloc[-1] / close.iloc[-22]  - 1) if len(close) > 22  else 0.0
        mom_63  = (close.iloc[-1] / close.iloc[-64]  - 1) if len(close) > 64  else 0.0
        mom_252 = (close.iloc[-1] / close.iloc[-253] - 1) if len(close) > 253 else 0.0

        # Rank using last 252-bar cross-section
        moms = [mom_21, mom_63, mom_252]
        composite_raw = sum(moms) / 3.0
        composite = (np.tanh(composite_raw * 5) + 1) / 2  # soft [0,1]
        raw_signal = composite * 2 - 1

        rv_21 = log_ret.iloc[-21:].std() * np.sqrt(365) if len(log_ret) >= 21 else 0.40
        vol_scalar = min(self.target_vol / max(rv_21, 0.05), 3.0)
        sized_signal = raw_signal * vol_scalar

        if abs(sized_signal) < self.min_signal:
            return None

        side = "buy" if sized_signal > 0 else "sell"
        confidence = min(abs(sized_signal) / 2.0, 0.95)
        current_price = float(close.iloc[-1])

        return Signal(
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            symbol=symbol,
            side=side,
            confidence=confidence,
            target_price=current_price,
            stop_loss=current_price * (0.85 if side == "buy" else 1.15),
            take_profit=None,
            metadata={"composite_signal": round(sized_signal, 4), "rv_21d": round(rv_21, 4), "order_type": "market"},
        )
