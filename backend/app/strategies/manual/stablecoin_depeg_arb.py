"""
Stablecoin Depeg Arbitrage

Source:
  Gorton & Zhang (2023) "Taming Wildcat Stablecoins"
    University of Chicago Law Review.
  Lyons & Viswanath-Natraj (2023) "Stablecoin Depegging and Currency Arbitrage"
    Journal of International Money and Finance, Vol. 131.

Edge: USDC, USDT, DAI and BUSD occasionally trade off their $1.00 peg on Binance
due to regional demand shocks, bank runs (SVB March 2023: USDC hit $0.88), or
liquidity crises. The peg is hard-enforced by arbitrageurs but with a lag of minutes
to hours. Lyons & Viswanath-Natraj (2023) document USDT/USD deviations with a
half-life of 4–12 hours on Binance. Round-trip friction is ~2 bps vs. deviations of
10–100 bps in normal regimes. In stress events, deviations reach 200–1200 bps with
the same reversion dynamic.

Risk: Structural depeg (issuer insolvency) causes 100% loss. Hard circuit breaker:
max 2% of AUM per stablecoin; never hold through a confirmed bank-run event.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class StablecoinDepegArbStrategy(AbstractStrategy):
    name = "stablecoin_depeg_arb"
    display_name = "Stablecoin Depeg Arbitrage"
    market_type = "crypto"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 300.0   # 5-minute bars

    # Minimum absolute premium (bps) to trade — must exceed 2× exchange fee
    ENTRY_BPS   = 15.0   # 15 bps minimum deviation to enter
    EXIT_BPS    = 2.0    # exit when within 2 bps of peg
    STOP_BPS    = 200.0  # stop if deviation widens to 200 bps (structural risk)
    Z_THRESH    = 2.0    # z-score confirmation threshold
    LOOKBACK    = 60     # 60 bars rolling window for z-score (~5 hours on 5-min bars)
    MIN_PERSIST = 3      # must persist for ≥3 bars before entry

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.entry_bps   = float(p.get("entry_bps",   self.ENTRY_BPS))
        self.exit_bps    = float(p.get("exit_bps",    self.EXIT_BPS))
        self.stop_bps    = float(p.get("stop_bps",    self.STOP_BPS))
        self.z_thresh    = float(p.get("z_thresh",    self.Z_THRESH))
        self.lookback    = int(p.get("lookback",      self.LOOKBACK))
        self.min_persist = int(p.get("min_persist",   self.MIN_PERSIST))

    def description(self) -> str:
        return (
            "Buys temporarily-depegged stablecoins (USDC, DAI, TUSD) vs USDT when "
            f"discount > {self.entry_bps} bps and z-score > {self.z_thresh}σ, "
            "targeting reversion to parity. Source: Lyons & Viswanath-Natraj (2023)."
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        false_series = pd.Series(False, index=df.index)
        default = BacktestSignals(
            entries=false_series,
            exits=false_series,
            short_entries=false_series,
            short_exits=false_series,
        )

        if "close" not in df.columns or len(df) < self.lookback + self.min_persist:
            return default

        # treat close as the stablecoin / USDT price (should be ~1.0)
        price = df["close"].astype(float)

        # Premium in bps: positive = stablecoin trading ABOVE USDT (sell opp)
        #                 negative = stablecoin trading BELOW USDT (buy opp)
        prem_bps = (price - 1.0) * 10_000

        # Rolling z-score
        roll_mean = prem_bps.rolling(self.lookback, min_periods=20).mean()
        roll_std  = prem_bps.rolling(self.lookback, min_periods=20).std().clip(lower=0.1)
        prem_z    = (prem_bps - roll_mean) / roll_std

        # Persistence: premium in same direction for min_persist consecutive bars
        deep_discount = (prem_bps < -self.entry_bps)
        deep_premium  = (prem_bps > self.entry_bps)
        persist_disc  = deep_discount.rolling(self.min_persist).sum() >= self.min_persist
        persist_prem  = deep_premium.rolling(self.min_persist).sum() >= self.min_persist

        # Hard stop: structural risk if deviation > stop_bps
        not_structural = prem_bps.abs() < self.stop_bps

        # Entry: buy cheap stablecoin (USDC/DAI discount → go long)
        entries       = (persist_disc & (prem_z < -self.z_thresh) & not_structural).shift(1).fillna(False).astype(bool)
        # Entry: sell expensive stablecoin (stablecoin premium → go short / hold USDT)
        short_entries = (persist_prem & (prem_z > self.z_thresh) & not_structural).shift(1).fillna(False).astype(bool)

        # Exit: premium reverts to near-zero
        exits       = (prem_bps.abs() < self.exit_bps).shift(1).fillna(True).astype(bool)
        short_exits = exits.copy()

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )

    async def analyze(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        if "close" not in df.columns or len(df) < self.lookback + self.min_persist:
            return None

        price = df["close"].astype(float)
        prem_bps = (price - 1.0) * 10_000

        roll_mean = prem_bps.rolling(self.lookback, min_periods=20).mean()
        roll_std  = prem_bps.rolling(self.lookback, min_periods=20).std().clip(lower=0.1)
        prem_z    = (prem_bps - roll_mean) / roll_std

        current_prem = float(prem_bps.iloc[-1])
        current_z    = float(prem_z.iloc[-1])
        current_price = float(price.iloc[-1])

        # Structural depeg safety gate
        if abs(current_prem) >= self.stop_bps:
            return None

        if current_prem < -self.entry_bps and current_z < -self.z_thresh:
            # Stablecoin is cheap → buy it (go long)
            confidence = min(abs(current_z) / (self.z_thresh * 3), 0.90)
            return Signal(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                symbol=symbol,
                side="buy",
                confidence=confidence,
                target_price=current_price,
                stop_loss=current_price * (1 - self.stop_bps / 10_000),
                take_profit=1.0,  # target: return to peg
                order_type="limit",
                quantity=None,
                metadata={"premium_bps": round(current_prem, 2), "z_score": round(current_z, 2)},
            )

        if current_prem > self.entry_bps and current_z > self.z_thresh:
            # Stablecoin is expensive → short it
            confidence = min(abs(current_z) / (self.z_thresh * 3), 0.90)
            return Signal(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                symbol=symbol,
                side="sell",
                confidence=confidence,
                target_price=current_price,
                stop_loss=current_price * (1 + self.stop_bps / 10_000),
                take_profit=1.0,
                order_type="limit",
                quantity=None,
                metadata={"premium_bps": round(current_prem, 2), "z_score": round(current_z, 2)},
            )

        return None
