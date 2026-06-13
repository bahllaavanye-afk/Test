"""Avellaneda-Stoikov market making strategy with inventory management."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class AvellanedaStoikovMM(AbstractStrategy):
    """
    Avellaneda-Stoikov optimal market making.

    reservation_price = mid - gamma * sigma^2 * (q - q_target) * T
    spread = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/kappa)

    Uses Binance 1-min OHLCV for sigma estimation.
    Falls back to symmetric quoting if no inventory data.
    """

    name = "avellaneda_stoikov_mm"
    display_name = "Avellaneda-Stoikov Market Making"
    market_type = "crypto"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 60.0  # 1-minute ticks

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        p = params or {}
        self.gamma = float(p.get("gamma", 0.1))       # risk aversion
        self.kappa = float(p.get("kappa", 1.5))        # order arrival rate
        self.T = float(p.get("T", 300.0))              # time horizon seconds
        self.max_inventory = float(p.get("max_inventory", 0.1))
        self.inventory: float = float(p.get("inventory", 0.0))

    def description(self) -> str:
        return (
            "Avellaneda-Stoikov optimal market making with inventory skew. "
            f"gamma={self.gamma}, kappa={self.kappa}, T={self.T}s. "
            "Source: Avellaneda & Stoikov (2008) 'High-frequency trading in a limit order book'."
        )

    def _estimate_sigma(self, df: pd.DataFrame) -> float:
        """Estimate short-term volatility from close returns."""
        if len(df) < 5:
            return 0.001
        returns = df["close"].pct_change().dropna()
        return float(returns.std()) if len(returns) > 0 else 0.001

    def _compute_quotes(
        self, mid: float, sigma: float, q: float
    ) -> tuple[float, float, float]:
        """
        Returns (bid, ask, spread_bps).

        reservation_price = mid - gamma * sigma^2 * q * T
        half_spread = (gamma * sigma^2 * T) / 2 + (1/gamma) * ln(1 + gamma/kappa)
        """
        kappa_safe = max(self.kappa, 1e-8)
        gamma_safe = max(self.gamma, 1e-8)
        half_spread = (gamma_safe * sigma ** 2 * self.T) / 2.0 + (
            1.0 / gamma_safe
        ) * math.log(1.0 + gamma_safe / kappa_safe)
        reservation = mid - gamma_safe * sigma ** 2 * q * self.T
        bid = reservation - half_spread
        ask = reservation + half_spread
        spread_bps = (ask - bid) / mid * 10_000.0 if mid > 0 else 0.0
        return bid, ask, spread_bps

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Emit a BUY signal when the computed spread is wide enough to be profitable.
        In production the execution layer places both bid and ask orders.
        """
        if data is None or "close" not in data.columns or len(data) < 10:
            return None

        mid = float(data["close"].iloc[-1])
        if mid <= 0:
            return None

        sigma = self._estimate_sigma(data)
        bid, ask, spread_bps = self._compute_quotes(mid, sigma, self.inventory)

        if spread_bps > 5.0:  # only trade when spread is wide enough to be profitable
            confidence = min(0.6 + spread_bps / 1000.0, 0.9)
            return Signal(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                symbol=symbol,
                side="buy",
                confidence=confidence,
                target_price=bid,
                metadata={
                    "bid": round(bid, 6),
                    "ask": round(ask, 6),
                    "spread_bps": round(spread_bps, 2),
                    "sigma": round(sigma, 6),
                    "inventory": self.inventory,
                    "reservation_price": round((bid + ask) / 2.0, 6),
                    "action": "post_bid_ask",
                    "order_type": "limit",
                },
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Vectorized backtest proxy:
        Signal = 1 (earn spread) whenever rolling spread_bps > 5 bp.
        Uses rolling 20-bar sigma estimate with shift(1) to prevent lookahead.
        """
        false_series = pd.Series(False, index=df.index)
        default = BacktestSignals(
            entries=false_series,
            exits=false_series,
            short_entries=false_series,
            short_exits=false_series,
        )

        if "close" not in df.columns or len(df) < 22:
            return default

        close = df["close"].astype(float)
        returns = close.pct_change()
        window = 20
        rolling_sigma = returns.rolling(window, min_periods=window // 2).std()
        mid = close

        # Compute spread_bps for each bar (q=0, neutral inventory)
        kappa_safe = max(self.kappa, 1e-8)
        gamma_safe = max(self.gamma, 1e-8)
        half_spread = (gamma_safe * rolling_sigma ** 2 * self.T) / 2.0 + (
            1.0 / gamma_safe
        ) * np.log1p(gamma_safe / kappa_safe)
        spread = half_spread * 2.0
        spread_bps = spread / mid.clip(lower=1e-8) * 10_000.0

        # shift(1) — no lookahead bias
        spread_bps_lag = spread_bps.shift(1)

        # Active (quoting) when spread is wide enough
        entries = (spread_bps_lag > 5.0).fillna(False).astype(bool)
        # Exit (stop quoting) when spread collapses
        exits = (spread_bps_lag <= 2.0).fillna(False).astype(bool)

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=false_series,
            short_exits=false_series,
        )
