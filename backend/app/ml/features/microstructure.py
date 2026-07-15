"""
Order book and microstructure features.

Computes limit order book (LOB) features from bid/ask depth data.
Used to enrich ML feature sets with market microstructure signals.

Features:
  - Order book imbalance (bid pressure vs ask pressure)
  - Bid-ask spread in basis points
  - Top-of-book depth ratio
  - PIN proxy (Probability of Informed Trading)
  - Kyle's lambda (price impact coefficient)

Signal utilities:
  - Generate entry/exit signals based on tightened microstructure
    conditions with confirmation filters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class OrderBookFeatures:
    """Compute LOB features from real-time bid/ask depth."""

    def compute_imbalance(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        levels: int = 5,
    ) -> float:
        """
        Order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol).

        Returns value in [-1, 1]. Positive = bid‑heavy (buying pressure).

        Args:
            bids: list of (price, size) pairs, best bid first.
            asks: list of (price, size) pairs, best ask first.
            levels: how many price levels to include.
        """
        if not bids or not asks:
            return 0.0
        bid_vol = sum(float(sz) for _, sz in bids[:levels])
        ask_vol = sum(float(sz) for _, sz in asks[:levels])
        total = bid_vol + ask_vol
        if total <= 0.0:
            return 0.0
        return float((bid_vol - ask_vol) / total)

    def compute_spread_bps(self, best_bid: float, best_ask: float) -> float:
        """
        Bid‑ask spread in basis points: (ask - bid) / mid * 10_000.

        Returns 0.0 for invalid inputs.
        """
        if best_bid <= 0.0 or best_ask <= 0.0 or best_ask <= best_bid:
            return 0.0
        mid = (best_bid + best_ask) / 2.0
        return float((best_ask - best_bid) / mid * 10_000.0)

    def compute_depth_ratio(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> float:
        """
        Top‑of‑book depth ratio: best_bid_size / best_ask_size.

        Values > 1 indicate more liquidity on the bid side.
        Returns 1.0 if either side is empty or ask size is zero.
        """
        if not bids or not asks:
            return 1.0
        best_bid_size = float(bids[0][1]) if bids else 0.0
        best_ask_size = float(asks[0][1]) if asks else 0.0
        if best_ask_size <= 0.0:
            return 1.0
        return float(best_bid_size / best_ask_size)

    def compute_pin_proxy(self, buy_volume: float, sell_volume: float) -> float:
        """
        Probability of Informed Trading proxy.

        PIN = |buy_vol - sell_vol| / (buy_vol + sell_vol)

        Returns value in [0, 1]. Near 1 = highly informed order flow.
        """
        total = buy_volume + sell_volume
        if total <= 0.0:
            return 0.0
        return float(abs(buy_volume - sell_volume) / total)

    def compute_kyle_lambda(
        self,
        price_changes: np.ndarray,
        signed_volumes: np.ndarray,
    ) -> float:
        """
        Kyle's lambda (price impact coefficient).

        Estimated via OLS: delta_price = lambda * signed_volume + epsilon

        Returns lambda (bps per unit volume). Higher = less liquid.
        Returns 0.0 if insufficient data or numerical issues.
        """
        if len(price_changes) < 5 or len(signed_volumes) < 5:
            return 0.0
        try:
            vol = np.array(signed_volumes, dtype=float)
            dp = np.array(price_changes, dtype=float)
            var_vol = np.var(vol)
            if var_vol < 1e-12:
                return 0.0
            cov_matrix = np.cov(dp, vol)
            lam = float(cov_matrix[0, 1] / var_vol)
            return lam
        except Exception:
            return 0.0

    def features_from_snapshot(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        buy_volume: float = 0.0,
        sell_volume: float = 0.0,
        levels: int = 5,
    ) -> dict[str, float]:
        """
        Compute all microstructure features from a single LOB snapshot.

        Returns a dict with keys: imbalance, spread_bps, depth_ratio, pin_proxy.
        """
        best_bid = float(bids[0][0]) if bids else 0.0
        best_ask = float(asks[0][0]) if asks else 0.0

        return {
            "imbalance": self.compute_imbalance(bids, asks, levels),
            "spread_bps": self.compute_spread_bps(best_bid, best_ask),
            "depth_ratio": self.compute_depth_ratio(bids, asks),
            "pin_proxy": self.compute_pin_proxy(buy_volume, sell_volume),
        }


def add_microstructure_features(
    df: pd.DataFrame,
    imbalance_series: Optional[pd.Series] = None,
    spread_bps_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Add microstructure feature columns to an OHLCV DataFrame.

    If real‑time LOB series are provided they are aligned and added.
    Otherwise, proxy features are computed from OHLCV:
      - lob_imbalance: (close - open) / (high - low + 1e-9) — approximates buy/sell pressure
      - spread_bps: (high - low) / close * 10_000 — proxy for intraday spread
    """
    df = df.copy()

    if imbalance_series is not None:
        df["lob_imbalance"] = imbalance_series.reindex(df.index).fillna(0.0)
    else:
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        df["lob_imbalance"] = ((df["close"] - df["open"]) / rng).clip(-1, 1).fillna(0.0)

    if spread_bps_series is not None:
        df["spread_bps"] = spread_bps_series.reindex(df.index).fillna(0.0)
    else:
        df["spread_bps"] = (
            (df["high"] - df["low"])
            / df["close"].replace(0, np.nan)
            * 10_000
        ).fillna(0.0)

    return df


MICROSTRUCTURE_FEATURE_COLS = ["lob_imbalance", "spread_bps"]


class MicrostructureSignal:
    """
    Generate entry and exit signals from microstructure features.

    The logic is deliberately tightened:
      * Entry requires a sustained imbalance above a configurable threshold
        and a spread below a configurable threshold.
      * Confirmation filter enforces that both conditions hold for a
        user‑defined number of consecutive periods.
      * Exit triggers when either condition reverses or when a stop‑loss
        based on spread widening is hit.
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.2,
        spread_threshold_bps: float = 5.0,
        confirm_periods: int = 2,
        exit_spread_multiplier: float = 2.0,
    ) -> None:
        self.imbalance_threshold = imbalance_threshold
        self.spread_threshold_bps = spread_threshold_bps
        self.confirm_periods = max(1, confirm_periods)
        self.exit_spread_multiplier = exit_spread_multiplier

    def _sustained_condition(
        self,
        series: pd.Series,
        condition: pd.Series,
    ) -> pd.Series:
        """
        Apply a rolling window to ensure the condition holds for
        ``self.confirm_periods`` consecutive periods.
        """
        # Convert boolean series to int for rolling sum
        int_cond = condition.astype(int)
        rolling_sum = int_cond.rolling(self.confirm_periods, min_periods=self.confirm_periods).sum()
        return rolling_sum == self.confirm_periods

    def generate_signals(
        self,
        df: pd.DataFrame,
        imbalance_col: str = "lob_imbalance",
        spread_col: str = "spread_bps",
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Produce entry and exit signal series aligned with ``df``.

        Args:
            df: DataFrame containing microstructure feature columns.
            imbalance_col: Column name for imbalance feature.
            spread_col: Column name for spread feature.

        Returns:
            (entry_signal, exit_signal) where each series contains 1 for a
            signal and 0 otherwise.
        """
        if imbalance_col not in df.columns or spread_col not in df.columns:
            raise ValueError("Required microstructure columns missing from DataFrame")

        # Entry filters
        imbalance_ok = df[imbalance_col] >= self.imbalance_threshold
        spread_ok = df[spread_col] <= self.spread_threshold_bps

        entry_condition = imbalance_ok & spread_ok
        entry_signal = self._sustained_condition(df[imbalance_col], entry_condition).astype(int)

        # Exit filters
        # Exit when imbalance falls below half the entry threshold or spread exceeds
        # a multiple of the entry spread threshold.
        imbalance_exit = df[imbalance_col] < (self.imbalance_threshold / 2)
        spread_exit = df[spread_col] > (self.spread_threshold_bps * self.exit_spread_multiplier)

        exit_condition = imbalance_exit | spread_exit
        exit_signal = self._sustained_condition(df[spread_col], exit_condition).astype(int)

        return entry_signal, exit_signal