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
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from functools import lru_cache

# Constants
DEFAULT_LEVELS: int = 5
BASIS_POINTS_MULTIPLIER: float = 10_000.0
MIN_SAMPLE_SIZE: int = 5
VAR_VOLUME_EPS: float = 1e-12

COL_HIGH: str = "high"
COL_LOW: str = "low"
COL_CLOSE: str = "close"
COL_OPEN: str = "open"
COL_LOB_IMBALANCE: str = "lob_imbalance"
COL_SPREAD_BPS: str = "spread_bps"

MICROSTRUCTURE_FEATURE_COLS = [COL_LOB_IMBALANCE, COL_SPREAD_BPS]


class OrderBookFeatures:
    """Compute LOB features from real-time bid/ask depth."""

    @staticmethod
    @lru_cache(maxsize=1024)
    def _slice_volumes(
        data: tuple[tuple[float, float], ...],
        levels: int,
    ) -> np.ndarray:
        """Extract volumes up to `levels` and return as NumPy array.

        Handles cases where `levels` exceeds the length of `data` by truncating.
        """
        if not data or levels <= 0:
            return np.array([], dtype=float)

        effective_levels = min(levels, len(data))
        # np.fromiter with count=None allows variable length iteration safely
        arr = np.fromiter((sz for _, sz in data[:effective_levels]), dtype=float)
        return arr

    def compute_imbalance(
        self,
        bids: list[tuple[float, float]] | None,
        asks: list[tuple[float, float]] | None,
        levels: int = DEFAULT_LEVELS,
    ) -> float:
        """
        Order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol).
        Returns value in [-1, 1]. Positive = bid-heavy (buying pressure).

        Args:
            bids: list of (price, size) pairs, best bid first
            asks: list of (price, size) pairs, best ask first
            levels: how many price levels to include
        """
        if not bids or not asks or levels <= 0:
            return 0.0

        bid_tuple = tuple(bids)
        ask_tuple = tuple(asks)

        bid_vols = self._slice_volumes(bid_tuple, levels)
        ask_vols = self._slice_volumes(ask_tuple, levels)

        if bid_vols.size == 0 or ask_vols.size == 0:
            return 0.0

        bid_vol = float(bid_vols.sum())
        ask_vol = float(ask_vols.sum())
        total = bid_vol + ask_vol
        if total <= 0.0:
            return 0.0
        return float((bid_vol - ask_vol) / total)

    def compute_spread_bps(self, best_bid: float | None, best_ask: float | None) -> float:
        """
        Bid-ask spread in basis points: (ask - bid) / mid * BASIS_POINTS_MULTIPLIER.
        Returns 0.0 for invalid inputs.
        """
        if best_bid is None or best_ask is None:
            return 0.0
        if best_bid <= 0.0 or best_ask <= 0.0 or best_ask <= best_bid:
            return 0.0
        mid = (best_bid + best_ask) * 0.5
        return float((best_ask - best_bid) / mid * BASIS_POINTS_MULTIPLIER)

    def compute_depth_ratio(
        self,
        bids: list[tuple[float, float]] | None,
        asks: list[tuple[float, float]] | None,
    ) -> float:
        """
        Top-of-book depth ratio: best_bid_size / best_ask_size.
        Values > 1 indicate more liquidity on bid side.
        Returns 1.0 if either side is empty or invalid.
        """
        if not bids or not asks:
            return 1.0
        try:
            best_bid_size = float(bids[0][1])
            best_ask_size = float(asks[0][1])
        except (IndexError, TypeError, ValueError):
            return 1.0
        if best_ask_size <= 0.0:
            return 1.0
        return float(best_bid_size / best_ask_size)

    def compute_pin_proxy(self, buy_volume: float | None, sell_volume: float | None) -> float:
        """
        Probability of Informed Trading proxy.
        PIN = |buy_vol - sell_vol| / (buy_vol + sell_vol)
        Returns value in [0, 1]. Near 1 = highly informed order flow.
        """
        buy_volume = float(buy_volume) if buy_volume is not None else 0.0
        sell_volume = float(sell_volume) if sell_volume is not None else 0.0
        total = buy_volume + sell_volume
        if total <= 0.0:
            return 0.0
        return float(abs(buy_volume - sell_volume) / total)

    def compute_kyle_lambda(
        self,
        price_changes: np.ndarray | None,
        signed_volumes: np.ndarray | None,
    ) -> float:
        """
        Kyle's lambda (price impact coefficient).
        Estimated via OLS: delta_price = lambda * signed_volume + epsilon

        Returns lambda (bps per unit volume). Higher = less liquid.
        Returns 0.0 if insufficient or invalid data.
        """
        if price_changes is None or signed_volumes is None:
            return 0.0

        if price_changes.size < MIN_SAMPLE_SIZE or signed_volumes.size < MIN_SAMPLE_SIZE:
            return 0.0

        # Ensure both arrays have the same length; truncate to the shorter one.
        min_len = min(price_changes.size, signed_volumes.size)
        if min_len < MIN_SAMPLE_SIZE:
            return 0.0

        try:
            vol = np.asarray(signed_volumes[:min_len], dtype=float)
            dp = np.asarray(price_changes[:min_len], dtype=float)

            var_vol = np.var(vol)
            if var_vol < VAR_VOLUME_EPS:
                return 0.0

            cov = np.mean(dp * vol) - np.mean(dp) * np.mean(vol)
            lam = float(cov / var_vol)
            return lam
        except Exception:
            return 0.0

    def features_from_snapshot(
        self,
        bids: list[tuple[float, float]] | None,
        asks: list[tuple[float, float]] | None,
        buy_volume: float = 0.0,
        sell_volume: float = 0.0,
        levels: int = DEFAULT_LEVELS,
    ) -> dict[str, float]:
        """
        Compute all microstructure features from a single LOB snapshot.

        Returns:
            dict with keys: imbalance, spread_bps, depth_ratio, pin_proxy
        """
        bids = bids or []
        asks = asks or []

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
    imbalance_series: pd.Series | None = None,
    spread_bps_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Add microstructure feature columns to an OHLCV DataFrame.

    If real-time LOB series are provided they are aligned and added.
    Otherwise, proxy features are computed from OHLCV:
      - volume_imbalance_proxy: (close - open) / (high - low)  — approximates buy/sell pressure
      - spread_bps_proxy: (high - low) / close * BASIS_POINTS_MULTIPLIER — proxy for intraday spread
    """
    df = df.copy()

    # Ensure required columns exist; if missing, create with NaNs to avoid KeyError.
    for col in (COL_HIGH, COL_LOW, COL_CLOSE, COL_OPEN):
        if col not in df.columns:
            df[col] = np.nan

    if imbalance_series is not None:
        df[COL_LOB_IMBALANCE] = imbalance_series.reindex(df.index).fillna(0.0)
    else:
        rng = (df[COL_HIGH] - df[COL_LOW]).replace(0, np.nan)
        df[COL_LOB_IMBALANCE] = ((df[COL_CLOSE] - df[COL_OPEN]) / rng).clip(-1, 1).fillna(0.0)

    if spread_bps_series is not None:
        df[COL_SPREAD_BPS] = spread_bps_series.reindex(df.index).fillna(0.0)
    else:
        close_nonzero = df[COL_CLOSE].replace(0, np.nan)
        df[COL_SPREAD_BPS] = ((df[COL_HIGH] - df[COL_LOW]) / close_nonzero * BASIS_POINTS_MULTIPLIER).fillna(0.0)

    return df