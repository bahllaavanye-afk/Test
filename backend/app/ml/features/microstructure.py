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
    """Utility class for computing limit order book (LOB) microstructure features."""

    @staticmethod
    @lru_cache(maxsize=1024)
    def _slice_volumes(
        data: tuple[tuple[float, float], ...],
        levels: int,
    ) -> np.ndarray:
        """
        Extract volumes up to ``levels`` from a tuple of (price, size) pairs.

        Args:
            data: Tuple of (price, size) pairs representing a side of the order book.
            levels: Number of price levels to include.

        Returns:
            NumPy array of the extracted sizes (float dtype).
        """
        arr = np.fromiter((sz for _, sz in data[:levels]), dtype=float, count=levels)
        return arr

    def compute_imbalance(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        levels: int = DEFAULT_LEVELS,
    ) -> float:
        """
        Order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol).

        Returns a value in [-1, 1]. Positive indicates bid‑heavy (buying pressure).

        Args:
            bids: List of (price, size) pairs, best bid first.
            asks: List of (price, size) pairs, best ask first.
            levels: Number of price levels to include in the calculation.

        Returns:
            Imbalance ratio as a float. Returns 0.0 when data is insufficient.
        """
        if not bids or not asks:
            return 0.0

        bid_tuple = tuple(bids)
        ask_tuple = tuple(asks)

        bid_vols = self._slice_volumes(bid_tuple, levels)
        ask_vols = self._slice_volumes(ask_tuple, levels)

        bid_vol = float(bid_vols.sum())
        ask_vol = float(ask_vols.sum())
        total = bid_vol + ask_vol
        if total <= 0.0:
            return 0.0
        return float((bid_vol - ask_vol) / total)

    def compute_spread_bps(self, best_bid: float, best_ask: float) -> float:
        """
        Compute the bid‑ask spread expressed in basis points.

        The spread is calculated as ``(ask - bid) / mid * BASIS_POINTS_MULTIPLIER``.
        Returns 0.0 for invalid inputs.

        Args:
            best_bid: Best bid price.
            best_ask: Best ask price.

        Returns:
            Spread in basis points as a float.
        """
        if best_bid <= 0.0 or best_ask <= 0.0 or best_ask <= best_bid:
            return 0.0
        mid = (best_bid + best_ask) * 0.5
        return float((best_ask - best_bid) / mid * BASIS_POINTS_MULTIPLIER)

    def compute_depth_ratio(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> float:
        """
        Top‑of‑book depth ratio: ``best_bid_size / best_ask_size``.

        Values greater than 1 indicate more liquidity on the bid side.
        Returns 1.0 if either side is empty or the ask size is non‑positive.

        Args:
            bids: List of (price, size) pairs for the bid side.
            asks: List of (price, size) pairs for the ask side.

        Returns:
            Depth ratio as a float.
        """
        if not bids or not asks:
            return 1.0
        best_bid_size = float(bids[0][1])
        best_ask_size = float(asks[0][1])
        if best_ask_size <= 0.0:
            return 1.0
        return float(best_bid_size / best_ask_size)

    def compute_pin_proxy(self, buy_volume: float, sell_volume: float) -> float:
        """
        Probability of Informed Trading (PIN) proxy.

        PIN = |buy_vol - sell_vol| / (buy_vol + sell_vol).

        Args:
            buy_volume: Total buy volume.
            sell_volume: Total sell volume.

        Returns:
            PIN proxy in the range [0, 1]. Returns 0.0 when total volume is zero.
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
        Estimate Kyle's lambda (price impact coefficient) via ordinary least squares.

        The regression model is ``delta_price = lambda * signed_volume + epsilon``.
        The function returns lambda measured in basis points per unit volume.

        Args:
            price_changes: Array of price differences (Δprice).
            signed_volumes: Array of signed trade volumes.

        Returns:
            Estimated lambda as a float. Returns 0.0 if data is insufficient or
            variance of volumes is too low.
        """
        if price_changes.size < MIN_SAMPLE_SIZE or signed_volumes.size < MIN_SAMPLE_SIZE:
            return 0.0
        try:
            vol = np.asarray(signed_volumes, dtype=float)
            dp = np.asarray(price_changes, dtype=float)

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
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        buy_volume: float = 0.0,
        sell_volume: float = 0.0,
        levels: int = DEFAULT_LEVELS,
    ) -> dict[str, float]:
        """
        Compute all microstructure features from a single LOB snapshot.

        Args:
            bids: List of (price, size) pairs for the bid side.
            asks: List of (price, size) pairs for the ask side.
            buy_volume: Aggregated buy volume for the PIN proxy.
            sell_volume: Aggregated sell volume for the PIN proxy.
            levels: Number of order‑book levels to include in the imbalance metric.

        Returns:
            Dictionary with keys ``imbalance``, ``spread_bps``, ``depth_ratio``,
            and ``pin_proxy`` mapping to their respective float values.
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
    imbalance_series: pd.Series | None = None,
    spread_bps_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Enrich an OHLCV DataFrame with microstructure feature columns.

    If real‑time LOB series are supplied they are aligned to the DataFrame's index
    and used directly. Otherwise, proxy features are derived from the OHLCV data:

    * ``volume_imbalance_proxy`` – approximates buy/sell pressure:
      ``(close - open) / (high - low)`` clipped to ``[-1, 1]``.
    * ``spread_bps_proxy`` – approximates intraday spread:
      ``(high - low) / close * BASIS_POINTS_MULTIPLIER``.

    Args:
        df: Input DataFrame containing at least ``high``, ``low``, ``open``,
            ``close`` columns.
        imbalance_series: Optional Series of pre‑computed LOB imbalance values.
        spread_bps_series: Optional Series of pre‑computed spread in basis points.

    Returns:
        A new DataFrame copy with ``lob_imbalance`` and ``spread_bps`` columns added.
    """
    df = df.copy()

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