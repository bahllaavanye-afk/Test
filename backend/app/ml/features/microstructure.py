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

        Parameters
        ----------
        data: tuple[tuple[float, float], ...]
            Sequence of (price, size) tuples ordered by price level.
        levels: int
            Number of price levels to include.

        Returns
        -------
        np.ndarray
            Array of sizes (volumes) for the requested depth levels.
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
        Compute order book imbalance.

        Imbalance is defined as ``(bid_vol - ask_vol) / (bid_vol + ask_vol)`` and
        ranges between -1 and 1. Positive values indicate buying pressure.

        Parameters
        ----------
        bids: list[tuple[float, float]]
            List of (price, size) pairs for the bid side, best bid first.
        asks: list[tuple[float, float]]
            List of (price, size) pairs for the ask side, best ask first.
        levels: int, optional
            Number of depth levels to consider (default is ``DEFAULT_LEVELS``).

        Returns
        -------
        float
            Imbalance value in ``[-1, 1]``. Returns ``0.0`` when data is missing
            or total volume is zero.
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

        Parameters
        ----------
        best_bid: float
            Best bid price.
        best_ask: float
            Best ask price.

        Returns
        -------
        float
            Spread in basis points, or ``0.0`` for invalid inputs.
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
        Compute the top‑of‑book depth ratio.

        Depth ratio is ``best_bid_size / best_ask_size``. Values greater than one
        indicate more liquidity on the bid side.

        Parameters
        ----------
        bids: list[tuple[float, float]]
            List of (price, size) pairs for the bid side.
        asks: list[tuple[float, float]]
            List of (price, size) pairs for the ask side.

        Returns
        -------
        float
            Depth ratio, or ``1.0`` when either side is empty or ask size is zero.
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
        Compute a proxy for the Probability of Informed Trading (PIN).

        PIN proxy is ``|buy_vol - sell_vol| / (buy_vol + sell_vol)`` and ranges from
        0 to 1. Values near 1 suggest a high proportion of informed order flow.

        Parameters
        ----------
        buy_volume: float
            Total buy‑side volume.
        sell_volume: float
            Total sell‑side volume.

        Returns
        -------
        float
            PIN proxy value in ``[0, 1]``. Returns ``0.0`` when total volume is zero.
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

        The model assumes ``delta_price = lambda * signed_volume + epsilon``.
        The estimated lambda is ``cov(delta_price, signed_volume) / var(signed_volume)``.

        Parameters
        ----------
        price_changes: np.ndarray
            Array of price changes (delta price) observations.
        signed_volumes: np.ndarray
            Corresponding array of signed trade volumes.

        Returns
        -------
        float
            Estimated lambda (basis points per unit volume). Returns ``0.0`` when
            there is insufficient data or the variance of volumes is negligible.
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

        Parameters
        ----------
        bids: list[tuple[float, float]]
            Bid side depth data.
        asks: list[tuple[float, float]]
            Ask side depth data.
        buy_volume: float, optional
            Aggregated buy volume for PIN proxy (default ``0.0``).
        sell_volume: float, optional
            Aggregated sell volume for PIN proxy (default ``0.0``).
        levels: int, optional
            Number of depth levels to include for imbalance calculation.

        Returns
        -------
        dict[str, float]
            Mapping with keys ``imbalance``, ``spread_bps``, ``depth_ratio``,
            and ``pin_proxy``.
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
    Append microstructure feature columns to an OHLCV DataFrame.

    If real‑time LOB series are supplied, they are aligned with the DataFrame
    index and used directly. Otherwise, proxy features are derived from the OHLCV
    data:

    * ``volume_imbalance_proxy`` (``lob_imbalance``):
      ``(close - open) / (high - low)`` – approximates buy/sell pressure.
    * ``spread_bps_proxy`` (``spread_bps``):
      ``(high - low) / close * BASIS_POINTS_MULTIPLIER`` – proxy for intraday
      spread.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame containing at least the columns defined by
        ``COL_HIGH``, ``COL_LOW``, ``COL_CLOSE``, and ``COL_OPEN``.
    imbalance_series: pd.Series | None, optional
        Optional pre‑computed LOB imbalance series indexed like ``df``.
    spread_bps_series: pd.Series | None, optional
        Optional pre‑computed spread series indexed like ``df``.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with ``lob_imbalance`` and ``spread_bps`` columns added.
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