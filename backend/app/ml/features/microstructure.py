"""
Order book and microstructure features.

This module provides utilities to compute limit order book (LOB) microstructure
features from bid/ask depth data and to augment OHLCV DataFrames with proxy
features when real‑time LOB data is unavailable.

Features implemented:
  - Order book imbalance (bid pressure vs ask pressure)
  - Bid‑ask spread expressed in basis points
  - Top‑of‑book depth ratio
  - PIN proxy (Probability of Informed Trading)
  - Kyle's lambda (price impact coefficient)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict


class OrderBookFeatures:
    """Compute LOB features from real‑time bid/ask depth.

    The class groups a set of pure‑Python calculations that operate on standard
    Python containers (lists of price/size tuples) or NumPy arrays.  All methods
    are side‑effect free and return a single scalar value.
    """

    def compute_imbalance(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        levels: int = 5,
    ) -> float:
        """Calculate order‑book imbalance.

        Imbalance is defined as ``(bid_vol - ask_vol) / (bid_vol + ask_vol)`` where
        ``bid_vol`` and ``ask_vol`` are the summed volumes across the best *levels*
        price levels.  The result lies in ``[-1, 1]``; positive values indicate
        buying pressure.

        Args:
            bids: List of ``(price, size)`` pairs, ordered best‑bid first.
            asks: List of ``(price, size)`` pairs, ordered best‑ask first.
            levels: Number of depth levels to include in the calculation.

        Returns:
            Imbalance as a float. Returns ``0.0`` when either side is empty or the
            total volume is non‑positive.
        """
        if not bids or not asks:
            return 0.0
        bid_vol = sum(float(sz) for _, sz in bids[:levels])
        ask_vol = sum(float(sz) for _, sz in asks[:levels])
        total = bid_vol + ask_vol
        if total <= 0:
            return 0.0
        return float((bid_vol - ask_vol) / total)

    def compute_spread_bps(self, best_bid: float, best_ask: float) -> float:
        """Calculate the bid‑ask spread in basis points.

        The spread is ``(ask - bid) / mid * 10_000`` where *mid* is the midpoint
        price.  Returns ``0.0`` for invalid or non‑positive inputs.

        Args:
            best_bid: Best bid price.
            best_ask: Best ask price.

        Returns:
            Spread expressed in basis points.
        """
        if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
            return 0.0
        mid = (best_bid + best_ask) / 2.0
        return float((best_ask - best_bid) / mid * 10_000.0)

    def compute_depth_ratio(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> float:
        """Calculate the top‑of‑book depth ratio.

        Defined as ``best_bid_size / best_ask_size``.  Values greater than one
        indicate more liquidity on the bid side.  If either side is missing or
        the ask size is non‑positive, the function returns ``1.0``.

        Args:
            bids: List of ``(price, size)`` pairs, best bid first.
            asks: List of ``(price, size)`` pairs, best ask first.

        Returns:
            Depth ratio as a float.
        """
        if not bids or not asks:
            return 1.0
        best_bid_size = float(bids[0][1]) if bids else 0.0
        best_ask_size = float(asks[0][1]) if asks else 0.0
        if best_ask_size <= 0:
            return 1.0
        return float(best_bid_size / best_ask_size)

    def compute_pin_proxy(self, buy_volume: float, sell_volume: float) -> float:
        """Compute a proxy for the Probability of Informed Trading (PIN).

        The proxy is ``|buy_vol - sell_vol| / (buy_vol + sell_vol)`` and lies in
        ``[0, 1]``.  Higher values suggest a higher proportion of informed order
        flow.

        Args:
            buy_volume: Total buy‑side volume.
            sell_volume: Total sell‑side volume.

        Returns:
            PIN proxy as a float. Returns ``0.0`` when the total volume is
            non‑positive.
        """
        total = buy_volume + sell_volume
        if total <= 0:
            return 0.0
        return float(abs(buy_volume - sell_volume) / total)

    def compute_kyle_lambda(
        self,
        price_changes: np.ndarray,
        signed_volumes: np.ndarray,
    ) -> float:
        """Estimate Kyle's lambda (price impact coefficient).

        The coefficient is obtained via ordinary least squares regression of
        ``Δprice = λ * signed_volume + ε``.  The function returns the estimated
        ``λ`` in basis points per unit volume.  If insufficient data is provided
        or the variance of the volume series is near zero, ``0.0`` is returned.

        Args:
            price_changes: Array of price changes (Δprice).
            signed_volumes: Corresponding signed trade volumes.

        Returns:
            Estimated lambda as a float, or ``0.0`` on failure.
        """
        if len(price_changes) < 5 or len(signed_volumes) < 5:
            return 0.0
        try:
            vol = np.array(signed_volumes, dtype=float)
            dp = np.array(price_changes, dtype=float)
            var_vol = np.var(vol)
            if var_vol < 1e-12:
                return 0.0
            lam = float(np.cov(dp, vol)[0, 1] / var_vol)
            return lam
        except Exception:
            return 0.0

    def features_from_snapshot(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        buy_volume: float = 0.0,
        sell_volume: float = 0.0,
        levels: int = 5,
    ) -> Dict[str, float]:
        """Compute all microstructure features from a single LOB snapshot.

        The method aggregates the individual feature calculations into a single
        dictionary for convenient downstream consumption.

        Args:
            bids: List of ``(price, size)`` pairs, best bid first.
            asks: List of ``(price, size)`` pairs, best ask first.
            buy_volume: Total buy‑side volume for the snapshot.
            sell_volume: Total sell‑side volume for the snapshot.
            levels: Number of depth levels to consider for imbalance.

        Returns:
            Mapping of feature names to their computed float values. Keys are:
            ``imbalance``, ``spread_bps``, ``depth_ratio``, and ``pin_proxy``.
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
    """Add microstructure feature columns to an OHLCV DataFrame.

    When real‑time LOB series are supplied they are aligned to the DataFrame
    index and used directly.  If they are omitted, simple proxy calculations are
    performed on the OHLCV data:

    * ``lob_imbalance`` – proxy for volume imbalance:
      ``(close - open) / (high - low)`` clipped to ``[-1, 1]``.
    * ``spread_bps`` – proxy for intraday spread:
      ``(high - low) / close * 10_000``.

    Args:
        df: OHLCV DataFrame containing at least ``open``, ``high``, ``low`` and
            ``close`` columns.
        imbalance_series: Optional series of pre‑computed LOB imbalance values.
        spread_bps_series: Optional series of pre‑computed spread values in bps.

    Returns:
        A new DataFrame (original is not mutated) with the added columns
        ``lob_imbalance`` and ``spread_bps``.
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