"""
Microstructure feature extraction utilities.

This module provides tools to compute a set of microstructure‑related
features from limit order book (LOB) snapshots and to augment OHLCV
DataFrames with proxy features when real‑time LOB data is unavailable.

Features include:
- Order book imbalance (bid vs. ask volume)
- Bid‑ask spread expressed in basis points
- Top‑of‑book depth ratio
- Probability of Informed Trading (PIN) proxy
- Kyle’s lambda (price impact coefficient)

The public API consists of the :class:`OrderBookFeatures` class for
computing features from a single snapshot and the
:func:`add_microstructure_features` function for appending feature columns
to a DataFrame.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class OrderBookFeatures:
    """Utility class for computing LOB‑derived microstructure features."""

    def compute_imbalance(
        self,
        bids: Sequence[Tuple[float, float]],
        asks: Sequence[Tuple[float, float]],
        levels: int = 5,
    ) -> float:
        """
        Compute the order‑book imbalance.

        The imbalance is defined as

        ``(bid_volume - ask_volume) / (bid_volume + ask_volume)``

        where ``bid_volume`` and ``ask_volume`` are the summed sizes of the
        best *levels* price levels on each side of the book.

        Parameters
        ----------
        bids : Sequence[Tuple[float, float]]
            List of *(price, size)* tuples for the bid side,
            ordered from best bid to worst.
        asks : Sequence[Tuple[float, float]]
            List of *(price, size)* tuples for the ask side,
            ordered from best ask to worst.
        levels : int, optional
            Number of price levels to include in the calculation. Defaults to 5.

        Returns
        -------
        float
            Imbalance in the range ``[-1, 1]``. Positive values indicate
            bid‑heavy pressure (potential buying interest). Returns ``0.0``
            when no depth is available or the total volume is zero.
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
        """
        Compute the bid‑ask spread expressed in basis points.

        The spread is calculated as

        ``(ask - bid) / mid * 10_000``

        where ``mid`` is the mid‑price of the best bid and ask.

        Parameters
        ----------
        best_bid : float
            Best bid price. Must be positive.
        best_ask : float
            Best ask price. Must be positive and greater than ``best_bid``.

        Returns
        -------
        float
            Spread in basis points. Returns ``0.0`` for invalid inputs.
        """
        if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
            return 0.0
        mid = (best_bid + best_ask) / 2.0
        return float((best_ask - best_bid) / mid * 10_000.0)

    def compute_depth_ratio(
        self,
        bids: Sequence[Tuple[float, float]],
        asks: Sequence[Tuple[float, float]],
    ) -> float:
        """
        Compute the top‑of‑book depth ratio.

        The ratio is ``best_bid_size / best_ask_size``. Values greater than
        one indicate more liquidity on the bid side.

        Parameters
        ----------
        bids : Sequence[Tuple[float, float]]
            Bid side depth, best level first.
        asks : Sequence[Tuple[float, float]]
            Ask side depth, best level first.

        Returns
        -------
        float
            Depth ratio. Returns ``1.0`` when either side is empty or the
            best ask size is non‑positive.
        """
        if not bids or not asks:
            return 1.0
        best_bid_size = float(bids[0][1]) if bids else 0.0
        best_ask_size = float(asks[0][1]) if asks else 0.0
        if best_ask_size <= 0:
            return 1.0
        return float(best_bid_size / best_ask_size)

    def compute_pin_proxy(self, buy_volume: float, sell_volume: float) -> float:
        """
        Compute a proxy for the Probability of Informed Trading (PIN).

        The proxy is defined as

        ``|buy_volume - sell_volume| / (buy_volume + sell_volume)``

        Parameters
        ----------
        buy_volume : float
            Total buy‑side volume.
        sell_volume : float
            Total sell‑side volume.

        Returns
        -------
        float
            PIN proxy in ``[0, 1]``. Returns ``0.0`` when the total volume is
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
        """
        Estimate Kyle’s lambda (price impact coefficient) via ordinary least squares.

        The model assumes

        ``Δprice = λ * signed_volume + ε``

        where ``signed_volume`` is positive for buys and negative for sells.

        Parameters
        ----------
        price_changes : np.ndarray
            Array of price changes (Δprice). Must be numeric.
        signed_volumes : np.ndarray
            Corresponding array of signed volumes. Must be numeric.

        Returns
        -------
        float
            Estimated λ (basis points per unit volume). Returns ``0.0`` when
            insufficient data is provided or the variance of ``signed_volumes``
            is effectively zero.
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
        bids: Sequence[Tuple[float, float]],
        asks: Sequence[Tuple[float, float]],
        buy_volume: float = 0.0,
        sell_volume: float = 0.0,
        levels: int = 5,
    ) -> Dict[str, float]:
        """
        Compute the full set of microstructure features from a single LOB snapshot.

        Parameters
        ----------
        bids : Sequence[Tuple[float, float]]
            Bid side depth, best level first.
        asks : Sequence[Tuple[float, float]]
            Ask side depth, best level first.
        buy_volume : float, optional
            Aggregated buy‑side volume for the snapshot. Defaults to ``0.0``.
        sell_volume : float, optional
            Aggregated sell‑side volume for the snapshot. Defaults to ``0.0``.
        levels : int, optional
            Number of price levels to consider for the imbalance metric.
            Defaults to ``5``.

        Returns
        -------
        Dict[str, float]
            Mapping with keys ``'imbalance'``, ``'spread_bps'``,
            ``'depth_ratio'``, and ``'pin_proxy'``. Each value is a float.
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
    Append microstructure feature columns to an OHLCV DataFrame.

    When real‑time LOB series are supplied they are aligned to the DataFrame’s
    index and used directly.  If they are omitted, proxy features are derived
    from the OHLCV data:

    * ``lob_imbalance`` – approximates order‑book imbalance using
      ``(close - open) / (high - low)``.
    * ``spread_bps`` – approximates the bid‑ask spread using
      ``(high - low) / close * 10_000``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least the columns ``'open'``, ``'high'``,
        ``'low'``, ``'close'``, and optionally ``'volume'``.
    imbalance_series : pd.Series | None, optional
        Series of pre‑computed LOB imbalance values indexed by timestamp.
        If ``None``, a proxy is calculated from the OHLCV data.
    spread_bps_series : pd.Series | None, optional
        Series of pre‑computed spread‑in‑bps values indexed by timestamp.
        If ``None``, a proxy is calculated from the OHLCV data.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with the additional columns ``'lob_imbalance'`` and
        ``'spread_bps'``.
    """
    df = df.copy()

    if imbalance_series is not None:
        df["lob_imbalance"] = imbalance_series.reindex(df.index).fillna(0.0)
    else:
        # Proxy: (close - open) / (high - low)
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


MICROSTRUCTURE_FEATURE_COLS: List[str] = ["lob_imbalance", "spread_bps"]