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

import logging
import time
from functools import lru_cache

import numpy as np
import pandas as pd

# Logger configuration (structured logging)
logger = logging.getLogger(__name__)

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
        """Extract volumes up to `levels` and return as NumPy array."""
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
        Returns value in [-1, 1]. Positive = bid-heavy (buying pressure).

        Args:
            bids: list of (price, size) pairs, best bid first
            asks: list of (price, size) pairs, best ask first
            levels: how many price levels to include
        """
        start = time.perf_counter()
        if not bids or not asks:
            result = 0.0
        else:
            bid_tuple = tuple(bids)
            ask_tuple = tuple(asks)

            bid_vols = self._slice_volumes(bid_tuple, levels)
            ask_vols = self._slice_volumes(ask_tuple, levels)

            bid_vol = float(bid_vols.sum())
            ask_vol = float(ask_vols.sum())
            total = bid_vol + ask_vol
            result = 0.0 if total <= 0.0 else float((bid_vol - ask_vol) / total)

        duration_ms = (time.perf_counter() - start) * 1_000
        logger.info(
            "compute_imbalance",
            extra={
                "signal": "lob_imbalance",
                "value": result,
                "execution_time_ms": duration_ms,
            },
        )
        return result

    def compute_spread_bps(self, best_bid: float, best_ask: float) -> float:
        """
        Bid-ask spread in basis points: (ask - bid) / mid * BASIS_POINTS_MULTIPLIER.
        Returns 0.0 for invalid inputs.
        """
        start = time.perf_counter()
        if best_bid <= 0.0 or best_ask <= 0.0 or best_ask <= best_bid:
            result = 0.0
        else:
            mid = (best_bid + best_ask) * 0.5
            result = float((best_ask - best_bid) / mid * BASIS_POINTS_MULTIPLIER)

        duration_ms = (time.perf_counter() - start) * 1_000
        logger.info(
            "compute_spread_bps",
            extra={
                "signal": "spread_bps",
                "value": result,
                "execution_time_ms": duration_ms,
            },
        )
        return result

    def compute_depth_ratio(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> float:
        """
        Top-of-book depth ratio: best_bid_size / best_ask_size.
        Values > 1 indicate more liquidity on bid side.
        Returns 1.0 if either side is empty.
        """
        start = time.perf_counter()
        if not bids or not asks:
            result = 1.0
        else:
            best_bid_size = float(bids[0][1])
            best_ask_size = float(asks[0][1])
            result = 1.0 if best_ask_size <= 0.0 else float(best_bid_size / best_ask_size)

        duration_ms = (time.perf_counter() - start) * 1_000
        logger.info(
            "compute_depth_ratio",
            extra={
                "signal": "depth_ratio",
                "value": result,
                "execution_time_ms": duration_ms,
            },
        )
        return result

    def compute_pin_proxy(self, buy_volume: float, sell_volume: float) -> float:
        """
        Probability of Informed Trading proxy.
        PIN = |buy_vol - sell_vol| / (buy_vol + sell_vol)
        Returns value in [0, 1]. Near 1 = highly informed order flow.
        """
        start = time.perf_counter()
        total = buy_volume + sell_volume
        result = 0.0 if total <= 0.0 else float(abs(buy_volume - sell_volume) / total)
        duration_ms = (time.perf_counter() - start) * 1_000
        logger.info(
            "compute_pin_proxy",
            extra={
                "signal": "pin_proxy",
                "value": result,
                "execution_time_ms": duration_ms,
            },
        )
        return result

    def compute_kyle_lambda(
        self,
        price_changes: np.ndarray,
        signed_volumes: np.ndarray,
    ) -> float:
        """
        Kyle's lambda (price impact coefficient).
        Estimated via OLS: delta_price = lambda * signed_volume + epsilon

        Returns lambda (bps per unit volume). Higher = less liquid.
        Returns 0.0 if insufficient data.
        """
        start = time.perf_counter()
        if price_changes.size < MIN_SAMPLE_SIZE or signed_volumes.size < MIN_SAMPLE_SIZE:
            result = 0.0
        else:
            try:
                vol = np.asarray(signed_volumes, dtype=float)
                dp = np.asarray(price_changes, dtype=float)

                var_vol = np.var(vol)
                if var_vol < VAR_VOLUME_EPS:
                    result = 0.0
                else:
                    cov = np.mean(dp * vol) - np.mean(dp) * np.mean(vol)
                    result = float(cov / var_vol)
            except Exception:
                result = 0.0

        duration_ms = (time.perf_counter() - start) * 1_000
        logger.info(
            "compute_kyle_lambda",
            extra={
                "signal": "kyle_lambda",
                "value": result,
                "execution_time_ms": duration_ms,
            },
        )
        return result

    def features_from_snapshot(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        buy_volume: float = 0.0,
        sell_volume: float = 0.0,
        levels: int = DEFAULT_LEVELS,
        pnl: float | None = None,
    ) -> dict[str, float]:
        """
        Compute all microstructure features from a single LOB snapshot.

        Returns:
            dict with keys: imbalance, spread_bps, depth_ratio, pin_proxy
        """
        start = time.perf_counter()
        best_bid = float(bids[0][0]) if bids else 0.0
        best_ask = float(asks[0][0]) if asks else 0.0

        result = {
            "imbalance": self.compute_imbalance(bids, asks, levels),
            "spread_bps": self.compute_spread_bps(best_bid, best_ask),
            "depth_ratio": self.compute_depth_ratio(bids, asks),
            "pin_proxy": self.compute_pin_proxy(buy_volume, sell_volume),
        }

        duration_ms = (time.perf_counter() - start) * 1_000
        logger.info(
            "features_from_snapshot",
            extra={
                "signal_count": len(result),
                "execution_time_ms": duration_ms,
                "pnl": pnl,
            },
        )
        return result


def add_microstructure_features(
    df: pd.DataFrame,
    imbalance_series: pd.Series | None = None,
    spread_bps_series: pd.Series | None = None,
    pnl: float | None = None,
) -> pd.DataFrame:
    """
    Add microstructure feature columns to an OHLCV DataFrame.

    If real-time LOB series are provided they are aligned and added.
    Otherwise, proxy features are computed from OHLCV:
      - volume_imbalance_proxy: (close - open) / (high - low)  — approximates buy/sell pressure
      - spread_bps_proxy: (high - low) / close * BASIS_POINTS_MULTIPLIER — proxy for intraday spread
    """
    start = time.perf_counter()
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

    duration_ms = (time.perf_counter() - start) * 1_000
    logger.info(
        "add_microstructure_features",
        extra={
            "signal_count": len(df),
            "execution_time_ms": duration_ms,
            "pnl": pnl,
        },
    )
    return df