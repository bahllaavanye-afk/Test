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

logger = logging.getLogger(__name__)


class OrderBookFeatures:
    """Compute LOB features from real-time bid/ask depth."""

    @staticmethod
    @lru_cache(maxsize=1024)
    def _slice_volumes(
        data: tuple[tuple[float, float], ...],
        levels: int,
    ) -> np.ndarray:
        """Extract volumes up to `levels` and return as NumPy array."""
        # Convert to NumPy array of shape (n, 2) and slice volumes
        arr = np.fromiter((sz for _, sz in data[:levels]), dtype=float, count=levels)
        return arr

    def compute_imbalance(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        levels: int = 5,
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
            logger.info(
                "compute_imbalance called with empty bids or asks",
                extra={"signal_count": 0, "execution_time_ms": (time.perf_counter() - start) * 1000},
            )
            return 0.0

        # Convert to immutable tuple for caching
        bid_tuple = tuple(bids)
        ask_tuple = tuple(asks)

        bid_vols = self._slice_volumes(bid_tuple, levels)
        ask_vols = self._slice_volumes(ask_tuple, levels)

        bid_vol = float(bid_vols.sum())
        ask_vol = float(ask_vols.sum())
        total = bid_vol + ask_vol
        if total <= 0.0:
            logger.info(
                "compute_imbalance total volume zero",
                extra={"signal_count": 0, "execution_time_ms": (time.perf_counter() - start) * 1000},
            )
            return 0.0
        result = float((bid_vol - ask_vol) / total)
        logger.info(
            "compute_imbalance completed",
            extra={"signal_count": 1, "execution_time_ms": (time.perf_counter() - start) * 1000},
        )
        return result

    def compute_spread_bps(self, best_bid: float, best_ask: float) -> float:
        """
        Bid-ask spread in basis points: (ask - bid) / mid * 10_000.
        Returns 0.0 for invalid inputs.
        """
        start = time.perf_counter()
        if best_bid <= 0.0 or best_ask <= 0.0 or best_ask <= best_bid:
            logger.info(
                "compute_spread_bps received invalid inputs",
                extra={"signal_count": 0, "execution_time_ms": (time.perf_counter() - start) * 1000},
            )
            return 0.0
        mid = (best_bid + best_ask) * 0.5
        result = float((best_ask - best_bid) / mid * 10_000.0)
        logger.info(
            "compute_spread_bps completed",
            extra={"signal_count": 1, "execution_time_ms": (time.perf_counter() - start) * 1000},
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
            logger.info(
                "compute_depth_ratio called with empty bids or asks",
                extra={"signal_count": 0, "execution_time_ms": (time.perf_counter() - start) * 1000},
            )
            return 1.0
        best_bid_size = float(bids[0][1])
        best_ask_size = float(asks[0][1])
        if best_ask_size <= 0.0:
            logger.info(
                "compute_depth_ratio best ask size non-positive",
                extra={"signal_count": 0, "execution_time_ms": (time.perf_counter() - start) * 1000},
            )
            return 1.0
        result = float(best_bid_size / best_ask_size)
        logger.info(
            "compute_depth_ratio completed",
            extra={"signal_count": 1, "execution_time_ms": (time.perf_counter() - start) * 1000},
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
        if total <= 0.0:
            logger.info(
                "compute_pin_proxy total volume zero",
                extra={"signal_count": 0, "execution_time_ms": (time.perf_counter() - start) * 1000},
            )
            return 0.0
        result = float(abs(buy_volume - sell_volume) / total)
        logger.info(
            "compute_pin_proxy completed",
            extra={"signal_count": 1, "execution_time_ms": (time.perf_counter() - start) * 1000},
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
        if price_changes.size < 5 or signed_volumes.size < 5:
            logger.info(
                "compute_kyle_lambda insufficient data",
                extra={"signal_count": 0, "execution_time_ms": (time.perf_counter() - start) * 1000},
            )
            return 0.0
        try:
            vol = np.asarray(signed_volumes, dtype=float)
            dp = np.asarray(price_changes, dtype=float)

            var_vol = np.var(vol)
            if var_vol < 1e-12:
                logger.info(
                    "compute_kyle_lambda variance of volume too low",
                    extra={"signal_count": 0, "execution_time_ms": (time.perf_counter() - start) * 1000},
                )
                return 0.0

            # Covariance via means to avoid np.cov overhead
            cov = np.mean(dp * vol) - np.mean(dp) * np.mean(vol)
            lam = float(cov / var_vol)
            logger.info(
                "compute_kyle_lambda completed",
                extra={"signal_count": 1, "execution_time_ms": (time.perf_counter() - start) * 1000},
            )
            return lam
        except Exception:
            logger.exception(
                "compute_kyle_lambda encountered an error",
                extra={"signal_count": 0, "execution_time_ms": (time.perf_counter() - start) * 1000},
            )
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
        logger.info(
            "features_from_snapshot computed",
            extra={
                "signal_count": len(result),
                "execution_time_ms": (time.perf_counter() - start) * 1000,
                "pnl": 0.0,
            },
        )
        return result


def add_microstructure_features(
    df: pd.DataFrame,
    imbalance_series: pd.Series | None = None,
    spread_bps_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Add microstructure feature columns to an OHLCV DataFrame.

    If real-time LOB series are provided they are aligned and added.
    Otherwise, proxy features are computed from OHLCV:
      - volume_imbalance_proxy: (close - open) / (high - low + 1e-9)  — approximates buy/sell pressure
      - spread_bps_proxy: (high - low) / close * 10_000               — proxy for intraday spread
    """
    start = time.perf_counter()
    df = df.copy()

    if imbalance_series is not None:
        df["lob_imbalance"] = imbalance_series.reindex(df.index).fillna(0.0)
    else:
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        df["lob_imbalance"] = ((df["close"] - df["open"]) / rng).clip(-1, 1).fillna(0.0)

    if spread_bps_series is not None:
        df["spread_bps"] = spread_bps_series.reindex(df.index).fillna(0.0)
    else:
        close_nonzero = df["close"].replace(0, np.nan)
        df["spread_bps"] = ((df["high"] - df["low"]) / close_nonzero * 10_000).fillna(0.0)

    # Structured logging
    signal_count = len(df)
    pnl = 0.0
    if not df.empty and "close" in df.columns:
        pnl = float(df["close"].iloc[-1] - df["close"].iloc[0])
    logger.info(
        "add_microstructure_features applied",
        extra={
            "signal_count": signal_count,
            "execution_time_ms": (time.perf_counter() - start) * 1000,
            "pnl": pnl,
        },
    )
    return df


MICROSTRUCTURE_FEATURE_COLS = ["lob_imbalance", "spread_bps"]