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

Signal logic:
  - Tightened entry conditions based on imbalance, spread and PIN proxy.
  - Confirmation filter requiring conditions to persist for a configurable window.
  - Exit logic triggered by reversal of imbalance or deterioration of spread/PIN.
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

# Feature column names
COL_HIGH: str = "high"
COL_LOW: str = "low"
COL_CLOSE: str = "close"
COL_OPEN: str = "open"
COL_LOB_IMBALANCE: str = "lob_imbalance"
COL_SPREAD_BPS: str = "spread_bps"
COL_DEPTH_RATIO: str = "depth_ratio"
COL_PIN_PROXY: str = "pin_proxy"

MICROSTRUCTURE_FEATURE_COLS = [
    COL_LOB_IMBALANCE,
    COL_SPREAD_BPS,
    COL_DEPTH_RATIO,
    COL_PIN_PROXY,
]

# Default signal thresholds
IMBALANCE_ENTRY_THRESHOLD: float = 0.20   # absolute imbalance needed for entry
SPREAD_BPS_MAX: float = 5.0               # max spread in bps for entry
PIN_PROXY_MAX: float = 0.50               # max PIN proxy for entry
CONFIRMATION_WINDOW: int = 3              # number of consecutive bars to confirm entry


class OrderBookFeatures:
    """Compute LOB features from real-time bid/ask depth."""

    @staticmethod
    @lru_cache(maxsize=1024)
    def _slice_volumes(
        data: tuple[tuple[float, float], ...],
        levels: int,
    ) -> np.ndarray:
        """Extract volumes up to `levels` and return as NumPy array."""
        # Guard against requesting more levels than available
        effective_levels = min(levels, len(data))
        arr = np.fromiter((sz for _, sz in data[:effective_levels]), dtype=float, count=effective_levels)
        # Pad with zeros if fewer levels than requested (maintains shape)
        if effective_levels < levels:
            arr = np.pad(arr, (0, levels - effective_levels), constant_values=0.0)
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
        if not bids or not asks:
            return 0.0

        # Ensure a sensible level count
        levels = max(1, int(levels))

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
        Bid-ask spread in basis points: (ask - bid) / mid * BASIS_POINTS_MULTIPLIER.
        Returns 0.0 for invalid inputs.
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
        Top-of-book depth ratio: best_bid_size / best_ask_size.
        Values > 1 indicate more liquidity on bid side.
        Returns 1.0 if either side is empty.
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
        Returns 0.0 if insufficient data.
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

        Returns:
            dict with keys: imbalance, spread_bps, depth_ratio, pin_proxy
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
    Add microstructure feature columns to an OHLCV DataFrame.

    If real-time LOB series are provided they are aligned and added.
    Otherwise, proxy features are computed from OHLCV:
      - volume_imbalance_proxy: (close - open) / (high - low)  — approximates buy/sell pressure
      - spread_bps_proxy: (high - low) / close * BASIS_POINTS_MULTIPLIER — proxy for intraday spread
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

    # Depth ratio and PIN proxy are optional; compute simple proxies if missing
    if COL_DEPTH_RATIO not in df.columns:
        df[COL_DEPTH_RATIO] = 1.0  # placeholder – real depth ratio requires LOB data
    if COL_PIN_PROXY not in df.columns:
        df[COL_PIN_PROXY] = 0.0   # placeholder – real PIN proxy requires order flow data

    return df


def generate_microstructure_signals(
    df: pd.DataFrame,
    imbalance_thresh: float = IMBALANCE_ENTRY_THRESHOLD,
    spread_bps_max: float = SPREAD_BPS_MAX,
    pin_proxy_max: float = PIN_PROXY_MAX,
    confirmation_window: int = CONFIRMATION_WINDOW,
) -> pd.DataFrame:
    """
    Derive entry/exit signals from microstructure features.

    Entry (long) is signaled when:
      * Imbalance >= `imbalance_thresh`
      * Spread_bps <= `spread_bps_max`
      * Pin_proxy <= `pin_proxy_max`
    The three conditions must hold for `confirmation_window` consecutive rows.

    Exit (flat) is triggered when any of the entry conditions is violated
    after a position has been opened.

    Returns a copy of ``df`` with an additional column ``signal``:
        1  → long entry,
        0  → flat / exit,
       -1  → short entry (currently not used, but kept for symmetry).
    """
    df = df.copy()

    # Ensure required columns exist
    required = [COL_LOB_IMBALANCE, COL_SPREAD_BPS, COL_PIN_PROXY]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is required for signal generation.")

    # Boolean masks for entry criteria
    imbalance_ok = df[COL_LOB_IMBALANCE] >= imbalance_thresh
    spread_ok = df[COL_SPREAD_BPS] <= spread_bps_max
    pin_ok = df[COL_PIN_PROXY] <= pin_proxy_max

    entry_condition = imbalance_ok & spread_ok & pin_ok

    # Confirmation filter: require condition to be true for N consecutive rows
    confirmed_entry = entry_condition.rolling(window=confirmation_window, min_periods=confirmation_window).apply(
        lambda x: 1.0 if x.all() else 0.0, raw=True
    ).astype(bool)

    # Initialise signal column
    df["signal"] = 0

    # Long entry where confirmation is met and we are currently flat
    df.loc[confirmed_entry & (df["signal"].shift(fill_value=0) == 0), "signal"] = 1

    # Propagate existing long position until exit condition occurs
    df["signal"] = df["signal"].replace(to_replace=0, method="ffill")
    # Exit when any entry condition fails
    exit_condition = ~entry_condition
    df.loc[exit_condition & (df["signal"] == 1), "signal"] = 0

    # Ensure flat after exit
    df["signal"] = df["signal"].replace(to_replace=0, method="ffill").where(df["signal"] != 0, 0)

    return df