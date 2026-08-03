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
  - Tightened entry conditions based on imbalance, spread and depth ratio.
  - Confirmation filter requiring the conditions to hold for a configurable
    number of consecutive snapshots.
  - Exit logic based on reversal of imbalance, spread widening or depth ratio
    deterioration.
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

MICROSTRUCTURE_FEATURE_COLS = [COL_LOB_IMBALANCE, COL_SPREAD_BPS]

# Signal thresholds – these can be tuned per instrument
IMBALANCE_ENTRY_THRESHOLD: float = 0.20   # minimum absolute imbalance to consider entry
SPREAD_ENTRY_MAX_BPS: float = 5.0        # max spread (bps) for entry
DEPTH_RATIO_ENTRY_MIN: float = 1.2      # minimum depth ratio for entry
PIN_PROXY_MAX: float = 0.30              # max PIN proxy (high values suggest informed flow)

IMBALANCE_EXIT_THRESHOLD: float = 0.10   # imbalance magnitude below which we exit
SPREAD_EXIT_MAX_BPS: float = 10.0        # spread widening trigger for exit
DEPTH_RATIO_EXIT_MIN: float = 0.9        # depth ratio deterioration trigger for exit

DEFAULT_CONFIRMATION_LOOKBACK: int = 3   # number of consecutive bars required


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

    # --------------------------------------------------------------------- #
    # Signal generation utilities
    # --------------------------------------------------------------------- #

    def _entry_conditions_met(self, row: pd.Series) -> bool:
        """
        Evaluate tightened entry conditions on a single row of features.
        """
        if abs(row["imbalance"]) < IMBALANCE_ENTRY_THRESHOLD:
            return False
        if row["spread_bps"] > SPREAD_ENTRY_MAX_BPS:
            return False
        if row.get("depth_ratio", 1.0) < DEPTH_RATIO_ENTRY_MIN:
            return False
        if row.get("pin_proxy", 0.0) > PIN_PROXY_MAX:
            return False
        return True

    def _exit_conditions_met(self, row: pd.Series) -> bool:
        """
        Evaluate exit conditions on a single row of features.
        """
        if abs(row["imbalance"]) < IMBALANCE_EXIT_THRESHOLD:
            return True
        if row["spread_bps"] > SPREAD_EXIT_MAX_BPS:
            return True
        if row.get("depth_ratio", 1.0) < DEPTH_RATIO_EXIT_MIN:
            return True
        return False

    def generate_signals(
        self,
        df: pd.DataFrame,
        confirmation_lookback: int = DEFAULT_CONFIRMATION_LOOKBACK,
    ) -> pd.DataFrame:
        """
        Generate entry/exit signals based on microstructure features.

        The method adds two columns to the DataFrame:
            * ``signal`` – 1 for long entry, -1 for short entry, 0 for no position.
            * ``signal_reason`` – short textual description for debugging.

        Entry requires that the tightened entry conditions hold for
        ``confirmation_lookback`` consecutive rows.  Exit triggers as soon
        as any exit condition is satisfied.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least the columns defined in ``MICROSTRUCTURE_FEATURE_COLS``.
        confirmation_lookback : int, optional
            Number of consecutive bars that must satisfy entry conditions.
            Default is ``DEFAULT_CONFIRMATION_LOOKBACK``.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with the added ``signal`` and ``signal_reason`` columns.
        """
        df = df.copy()

        # Ensure required columns exist
        missing = [c for c in MICROSTRUCTURE_FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required microstructure feature columns: {missing}")

        # Compute boolean masks for entry and exit conditions
        entry_mask = df.apply(self._entry_conditions_met, axis=1)
        exit_mask = df.apply(self._exit_conditions_met, axis=1)

        # Confirmation filter: entry only if the mask is True for the last N rows
        confirmed_entry = entry_mask.rolling(window=confirmation_lookback, min_periods=confirmation_lookback).apply(
            lambda x: 1.0 if x.all() else 0.0, raw=False
        ).astype(bool)

        # Initialise signal columns
        df["signal"] = 0
        df["signal_reason"] = ""

        # Long vs short based on sign of imbalance
        long_entries = confirmed_entry & (df["imbalance"] > 0)
        short_entries = confirmed_entry & (df["imbalance"] < 0)

        df.loc[long_entries, "signal"] = 1
        df.loc[short_entries, "signal"] = -1
        df.loc[long_entries, "signal_reason"] = "entry_long_imbalance"
        df.loc[short_entries, "signal_reason"] = "entry_short_imbalance"

        # Override with exit signals where applicable
        exit_positions = (df["signal"] != 0) & exit_mask
        df.loc[exit_positions, "signal"] = 0
        df.loc[exit_positions, "signal_reason"] = "exit_condition"

        return df


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

    return df