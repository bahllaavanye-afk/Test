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

Additionally provides a lightweight microstructure‑based signal generator that
applies tighter entry conditions, confirmation filters and improved exit logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Iterable, List, Tuple, Union


class OrderBookFeatures:
    """Compute LOB features from real‑time bid/ask depth."""

    def compute_imbalance(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        levels: int = 5,
    ) -> float:
        """
        Order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol).
        Returns value in [-1, 1]. Positive = bid‑heavy (buying pressure).

        Args:
            bids: list of (price, size) pairs, best bid first
            asks: list of (price, size) pairs, best ask first
            levels: how many price levels to include
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
        Bid‑ask spread in basis points: (ask - bid) / mid * 10_000.
        Returns 0.0 for invalid inputs.
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
        """
        Top‑of‑book depth ratio: best_bid_size / best_ask_size.
        Values > 1 indicate more liquidity on bid side.
        Returns 1.0 if either side is empty.
        """
        if not bids or not asks:
            return 1.0
        best_bid_size = float(bids[0][1])
        best_ask_size = float(asks[0][1])
        if best_ask_size <= 0:
            return 1.0
        return float(best_bid_size / best_ask_size)

    def compute_pin_proxy(self, buy_volume: float, sell_volume: float) -> float:
        """
        Probability of Informed Trading proxy.
        PIN = |buy_vol - sell_vol| / (buy_vol + sell_vol)
        Returns value in [0, 1]. Near 1 = highly informed order flow.
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
        Kyle's lambda (price impact coefficient).
        Estimated via OLS: delta_price = lambda * signed_volume + epsilon

        Returns lambda (bps per unit volume). Higher = less liquid.
        Returns 0.0 if insufficient data.
        """
        if len(price_changes) < 5 or len(signed_volumes) < 5:
            return 0.0
        try:
            vol = np.asarray(signed_volumes, dtype=float)
            dp = np.asarray(price_changes, dtype=float)
            var_vol = np.var(vol)
            if var_vol < 1e-12:
                return 0.0
            cov_matrix = np.cov(dp, vol, ddof=0)
            # cov_matrix shape is (2,2); element [0,1] is cov(dp, vol)
            lam = float(cov_matrix[0, 1] / var_vol)
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


class VPINFeatures:
    """
    Volume‑Synchronized Probability of Informed Trading (VPIN).

    VPIN measures order‑flow toxicity by classifying trades as buyer‑ or
    seller‑initiated using the Lee‑Ready rule, then computing the imbalance
    within equal‑volume buckets. High VPIN (> 0.5) signals elevated
    informed‑trading risk and often precedes adverse price moves.

    Reference: Easley, López de Prado & O'Hara (2012).
    """

    def classify_trades_lee_ready(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        prev_close: float | None = None,
    ) -> np.ndarray:
        """
        Classify each trade as buy (+volume) or sell (-volume) using Lee‑Ready.

        Rule:
          - If price > prev_price → buy
          - If price < prev_price → sell
          - If price == prev_price (tick test) → inherit last non‑zero direction

        Returns signed_volumes array of same length as prices.
        """
        prices = np.asarray(prices, dtype=float)
        volumes = np.asarray(volumes, dtype=float)
        n = len(prices)
        signed = np.zeros(n, dtype=float)

        last_direction = 1  # default: buy
        prev = prev_close if prev_close is not None else prices[0]

        for i in range(n):
            if prices[i] > prev:
                last_direction = 1
            elif prices[i] < prev:
                last_direction = -1
            signed[i] = last_direction * volumes[i]
            prev = prices[i]

        return signed

    def compute_vpin(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        bucket_size: float | None = None,
        n_buckets: int = 50,
        prev_close: float | None = None,
    ) -> float:
        """
        Compute VPIN over the provided trade data.

        Args:
            prices: trade prices (chronological order)
            volumes: trade volumes (same order as prices)
            bucket_size: target volume per bucket; if None, auto‑computed as
                         total_volume / n_buckets
            n_buckets: number of equal‑volume buckets to use; only used when
                       bucket_size is None
            prev_close: optional previous close price for the first tick test

        Returns:
            VPIN value in [0, 1]. 0 indicates no imbalance, 1 extreme imbalance.
        """
        if len(prices) == 0 or len(volumes) == 0:
            return 0.0

        signed = self.classify_trades_lee_ready(prices, volumes, prev_close)

        total_vol = np.sum(np.abs(signed))
        if total_vol == 0:
            return 0.0

        if bucket_size is None:
            bucket_size = total_vol / max(1, n_buckets)

        # Accumulate into volume‑synchronized buckets
        cum_vol = 0.0
        bucket_imbalance = []
        bucket_buy = 0.0
        bucket_sell = 0.0

        for sv in signed:
            vol = abs(sv)
            direction = 1 if sv > 0 else -1
            while cum_vol + vol >= bucket_size:
                portion = bucket_size - cum_vol
                if direction > 0:
                    bucket_buy += portion
                else:
                    bucket_sell += portion
                # record bucket
                imbalance = abs(bucket_buy - bucket_sell) / (bucket_buy + bucket_sell)
                bucket_imbalance.append(imbalance)
                # reset for next bucket
                vol -= portion
                cum_vol = 0.0
                bucket_buy = bucket_sell = 0.0
            # remaining volume stays in current bucket
            if vol > 0:
                if direction > 0:
                    bucket_buy += vol
                else:
                    bucket_sell += vol
                cum_vol += vol

        if not bucket_imbalance:
            return 0.0
        return float(np.mean(bucket_imbalance))


class MicrostructureSignalGenerator:
    """
    Lightweight signal generator based on microstructure features.

    The logic aims for higher signal quality by:
      * Tightening entry thresholds (e.g., stronger imbalance, tighter spread)
      * Adding confirmation filters (depth ratio and PIN proxy)
      * Using a clear exit rule (reversal of imbalance or spread widening)

    The implementation is deliberately simple and fully vectorised to work
    efficiently on pandas DataFrames.
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.25,
        spread_bps_max: float = 5.0,
        depth_ratio_min: float = 1.1,
        pin_proxy_max: float = 0.4,
        exit_imbalance_threshold: float = 0.1,
        exit_spread_bps_increment: float = 2.0,
    ):
        """
        Initialise the generator with configurable thresholds.

        All thresholds are expressed in the same units as the underlying
        features (e.g., imbalance ∈ [-1, 1], spread_bps in basis points).

        Args:
            imbalance_threshold: minimum absolute imbalance to consider entry.
            spread_bps_max: maximum spread (bps) for a valid entry.
            depth_ratio_min: minimum depth ratio (>1) confirming liquidity on the bid side.
            pin_proxy_max: maximum PIN proxy allowed for entry (lower toxicity).
            exit_imbalance_threshold: imbalance magnitude below which a position is closed.
            exit_spread_bps_increment: additional spread bps over the entry level that triggers exit.
        """
        self.imbalance_threshold = imbalance_threshold
        self.spread_bps_max = spread_bps_max
        self.depth_ratio_min = depth_ratio_min
        self.pin_proxy_max = pin_proxy_max
        self.exit_imbalance_threshold = exit_imbalance_threshold
        self.exit_spread_bps_increment = exit_spread_bps_increment

    def _generate_raw_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Produce a raw directional signal (+1 long, -1 short, 0 flat) based on
        the entry criteria alone.
        """
        cond_long = (
            (df["lob_imbalance"] >= self.imbalance_threshold)
            & (df["spread_bps"] <= self.spread_bps_max)
            & (df["depth_ratio"] >= self.depth_ratio_min)
            & (df["pin_proxy"] <= self.pin_proxy_max)
        )
        cond_short = (
            (df["lob_imbalance"] <= -self.imbalance_threshold)
            & (df["spread_bps"] <= self.spread_bps_max)
            & (df["depth_ratio"] >= self.depth_ratio_min)
            & (df["pin_proxy"] <= self.pin_proxy_max)
        )
        signal = pd.Series(0, index=df.index, dtype=int)
        signal[cond_long] = 1
        signal[cond_short] = -1
        return signal

    def _apply_exit_logic(self, df: pd.DataFrame, raw_signal: pd.Series) -> pd.Series:
        """
        Convert the raw entry signal into a realistic position series by
        applying exit rules. The position is held until an exit condition
        is met, after which it reverts to flat (0) until a new entry signal
        appears.
        """
        position = pd.Series(0, index=df.index, dtype=int)

        current_pos = 0
        entry_spread = np.nan

        for idx in df.index:
            sig = raw_signal.loc[idx]

            # If we are flat and a new entry signal appears, open position
            if current_pos == 0 and sig != 0:
                current_pos = sig
                entry_spread = df.at[idx, "spread_bps"]
                position.at[idx] = current_pos
                continue

            # If we already have a position, check exit conditions
            if current_pos != 0:
                # 1) Imbalance weakens below exit threshold (same sign)
                imbalance = df.at[idx, "lob_imbalance"]
                if (
                    (current_pos == 1 and imbalance < self.exit_imbalance_threshold)
                    or (current_pos == -1 and imbalance > -self.exit_imbalance_threshold)
                ):
                    current_pos = 0
                    entry_spread = np.nan
                    position.at[idx] = 0
                    continue

                # 2) Spread widens beyond entry level + increment
                spread = df.at[idx, "spread_bps"]
                if not np.isnan(entry_spread) and spread > entry_spread + self.exit_spread_bps_increment:
                    current_pos = 0
                    entry_spread = np.nan
                    position.at[idx] = 0
                    continue

                # No exit triggered; maintain existing position
                position.at[idx] = current_pos
            else:
                # Flat and no entry
                position.at[idx] = 0

        return position

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Public API: return a DataFrame with a 'microstructure_signal' column.

        The input DataFrame must contain the columns:
          - lob_imbalance
          - spread_bps
          - depth_ratio
          - pin_proxy

        Missing columns are filled with safe defaults (0 for imbalance and spread,
        1 for depth_ratio, 0 for pin_proxy) to avoid runtime errors.
        """
        required = ["lob_imbalance", "spread_bps", "depth_ratio", "pin_proxy"]
        for col in required:
            if col not in df.columns:
                # Provide sensible defaults
                if col == "depth_ratio":
                    df[col] = 1.0
                else:
                    df[col] = 0.0

        raw = self._generate_raw_signal(df)
        position = self._apply_exit_logic(df, raw)
        result = df.copy()
        result["microstructure_signal"] = position
        return result