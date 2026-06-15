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


class OrderBookFeatures:
    """Compute LOB features from real-time bid/ask depth."""

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
        Bid-ask spread in basis points: (ask - bid) / mid * 10_000.
        Returns 0.0 for invalid inputs.
        """
        if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
            return 0.0
        mid = (best_bid + best_ask) / 2.0
        return float((best_ask - best_bid) / mid * 10_000.0)

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
        best_bid_size = float(bids[0][1]) if bids else 0.0
        best_ask_size = float(asks[0][1]) if asks else 0.0
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
            vol = np.array(signed_volumes, dtype=float)
            dp = np.array(price_changes, dtype=float)
            # OLS: lambda = cov(dp, vol) / var(vol)
            var_vol = np.var(vol)
            if var_vol < 1e-12:
                return 0.0
            lam = float(np.cov(dp, vol)[0, 1] / var_vol)
            return lam
        except Exception:
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
      - volume_imbalance_proxy: (close - open) / (high - low + 1e-9)  — approximates buy/sell pressure
      - spread_bps_proxy: (high - low) / close * 10_000               — proxy for intraday spread
    """
    df = df.copy()

    if imbalance_series is not None:
        df["lob_imbalance"] = imbalance_series.reindex(df.index).fillna(0.0)
    else:
        # Proxy: (close - open) / range
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        df["lob_imbalance"] = ((df["close"] - df["open"]) / rng).clip(-1, 1).fillna(0.0)

    if spread_bps_series is not None:
        df["spread_bps"] = spread_bps_series.reindex(df.index).fillna(0.0)
    else:
        df["spread_bps"] = ((df["high"] - df["low"]) / df["close"].replace(0, np.nan) * 10_000).fillna(0.0)

    return df


MICROSTRUCTURE_FEATURE_COLS = ["lob_imbalance", "spread_bps"]


class VPINFeatures:
    """
    Volume-Synchronized Probability of Informed Trading (VPIN).

    VPIN measures order-flow toxicity by classifying trades as buyer- or
    seller-initiated using the Lee-Ready rule, then computing the imbalance
    within equal-volume buckets. High VPIN (> 0.5) signals elevated
    informed-trading risk and often precedes adverse price moves.

    Reference: Easley, López de Prado & O'Hara (2012).
    """

    def classify_trades_lee_ready(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        prev_close: float | None = None,
    ) -> np.ndarray:
        """
        Classify each trade as buy (+volume) or sell (-volume) using Lee-Ready.

        Rule:
          - If price > prev_price → buy
          - If price < prev_price → sell
          - If price == prev_price (tick test) → inherit last non-zero direction

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
            # else: tie — keep last_direction (tick test)
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
            bucket_size: target volume per bucket; if None, auto-computed as
                         total_volume / n_buckets
            n_buckets: number of equal-volume buckets to use; only used when
                       bucket_size is None
            prev_close: previous bar's close for the first Lee-Ready comparison

        Returns:
            VPIN in [0, 1]: probability of informed trading.
            Returns 0.0 if insufficient data.
        """
        prices = np.asarray(prices, dtype=float)
        volumes = np.asarray(volumes, dtype=float)

        if len(prices) < 10 or volumes.sum() <= 0:
            return 0.0

        signed = self.classify_trades_lee_ready(prices, volumes, prev_close)
        total_volume = volumes.sum()

        if bucket_size is None:
            bucket_size = total_volume / max(n_buckets, 1)

        # Fill equal-volume buckets
        bucket_buy: list[float] = []
        bucket_sell: list[float] = []
        cur_buy = 0.0
        cur_sell = 0.0
        cur_vol = 0.0

        for sv in signed:
            vol = abs(sv)
            if sv > 0:
                cur_buy += vol
            else:
                cur_sell += vol
            cur_vol += vol

            while cur_vol >= bucket_size:
                # Complete one bucket
                fraction = bucket_size / cur_vol
                bucket_buy.append(cur_buy * fraction)
                bucket_sell.append(cur_sell * fraction)

                remaining = 1.0 - fraction
                cur_buy *= remaining
                cur_sell *= remaining
                cur_vol -= bucket_size

        if not bucket_buy:
            return 0.0

        buy_arr = np.array(bucket_buy)
        sell_arr = np.array(bucket_sell)
        total_arr = buy_arr + sell_arr
        valid = total_arr > 0
        if not valid.any():
            return 0.0

        vpin = float(np.mean(np.abs(buy_arr[valid] - sell_arr[valid]) / total_arr[valid]))
        return min(max(vpin, 0.0), 1.0)

    def compute_vpin_series(
        self,
        df: pd.DataFrame,
        window: int = 50,
        bucket_size: float | None = None,
    ) -> pd.Series:
        """
        Compute a rolling VPIN series from an OHLCV DataFrame.

        Uses VWAP as a proxy for trade prices and volume as trade volume.
        Produces one VPIN value per bar using the trailing `window` bars.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
            window: number of bars in the rolling window
            bucket_size: target volume per bucket (auto-computed if None)

        Returns:
            pd.Series of VPIN values aligned with df.index, NaN before warmup.
        """
        if "volume" not in df.columns:
            return pd.Series(np.nan, index=df.index)

        vwap = (df["high"] + df["low"] + df["close"]) / 3.0
        volumes = df["volume"].fillna(0.0)

        vpin_vals = np.full(len(df), np.nan)
        for i in range(window - 1, len(df)):
            p_slice = vwap.iloc[i - window + 1 : i + 1].to_numpy()
            v_slice = volumes.iloc[i - window + 1 : i + 1].to_numpy()
            prev = vwap.iloc[i - window] if i - window >= 0 else None
            vpin_vals[i] = self.compute_vpin(
                p_slice, v_slice, bucket_size=bucket_size, prev_close=float(prev) if prev is not None else None
            )

        return pd.Series(vpin_vals, index=df.index, name="vpin")


def add_vpin_feature(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Convenience wrapper: compute VPIN and add it as a column.
    Requires 'volume' column in df. NaN for the first `window-1` rows.
    """
    featurizer = VPINFeatures()
    df = df.copy()
    df["vpin"] = featurizer.compute_vpin_series(df, window=window)
    return df
