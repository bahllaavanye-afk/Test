"""
Kalman Filter Pairs Trading
============================
Dynamic hedge ratio estimation using Kalman filter.

Problem with static OLS pairs: the hedge ratio β drifts over time as business
conditions change (e.g., AAPL/MSFT ratio shifts with product cycles).

Solution: Kalman filter estimates β online, treating it as a latent variable:
  Observation:  y_t = β_t × x_t + ε_t      (ε ~ N(0, R))
  State:        β_t = β_{t-1} + w_t          (w ~ N(0, Q))

Parameters tuned to financial data:
- Q (process noise): 1e-5 (hedge ratio changes slowly)
- R (observation noise): 1e-3 (price relationship is noisy)
- Initial state: OLS estimate on first 60 days
- Entry: |spread_z| > 2.0
- Exit: |spread_z| < 0.5
- Stop: |spread_z| > 3.5

Academic: Elliott et al. (2005) "Pairs Trading", Kalman & Bucy (1961)
Documented Sharpe: 1.5-2.5 (market neutral)
"""
import numpy as np
import pandas as pd
import httpx
import asyncio
from datetime import date, timedelta
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal
from app.config import settings


def kalman_filter_regression(y: np.ndarray, x: np.ndarray,
                              Q: float = 1e-5, R: float = 1e-3) -> tuple:
    """
    Run Kalman filter to estimate dynamic hedge ratio β.
    Returns: (beta_series, spread_series, spread_zscore)
    """
    n = len(y)
    # State = [β, α] (hedge ratio and intercept)
    beta = np.zeros(n)
    alpha = np.zeros(n)
    P = np.eye(2)  # State covariance (2x2 for β and α)

    # Initialize with OLS on first 30 observations
    init = min(30, n)
    X_init = np.column_stack([x[:init], np.ones(init)])
    ols = np.linalg.lstsq(X_init, y[:init], rcond=None)[0]
    state = np.array([ols[0], ols[1]])  # [β, α]

    spreads = np.zeros(n)

    for t in range(n):
        H = np.array([x[t], 1.0])  # Observation matrix

        # Predict
        # (State transition = identity, no drift)
        # P_pred = P + Q*I
        P = P + Q * np.eye(2)

        # Update
        innovation = y[t] - H @ state  # prediction error
        S = H @ P @ H + R              # innovation variance
        K = P @ H / S                  # Kalman gain

        state = state + K * innovation  # state update
        P = (np.eye(2) - np.outer(K, H)) @ P  # covariance update

        beta[t] = state[0]
        alpha[t] = state[1]
        spreads[t] = y[t] - state[0] * x[t] - state[1]

    # Z-score of spread using rolling 60-bar window
    spread_series = pd.Series(spreads)
    spread_mean = spread_series.rolling(60, min_periods=20).mean()
    spread_std = spread_series.rolling(60, min_periods=20).std()
    spread_z = (spread_series - spread_mean) / spread_std.clip(lower=1e-8)

    return beta, spreads, spread_z.values


class KalmanPairsStrategy(AbstractStrategy):
    """
    Pairs trading with dynamic (Kalman) hedge ratio.
    Tracks cointegration in real-time — adapts to regime shifts.
    """
    name = "kalman_pairs"
    display_name = "Kalman Filter Pairs Trading"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 3600.0

    # Default pairs: economically linked, historically cointegrated
    DEFAULT_PAIRS = [
        ("XOM", "CVX"),   # Oil majors
        ("KO", "PEP"),    # Cola duopoly
        ("GS", "MS"),     # Investment banks
        ("AMD", "NVDA"),  # GPU/CPU chips
        ("V", "MA"),      # Payment networks
        ("AAPL", "MSFT"), # Mega-cap tech
    ]

    ENTRY_Z = 2.0    # Enter when |z| > 2.0
    EXIT_Z  = 0.5    # Exit when |z| < 0.5
    STOP_Z  = 3.5    # Stop loss when |z| > 3.5

    _DATA_BASE = "https://data.alpaca.markets"

    def __init__(self, params: dict | None = None):
        super().__init__(params)

    def _headers(self):
        return {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        }

    async def _fetch_bars(self, symbol: str, days: int = 252) -> pd.Series:
        start = (date.today() - timedelta(days=days)).isoformat()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self._DATA_BASE}/v2/stocks/{symbol}/bars",
                params={"timeframe": "1Day", "start": start, "limit": days},
                headers=self._headers(),
            )
        if resp.status_code != 200:
            return pd.Series(dtype=float)
        bars = resp.json().get("bars", [])
        if not bars:
            return pd.Series(dtype=float)
        s = pd.Series(
            {b["t"]: float(b["c"]) for b in bars},
            name=symbol,
        )
        s.index = pd.to_datetime(s.index)
        return s

    async def analyze(self, data: pd.DataFrame, symbol: str = "XOM") -> Signal | None:
        # Find which pair contains this symbol
        pair = None
        other = None
        for a, b in self.DEFAULT_PAIRS:
            if symbol == a:
                pair, other = a, b
                break
            if symbol == b:
                pair, other = b, a
                break
        if pair is None:
            # Default to first pair
            pair, other = self.DEFAULT_PAIRS[0]

        # Fetch price series for both legs
        y_series, x_series = await asyncio.gather(
            self._fetch_bars(pair),
            self._fetch_bars(other),
        )
        if y_series.empty or x_series.empty:
            return None

        # Align
        common = y_series.index.intersection(x_series.index)
        if len(common) < 60:
            return None
        y = np.log(y_series[common].values.astype(float))
        x = np.log(x_series[common].values.astype(float))

        # Run Kalman filter
        beta, spreads, z_scores = kalman_filter_regression(y, x)
        current_z = z_scores[-1]
        current_beta = beta[-1]

        if abs(current_z) > self.STOP_Z:
            return None  # Stop: relationship may be broken

        if abs(current_z) < self.ENTRY_Z:
            return None  # No signal

        # z > 0: y is expensive relative to x → sell y, buy x
        # z < 0: y is cheap relative to x → buy y, sell x
        side = "sell" if current_z > 0 else "buy"
        confidence = min((abs(current_z) - self.ENTRY_Z) / (self.STOP_Z - self.ENTRY_Z), 1.0)

        return Signal(
            symbol=pair,
            side=side,
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "strategy": "kalman_pairs",
                "pair": f"{pair}/{other}",
                "z_score": round(float(current_z), 3),
                "hedge_ratio": round(float(current_beta), 4),
                "entry_threshold": self.ENTRY_Z,
                "exit_threshold": self.EXIT_Z,
                "hedge_symbol": other,
                "hedge_side": "buy" if side == "sell" else "sell",
            },
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Vectorized backtest using precomputed spread if df has 'y' and 'x' columns."""
        if "y" in df.columns and "x" in df.columns:
            y = np.log(df["y"].values)
            x = np.log(df["x"].values)
        elif "close" in df.columns:
            y = np.log(df["close"].values)
            x = np.roll(y, 1)  # degenerate: self-correlation (for API compatibility)
        else:
            return BacktestSignals(
                entries=pd.Series(False, index=df.index),
                exits=pd.Series(False, index=df.index),
            )

        _, _, z_scores = kalman_filter_regression(y, x)
        z = pd.Series(z_scores, index=df.index)

        # Long entry: z very negative (y cheap vs x)
        entries = (z.shift(1) < -self.ENTRY_Z).fillna(False)
        exits = (z.shift(1) > -self.EXIT_Z).fillna(False)
        short_entries = (z.shift(1) > self.ENTRY_Z).fillna(False)
        short_exits = (z.shift(1) < self.EXIT_Z).fillna(False)

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )
