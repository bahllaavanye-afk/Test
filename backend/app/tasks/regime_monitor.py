"""
Market Regime Monitor — runs every 5 minutes.

Fits a 3-state HMM on SPY daily returns + volatility.
Writes current regime to Redis key 'market:regime':
  0 = bear (negative drift, high vol)
  1 = sideways (near-zero drift, moderate vol)
  2 = bull (positive drift, low vol)

Strategy runner reads this key to gate directional strategies.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from app.utils.logging import logger

try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False


def _fit_regime(returns: np.ndarray) -> int:
    """
    Fit Gaussian HMM and return the current regime (0/1/2).
    Falls back to vol-rank heuristic if hmmlearn is unavailable.
    """
    n = len(returns)
    if n < 60:
        return 1  # insufficient data → sideways

    vol_20 = pd.Series(returns).rolling(20).std().bfill().values
    features = np.column_stack([returns, vol_20])

    if _HMM_AVAILABLE:
        try:
            model = GaussianHMM(n_components=3, covariance_type="diag",
                                n_iter=200, random_state=42)
            model.fit(features)
            states = model.predict(features)
            # Label states by mean return: highest → bull(2), lowest → bear(0)
            means = [features[states == s, 0].mean() for s in range(3)]
            order = np.argsort(means)  # indices sorted by mean return ascending
            label = {int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}
            return int(label[int(states[-1])])
        except Exception as exc:
            logger.warning("HMM fit failed, using heuristic", error=str(exc))

    # Heuristic: vol rank + recent momentum
    recent_vol = float(np.std(returns[-20:]))
    long_vol = float(np.std(returns[-252:]))
    vol_rank = recent_vol / max(long_vol, 1e-8)
    recent_return = float(np.mean(returns[-20:]))

    if vol_rank > 1.5 or recent_return < -0.002:
        return 0  # bear / crisis
    if recent_return > 0.001 and vol_rank < 0.9:
        return 2  # bull
    return 1  # sideways


async def _fetch_spy_returns() -> np.ndarray | None:
    """Fetch 1 year of SPY daily returns via yfinance."""
    try:
        import yfinance as yf  # type: ignore
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=400)
        df = yf.download("SPY", start=str(start), end=str(end),
                          progress=False, auto_adjust=True)
        if df is None or len(df) < 60:
            return None
        closes = df["Close"].dropna()
        returns = closes.pct_change().dropna().values.astype(float)
        return returns
    except Exception as exc:
        logger.warning("Regime monitor: SPY fetch failed", error=str(exc))
        return None


async def run_once(redis_client) -> int | None:
    """Fit regime, write to Redis, return regime int or None on failure."""
    returns = await _fetch_spy_returns()
    if returns is None:
        return None

    regime = _fit_regime(returns)
    labels = {0: "bear", 1: "sideways", 2: "bull"}

    try:
        await redis_client.set("market:regime", str(regime), ex=600)  # TTL 10 min
        logger.info("Regime updated", regime=regime, label=labels[regime])
    except Exception as exc:
        logger.warning("Regime monitor: Redis write failed", error=str(exc))
        return None

    return regime


class RegimeMonitor:
    """Background asyncio task — call start() in app lifespan."""

    INTERVAL_SECONDS = 300  # 5 minutes

    def __init__(self):
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="regime_monitor")

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        from app.redis_client import get_redis
        redis = get_redis()
        while True:
            try:
                await run_once(redis)
            except Exception as exc:
                logger.warning("Regime monitor loop error", error=str(exc))
            await asyncio.sleep(self.INTERVAL_SECONDS)
