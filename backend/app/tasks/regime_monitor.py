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
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app.utils.logging import logger

try:
    from hmmlearn.hmm import GaussianHMM

    _HMM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HMM_AVAILABLE = False

# Simple in‑memory cache to avoid redundant work within the same day
_cache: Dict[str, Any] = {
    "returns": None,  # np.ndarray of last fetched returns
    "regime": None,   # int of last computed regime
    "date": None,     # datetime.date of last fetch
}


def _compute_features(returns: np.ndarray) -> np.ndarray:
    """
    Build feature matrix for HMM: [return, rolling 20‑day volatility].

    Parameters
    ----------
    returns: np.ndarray
        Daily returns series.

    Returns
    -------
    np.ndarray
        2‑column feature array.
    """
    vol_20 = pd.Series(returns).rolling(20).std().bfill().values
    return np.column_stack([returns, vol_20])


def _fit_hmm(features: np.ndarray) -> Optional[np.ndarray]:
    """
    Fit a Gaussian HMM with 3 components and return the inferred state
    sequence. Returns ``None`` if fitting fails.

    Parameters
    ----------
    features: np.ndarray
        Feature matrix for the HMM.

    Returns
    -------
    np.ndarray | None
        Predicted state indices, or ``None`` on error.
    """
    if not _HMM_AVAILABLE:
        return None
    try:
        model = GaussianHMM(
            n_components=3,
            covariance_type="diag",
            n_iter=200,
            random_state=42,
        )
        model.fit(features)
        return model.predict(features)
    except Exception as exc:  # pragma: no cover
        logger.warning("HMM fit failed, using heuristic", error=str(exc))
        return None


def _label_states(states: np.ndarray, features: np.ndarray) -> int:
    """
    Map raw HMM state indices to regime labels (0 = bear, 1 = sideways, 2 = bull)
    based on the mean return of each state.

    Parameters
    ----------
    states: np.ndarray
        HMM state sequence.
    features: np.ndarray
        Feature matrix (first column is return).

    Returns
    -------
    int
        Regime label for the most recent observation.
    """
    means = [features[states == s, 0].mean() for s in range(3)]
    order = np.argsort(means)
    label_map = {int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}
    return int(label_map[int(states[-1])])


def _heuristic_regime(returns: np.ndarray) -> int:
    """
    Simple fallback heuristic based on recent volatility rank and momentum.

    Parameters
    ----------
    returns: np.ndarray
        Daily returns series.

    Returns
    -------
    int
        Regime label (0, 1, or 2).
    """
    recent_vol = float(np.std(returns[-20:]))
    long_vol = float(np.std(returns[-252:]))
    vol_rank = recent_vol / max(long_vol, 1e-8)
    recent_return = float(np.mean(returns[-20:]))

    if vol_rank > 1.5 or recent_return < -0.002:
        return 0  # bear / crisis
    if recent_return > 0.001 and vol_rank < 0.9:
        return 2  # bull
    return 1  # sideways


def _fit_regime(returns: np.ndarray) -> int:
    """
    Determine the current market regime from a returns series.

    The function first attempts a Gaussian HMM fit; if that fails or the
    required library is missing, it falls back to a lightweight heuristic.

    Parameters
    ----------
    returns: np.ndarray
        Daily returns series.

    Returns
    -------
    int
        Regime label (0 = bear, 1 = sideways, 2 = bull).
    """
    if len(returns) < 60:
        return 1  # insufficient data → sideways

    features = _compute_features(returns)
    states = _fit_hmm(features)

    if states is not None:
        return _label_states(states, features)

    return _heuristic_regime(returns)


def _fetch_spy_returns_sync() -> Optional[np.ndarray]:
    """Sync yfinance fetch — must be called via run_in_executor."""
    try:
        import yfinance as yf  # type: ignore

        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=400)
        df = yf.download(
            "SPY",
            start=str(start),
            end=str(end),
            progress=False,
            auto_adjust=True,
        )
        if df is None or len(df) < 60:
            return None
        closes = df["Close"].dropna()
        return closes.pct_change().dropna().values.astype(float)
    except Exception as exc:  # pragma: no cover
        logger.warning("Regime monitor: SPY fetch failed", error=str(exc))
        return None


def _synthetic_spy_returns(n: int = 300) -> np.ndarray:
    """
    Generate deterministic synthetic SPY returns using a geometric Brownian
    motion model. Useful when live data cannot be fetched.

    Parameters
    ----------
    n: int, default 300
        Number of synthetic daily returns to produce.

    Returns
    -------
    np.ndarray
        Synthetic returns array.
    """
    seed = int(datetime.now(timezone.utc).strftime("%Y%m%d"))
    rng = np.random.default_rng(seed)
    daily_mu = 0.0003
    daily_sigma = 0.01
    return rng.normal(daily_mu, daily_sigma, n).astype(float)


async def _fetch_spy_returns() -> Optional[np.ndarray]:
    """Fetch 1 year of SPY daily returns without blocking the event loop."""
    today = datetime.now(timezone.utc).date()
    # Return cached data if we already fetched for today
    if _cache["date"] == today and isinstance(_cache["returns"], np.ndarray):
        return _cache["returns"]

    loop = asyncio.get_running_loop()
    returns = await loop.run_in_executor(None, _fetch_spy_returns_sync)

    if returns is not None:
        _cache["returns"] = returns
        _cache["date"] = today
    return returns


async def run_once(redis_client) -> Optional[int]:
    """Fit regime, write to Redis, return regime int or None on failure."""
    returns = await _fetch_spy_returns()
    if returns is None:
        logger.info(
            "Regime monitor: using synthetic SPY returns (live data unavailable)"
        )
        returns = _synthetic_spy_returns()

    # If returns haven't changed since last computation, reuse cached regime
    if (
        _cache["returns"] is not None
        and np.array_equal(returns, _cache["returns"])
        and isinstance(_cache["regime"], int)
    ):
        regime = _cache["regime"]
    else:
        regime = _fit_regime(returns)
        _cache["regime"] = regime
        _cache["returns"] = returns

    labels = {0: "bear", 1: "sideways", 2: "bull"}

    try:
        await redis_client.set("market:regime", str(regime), ex=600)  # TTL 10 min
        logger.info("Regime updated", regime=regime, label=labels[regime])
    except Exception as exc:  # pragma: no cover
        logger.warning("Regime monitor: Redis write failed", error=str(exc))
        return None

    return regime


class RegimeMonitor:
    """Background asyncio task — call start() in app lifespan."""

    INTERVAL_SECONDS = 300  # 5 minutes

    def __init__(self):
        self._task: Optional[asyncio.Task] = None

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
            except Exception as exc:  # pragma: no cover
                logger.warning("Regime monitor loop error", error=str(exc))
            await asyncio.sleep(self.INTERVAL_SECONDS)