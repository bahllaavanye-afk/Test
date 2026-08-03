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
from typing import Any

import numpy as np
import pandas as pd

from app.utils.logging import logger

try:
    from hmmlearn.hmm import GaussianHMM

    _HMM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HMM_AVAILABLE = False


def _compute_features(returns: np.ndarray) -> np.ndarray:
    """
    Build feature matrix for HMM: [return, rolling 20‑day volatility].

    Parameters
    ----------
    returns: np.ndarray
        Daily returns series. Must be a 1‑dimensional numeric array.

    Returns
    -------
    np.ndarray
        2‑column feature array.
    """
    if not isinstance(returns, np.ndarray):
        raise ValueError("returns must be a numpy.ndarray")
    if returns.ndim != 1:
        raise ValueError("returns must be a 1‑dimensional array")
    if returns.size == 0:
        raise ValueError("returns array cannot be empty")

    vol_20 = pd.Series(returns).rolling(20).std().bfill().values
    return np.column_stack([returns, vol_20])


def _fit_hmm(features: np.ndarray) -> np.ndarray | None:
    """
    Fit a Gaussian HMM with 3 components and return the inferred state
    sequence. Returns ``None`` if fitting fails.

    Parameters
    ----------
    features: np.ndarray
        Feature matrix for the HMM. Must be a 2‑column numeric array.

    Returns
    -------
    np.ndarray | None
        Predicted state indices, or ``None`` on error.
    """
    if not isinstance(features, np.ndarray):
        raise ValueError("features must be a numpy.ndarray")
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError("features must be a 2‑dimensional array with shape (n, 2)")

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
        HMM state sequence. Must be a 1‑dimensional integer array.
    features: np.ndarray
        Feature matrix (first column is return). Must have the same number of rows
        as ``states`` and exactly two columns.

    Returns
    -------
    int
        Regime label for the most recent observation.
    """
    if not isinstance(states, np.ndarray):
        raise ValueError("states must be a numpy.ndarray")
    if states.ndim != 1:
        raise ValueError("states must be a 1‑dimensional array")
    if not isinstance(features, np.ndarray):
        raise ValueError("features must be a numpy.ndarray")
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError("features must be a 2‑dimensional array with shape (n, 2)")
    if len(states) != features.shape[0]:
        raise ValueError("states length must match number of feature rows")

    # Compute mean return per state
    means = [features[states == s, 0].mean() for s in range(3)]
    # Order states by ascending mean return
    order = np.argsort(means)
    # Create mapping: lowest mean → bear (0), middle → sideways (1), highest → bull (2)
    label_map = {int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}
    return int(label_map[int(states[-1])])


def _heuristic_regime(returns: np.ndarray) -> int:
    """
    Simple fallback heuristic based on recent volatility rank and momentum.

    Parameters
    ----------
    returns: np.ndarray
        Daily returns series. Must be a 1‑dimensional numeric array with at
        least 20 elements.

    Returns
    -------
    int
        Regime label (0, 1, or 2).
    """
    if not isinstance(returns, np.ndarray):
        raise ValueError("returns must be a numpy.ndarray")
    if returns.ndim != 1:
        raise ValueError("returns must be a 1‑dimensional array")
    if returns.size < 20:
        raise ValueError("returns array must contain at least 20 elements for heuristic")

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
        Daily returns series. Must be a 1‑dimensional numeric array.

    Returns
    -------
    int
        Regime label (0 = bear, 1 = sideways, 2 = bull).
    """
    if not isinstance(returns, np.ndarray):
        raise ValueError("returns must be a numpy.ndarray")
    if returns.ndim != 1:
        raise ValueError("returns must be a 1‑dimensional array")
    if returns.size < 60:
        return 1  # insufficient data → sideways

    features = _compute_features(returns)
    states = _fit_hmm(features)

    if states is not None:
        return _label_states(states, features)

    return _heuristic_regime(returns)


def _fetch_spy_returns_sync() -> np.ndarray | None:
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
    except Exception as exc:
        logger.warning("Regime monitor: SPY fetch failed", error=str(exc))
        return None


def _synthetic_spy_returns(n: int = 300) -> np.ndarray:
    """
    Generate deterministic synthetic SPY returns using a geometric Brownian
    motion model. Useful when live data cannot be fetched.

    Parameters
    ----------
    n: int, default 300
        Number of synthetic daily returns to produce. Must be a positive integer.

    Returns
    -------
    np.ndarray
        Synthetic returns array.
    """
    if not isinstance(n, int):
        raise ValueError("n must be an integer")
    if n <= 0:
        raise ValueError("n must be a positive integer")

    seed = int(datetime.now(timezone.utc).strftime("%Y%m%d"))
    rng = np.random.default_rng(seed)
    daily_mu = 0.0003
    daily_sigma = 0.01
    return rng.normal(daily_mu, daily_sigma, n).astype(float)


async def _fetch_spy_returns() -> np.ndarray | None:
    """Fetch 1 year of SPY daily returns without blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _fetch_spy_returns_sync)


async def run_once(redis_client: Any) -> int | None:
    """
    Fit regime, write to Redis, and return the regime integer.

    Parameters
    ----------
    redis_client: Any
        An asynchronous Redis client with a ``set`` coroutine method.

    Returns
    -------
    int | None
        Regime label on success, or ``None`` if the operation fails.
    """
    if redis_client is None:
        raise ValueError("redis_client cannot be None")
    if not hasattr(redis_client, "set"):
        raise ValueError("redis_client must have a 'set' method")

    returns = await _fetch_spy_returns()
    if returns is None:
        logger.info(
            "Regime monitor: using synthetic SPY returns (live data unavailable)"
        )
        returns = _synthetic_spy_returns()

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
        """Create and schedule the monitor loop."""
        self._task = asyncio.create_task(self._loop(), name="regime_monitor")

    def stop(self) -> None:
        """Cancel the monitor loop if it is running."""
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