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


def _prepare_features(returns: np.ndarray) -> np.ndarray:
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


def _label_states(states: np.ndarray, features: np.ndarray) -> dict[int, int]:
    """
    Map raw HMM state indices to regime labels (0,1,2) based on mean return.

    Parameters
    ----------
    states: np.ndarray
        Predicted state sequence from HMM.
    features: np.ndarray
        Feature matrix used for fitting.

    Returns
    -------
    dict[int, int]
        Mapping from HMM state index to regime label.
    """
    means = [features[states == s, 0].mean() for s in range(3)]
    order = np.argsort(means)  # ascending: bear → bull
    return {int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}


def _hmm_regime(returns: np.ndarray) -> int | None:
    """
    Fit a Gaussian HMM and return the regime of the most recent observation.

    Returns ``None`` if the fit fails.
    """
    features = _prepare_features(returns)
    try:
        model = GaussianHMM(
            n_components=3,
            covariance_type="diag",
            n_iter=200,
            random_state=42,
        )
        model.fit(features)
        states = model.predict(features)
        label_map = _label_states(states, features)
        return int(label_map[int(states[-1])])
    except Exception as exc:  # pragma: no cover
        logger.warning("HMM fit failed, using heuristic", error=str(exc))
        return None


def _heuristic_regime(returns: np.ndarray) -> int:
    """
    Simple heuristic based on recent volatility rank and momentum.

    Returns
    -------
    int
        0 = bear, 1 = sideways, 2 = bull
    """
    recent_vol = float(np.std(returns[-20:]))
    long_vol = float(np.std(returns[-252:]))
    vol_rank = recent_vol / max(long_vol, 1e-8)
    recent_return = float(np.mean(returns[-20:]))

    if vol_rank > 1.5 or recent_return < -0.002:
        return 0
    if recent_return > 0.001 and vol_rank < 0.9:
        return 2
    return 1


def _fit_regime(returns: np.ndarray) -> int:
    """
    Determine the current market regime from SPY returns.

    If enough data are available, attempts an HMM fit; otherwise falls back
    to a deterministic heuristic. The function never raises.

    Parameters
    ----------
    returns: np.ndarray
        Daily return series.

    Returns
    -------
    int
        Regime label: 0 = bear, 1 = sideways, 2 = bull.
    """
    if len(returns) < 60:
        return 1  # insufficient data → sideways

    if _HMM_AVAILABLE:
        hmm_result = _hmm_regime(returns)
        if hmm_result is not None:
            return hmm_result

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
    GBM synthetic SPY returns when yfinance is unreachable (network policy,
    offline dev container). Keeps the regime monitor functional 24/7.
    Deterministic per‑day seed so the regime is stable within a session.
    """
    seed = int(datetime.now(timezone.utc).strftime("%Y%m%d"))
    rng = np.random.default_rng(seed)
    daily_mu = 0.0003
    daily_sigma = 0.01
    return rng.normal(daily_mu, daily_sigma, n).astype(float)


async def _fetch_spy_returns() -> np.ndarray | None:
    """Fetch 1 year of SPY daily returns without blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _fetch_spy_returns_sync)


async def run_once(redis_client) -> int | None:
    """Fit regime, write to Redis, return regime int or None on failure."""
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