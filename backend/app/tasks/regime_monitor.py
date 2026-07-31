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

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# HMM configuration
HMM_COMPONENTS: int = 3
HMM_COVARIANCE_TYPE: str = "diag"
HMM_ITERATIONS: int = 200
HMM_RANDOM_STATE: int = 42

# Data fetch parameters
SPY_TICKER: str = "SPY"
FETCH_DAYS: int = 400
MIN_REQUIRED_DAYS: int = 60  # Minimum returns length to consider regime
MIN_DF_LENGTH: int = 60

# Heuristic thresholds
RECENT_WINDOW_DAYS: int = 20
LONG_WINDOW_DAYS: int = 252
VOL_RANK_BEAR_THRESHOLD: float = 1.5
VOL_RANK_BULL_THRESHOLD: float = 0.9
RETURN_BEAR_THRESHOLD: float = -0.002
RETURN_BULL_THRESHOLD: float = 0.001

# Synthetic returns generation
SYNTH_RETURNS_DEFAULT_N: int = 300
DAILY_MU: float = 0.0003
DAILY_SIGMA: float = 0.01

# Redis settings
REDIS_REGIME_KEY: str = "market:regime"
REDIS_TTL_SECONDS: int = 600  # 10 minutes

# Regime label mapping
REGIME_LABELS: dict[int, str] = {0: "bear", 1: "sideways", 2: "bull"}

# --------------------------------------------------------------------------- #

try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False


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
    vol_20 = pd.Series(returns).rolling(RECENT_WINDOW_DAYS).std().bfill().values
    return np.column_stack([returns, vol_20])


def _fit_hmm(features: np.ndarray) -> np.ndarray | None:
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
            n_components=HMM_COMPONENTS,
            covariance_type=HMM_COVARIANCE_TYPE,
            n_iter=HMM_ITERATIONS,
            random_state=HMM_RANDOM_STATE,
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
    # Compute mean return per state
    means = [features[states == s, 0].mean() for s in range(HMM_COMPONENTS)]
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
        Daily returns series.

    Returns
    -------
    int
        Regime label (0, 1, or 2).
    """
    recent_vol = float(np.std(returns[-RECENT_WINDOW_DAYS:]))
    long_vol = float(np.std(returns[-LONG_WINDOW_DAYS:]))
    vol_rank = recent_vol / max(long_vol, 1e-8)
    recent_return = float(np.mean(returns[-RECENT_WINDOW_DAYS:]))

    if vol_rank > VOL_RANK_BEAR_THRESHOLD or recent_return < RETURN_BEAR_THRESHOLD:
        return 0  # bear / crisis
    if recent_return > RETURN_BULL_THRESHOLD and vol_rank < VOL_RANK_BULL_THRESHOLD:
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
    if len(returns) < MIN_REQUIRED_DAYS:
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
        start = end - timedelta(days=FETCH_DAYS)
        df = yf.download(
            SPY_TICKER,
            start=str(start),
            end=str(end),
            progress=False,
            auto_adjust=True,
        )
        if df is None or len(df) < MIN_DF_LENGTH:
            return None
        closes = df["Close"].dropna()
        return closes.pct_change().dropna().values.astype(float)
    except Exception as exc:
        logger.warning("Regime monitor: SPY fetch failed", error=str(exc))
        return None


def _synthetic_spy_returns(n: int = SYNTH_RETURNS_DEFAULT_N) -> np.ndarray:
    """
    Generate deterministic synthetic SPY returns using a geometric Brownian
    motion model. Useful when live data cannot be fetched.

    Parameters
    ----------
    n: int, default SYNTH_RETURNS_DEFAULT_N
        Number of synthetic daily returns to produce.

    Returns
    -------
    np.ndarray
        Synthetic returns array.
    """
    seed = int(datetime.now(timezone.utc).strftime("%Y%m%d"))
    rng = np.random.default_rng(seed)
    return rng.normal(DAILY_MU, DAILY_SIGMA, n).astype(float)


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

    try:
        await redis_client.set(REDIS_REGIME_KEY, str(regime), ex=REDIS_TTL_SECONDS)
        logger.info("Regime updated", regime=regime, label=REGIME_LABELS[regime])
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