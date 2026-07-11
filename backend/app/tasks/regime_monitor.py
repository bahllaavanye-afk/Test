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

# -----------------------------
# Constants
# -----------------------------
MIN_DATA_LENGTH = 60
VOL_WINDOW = 20

HMM_COMPONENTS = 3
HMM_COV_TYPE = "diag"
HMM_ITER = 200
HMM_RANDOM_STATE = 42

VOL_RANK_BEAR_THRESHOLD = 1.5
RETURN_BEAR_THRESHOLD = -0.002
RETURN_BULL_THRESHOLD = 0.001
VOL_RANK_BULL_MAX = 0.9

FETCH_DAYS = 400
SYMBOL_SPY = "SPY"

SYNTHETIC_DAYS = 300
SYNTHETIC_DAILY_MU = 0.0003
SYNTHETIC_DAILY_SIGMA = 0.01

REDIS_KEY = "market:regime"
REDIS_TTL = 600  # seconds

LABELS = {0: "bear", 1: "sideways", 2: "bull"}

MONITOR_INTERVAL = 300  # seconds

# -----------------------------
# Optional dependency
# -----------------------------
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
    if n < MIN_DATA_LENGTH:
        return 1  # insufficient data → sideways

    vol_20 = pd.Series(returns).rolling(VOL_WINDOW).std().bfill().values
    features = np.column_stack([returns, vol_20])

    if _HMM_AVAILABLE:
        try:
            model = GaussianHMM(
                n_components=HMM_COMPONENTS,
                covariance_type=HMM_COV_TYPE,
                n_iter=HMM_ITER,
                random_state=HMM_RANDOM_STATE,
            )
            model.fit(features)
            states = model.predict(features)
            # Label states by mean return: highest → bull(2), lowest → bear(0)
            means = [features[states == s, 0].mean() for s in range(HMM_COMPONENTS)]
            order = np.argsort(means)  # indices sorted by mean return ascending
            label = {int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}
            return int(label[int(states[-1])])
        except Exception as exc:
            logger.warning("HMM fit failed, using heuristic", error=str(exc))

    # Heuristic: vol rank + recent momentum
    recent_vol = float(np.std(returns[-VOL_WINDOW:]))
    long_vol = float(np.std(returns[-252:]))
    vol_rank = recent_vol / max(long_vol, 1e-8)
    recent_return = float(np.mean(returns[-VOL_WINDOW:]))

    if vol_rank > VOL_RANK_BEAR_THRESHOLD or recent_return < RETURN_BEAR_THRESHOLD:
        return 0  # bear / crisis
    if recent_return > RETURN_BULL_THRESHOLD and vol_rank < VOL_RANK_BULL_MAX:
        return 2  # bull
    return 1  # sideways


def _fetch_spy_returns_sync() -> np.ndarray | None:
    """Sync yfinance fetch — must be called via run_in_executor."""
    try:
        import yfinance as yf  # type: ignore
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=FETCH_DAYS)
        df = yf.download(
            SYMBOL_SPY,
            start=str(start),
            end=str(end),
            progress=False,
            auto_adjust=True,
        )
        if df is None or len(df) < MIN_DATA_LENGTH:
            return None
        closes = df["Close"].dropna()
        return closes.pct_change().dropna().values.astype(float)
    except Exception as exc:
        logger.warning("Regime monitor: SPY fetch failed", error=str(exc))
        return None


def _synthetic_spy_returns(n: int = SYNTHETIC_DAYS) -> np.ndarray:
    """
    GBM synthetic SPY returns when yfinance is unreachable (network policy,
    offline dev container). Keeps the regime monitor functional 24/7.
    Deterministic per-day seed so the regime is stable within a session.
    """
    seed = int(datetime.now(timezone.utc).strftime("%Y%m%d"))
    rng = np.random.default_rng(seed)
    return rng.normal(SYNTHETIC_DAILY_MU, SYNTHETIC_DAILY_SIGMA, n).astype(float)


async def _fetch_spy_returns() -> np.ndarray | None:
    """Fetch 1 year of SPY daily returns without blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _fetch_spy_returns_sync)


async def run_once(redis_client) -> int | None:
    """Fit regime, write to Redis, return regime int or None on failure."""
    returns = await _fetch_spy_returns()
    if returns is None:
        # Network blocked / offline — fall back to synthetic returns so the
        # regime signal stays live instead of going stale at 'unknown'.
        logger.info("Regime monitor: using synthetic SPY returns (live data unavailable)")
        returns = _synthetic_spy_returns()

    regime = _fit_regime(returns)

    try:
        await redis_client.set(REDIS_KEY, str(regime), ex=REDIS_TTL)
        logger.info("Regime updated", regime=regime, label=LABELS[regime])
    except Exception as exc:
        logger.warning("Regime monitor: Redis write failed", error=str(exc))
        return None

    return regime


class RegimeMonitor:
    """Background asyncio task — call start() in app lifespan."""

    INTERVAL_SECONDS = MONITOR_INTERVAL  # 5 minutes

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