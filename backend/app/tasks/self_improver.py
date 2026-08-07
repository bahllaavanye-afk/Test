"""
Self‑improvement autoloop for QuantEdge.

The loop continuously searches for better parameter configurations for a set of
target strategies. For each target (strategy, symbol) it:

1. Samples a small number of random parameter sets.
2. Evaluates each set via a quick back‑test.
3. Promotes the best configuration if it improves the Sharpe ratio by a
   configurable factor and exceeds a minimum Sharpe threshold.
4. Persists promotion events to ``experiments/results/self_improver.json``.
5. Sleeps for a configurable interval before repeating.

The implementation is deliberately lightweight: it avoids any external paid
APIs, keeps all I/O local, and logs extensively for observability.
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.utils.logging import logger

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DEFAULT_INTERVAL_SECONDS: int = 900
CONFIGS_PER_ITERATION: int = 5
IMPROVEMENT_FACTOR: float = 1.10
MIN_SHARPE_THRESHOLD: float = 0.5
MAX_HISTORY_LENGTH: int = 300
BACKTEST_DAYS: int = 730
MIN_HISTORY_LENGTH: int = 60
MIN_SIGNALS_LENGTH: int = 30

RESULTS_FILE = Path(__file__).parents[3] / "experiments" / "results" / "self_improver.json"
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Parameter search spaces per strategy
PARAM_SPACES: Dict[str, Dict[str, List[Any]]] = {
    "momentum": {
        "lookback_months": [3, 6, 9, 12],
        "min_score": [0.1, 0.2, 0.3, 0.5],
    },
    "mean_reversion": {
        "bb_period": [10, 20, 30],
        "bb_std": [1.5, 2.0, 2.5],
        "rsi_oversold": [20, 25, 30],
    },
    "rsi_macd": {
        "rsi_period": [9, 14, 21],
        "rsi_oversold": [25, 30, 35],
    },
    "breakout": {
        "high_period": [50, 100, 252],
        "volume_mult": [1.2, 1.5, 2.0],
    },
    "supertrend": {
        "atr_period": [10, 14, 20],
        "multiplier": [2.0, 3.0, 4.0],
    },
}

# Target strategies and symbols
TARGETS: List[tuple[str, str]] = [
    ("momentum", "SPY"),
    ("momentum", "QQQ"),
    ("mean_reversion", "AAPL"),
    ("rsi_macd", "MSFT"),
    ("breakout", "NVDA"),
    ("supertrend", "SPY"),
]


class SelfImprover:
    """Continuously search for better strategy parameters.

    Args:
        algo_agent: Optional external agent used to retrieve leaderboard data.
            The current implementation does not depend on it directly.
        interval_seconds: Sleep interval between iterations (default
            ``DEFAULT_INTERVAL_SECONDS``).
    """

    def __init__(self, algo_agent: Any = None, interval_seconds: int = DEFAULT_INTERVAL_SECONDS) -> None:
        self.algo_agent = algo_agent
        self.interval_seconds = interval_seconds
        self._best_params: Dict[str, Dict[str, Any]] = {}  # strategy:symbol → best params
        self._best_sharpe: Dict[str, float] = {}          # strategy:symbol → best Sharpe
        self._running: bool = False
        self._iteration: int = 0

    def _sample_params(self, strategy: str) -> Dict[str, Any]:
        """Return a random parameter configuration for *strategy*.

        The configuration is sampled uniformly from the predefined ``PARAM_SPACES``
        for the given strategy. If the strategy is unknown an empty dictionary is
        returned.
        """
        space = PARAM_SPACES.get(strategy, {})
        return {k: random.choice(v) for k, v in space.items()}

    async def _evaluate(self, strategy: str, symbol: str, params: Dict[str, Any]) -> float:
        """Execute a quick back‑test for *strategy* on *symbol* with *params*.

        The function downloads recent daily price data via ``yfinance``, constructs
        the strategy instance, generates signals, and runs the back‑test engine.
        It returns the Sharpe ratio of the resulting equity curve, or ``0.0`` on
        any failure or if the data is insufficient.
        """
        try:
            import pandas as pd
            import yfinance as yf
            from app.backtest.engine import run_backtest
            from app.strategies import STRATEGY_REGISTRY

            end = datetime.now(timezone.utc)
            start = end - timedelta(days=BACKTEST_DAYS)
            loop = asyncio.get_running_loop()
            hist = await loop.run_in_executor(
                None,
                lambda: yf.download(
                    symbol,
                    start=str(start.date()),
                    end=str(end.date()),
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                ),
            )
            if hist is None or len(hist) < MIN_HISTORY_LENGTH:
                return 0.0

            close = hist["Close"].squeeze() if hasattr(hist["Close"], "squeeze") else hist["Close"]

            cls = STRATEGY_REGISTRY.get(strategy)
            if not cls:
                return 0.0

            try:
                strat = cls(**params)
            except TypeError:
                strat = cls()  # ignore params if constructor doesn't accept them

            signals = strat.backtest_signals(hist)
            if signals is None or (hasattr(signals, "__len__") and len(signals) < MIN_SIGNALS_LENGTH):
                return 0.0

            sig_series = signals if hasattr(signals, "values") else pd.Series(signals, index=hist.index)
            metrics = run_backtest(sig_series, close)
            return float(metrics.sharpe)
        except Exception as e:
            logger.debug("Self-improver eval failed", strategy=strategy, error=str(e))
            return 0.0

    async def _improve_strategy(self, strategy: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Search for a better configuration for *strategy* on *symbol*.

        The method samples ``CONFIGS_PER_ITERATION`` random configurations,
        evaluates each, and promotes the best one if it improves the stored Sharpe
        ratio by at least ``IMPROVEMENT_FACTOR`` and exceeds ``MIN_SHARPE_THRESHOLD``.
        The promotion details are persisted to disk and also returned.
        """
        space = PARAM_SPACES.get(strategy)
        if not space:
            return None

        current_best = self._best_sharpe.get(f"{strategy}:{symbol}", 0.0)
        best_iter_sharpe = current_best
        best_iter_params: Optional[Dict[str, Any]] = None

        for _ in range(CONFIGS_PER_ITERATION):
            params = self._sample_params(strategy)
            sharpe = await self._evaluate(strategy, symbol, params)
            if sharpe > best_iter_sharpe:
                best_iter_sharpe = sharpe
                best_iter_params = params

        if (
            best_iter_params
            and best_iter_sharpe > current_best * IMPROVEMENT_FACTOR
            and best_iter_sharpe > MIN_SHARPE_THRESHOLD
        ):
            key = f"{strategy}:{symbol}"
            self._best_params[key] = best_iter_params
            self._best_sharpe[key] = best_iter_sharpe
            promotion = {
                "id": str(uuid.uuid4()),
                "strategy": strategy,
                "symbol": symbol,
                "params": best_iter_params,
                "new_sharpe": round(best_iter_sharpe, 4),
                "previous_sharpe": round(current_best, 4),
                "improvement_pct": round(
                    (best_iter_sharpe - current_best) / max(abs(current_best), 0.1), 4
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._persist(promotion)
            logger.info("Self-improver PROMOTED params", **promotion)
            return promotion
        return None

    def _persist(self, entry: Dict[str, Any]) -> None:
        """Append *entry* to the JSON results file, keeping the history bounded."""
        try:
            history = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else []
            history.append(entry)
            history = history[-MAX_HISTORY_LENGTH:]
            RESULTS_FILE.write_text(json.dumps(history, indent=2))
        except Exception as exc:
            logger.debug("self_improver persist failed", error=str(exc))

    def get_best_params(self, strategy: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve the best known parameters for *strategy* on *symbol*."""
        return self._best_params.get(f"{strategy}:{symbol}")

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the full promotion history stored on disk."""
        if not RESULTS_FILE.exists():
            return []
        try:
            return json.loads(RESULTS_FILE.read_text())
        except Exception:
            return []

    async def run(self) -> None:
        """Main loop that iterates over ``TARGETS`` until stopped."""
        self._running = True
        logger.info("SelfImprover started", interval=self.interval_seconds)

        while self._running:
            self._iteration += 1
            logger.info("SelfImprover iteration", n=self._iteration)
            for strategy, symbol in TARGETS:
                try:
                    await self._improve_strategy(strategy, symbol)
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    logger.warning(
                        "Self-improver target failed",
                        strategy=strategy,
                        symbol=symbol,
                        error=str(e),
                    )
            await asyncio.sleep(self.interval_seconds)

    async def stop(self) -> None:
        """Signal the main loop to exit after the current iteration."""
        self._running = False