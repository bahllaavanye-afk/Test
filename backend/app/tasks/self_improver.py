"""
Self-improvement autoloop. Runs forever, looking for ways to improve the platform:
  1. Take the top-3 strategies from AlgoAgent leaderboard
  2. Sweep their parameters (Optuna-style) — run a set number of random configs each
  3. If a config beats the current best Sharpe by > IMPROVEMENT_FACTOR, promote it
  4. Log everything to experiments/results/self_improver.json
  5. Sleep, then repeat
"""
from __future__ import annotations

import asyncio
import json
import random
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

from app.utils.logging import logger

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DEFAULT_INTERVAL_SECONDS = 900
CONFIGS_PER_ITERATION = 5
IMPROVEMENT_FACTOR = 1.10
MIN_SHARPE_THRESHOLD = 0.5
MAX_HISTORY_LENGTH = 300
BACKTEST_DAYS = 730
MIN_HISTORY_LENGTH = 60
MIN_SIGNALS_LENGTH = 30

RESULTS_FILE = Path(__file__).parents[3] / "experiments" / "results" / "self_improver.json"
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Parameter search spaces per strategy
PARAM_SPACES = {
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
TARGETS = [
    ("momentum", "SPY"),
    ("momentum", "QQQ"),
    ("mean_reversion", "AAPL"),
    ("rsi_macd", "MSFT"),
    ("breakout", "NVDA"),
    ("supertrend", "SPY"),
]


class SelfImprover:
    def __init__(self, algo_agent=None, interval_seconds: int = DEFAULT_INTERVAL_SECONDS):
        self.algo_agent = algo_agent
        self.interval_seconds = interval_seconds
        self._best_params: dict[str, dict] = {}    # strategy → best params dict
        self._best_sharpe: dict[str, float] = {}   # strategy → best Sharpe
        self._running = False
        self._iteration = 0

    def _sample_params(self, strategy: str) -> dict:
        """Random sample from PARAM_SPACES."""
        space = PARAM_SPACES.get(strategy, {})
        return {k: random.choice(v) for k, v in space.items()}

    async def _evaluate(self, strategy: str, symbol: str, params: dict) -> float:
        """Run a quick backtest with the given params. Returns Sharpe."""
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

    # ------------------------------------------------------------------
    # Helper methods for _improve_strategy
    # ------------------------------------------------------------------
    def _get_param_space(self, strategy: str) -> dict | None:
        return PARAM_SPACES.get(strategy)

    def _key(self, strategy: str, symbol: str) -> str:
        return f"{strategy}:{symbol}"

    async def _search_best_params(self, strategy: str, symbol: str, current_best: float) -> tuple[float, dict | None]:
        """Randomly sample CONFIGS_PER_ITERATION configs and return the best Sharpe with its params."""
        best_sharpe = current_best
        best_params = None
        for _ in range(CONFIGS_PER_ITERATION):
            params = self._sample_params(strategy)
            sharpe = await self._evaluate(strategy, symbol, params)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
        return best_sharpe, best_params

    def _should_promote(self, candidate_sharpe: float, current_best: float) -> bool:
        """Determine if the candidate Sharpe justifies promotion."""
        return (
            candidate_sharpe > current_best * IMPROVEMENT_FACTOR
            and candidate_sharpe > MIN_SHARPE_THRESHOLD
        )

    def _store_promotion(self, key: str, params: dict, sharpe: float) -> None:
        """Persist the new best parameters and Sharpe in memory."""
        self._best_params[key] = params
        self._best_sharpe[key] = sharpe

    def _build_promotion_entry(
        self,
        strategy: str,
        symbol: str,
        params: dict,
        new_sharpe: float,
        previous_sharpe: float,
    ) -> dict:
        """Create a dict describing the promotion event."""
        improvement_pct = round(
            (new_sharpe - previous_sharpe) / max(abs(previous_sharpe), 0.1), 4
        )
        return {
            "id": str(uuid.uuid4()),
            "strategy": strategy,
            "symbol": symbol,
            "params": params,
            "new_sharpe": round(new_sharpe, 4),
            "previous_sharpe": round(previous_sharpe, 4),
            "improvement_pct": improvement_pct,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def _improve_strategy(self, strategy: str, symbol: str) -> dict | None:
        """Sweep params for one strategy. Returns promoted result or None."""
        if not self._get_param_space(strategy):
            return None

        key = self._key(strategy, symbol)
        current_best = self._best_sharpe.get(key, 0.0)

        best_sharpe, best_params = await self._search_best_params(strategy, symbol, current_best)

        if best_params is None or not self._should_promote(best_sharpe, current_best):
            return None

        self._store_promotion(key, best_params, best_sharpe)

        promotion = self._build_promotion_entry(
            strategy=strategy,
            symbol=symbol,
            params=best_params,
            new_sharpe=best_sharpe,
            previous_sharpe=current_best,
        )
        self._persist(promotion)
        logger.info("Self-improver PROMOTED params", **promotion)
        return promotion

    def _persist(self, entry: dict) -> None:
        try:
            history = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else []
            history.append(entry)
            history = history[-MAX_HISTORY_LENGTH:]
            RESULTS_FILE.write_text(json.dumps(history, indent=2))
        except Exception as exc:
            logger.debug("self_improver persist failed", error=str(exc))

    def get_best_params(self, strategy: str, symbol: str) -> dict | None:
        return self._best_params.get(self._key(strategy, symbol))

    def get_history(self) -> list[dict]:
        if not RESULTS_FILE.exists():
            return []
        try:
            return json.loads(RESULTS_FILE.read_text())
        except Exception:
            return []

    async def run(self) -> None:
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
        self._running = False