"""
Self-improvement autoloop. Runs forever, looking for ways to improve the platform:
  1. Take the top-3 strategies from AlgoAgent leaderboard
  2. Sweep their parameters (Optuna-style) — run 5 random configs each
  3. If a config beats the current best Sharpe by > 10%, promote it
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


class SelfImprover:
    def __init__(self, algo_agent=None, interval_seconds: int = 900):
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
            start = end - timedelta(days=730)
            loop = asyncio.get_event_loop()
            hist = await loop.run_in_executor(
                None,
                lambda: yf.download(symbol, start=str(start.date()), end=str(end.date()),
                                    interval="1d", auto_adjust=True, progress=False)
            )
            if hist is None or len(hist) < 60:
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
            if signals is None or (hasattr(signals, "__len__") and len(signals) < 30):
                return 0.0

            sig_series = signals if hasattr(signals, "values") else pd.Series(signals, index=hist.index)
            metrics = run_backtest(sig_series, close)
            return float(metrics.sharpe)
        except Exception as e:
            logger.debug("Self-improver eval failed", strategy=strategy, error=str(e))
            return 0.0

    async def _improve_strategy(self, strategy: str, symbol: str) -> dict | None:
        """Sweep params for one strategy. Returns promoted result or None."""
        space = PARAM_SPACES.get(strategy)
        if not space:
            return None

        current_best = self._best_sharpe.get(f"{strategy}:{symbol}", 0.0)
        best_iter_sharpe = current_best
        best_iter_params = None

        # 5 random configs per iteration
        for _ in range(5):
            params = self._sample_params(strategy)
            sharpe = await self._evaluate(strategy, symbol, params)
            if sharpe > best_iter_sharpe:
                best_iter_sharpe = sharpe
                best_iter_params = params

        # Promote if improvement > 10%
        if best_iter_params and best_iter_sharpe > current_best * 1.10 and best_iter_sharpe > 0.5:
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
                "improvement_pct": round((best_iter_sharpe - current_best) / max(abs(current_best), 0.1), 4),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._persist(promotion)
            logger.info("Self-improver PROMOTED params", **promotion)
            return promotion
        return None

    def _persist(self, entry: dict) -> None:
        try:
            history = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else []
            history.append(entry)
            history = history[-300:]
            RESULTS_FILE.write_text(json.dumps(history, indent=2))
        except Exception:
            pass

    def get_best_params(self, strategy: str, symbol: str) -> dict | None:
        return self._best_params.get(f"{strategy}:{symbol}")

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

        # Symbol coverage
        TARGETS = [("momentum", "SPY"), ("momentum", "QQQ"), ("mean_reversion", "AAPL"),
                   ("rsi_macd", "MSFT"), ("breakout", "NVDA"), ("supertrend", "SPY")]

        while self._running:
            self._iteration += 1
            logger.info("SelfImprover iteration", n=self._iteration)
            for strategy, symbol in TARGETS:
                try:
                    await self._improve_strategy(strategy, symbol)
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    logger.warning("Self-improver target failed", strategy=strategy, symbol=symbol, error=str(e))
            await asyncio.sleep(self.interval_seconds)

    async def stop(self) -> None:
        self._running = False
