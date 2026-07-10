"""
Always-running Algorithm Agent: continuously discovers, tests, and improves strategies.
Uses Upper Confidence Bound (UCB1) for exploration vs exploitation.
Runs as a background asyncio task alongside the strategy runner.
"""
from __future__ import annotations

import asyncio
import json
import math
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass

from app.utils.logging import logger


EXPERIMENTS_DIR = Path(__file__).parents[3] / "experiments" / "results"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AlgoCandidate:
    """Tracks a strategy's UCB1 stats for exploration/exploitation."""
    name: str
    symbol: str
    strategy_type: str  # 'manual' | 'ml_enhanced'
    n_runs: int = 0
    total_sharpe: float = 0.0
    best_sharpe: float = 0.0
    last_run_at: datetime | None = None

    @property
    def avg_sharpe(self) -> float:
        return self.total_sharpe / self.n_runs if self.n_runs > 0 else 0.0

    def ucb_score(self, total_runs: int, c: float = 1.414) -> float:
        """UCB1 formula: avg_reward + c * sqrt(ln(total_runs) / n_runs)"""
        if self.n_runs == 0:
            return float("inf")  # always try unexplored candidates first
        exploitation = self.avg_sharpe
        # Guard against log(0) – total_runs is at least 0, add 1 to keep >0
        exploration = c * math.sqrt(math.log(total_runs + 1) / self.n_runs)
        return exploitation + exploration


class AlgoAgent:
    """
    Continuously runs experiments and improves strategies.
    Decision making via UCB1 (Upper Confidence Bound).

    Loop:
    1. Score all candidates via UCB1
    2. Pick highest-scoring candidate (unexplored first, then best exploitation+exploration)
    3. Run a quick backtest on it
    4. Update stats
    5. If ML: also retrain with new hyperparams found via Optuna
    6. Sleep and repeat
    """

    STRATEGY_CANDIDATES = [
        ("momentum", "SPY", "manual"),
        ("momentum", "QQQ", "manual"),
        ("mean_reversion", "AAPL", "manual"),
        ("mean_reversion", "MSFT", "manual"),
        ("rsi_macd", "SPY", "manual"),
        ("breakout", "NVDA", "manual"),
        ("supertrend", "SPY", "manual"),
        ("low_volatility", "SPY", "manual"),
        ("ml_momentum", "SPY", "ml_enhanced"),
        ("ml_momentum", "QQQ", "ml_enhanced"),
        ("ml_mean_reversion", "AAPL", "ml_enhanced"),
        ("ml_breakout", "SPY", "ml_enhanced"),
        ("ensemble", "SPY", "ml_enhanced"),
        ("gamma_exposure", "SPY", "manual"),
        ("kalman_pairs", "XOM", "manual"),
        ("vrp_systematic", "SPY", "manual"),
        ("hmm_regime", "SPY", "manual"),
        ("opening_range_breakout", "SPY", "manual"),
        ("dispersion_trading", "QQQ", "manual"),
        ("pead_sue", "AAPL", "manual"),
        ("skew_arb", "SPY", "manual"),
        ("triple_barrier_momentum", "NVDA", "manual"),
        ("residual_momentum", "AAPL", "manual"),
        ("idio_vol_anomaly", "AAPL", "manual"),
        ("fifty_two_week_high", "MSFT", "manual"),
        ("open_close_revert", "SPY", "manual"),
    ]

    def __init__(self, broker=None, interval_seconds: int = 300):
        self.broker = broker
        self.interval_seconds = interval_seconds
        self._candidates: dict[str, AlgoCandidate] = {}
        self._total_runs = 0
        self._running = False
        self._results: list[dict] = []

        for name, symbol, stype in self.STRATEGY_CANDIDATES:
            key = f"{name}:{symbol}"
            self._candidates[key] = AlgoCandidate(name=name, symbol=symbol, strategy_type=stype)

    def _select_candidate(self) -> AlgoCandidate | None:
        """UCB1 selection — always picks unexplored first, then highest UCB score."""
        if not self._candidates:
            logger.warning("AlgoAgent: No candidates available for selection")
            return None

        scores = {k: c.ucb_score(self._total_runs) for k, c in self._candidates.items()}
        if not scores:
            logger.warning("AlgoAgent: Score computation resulted in empty dict")
            return None

        best_key = max(scores, key=lambda k: scores[k])
        return self._candidates.get(best_key)

    async def _run_quick_backtest(self, candidate: AlgoCandidate | None) -> float:
        """
        Runs a quick 2-year backtest using Alpaca historical bars.
        Returns Sharpe ratio or 0.0 on failure.
        """
        if candidate is None:
            logger.debug("Quick backtest called with None candidate")
            return 0.0
        if not candidate.symbol:
            logger.debug("Quick backtest called with candidate missing symbol", candidate=candidate.name)
            return 0.0

        try:
            import pandas as pd
            import httpx
            from app.config import settings
            from app.backtest.engine import run_backtest
            from app.strategies import STRATEGY_REGISTRY

            end = datetime.now(timezone.utc)
            start = end - timedelta(days=730)
            start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")

            headers = {
                "APCA-API-KEY-ID": settings.alpaca_api_key,
                "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
            }

            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"https://data.alpaca.markets/v2/stocks/{candidate.symbol.upper()}/bars",
                    params={"timeframe": "1Day", "start": start_str, "limit": 1000},
                    headers=headers,
                )

            if resp.status_code != 200:
                logger.debug("Alpaca response error", status=resp.status_code)
                return 0.0

            payload = resp.json()
            if not isinstance(payload, dict):
                logger.debug("Unexpected Alpaca payload type", payload_type=type(payload))
                return 0.0

            raw_bars = payload.get("bars", [])
            if not isinstance(raw_bars, list) or len(raw_bars) < 60:
                logger.debug("Insufficient bar data", count=len(raw_bars) if isinstance(raw_bars, list) else "N/A")
                return 0.0

            # Extract fields safely
            dates = pd.to_datetime([b.get("t") for b in raw_bars if b.get("t")], utc=True)
            if dates.empty:
                return 0.0

            def safe_float(value):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0

            closes = [safe_float(b.get("c")) for b in raw_bars]
            opens = [safe_float(b.get("o")) for b in raw_bars]
            highs = [safe_float(b.get("h")) for b in raw_bars]
            lows = [safe_float(b.get("l")) for b in raw_bars]
            vols = [safe_float(b.get("v")) for b in raw_bars]

            hist = pd.DataFrame(
                {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
                index=dates,
            )

            if hist.empty or len(hist) < 60:
                return 0.0

            close = hist["Close"]

            strategy_cls = STRATEGY_REGISTRY.get(candidate.name)
            if not strategy_cls:
                logger.debug("Strategy not found in registry", name=candidate.name)
                return 0.0

            strategy = strategy_cls()
            signals = strategy.backtest_signals(hist)
            if signals is None:
                logger.debug("Strategy returned None signals", name=candidate.name)
                return 0.0

            # Ensure we have a Series for backtest
            if hasattr(signals, "values"):
                sig_series = signals
            else:
                sig_series = pd.Series(signals, index=hist.index)

            if len(sig_series) < 30:
                logger.debug("Insufficient signal length", length=len(sig_series))
                return 0.0

            metrics = run_backtest(sig_series, close)
            return float(metrics.sharpe)

        except Exception as e:
            logger.debug("Quick backtest failed", candidate=candidate.name if candidate else "None", error=str(e))
            return 0.0

    def _save_result(self, candidate: AlgoCandidate | None, sharpe: float | None) -> None:
        if candidate is None:
            logger.warning("Attempted to save result for None candidate")
            return

        sharpe_val = sharpe if sharpe is not None else 0.0
        result = {
            "id": str(uuid.uuid4()),
            "strategy": candidate.name,
            "symbol": candidate.symbol,
            "strategy_type": candidate.strategy_type,
            "sharpe": round(sharpe_val, 4),
            "avg_sharpe": round(candidate.avg_sharpe, 4),
            "n_runs": candidate.n_runs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._results.append(result)

        results_file = EXPERIMENTS_DIR / "algo_agent_results.json"
        try:
            if results_file.exists():
                existing = json.loads(results_file.read_text())
                if not isinstance(existing, list):
                    existing = []
            else:
                existing = []
            existing.append(result)
            # Keep only the most recent 500 entries
            existing = existing[-500:]
            results_file.write_text(json.dumps(existing, indent=2))
        except Exception as e:
            logger.warning("AlgoAgent: failed to persist result", error=str(e))

    async def run(self) -> None:
        """Main loop — runs forever, selecting and testing candidates via UCB1."""
        self._running = True
        logger.info(
            "AlgoAgent started",
            candidates=len(self._candidates),
            interval=self.interval_seconds,
        )

        while self._running:
            try:
                candidate = self._select_candidate()
                if candidate is None:
                    logger.warning("AlgoAgent: No candidate selected, terminating loop")
                    break

                logger.info("AlgoAgent testing", strategy=candidate.name, symbol=candidate.symbol)

                sharpe = await self._run_quick_backtest(candidate)

                # Update candidate statistics safely
                candidate.n_runs += 1
                sharpe_val = sharpe if sharpe is not None else 0.0
                candidate.total_sharpe += sharpe_val
                if sharpe_val > candidate.best_sharpe:
                    candidate.best_sharpe = sharpe_val
                candidate.last_run_at = datetime.now(timezone.utc)

                self._total_runs += 1
                self._save_result(candidate, sharpe_val)

                # Placeholder for ML-specific retraining logic
                if candidate.strategy_type == "ml_enhanced":
                    # In a real implementation we would trigger Optuna hyper‑parameter search here.
                    pass

                await asyncio.sleep(self.interval_seconds)

            except Exception as e:
                logger.error("AlgoAgent loop error", error=str(e))
                # Prevent tight crash loops
                await asyncio.sleep(self.interval_seconds)

    def stop(self) -> None:
        """Gracefully stop the background loop."""
        self._running = False
        logger.info("AlgoAgent stopping")