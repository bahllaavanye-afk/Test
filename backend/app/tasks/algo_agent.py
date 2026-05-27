"""
Always-running Algorithm Agent: continuously discovers, tests, and improves strategies.
Uses Upper Confidence Bound (UCB1) for exploration vs exploitation.
Runs as a background asyncio task alongside the strategy runner.
"""
from __future__ import annotations
import asyncio
import math
import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field

from app.utils.logging import logger

EXPERIMENTS_DIR = Path(__file__).parents[3] / "experiments" / "results"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AlgoCandidate:
    """Tracks a strategy's UCB1 stats for exploration/exploitation."""
    name: str
    symbol: str
    strategy_type: str           # 'manual' | 'ml_enhanced'
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

    def _select_candidate(self) -> AlgoCandidate:
        """UCB1 selection — always picks unexplored first, then highest UCB score."""
        scores = {k: c.ucb_score(self._total_runs) for k, c in self._candidates.items()}
        best_key = max(scores, key=lambda k: scores[k])
        return self._candidates[best_key]

    async def _run_quick_backtest(self, candidate: AlgoCandidate) -> float:
        """
        Runs a quick 2-year backtest using Alpaca historical bars.
        Returns Sharpe ratio or 0.0 on failure.
        """
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
                return 0.0

            raw_bars = resp.json().get("bars", [])
            if not raw_bars or len(raw_bars) < 60:
                return 0.0

            dates = pd.to_datetime([b["t"] for b in raw_bars], utc=True)
            closes = [float(b["c"]) for b in raw_bars]
            opens  = [float(b["o"]) for b in raw_bars]
            highs  = [float(b["h"]) for b in raw_bars]
            lows   = [float(b["l"]) for b in raw_bars]
            vols   = [float(b["v"]) for b in raw_bars]

            hist = pd.DataFrame(
                {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
                index=dates,
            )

            if hist is None or len(hist) < 60:
                return 0.0

            close = hist["Close"]

            strategy_cls = STRATEGY_REGISTRY.get(candidate.name)
            if not strategy_cls:
                return 0.0

            strategy = strategy_cls()
            signals = strategy.backtest_signals(hist)
            if signals is None or len(signals) < 30:
                return 0.0

            if hasattr(signals, "values"):
                sig_series = signals
            else:
                sig_series = pd.Series(signals, index=hist.index)

            metrics = run_backtest(sig_series, close)
            return float(metrics.sharpe)

        except Exception as e:
            logger.debug("Quick backtest failed", candidate=candidate.name, error=str(e))
            return 0.0

    def _save_result(self, candidate: AlgoCandidate, sharpe: float) -> None:
        result = {
            "id": str(uuid.uuid4()),
            "strategy": candidate.name,
            "symbol": candidate.symbol,
            "strategy_type": candidate.strategy_type,
            "sharpe": round(sharpe, 4),
            "avg_sharpe": round(candidate.avg_sharpe, 4),
            "n_runs": candidate.n_runs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._results.append(result)

        # Persist to disk
        results_file = EXPERIMENTS_DIR / "algo_agent_results.json"
        try:
            if results_file.exists():
                existing = json.loads(results_file.read_text())
            else:
                existing = []
            existing.append(result)
            existing = existing[-500:]  # keep last 500
            results_file.write_text(json.dumps(existing, indent=2))
        except Exception:
            pass

    async def run(self) -> None:
        """Main loop — runs forever, selecting and testing candidates via UCB1."""
        self._running = True
        logger.info("AlgoAgent started", candidates=len(self._candidates), interval=self.interval_seconds)

        while self._running:
            try:
                candidate = self._select_candidate()
                logger.info("AlgoAgent testing", strategy=candidate.name, symbol=candidate.symbol,
                            ucb=round(candidate.ucb_score(self._total_runs), 3))

                sharpe = await self._run_quick_backtest(candidate)
                candidate.n_runs += 1
                candidate.total_sharpe += sharpe
                candidate.best_sharpe = max(candidate.best_sharpe, sharpe)
                candidate.last_run_at = datetime.now(timezone.utc)
                self._total_runs += 1

                self._save_result(candidate, sharpe)

                logger.info("AlgoAgent result", strategy=candidate.name, symbol=candidate.symbol,
                            sharpe=round(sharpe, 3), avg=round(candidate.avg_sharpe, 3),
                            n_runs=candidate.n_runs)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("AlgoAgent error", error=str(e))

            await asyncio.sleep(self.interval_seconds)

    async def stop(self) -> None:
        self._running = False

    def get_leaderboard(self) -> list[dict]:
        """Return candidates sorted by average Sharpe descending."""
        def _safe(v: float) -> float:
            import math
            if math.isinf(v) or math.isnan(v):
                return 9999.0 if v > 0 else -9999.0
            return round(v, 3)

        return sorted(
            [{"key": k, "strategy": c.name, "symbol": c.symbol, "type": c.strategy_type,
              "avg_sharpe": round(c.avg_sharpe, 3), "best_sharpe": round(c.best_sharpe, 3),
              "n_runs": c.n_runs, "ucb": _safe(c.ucb_score(self._total_runs))}
             for k, c in self._candidates.items()],
            key=lambda x: x["avg_sharpe"],
            reverse=True,
        )
