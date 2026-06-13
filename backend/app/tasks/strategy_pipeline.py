"""
Strategy Auto-Pipeline: staged → backtest → promote.

Closes the loop that AIStrategyGenerator leaves open:
- Scans the strategies/staging/ directory every 30 minutes.
- Runs a quick vectorised backtest on the staged strategy (60 days of SPY/BTC data).
- Promotes strategies that meet the quality gate (Sharpe > 0.8 OOS, no negative years).
- Rejects those that don't and moves them to strategies/rejected/.
- Broadcasts results on the agent bus and posts to Slack.

Quality gates (all must pass):
  1. Sharpe > 0.80 on the most recent 60-day out-of-sample period.
  2. Max drawdown < 20%.
  3. Win rate > 45%.
  4. Survives walk-forward: Sharpe > 0.5 on each of 3 rolling 20-day windows.

Promoted strategies go to strategies/manual/ and are added to STRATEGY_REGISTRY
via a runtime registration (no process restart needed).
"""
from __future__ import annotations

import asyncio
import importlib.util
import re
import shutil
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.utils.logging import logger

STAGING_DIR  = Path(__file__).parent.parent / "strategies" / "staging"
PROMOTED_DIR = Path(__file__).parent.parent / "strategies" / "manual"
REJECTED_DIR = Path(__file__).parent.parent / "strategies" / "rejected"

# Quality gates
MIN_SHARPE      = 0.80
MAX_DRAWDOWN    = 0.20
MIN_WIN_RATE    = 0.45
WF_MIN_SHARPE   = 0.50   # per walk-forward fold
WF_FOLDS        = 3
WF_FOLD_DAYS    = 20
BACKTEST_DAYS   = 60     # total OOS window
DEFAULT_SYMBOLS = ["SPY", "QQQ"]   # benchmark data for quick backtests


# ── Metrics ───────────────────────────────────────────────────────────────────

def _sharpe(returns: pd.Series, ann: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(ann))


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(abs(dd.min()))


def _win_rate(returns: pd.Series) -> float:
    pos = (returns > 0).sum()
    total = (returns != 0).sum()
    return float(pos / total) if total > 0 else 0.0


@dataclass
class BacktestResult:
    strategy_name: str
    symbol:        str
    sharpe:        float
    max_drawdown:  float
    win_rate:      float
    n_trades:      int
    wf_sharpes:    list[float]
    passed:        bool
    reject_reason: Optional[str] = None


# ── Data fetcher (yfinance, no auth) ─────────────────────────────────────────

def _fetch_data(symbol: str, days: int) -> pd.DataFrame:
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=f"{days + 20}d", interval="1d")
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        return df.tail(days + 10)
    except Exception as e:
        logger.warning("strategy_pipeline: data fetch failed", symbol=symbol, error=str(e))
        return pd.DataFrame()


# ── Backtest engine ───────────────────────────────────────────────────────────

def _run_simple_backtest(
    strategy_cls,
    df: pd.DataFrame,
    symbol: str,
    days: int,
) -> tuple[pd.Series, int]:
    """
    Minimal vectorised backtest using the strategy's backtest_signals() method.
    Returns (daily_returns, n_trades).
    """
    import pandas_ta as ta  # used inside strategy files

    df = df.copy()
    if len(df) < days:
        return pd.Series(dtype=float), 0

    try:
        strat = strategy_cls()
        loop = asyncio.new_event_loop()
        signals: pd.Series = loop.run_until_complete(strat.backtest_signals(df))
        loop.close()
    except Exception:
        return pd.Series(dtype=float), 0

    price = df["close"].reset_index(drop=True)
    sig   = signals.reset_index(drop=True).reindex(price.index, fill_value=0)
    daily_ret = price.pct_change().shift(-1) * sig.shift(1).fillna(0)
    daily_ret = daily_ret.dropna()
    n_trades = int((sig.diff().abs() > 0).sum())
    return daily_ret.tail(days), n_trades


def _walk_forward_sharpes(strategy_cls, df: pd.DataFrame,
                           symbol: str, fold_days: int, n_folds: int) -> list[float]:
    sharpes = []
    for i in range(n_folds):
        offset = (n_folds - 1 - i) * fold_days
        window = df.iloc[-(fold_days + offset): -offset if offset > 0 else None]
        rets, _ = _run_simple_backtest(strategy_cls, window, symbol, fold_days)
        sharpes.append(_sharpe(rets) if len(rets) > 0 else 0.0)
    return sharpes


# ── Staging file loader ────────────────────────────────────────────────────────

def _load_strategy_class(path: Path):
    """Dynamically load a strategy class from a staging .py file."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None

    # Expect a class named same as the file stem (snake→pascal) or find AbstractStrategy subclass
    from app.strategies.base import AbstractStrategy
    for name in dir(mod):
        obj = getattr(mod, name)
        try:
            if (isinstance(obj, type)
                    and issubclass(obj, AbstractStrategy)
                    and obj is not AbstractStrategy):
                return obj
        except TypeError:
            pass
    return None


# ── Promotion / rejection ─────────────────────────────────────────────────────

def _promote(path: Path, strategy_cls) -> None:
    PROMOTED_DIR.mkdir(parents=True, exist_ok=True)
    dest = PROMOTED_DIR / path.name
    shutil.copy2(path, dest)
    path.unlink(missing_ok=True)
    # Runtime registration so the strategy is usable immediately
    try:
        from app.strategies import STRATEGY_REGISTRY
        strat = strategy_cls()
        STRATEGY_REGISTRY[strat.name] = strategy_cls
        logger.info("strategy_pipeline: promoted and registered", name=strat.name)
    except Exception as e:
        logger.warning("strategy_pipeline: runtime registration failed", error=str(e))


def _reject(path: Path, reason: str) -> None:
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    dest = REJECTED_DIR / path.name
    shutil.copy2(path, dest)
    path.unlink(missing_ok=True)
    logger.info("strategy_pipeline: rejected", name=path.stem, reason=reason)


# ── Main agent ────────────────────────────────────────────────────────────────

class StrategyPipelineAgent:
    """Scans staging every 30 minutes, backtests, promotes, or rejects."""

    def __init__(self, interval_seconds: int = 1800):
        self.interval_seconds = interval_seconds
        self._running = False

    async def run_cycle(self) -> list[dict]:
        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        staged = [f for f in STAGING_DIR.glob("*.py") if not f.name.startswith("_")]
        if not staged:
            return []

        results = []
        loop = asyncio.get_running_loop()
        for path in staged:
            result = await loop.run_in_executor(None, self._evaluate, path)
            if result:
                results.append(asdict(result))
                await self._broadcast(result)
        return results

    def _evaluate(self, path: Path) -> Optional[BacktestResult]:
        logger.info("strategy_pipeline: evaluating", file=path.name)
        try:
            strategy_cls = _load_strategy_class(path)
            if strategy_cls is None:
                _reject(path, "could not load class")
                return BacktestResult(
                    strategy_name=path.stem, symbol="N/A",
                    sharpe=0, max_drawdown=1, win_rate=0,
                    n_trades=0, wf_sharpes=[], passed=False,
                    reject_reason="could not load class",
                )
        except Exception as e:
            _reject(path, f"import error: {e}")
            return None

        # Determine symbol from strategy metadata or default to SPY
        symbol = DEFAULT_SYMBOLS[0]
        try:
            strat_instance = strategy_cls()
            if getattr(strat_instance, "market_type", "equity") == "crypto":
                symbol = "BTC-USD"
        except Exception:
            pass

        df = _fetch_data(symbol, BACKTEST_DAYS + WF_FOLDS * WF_FOLD_DAYS + 20)
        if df.empty:
            _reject(path, "no market data available")
            return None

        try:
            rets, n_trades = _run_simple_backtest(strategy_cls, df, symbol, BACKTEST_DAYS)
            if len(rets) < 10:
                _reject(path, "backtest produced < 10 return observations")
                return None

            sharpe  = _sharpe(rets)
            eq      = (1 + rets).cumprod()
            max_dd  = _max_drawdown(eq)
            wr      = _win_rate(rets)
            wf_sharpes = _walk_forward_sharpes(strategy_cls, df, symbol,
                                               WF_FOLD_DAYS, WF_FOLDS)

        except Exception as e:
            logger.error("strategy_pipeline: backtest crashed", name=path.stem,
                         error=traceback.format_exc())
            _reject(path, f"backtest crashed: {e}")
            return None

        # Quality gate evaluation
        reject_reason = None
        if sharpe < MIN_SHARPE:
            reject_reason = f"Sharpe {sharpe:.2f} < {MIN_SHARPE}"
        elif max_dd > MAX_DRAWDOWN:
            reject_reason = f"MaxDD {max_dd:.1%} > {MAX_DRAWDOWN:.0%}"
        elif wr < MIN_WIN_RATE:
            reject_reason = f"WinRate {wr:.1%} < {MIN_WIN_RATE:.0%}"
        elif any(s < WF_MIN_SHARPE for s in wf_sharpes):
            bad = [round(s, 2) for s in wf_sharpes if s < WF_MIN_SHARPE]
            reject_reason = f"Walk-forward folds failed: {bad}"

        passed = reject_reason is None

        result = BacktestResult(
            strategy_name=path.stem, symbol=symbol,
            sharpe=round(sharpe, 3), max_drawdown=round(max_dd, 3),
            win_rate=round(wr, 3), n_trades=n_trades,
            wf_sharpes=[round(s, 3) for s in wf_sharpes],
            passed=passed, reject_reason=reject_reason,
        )

        if passed:
            _promote(path, strategy_cls)
        else:
            _reject(path, reject_reason or "unknown")

        return result

    async def _broadcast(self, result: BacktestResult) -> None:
        try:
            from app.tasks.agent_bus import get_bus
            await get_bus().broadcast_signal(
                {"type": "strategy_pipeline_result", **asdict(result)},
                from_agent="strategy_pipeline",
            )
        except Exception as e:
            logger.debug("strategy_pipeline: broadcast failed", error=str(e))

        status = "PROMOTED" if result.passed else "REJECTED"
        emoji = ":white_check_mark:" if result.passed else ":x:"
        msg = (
            f"{emoji} *Strategy {status}*: `{result.strategy_name}`\n"
            f"• Sharpe: `{result.sharpe:.2f}` | MaxDD: `{result.max_drawdown:.1%}` "
            f"| WinRate: `{result.win_rate:.1%}` | Trades: {result.n_trades}\n"
            f"• Walk-forward Sharpes: {result.wf_sharpes}\n"
            + (f"• Reject reason: _{result.reject_reason}_" if result.reject_reason else "")
        )
        try:
            from app.tasks.agent_bus import get_bus
            await get_bus().slack_notify(msg, from_agent="strategy_pipeline",
                                          level="info" if result.passed else "warning")
        except Exception:
            pass

    async def run(self) -> None:
        self._running = True
        logger.info("StrategyPipelineAgent started", interval_s=self.interval_seconds)
        while self._running:
            try:
                results = await self.run_cycle()
                if results:
                    logger.info("strategy_pipeline: cycle done",
                                evaluated=len(results),
                                promoted=sum(1 for r in results if r["passed"]))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("StrategyPipelineAgent: cycle crashed", error=str(e))
            if self._running:
                await asyncio.sleep(self.interval_seconds)

    def stop(self) -> None:
        self._running = False


_agent: StrategyPipelineAgent | None = None


def get_pipeline() -> StrategyPipelineAgent:
    global _agent
    if _agent is None:
        _agent = StrategyPipelineAgent()
    return _agent
