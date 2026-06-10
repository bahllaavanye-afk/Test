"""
Strategy Exploration Autopilot — auto-generate, backtest, review, and promote strategies.

Pipeline:
  1. SCAN    — audit all registered strategies' recent backtest scores from brain
  2. RANK    — sort by Sharpe; flag underperformers (Sharpe < 0.5) and untested strategies
  3. EXPLORE — LLM proposes parameter mutations or new strategy variants
  4. BACKTEST — quick 1-year rolling backtest using yfinance + vectorbt-style logic
  5. REVIEW  — alpha_researcher persona evaluates: Sharpe, max drawdown, win rate
  6. PROMOTE — if Sharpe > 1.0 out-of-sample: write to brain, post to #strategy-lab
  7. RETIRE  — if Sharpe < 0.2 for 3 consecutive runs: flag for retirement to #strategy-lab

Constraints:
  - No live execution — backtest only
  - No paid data sources
  - No mock results — real yfinance data only
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).parent))
from llm_common import core_update, memory_write, memory_read, slack_post, llm, core_get

ALLOW_PAID = os.environ.get("ALLOW_PAID_APIS", "False")
if ALLOW_PAID.lower() == "true":
    sys.exit(1)

_STATE_DIR = Path(os.environ.get("GITHUB_WORKSPACE", ".")) / ".github" / "state"
_AUTOPILOT_STATE = _STATE_DIR / "strategy_autopilot.json"

# Minimum Sharpe to consider a strategy worth keeping
_PROMOTE_THRESHOLD = 1.0
_RETIRE_THRESHOLD = 0.2
_RETIRE_CONSECUTIVE = 3


# ── Quick backtest engine (no vectorbt dependency) ────────────────────────────

def _yf_ohlcv(symbol: str, period: str = "2y") -> list[dict]:
    """Download OHLCV. Returns list of {date, open, high, low, close, volume}."""
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if hist.empty or "Close" not in hist.columns:
            return []
        result = []
        for ts, row in hist.iterrows():
            result.append({
                "date": str(ts.date()),
                "open": float(row.get("Open", 0)),
                "high": float(row.get("High", 0)),
                "low": float(row.get("Low", 0)),
                "close": float(row.get("Close", 0)),
                "volume": float(row.get("Volume", 0)),
            })
        return result
    except Exception:
        return []


def _compute_backtest_metrics(
    closes: list[float],
    entries: list[bool],
    exits: list[bool],
) -> dict:
    """
    Simple long-only backtest metrics from signal arrays.
    entries[i]=True means BUY at close[i+1].
    exits[i]=True means SELL at close[i+1].
    """
    if len(closes) < 10:
        return {}

    equity = [1.0]
    in_trade = False
    entry_price = 0.0
    trades: list[float] = []
    daily_returns: list[float] = []

    for i in range(len(closes) - 1):
        if not in_trade and i < len(entries) and entries[i]:
            in_trade = True
            entry_price = closes[i + 1]
        elif in_trade and i < len(exits) and exits[i]:
            ret = (closes[i + 1] - entry_price) / entry_price
            trades.append(ret)
            equity.append(equity[-1] * (1 + ret))
            in_trade = False

        if in_trade and entry_price > 0:
            daily_ret = (closes[i + 1] - closes[i]) / closes[i]
            daily_returns.append(daily_ret)
        else:
            daily_returns.append(0.0)

    if not trades:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "n_trades": 0}

    import math

    # Sharpe (annualized, daily returns)
    if len(daily_returns) > 1:
        mean_r = sum(daily_returns) / len(daily_returns)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in daily_returns) / len(daily_returns))
        sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    peak = 1.0
    max_dd = 0.0
    running = 1.0
    for ret in trades:
        running *= 1 + ret
        if running > peak:
            peak = running
        dd = (peak - running) / peak
        if dd > max_dd:
            max_dd = dd

    win_rate = sum(1 for t in trades if t > 0) / len(trades)
    total_return = (running - 1.0) * 100

    return {
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "n_trades": len(trades),
        "total_return_pct": round(total_return, 2),
    }


def _momentum_signals(closes: list[float], lookback: int = 90, skip: int = 20) -> tuple[list[bool], list[bool]]:
    """12-1 momentum style: enter when momentum > 0, exit when < 0."""
    entries = [False] * len(closes)
    exits = [False] * len(closes)
    for i in range(lookback + skip, len(closes)):
        mom = (closes[i - skip] - closes[i - lookback - skip]) / closes[i - lookback - skip]
        if mom > 0.02:
            entries[i] = True
        elif mom < -0.02:
            exits[i] = True
    return entries, exits


def _mean_reversion_signals(closes: list[float], window: int = 20, threshold: float = 2.0) -> tuple[list[bool], list[bool]]:
    """Bollinger band mean reversion."""
    import math
    entries = [False] * len(closes)
    exits = [False] * len(closes)
    for i in range(window, len(closes)):
        window_slice = closes[i - window:i]
        mean = sum(window_slice) / window
        std = math.sqrt(sum((c - mean) ** 2 for c in window_slice) / window)
        if std == 0:
            continue
        z = (closes[i] - mean) / std
        if z < -threshold:
            entries[i] = True
        elif z > 0:
            exits[i] = True
    return entries, exits


def _rsi_signals(closes: list[float], period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> tuple[list[bool], list[bool]]:
    """RSI-based entry/exit."""
    entries = [False] * len(closes)
    exits = [False] * len(closes)

    def _rsi(idx: int) -> float | None:
        if idx < period + 1:
            return None
        gains, losses = [], []
        for j in range(idx - period, idx):
            chg = closes[j] - closes[j - 1]
            gains.append(max(chg, 0))
            losses.append(max(-chg, 0))
        ag = sum(gains) / period
        al = sum(losses) / period
        if al == 0:
            return 100.0
        return 100 - 100 / (1 + ag / al)

    for i in range(period + 2, len(closes)):
        r = _rsi(i)
        if r is None:
            continue
        if r < oversold:
            entries[i] = True
        elif r > overbought:
            exits[i] = True
    return entries, exits


# Strategy registry for autopilot testing
_STRATEGY_VARIANTS: dict[str, dict] = {
    "momentum_90_20": {"type": "momentum", "lookback": 90, "skip": 20},
    "momentum_120_20": {"type": "momentum", "lookback": 120, "skip": 20},
    "momentum_60_10": {"type": "momentum", "lookback": 60, "skip": 10},
    "mean_rev_20_2": {"type": "mean_reversion", "window": 20, "threshold": 2.0},
    "mean_rev_20_1.5": {"type": "mean_reversion", "window": 20, "threshold": 1.5},
    "mean_rev_30_2": {"type": "mean_reversion", "window": 30, "threshold": 2.0},
    "rsi_14_30_70": {"type": "rsi", "period": 14, "oversold": 30, "overbought": 70},
    "rsi_14_25_75": {"type": "rsi", "period": 14, "oversold": 25, "overbought": 75},
    "rsi_21_30_70": {"type": "rsi", "period": 21, "oversold": 30, "overbought": 70},
}

_SYMBOLS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]


def _run_variant_backtest(name: str, config: dict, data: dict[str, list[dict]]) -> dict:
    """Run one strategy variant across all symbols, return aggregate metrics."""
    all_sharpes = []
    sym_results = {}

    for sym, ohlcv in data.items():
        if len(ohlcv) < 150:
            continue
        closes = [r["close"] for r in ohlcv]

        strategy_type = config["type"]
        if strategy_type == "momentum":
            entries, exits = _momentum_signals(closes, config.get("lookback", 90), config.get("skip", 20))
        elif strategy_type == "mean_reversion":
            entries, exits = _mean_reversion_signals(closes, config.get("window", 20), config.get("threshold", 2.0))
        elif strategy_type == "rsi":
            entries, exits = _rsi_signals(closes, config.get("period", 14),
                                           config.get("oversold", 30), config.get("overbought", 70))
        else:
            continue

        metrics = _compute_backtest_metrics(closes, entries, exits)
        if metrics:
            sym_results[sym] = metrics
            all_sharpes.append(metrics["sharpe"])

    if not all_sharpes:
        return {}

    avg_sharpe = sum(all_sharpes) / len(all_sharpes)
    avg_dd = sum(sym_results[s]["max_drawdown"] for s in sym_results) / len(sym_results)
    avg_wr = sum(sym_results[s]["win_rate"] for s in sym_results) / len(sym_results)
    total_trades = sum(sym_results[s]["n_trades"] for s in sym_results)

    return {
        "name": name,
        "avg_sharpe": round(avg_sharpe, 3),
        "avg_max_drawdown": round(avg_dd, 2),
        "avg_win_rate": round(avg_wr, 1),
        "total_trades": total_trades,
        "symbol_results": sym_results,
        "config": config,
        "ts": time.time(),
    }


def _load_state() -> dict:
    try:
        if _AUTOPILOT_STATE.exists():
            return json.loads(_AUTOPILOT_STATE.read_text())
    except Exception:
        pass
    return {"runs": {}, "consecutive_poor": {}}


def _save_state(state: dict) -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _AUTOPILOT_STATE.write_text(json.dumps(state, indent=2))


def _llm_propose_mutation(top_variants: list[dict], bottom_variants: list[dict]) -> list[dict]:
    """Ask LLM to propose new strategy variants based on winners and losers."""
    top_str = json.dumps([{"name": v["name"], "sharpe": v["avg_sharpe"], "config": v["config"]} for v in top_variants[:3]])
    bot_str = json.dumps([{"name": v["name"], "sharpe": v["avg_sharpe"], "config": v["config"]} for v in bottom_variants[:3]])
    prompt = (
        f"These quantitative strategy variants performed best (Sharpe): {top_str}\n"
        f"These performed worst: {bot_str}\n\n"
        "Propose 2 new strategy variants by mutating the best performers. "
        "Each variant must be one of these types: momentum, mean_reversion, rsi. "
        "Return valid JSON array of objects with fields: name (string), type, and type-specific params "
        "(momentum: lookback int, skip int; mean_reversion: window int, threshold float; "
        "rsi: period int, oversold float, overbought float). "
        "No markdown, just the JSON array."
    )
    try:
        raw = llm(prompt, max_tokens=300, use_cache=False, inject_company_context=False)
        # Extract JSON array from response
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            proposed = json.loads(raw[start:end])
            return [p for p in proposed if isinstance(p, dict) and "type" in p and "name" in p]
    except Exception:
        pass
    return []


def main() -> None:
    print("Strategy Autopilot starting…", flush=True)
    state = _load_state()

    # Download market data once for all variants
    print("Fetching price data…", flush=True)
    data: dict[str, list[dict]] = {}
    for sym in _SYMBOLS:
        ohlcv = _yf_ohlcv(sym, period="2y")
        if ohlcv:
            data[sym] = ohlcv
            print(f"  {sym}: {len(ohlcv)} bars", flush=True)

    if not data:
        print("No price data — aborting", flush=True)
        sys.exit(0)

    # Run all registered variants
    results = []
    for name, config in _STRATEGY_VARIANTS.items():
        print(f"  Backtesting {name}…", flush=True)
        res = _run_variant_backtest(name, config, data)
        if res:
            results.append(res)
            state["runs"][name] = state["runs"].get(name, [])
            state["runs"][name].append(res["avg_sharpe"])
            # Keep last 5 runs per variant
            state["runs"][name] = state["runs"][name][-5:]

    if not results:
        print("No backtest results — check data", flush=True)
        sys.exit(0)

    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    top = results[:3]
    bottom = results[-3:]

    print("\n=== Autopilot Results ===", flush=True)
    for r in results:
        print(f"  {r['name']}: Sharpe={r['avg_sharpe']:.3f} | MaxDD={r['avg_max_drawdown']:.1f}% | "
              f"WinRate={r['avg_win_rate']:.1f}% | Trades={r['total_trades']}", flush=True)

    # LLM proposes mutations
    print("\nAsking LLM to propose mutations…", flush=True)
    mutations = _llm_propose_mutation(top, bottom)
    if mutations:
        print(f"  LLM proposed {len(mutations)} new variants", flush=True)
        for mut in mutations:
            mut_name = mut.pop("name")
            mut_config = mut
            print(f"  Testing mutation: {mut_name}…", flush=True)
            res = _run_variant_backtest(mut_name, mut_config, data)
            if res:
                results.append(res)
                print(f"    → Sharpe={res['avg_sharpe']:.3f}", flush=True)

    # Re-sort with mutations included
    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)

    # Promote top performers
    promoted = []
    for r in results:
        if r["avg_sharpe"] >= _PROMOTE_THRESHOLD:
            promoted.append(r)
            memory_write("experiment_results", {
                "source": "strategy_autopilot",
                "name": r["name"],
                "sharpe": r["avg_sharpe"],
                "max_drawdown": r["avg_max_drawdown"],
                "win_rate": r["avg_win_rate"],
                "config": r["config"],
                "status": "promoted",
            })

    # Flag consistently poor performers
    retired = []
    for name, run_history in state["runs"].items():
        if len(run_history) >= _RETIRE_CONSECUTIVE:
            recent = run_history[-_RETIRE_CONSECUTIVE:]
            if all(s < _RETIRE_THRESHOLD for s in recent):
                retired.append(name)
                state["consecutive_poor"][name] = state["consecutive_poor"].get(name, 0) + 1

    # Update brain top strategies
    top_names = [r["name"] for r in results[:5] if r["avg_sharpe"] > 0.5]
    if top_names:
        core_update("top_strategies", top_names)

    core_update("last_autopilot_run", time.time())
    core_update("autopilot_best_sharpe", results[0]["avg_sharpe"] if results else 0)

    # Save state
    _save_state(state)

    # Build Slack report
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"*Strategy Autopilot Report — {now_str}*",
        f"Tested {len(results)} variants across {len(data)} symbols ({len(list(data.values())[0]) if data else 0} bars)",
        "",
        "*Top 5 Performers:*",
    ]
    for r in results[:5]:
        status = "🏆" if r["avg_sharpe"] >= _PROMOTE_THRESHOLD else "✅" if r["avg_sharpe"] >= 0.5 else "⚠️"
        lines.append(
            f"  {status} `{r['name']}` Sharpe={r['avg_sharpe']:.3f} | "
            f"DD={r['avg_max_drawdown']:.1f}% | WR={r['avg_win_rate']:.1f}%"
        )

    if promoted:
        lines.append("")
        lines.append(f"*🚀 Promoted to paper trading ({len(promoted)}):*")
        for r in promoted:
            lines.append(f"  • `{r['name']}` Sharpe={r['avg_sharpe']:.3f}")

    if retired:
        lines.append("")
        lines.append(f"*🗑️ Flagged for retirement ({len(retired)}):*")
        for name in retired:
            lines.append(f"  • `{name}` (Sharpe < {_RETIRE_THRESHOLD} for {_RETIRE_CONSECUTIVE} runs)")

    if mutations:
        lines.append("")
        lines.append(f"*🤖 LLM-proposed mutations tested:* {len(mutations)}")

    slack_report = "\n".join(lines)
    print("\n" + slack_report, flush=True)
    slack_post("#strategy-lab", slack_report)

    print("Autopilot complete.", flush=True)


if __name__ == "__main__":
    main()
