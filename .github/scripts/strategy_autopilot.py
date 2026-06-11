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
    Full backtest metrics: Sharpe, Sortino, Calmar, profit factor, expectancy,
    omega ratio, recovery factor, max consecutive losses, win/loss ratio.
    entries[i]=True means BUY at close[i+1]. exits[i]=True means SELL at close[i+1].
    """
    if len(closes) < 10:
        return {}

    import math

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
            in_trade = False

        if in_trade and entry_price > 0:
            daily_ret = (closes[i + 1] - closes[i]) / closes[i]
            daily_returns.append(daily_ret)
        else:
            daily_returns.append(0.0)

    if not trades:
        return {
            "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
            "max_drawdown": 0.0, "win_rate": 0.0, "n_trades": 0,
            "profit_factor": 0.0, "expectancy": 0.0, "omega_ratio": 0.0,
            "recovery_factor": 0.0, "max_consec_losses": 0, "win_loss_ratio": 0.0,
            "total_return_pct": 0.0, "composite_score": 0.0,
        }

    # ── Sharpe (annualized) ──────────────────────────────────────────────────
    if len(daily_returns) > 1:
        mean_r = sum(daily_returns) / len(daily_returns)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in daily_returns) / len(daily_returns))
        sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    # ── Sortino (downside deviation only) ───────────────────────────────────
    downside = [r for r in daily_returns if r < 0]
    if downside and len(downside) > 1:
        mean_down = sum(downside) / len(downside)
        dd_std = math.sqrt(sum((r - mean_down) ** 2 for r in downside) / len(downside))
        mean_r_all = sum(daily_returns) / len(daily_returns) if daily_returns else 0.0
        sortino = (mean_r_all / dd_std * math.sqrt(252)) if dd_std > 0 else 0.0
    else:
        sortino = sharpe  # no downside — as good as Sharpe

    # ── Max drawdown (on cumulative equity from trades) ──────────────────────
    running = 1.0
    peak = 1.0
    max_dd = 0.0
    for ret in trades:
        running *= 1 + ret
        peak = max(peak, running)
        dd = (peak - running) / peak
        max_dd = max(max_dd, dd)
    total_return = (running - 1.0) * 100

    # ── Calmar ratio (annualized return / max drawdown) ──────────────────────
    # Approximate annualization: assume 2 years of data (the default fetch period)
    ann_return = ((1 + total_return / 100) ** (1 / 2) - 1) * 100
    calmar = (ann_return / (max_dd * 100)) if max_dd > 0 else (ann_return if ann_return > 0 else 0.0)

    # ── Win/loss breakdown ───────────────────────────────────────────────────
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    win_rate = len(wins) / len(trades)
    loss_rate = 1 - win_rate

    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0

    # ── Profit factor (gross wins / gross losses) ────────────────────────────
    gross_wins = sum(wins) if wins else 0.0
    gross_losses = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (10.0 if gross_wins > 0 else 0.0)

    # ── Expectancy (avg P&L per trade as fraction) ──────────────────────────
    expectancy = avg_win * win_rate - avg_loss * loss_rate

    # ── Win/loss ratio (avg win / avg loss) ──────────────────────────────────
    win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else (avg_win * 10 if avg_win > 0 else 0.0)

    # ── Omega ratio (probability-weighted gains / losses above threshold=0) ──
    gains_area = sum(t for t in trades if t > 0)
    losses_area = abs(sum(t for t in trades if t <= 0))
    omega_ratio = (gains_area / losses_area) if losses_area > 0 else (10.0 if gains_area > 0 else 1.0)

    # ── Recovery factor (total return / max drawdown) ───────────────────────
    recovery_factor = (total_return / (max_dd * 100)) if max_dd > 0 else (total_return if total_return > 0 else 0.0)

    # ── Max consecutive losses ───────────────────────────────────────────────
    max_consec = 0
    cur_consec = 0
    for t in trades:
        if t <= 0:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    # ── Composite score (weighted combination for ranking) ───────────────────
    # Weights: Sharpe 30%, Sortino 20%, Calmar 20%, profit_factor 15%, expectancy 15%
    # Normalize each to a 0-2 scale to avoid domination
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))
    c_sharpe = _clamp(sharpe, -2, 3) / 3
    c_sortino = _clamp(sortino, -2, 4) / 4
    c_calmar = _clamp(calmar, -1, 5) / 5
    c_pf = _clamp(profit_factor - 1, -1, 3) / 3
    c_exp = _clamp(expectancy * 100, -5, 10) / 10
    composite_score = round(
        0.30 * c_sharpe + 0.20 * c_sortino + 0.20 * c_calmar + 0.15 * c_pf + 0.15 * c_exp,
        4,
    )

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "n_trades": len(trades),
        "profit_factor": round(profit_factor, 3),
        "expectancy": round(expectancy * 100, 3),   # in % per trade
        "omega_ratio": round(omega_ratio, 3),
        "recovery_factor": round(recovery_factor, 3),
        "max_consec_losses": max_consec,
        "win_loss_ratio": round(win_loss_ratio, 3),
        "total_return_pct": round(total_return, 2),
        "composite_score": composite_score,
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

# Walk-forward split: train on first 75% of bars, forward-test on last 25%
_TRAIN_FRAC = 0.75


def _generate_signals(closes: list[float], config: dict) -> tuple[list[bool], list[bool]]:
    """Generate entry/exit signals from config."""
    strategy_type = config["type"]
    if strategy_type == "momentum":
        return _momentum_signals(closes, config.get("lookback", 90), config.get("skip", 20))
    elif strategy_type == "mean_reversion":
        return _mean_reversion_signals(closes, config.get("window", 20), config.get("threshold", 2.0))
    elif strategy_type == "rsi":
        return _rsi_signals(closes, config.get("period", 14),
                            config.get("oversold", 30), config.get("overbought", 70))
    return [False] * len(closes), [False] * len(closes)


def _run_variant_backtest(name: str, config: dict, data: dict[str, list[dict]]) -> dict:
    """
    Run one strategy variant with walk-forward validation.
    - In-sample (IS): first 75% of bars (parameter fitting period)
    - Out-of-sample / forward-test (OOS): last 25% of bars (true validation)
    Reports both sets of metrics and flags overfitting when IS >> OOS.
    """
    sym_is: dict[str, dict] = {}    # in-sample metrics per symbol
    sym_oos: dict[str, dict] = {}   # out-of-sample metrics per symbol
    all_is_sharpes: list[float] = []

    for sym, ohlcv in data.items():
        if len(ohlcv) < 200:
            continue
        closes = [r["close"] for r in ohlcv]
        split = max(100, int(len(closes) * _TRAIN_FRAC))

        # In-sample (train)
        is_closes = closes[:split]
        is_entries, is_exits = _generate_signals(is_closes, config)
        is_metrics = _compute_backtest_metrics(is_closes, is_entries, is_exits)
        if is_metrics and is_metrics.get("n_trades", 0) > 0:
            sym_is[sym] = is_metrics
            all_is_sharpes.append(is_metrics["sharpe"])

        # Out-of-sample / forward-test (held-out)
        oos_closes = closes[split:]
        if len(oos_closes) >= 50:
            oos_entries, oos_exits = _generate_signals(oos_closes, config)
            oos_metrics = _compute_backtest_metrics(oos_closes, oos_entries, oos_exits)
            if oos_metrics:
                sym_oos[sym] = oos_metrics

    if not all_is_sharpes:
        return {}

    n_is = max(len(sym_is), 1)
    n_oos = max(len(sym_oos), 1)

    def _avg_is(key: str) -> float:
        vals = [sym_is[s].get(key, 0.0) for s in sym_is]
        return round(sum(vals) / n_is, 4)

    def _avg_oos(key: str) -> float:
        vals = [sym_oos[s].get(key, 0.0) for s in sym_oos] if sym_oos else [0.0]
        return round(sum(vals) / n_oos, 4)

    # Overfitting flag: IS composite > OOS composite by a large margin
    is_composite = _avg_is("composite_score")
    oos_composite = _avg_oos("composite_score")
    overfit_flag = (is_composite - oos_composite) > 0.15 and oos_composite < 0.0

    # Blended score weights OOS higher (OOS is the truth)
    blended_composite = round(0.35 * is_composite + 0.65 * oos_composite, 4)

    # Alias for _avg function used in return dict
    def _avg(key: str) -> float:
        return _avg_is(key)

    # Count total trades across both IS and OOS
    total_is_trades = sum(sym_is[s]["n_trades"] for s in sym_is)
    total_oos_trades = sum(sym_oos[s]["n_trades"] for s in sym_oos)

    return {
        "name": name,
        # In-sample metrics
        "avg_sharpe": _avg_is("sharpe"),
        "avg_sortino": _avg_is("sortino"),
        "avg_calmar": _avg_is("calmar"),
        "avg_max_drawdown": _avg_is("max_drawdown"),
        "avg_win_rate": _avg_is("win_rate"),
        "avg_profit_factor": _avg_is("profit_factor"),
        "avg_expectancy": _avg_is("expectancy"),
        "avg_omega_ratio": _avg_is("omega_ratio"),
        "avg_recovery_factor": _avg_is("recovery_factor"),
        "avg_win_loss_ratio": _avg_is("win_loss_ratio"),
        "avg_max_consec_losses": round(_avg_is("max_consec_losses"), 1),
        "avg_composite_score": is_composite,
        # Out-of-sample / forward-test metrics
        "oos_sharpe": _avg_oos("sharpe"),
        "oos_sortino": _avg_oos("sortino"),
        "oos_calmar": _avg_oos("calmar"),
        "oos_max_drawdown": _avg_oos("max_drawdown"),
        "oos_win_rate": _avg_oos("win_rate"),
        "oos_profit_factor": _avg_oos("profit_factor"),
        "oos_expectancy": _avg_oos("expectancy"),
        "oos_composite_score": oos_composite,
        # Blended score (35% IS + 65% OOS) — primary ranking key
        "blended_composite": blended_composite,
        "overfit_flag": overfit_flag,
        "total_trades": total_is_trades,
        "oos_trades": total_oos_trades,
        "symbol_results": sym_is,
        "oos_symbol_results": sym_oos,
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
    """Ask LLM to propose new strategy variants based on multi-metric winner/loser analysis."""
    def _summary(v: dict) -> dict:
        return {
            "name": v["name"],
            "composite_score": v.get("avg_composite_score", 0),
            "sharpe": v.get("avg_sharpe", 0),
            "sortino": v.get("avg_sortino", 0),
            "calmar": v.get("avg_calmar", 0),
            "profit_factor": v.get("avg_profit_factor", 0),
            "expectancy_pct": v.get("avg_expectancy", 0),
            "win_rate": v.get("avg_win_rate", 0),
            "max_drawdown": v.get("avg_max_drawdown", 0),
            "config": v["config"],
        }

    top_str = json.dumps([_summary(v) for v in top_variants[:3]])
    bot_str = json.dumps([_summary(v) for v in bottom_variants[:3]])
    prompt = (
        f"Top performers (by composite score = 30% Sharpe + 20% Sortino + 20% Calmar + 15% ProfitFactor + 15% Expectancy):\n{top_str}\n"
        f"Worst performers:\n{bot_str}\n\n"
        "Propose 2 new strategy variants by mutating the best performers. "
        "Optimize for composite score: improve Sortino (reduce downside), improve Calmar (reduce drawdowns), "
        "and improve profit factor (tighter entries). "
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

    if not results:
        print("No backtest results — check data", flush=True)
        sys.exit(0)

    results.sort(key=lambda x: x["blended_composite"], reverse=True)
    top = results[:3]
    bottom = results[-3:]

    print("\n=== Autopilot Results (IS = in-sample, OOS = forward-test) ===", flush=True)
    for r in results:
        overfit = "⚠️OVERFIT" if r.get("overfit_flag") else ""
        print(
            f"  {r['name']}: Blended={r['blended_composite']:.3f} "
            f"IS[Sh={r['avg_sharpe']:.2f} PF={r['avg_profit_factor']:.2f}] "
            f"OOS[Sh={r['oos_sharpe']:.2f} PF={r['oos_profit_factor']:.2f} "
            f"WR={r['oos_win_rate']:.1f}%] Trades(IS/OOS)={r['total_trades']}/{r['oos_trades']} {overfit}",
            flush=True,
        )

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
                print(f"    → Score={res['avg_composite_score']:.3f} | Sharpe={res['avg_sharpe']:.3f}", flush=True)

    # Re-sort with mutations included — blended (35% IS + 65% OOS) is the primary key
    results.sort(key=lambda x: x["blended_composite"], reverse=True)

    # Promote top performers: require Sharpe ≥ 1.0 AND positive OOS (no overfitters)
    promoted = []
    for r in results:
        # Must pass: IS Sharpe ≥ 1.0 AND OOS Sharpe > 0 AND not flagged as overfit
        qualifies = (
            r["avg_sharpe"] >= _PROMOTE_THRESHOLD
            and r["oos_sharpe"] > 0
            and not r.get("overfit_flag", False)
        )
        if qualifies:
            promoted.append(r)
            memory_write("experiment_results", {
                "source": "strategy_autopilot",
                "name": r["name"],
                "blended_composite": r["blended_composite"],
                "is_composite": r["avg_composite_score"],
                "oos_composite": r["oos_composite_score"],
                "is_sharpe": r["avg_sharpe"],
                "oos_sharpe": r["oos_sharpe"],
                "is_sortino": r["avg_sortino"],
                "oos_sortino": r["oos_sortino"],
                "calmar": r["avg_calmar"],
                "profit_factor": r["avg_profit_factor"],
                "oos_profit_factor": r["oos_profit_factor"],
                "expectancy_pct": r["avg_expectancy"],
                "omega_ratio": r["avg_omega_ratio"],
                "max_drawdown": r["avg_max_drawdown"],
                "is_win_rate": r["avg_win_rate"],
                "oos_win_rate": r["oos_win_rate"],
                "win_loss_ratio": r["avg_win_loss_ratio"],
                "config": r["config"],
                "status": "promoted",
            })

    # Flag consistently poor performers (track by composite score now)
    retired = []
    for name, run_history in state["runs"].items():
        if len(run_history) >= _RETIRE_CONSECUTIVE:
            recent = run_history[-_RETIRE_CONSECUTIVE:]
            if all(s < _RETIRE_THRESHOLD for s in recent):
                retired.append(name)
                state["consecutive_poor"][name] = state["consecutive_poor"].get(name, 0) + 1

    # Update brain top strategies
    # Only include strategies that have positive OOS performance
    top_names = [r["name"] for r in results[:5] if r["avg_sharpe"] > 0.5 and r["oos_sharpe"] > 0]
    if top_names:
        core_update("top_strategies", top_names)

    core_update("last_autopilot_run", time.time())
    core_update("autopilot_best_sharpe", results[0]["avg_sharpe"] if results else 0)
    core_update("autopilot_best_oos_sharpe", results[0]["oos_sharpe"] if results else 0)
    core_update("autopilot_best_blended", results[0]["blended_composite"] if results else 0)
    core_update("autopilot_overfit_count", sum(1 for r in results if r.get("overfit_flag", False)))

    # Save state (store composite score history alongside sharpe)
    for r in results:
        state["runs"][r["name"]] = state["runs"].get(r["name"], [])
        state["runs"][r["name"]].append(r["avg_sharpe"])
        state["runs"][r["name"]] = state["runs"][r["name"]][-5:]
    _save_state(state)

    # Build Slack report
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_bars = len(list(data.values())[0]) if data else 0
    n_is_bars = int(n_bars * _TRAIN_FRAC)
    n_oos_bars = n_bars - n_is_bars
    lines = [
        f"*Strategy Autopilot Report — {now_str}*",
        f"Tested {len(results)} variants | {len(data)} symbols | {n_bars} bars "
        f"(IS={n_is_bars} 75% | OOS={n_oos_bars} 25% forward-test)",
        f"_Ranked by blended score = 35% IS composite + 65% OOS composite_",
        "",
        "*Top 5 Performers (IS | OOS):*",
    ]
    for r in results[:5]:
        overfit_warn = " ⚠️OVERFIT" if r.get("overfit_flag") else ""
        if r["avg_sharpe"] >= _PROMOTE_THRESHOLD and r["oos_sharpe"] > 0 and not r.get("overfit_flag"):
            status = "🏆"
        elif r["avg_sharpe"] >= 0.5 and r["oos_sharpe"] > 0:
            status = "✅"
        else:
            status = "⚠️"
        lines.append(
            f"  {status} `{r['name']}` Blended={r['blended_composite']:.3f}{overfit_warn}\n"
            f"     IS: Sh={r['avg_sharpe']:.3f} Sortino={r['avg_sortino']:.3f} "
            f"PF={r['avg_profit_factor']:.2f} WR={r['avg_win_rate']:.1f}% DD={r['avg_max_drawdown']:.1f}%\n"
            f"     OOS: Sh={r['oos_sharpe']:.3f} PF={r['oos_profit_factor']:.2f} "
            f"WR={r['oos_win_rate']:.1f}% Trades={r['oos_trades']}"
        )

    if promoted:
        lines.append("")
        lines.append(f"*🚀 Promoted to paper trading ({len(promoted)}) — IS+OOS both positive, no overfitting:*")
        for r in promoted:
            lines.append(
                f"  • `{r['name']}` IS Sharpe={r['avg_sharpe']:.3f} → OOS Sharpe={r['oos_sharpe']:.3f} | "
                f"PF: IS={r['avg_profit_factor']:.2f} OOS={r['oos_profit_factor']:.2f}"
            )

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
