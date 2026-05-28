"""
Automated experiment runner for GitHub Actions.

For each strategy in EXPERIMENT_CONFIGS, fetches historical OHLCV via yfinance,
runs backtest_signals(), computes Sharpe/Sortino/drawdown, and writes a JSON result
to experiments/results/. On completion, posts a summary to Slack #ml-experiments.

No mocks. If data is unavailable the strategy result is skipped (not faked).
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Experiment definitions ────────────────────────────────────────────────────
# Each entry: (strategy_name, import_path, class_name, symbol, interval,
#              train_start, test_start, test_end, params)
EXPERIMENT_CONFIGS = [
    # Equities desk
    ("momentum",         "app.strategies.manual.momentum",         "MomentumStrategy",         "SPY",     "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("mean_reversion",   "app.strategies.manual.mean_reversion",   "MeanReversionStrategy",    "AAPL",    "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("breakout",         "app.strategies.manual.breakout",         "BreakoutStrategy",         "QQQ",     "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("rsi_macd",         "app.strategies.manual.rsi_macd",         "RSIMACDStrategy",          "MSFT",    "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("supertrend",       "app.strategies.manual.supertrend",       "SupertrendStrategy",       "NVDA",    "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("low_volatility",   "app.strategies.manual.low_volatility",   "LowVolatilityStrategy",    "XLU",     "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("fifty_two_week_high", "app.strategies.manual.fifty_two_week_high", "FiftyTwoWeekHighStrategy", "SPY", "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("time_series_momentum", "app.strategies.manual.time_series_momentum", "TimeSeriesMomentumStrategy", "SPY", "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("sector_rotation",  "app.strategies.manual.sector_rotation",  "SectorRotationStrategy",   "SPY",     "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("vix_mean_reversion","app.strategies.manual.vix_mean_reversion","VIXMeanReversionStrategy","SPY",    "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("overnight_return", "app.strategies.manual.overnight_return", "OvernightReturnStrategy",  "SPY",     "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("open_close_revert","app.strategies.manual.open_close_revert","OpenCloseRevertStrategy",  "SPY",     "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("residual_momentum","app.strategies.manual.residual_momentum","ResidualMomentumStrategy", "SPY",     "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("idio_vol_anomaly", "app.strategies.manual.idio_vol_anomaly", "IdiosyncraticVolAnomalyStrategy","SPY","1d","2020-01-01","2024-01-01","2025-01-01",{}),
    ("triple_barrier_momentum","app.strategies.manual.triple_barrier_momentum","TripleBarrierMomentumStrategy","QQQ","1d","2020-01-01","2024-01-01","2025-01-01",{}),
    ("hmm_regime",       "app.strategies.manual.hmm_regime",       "HMMRegimeStrategy",        "SPY",     "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
]

STRATEGY_FILTER = os.environ.get("STRATEGY_FILTER", "").strip()
SYMBOL_FILTER   = os.environ.get("SYMBOL_FILTER",   "").strip()
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL   = "#ml-experiments"


# ── Metrics helpers ────────────────────────────────────────────────────────────

# SPY benchmark cache: fetched once per run, shared across all experiments
_SPY_CACHE: dict[tuple[str, str], pd.Series] = {}

def _fetch_spy_returns(start: str, end: str) -> pd.Series | None:
    key = (start, end)
    if key in _SPY_CACHE:
        return _SPY_CACHE[key]
    spy_df = _fetch_ohlcv("SPY", start, end)
    if spy_df is None:
        return None
    r = spy_df["close"].pct_change().fillna(0.0)
    _SPY_CACHE[key] = r
    return r


def _compute_metrics(entries: pd.Series, exits: pd.Series, prices: pd.Series,
                     benchmark_returns: pd.Series | None = None) -> dict:
    """Compute full advanced metrics suite via app.ml.evaluation.metrics."""
    try:
        from app.ml.evaluation.metrics import metrics_from_signals
        m = metrics_from_signals(entries, exits, prices, benchmark_prices=None)
        # Inject benchmark returns directly if available
        if benchmark_returns is not None:
            from app.ml.evaluation.metrics import compute_metrics, _fill_benchmark_stats, _ann_factor
            r = prices.pct_change().fillna(0.0)
            pos = pd.Series(0, index=prices.index, dtype=float)
            in_trade = False
            for i in range(len(prices)):
                if not in_trade and entries.iloc[i]:
                    in_trade = True
                elif in_trade and exits.iloc[i]:
                    in_trade = False
                pos.iloc[i] = 1.0 if in_trade else 0.0
            strat_r = (pos.shift(1).fillna(0) * r)
            b = benchmark_returns.reindex(strat_r.index).fillna(0)
            _fill_benchmark_stats(m, strat_r, b, _ann_factor(strat_r), 0.0)
        return {**m.to_dict(), "n_bars": len(prices)}
    except Exception as exc:
        print(f"  ⚠ advanced metrics failed ({exc}), falling back to basic", flush=True)
        return _compute_metrics_basic(entries, exits, prices)


def _compute_metrics_basic(entries: pd.Series, exits: pd.Series, prices: pd.Series) -> dict:
    """Fallback basic metrics (no external deps)."""
    returns = prices.pct_change().fillna(0.0)
    position = pd.Series(0, index=prices.index, dtype=float)
    in_trade = False
    for i in range(len(prices)):
        if not in_trade and entries.iloc[i]:
            in_trade = True
        elif in_trade and exits.iloc[i]:
            in_trade = False
        position.iloc[i] = 1.0 if in_trade else 0.0
    strat_returns = (position.shift(1).fillna(0) * returns)
    cum = (1 + strat_returns).cumprod()
    total_return = float(cum.iloc[-1] - 1)
    af = 252
    mean_r = strat_returns.mean() * af
    std_r  = strat_returns.std() * (af ** 0.5)
    sharpe = float(mean_r / std_r) if std_r > 0 else 0.0
    neg_r  = strat_returns[strat_returns < 0]
    sortino_denom = neg_r.std() * (af ** 0.5) if len(neg_r) > 0 else 1e-9
    sortino = float(mean_r / sortino_denom)
    dd = (cum - cum.cummax()) / cum.cummax()
    return {
        "sharpe": round(sharpe, 4), "sortino": round(sortino, 4),
        "total_return": round(total_return, 4), "max_drawdown": round(float(dd.min()), 4),
        "n_trades": int(entries.sum()), "n_bars": len(prices),
    }


def _fetch_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            return None
        # Normalise column names
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            return None
        return df.dropna(subset=["close"])
    except Exception as exc:
        print(f"  ⚠ yfinance fetch failed for {symbol}: {exc}", flush=True)
        return None


def _import_strategy(module_path: str, class_name: str, params: dict):
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(params=params if params else None)


def _run_one(name: str, module_path: str, class_name: str, symbol: str,
             interval: str, train_start: str, test_start: str, test_end: str,
             params: dict) -> dict:
    if STRATEGY_FILTER and name != STRATEGY_FILTER:
        return {}
    effective_symbol = SYMBOL_FILTER if SYMBOL_FILTER else symbol

    print(f"\n{'─'*60}", flush=True)
    print(f"  {name} | {effective_symbol} | {test_start}→{test_end}", flush=True)

    # Fetch full data (train + test combined so indicators warm up)
    df = _fetch_ohlcv(effective_symbol, train_start, test_end)
    if df is None or len(df) < 60:
        print(f"  ✗ insufficient data ({len(df) if df is not None else 0} bars)", flush=True)
        return {}

    try:
        strategy = _import_strategy(module_path, class_name, params)
    except Exception as exc:
        print(f"  ✗ import failed: {exc}", flush=True)
        return {}

    try:
        result = strategy.backtest_signals(df)
        # Support both BacktestSignals dataclass and plain Series
        if hasattr(result, "entries"):
            entries = result.entries.astype(bool)
            exits   = result.exits.astype(bool)
        else:
            sig     = result.fillna(0)
            entries = (sig == 1)
            exits   = (sig == -1)
    except Exception as exc:
        print(f"  ✗ backtest_signals() failed: {exc}", flush=True)
        traceback.print_exc()
        return {}

    # Evaluate on test period only
    test_mask = df.index >= pd.Timestamp(test_start)
    if test_mask.sum() < 20:
        print(f"  ✗ too few test bars ({test_mask.sum()})", flush=True)
        return {}

    # Fetch SPY benchmark for relative metrics (skip if same symbol or fetch fails)
    spy_returns = None
    if effective_symbol != "SPY":
        spy_returns = _fetch_spy_returns(test_start, test_end)

    metrics = _compute_metrics(
        entries[test_mask], exits[test_mask],
        df["close"][test_mask],
        benchmark_returns=spy_returns,
    )
    calmar_str = f"  Calmar={metrics.get('calmar', 0):+.3f}" if "calmar" in metrics else ""
    alpha_str  = f"  α={metrics.get('alpha', 0):+.3f}" if "alpha" in metrics and metrics.get("alpha", 0) != 0 else ""
    print(
        f"  ✓ Sharpe={metrics['sharpe']:+.3f}  Sortino={metrics['sortino']:+.3f}"
        f"  MDD={metrics['max_drawdown']:+.2%}  trades={metrics['n_trades']}"
        f"{calmar_str}{alpha_str}",
        flush=True,
    )

    return {
        "experiment": {
            "name": f"{name}_{effective_symbol.lower()}_{interval}",
            "strategy": name,
            "symbol": effective_symbol,
            "interval": interval,
            "train_start": train_start,
            "test_start": test_start,
            "test_end": test_end,
            "params": params,
        },
        "results": metrics,
        "run_at": datetime.utcnow().isoformat() + "Z",
    }


def _save_result(result: dict) -> Path:
    exp_name = result["experiment"]["name"]
    ts       = datetime.utcnow().strftime("%Y%m%dT%H%M")
    fname    = RESULTS_DIR / f"{exp_name}_{ts}.json"
    fname.write_text(json.dumps(result, indent=2))
    return fname


def _post_slack(message: str) -> None:
    if not SLACK_BOT_TOKEN:
        print("  (no SLACK_BOT_TOKEN — skipping Slack post)", flush=True)
        return
    try:
        import urllib.request, urllib.parse
        payload = json.dumps({"channel": SLACK_CHANNEL, "text": message})
        req = urllib.request.Request(
            "https://slack.com/api/chat.postMessage",
            data=payload.encode(),
            headers={
                "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            if not body.get("ok"):
                print(f"  ⚠ Slack error: {body.get('error')}", flush=True)
    except Exception as exc:
        print(f"  ⚠ Slack post failed: {exc}", flush=True)


def main() -> None:
    print(f"QuantEdge Experiment Runner — {datetime.utcnow().isoformat()}Z", flush=True)
    print(f"STRATEGY_FILTER={STRATEGY_FILTER or '(all)'}  SYMBOL_FILTER={SYMBOL_FILTER or '(default)'}", flush=True)

    successes: list[dict] = []
    failures:  list[str]  = []

    for cfg in EXPERIMENT_CONFIGS:
        name, module_path, class_name, symbol, interval, train_start, test_start, test_end, params = cfg
        try:
            result = _run_one(name, module_path, class_name, symbol, interval,
                              train_start, test_start, test_end, params)
            if result:
                path = _save_result(result)
                print(f"  → saved {path.name}", flush=True)
                successes.append(result)
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            print(f"  ✗ UNHANDLED: {exc}", flush=True)
            traceback.print_exc()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}", flush=True)
    print(f"Completed: {len(successes)} succeeded, {len(failures)} failed", flush=True)

    if not successes and not STRATEGY_FILTER:
        print("WARNING: no experiment results — check yfinance connectivity", flush=True)

    # Build Slack report — rich advanced metrics
    lines = [f"*QuantEdge ML Experiments* — {date.today().isoformat()}"]
    lines.append(f"✅ {len(successes)} backtests completed  |  ❌ {len(failures)} failed")
    lines.append("")

    # Sort by Sharpe descending
    successes.sort(key=lambda r: r["results"].get("sharpe", 0), reverse=True)
    lines.append("*Top results by Sharpe (full advanced metrics):*")
    for r in successes[:10]:
        exp = r["experiment"]
        res = r["results"]
        sharpe  = res.get("sharpe", 0)
        calmar  = res.get("calmar", 0)
        omega   = res.get("omega", 0)
        sortino = res.get("sortino", 0)
        mdd     = res.get("max_drawdown", 0)
        ret     = res.get("total_return", 0)
        alpha   = res.get("alpha", 0)
        wr      = res.get("win_rate", 0)
        pf      = res.get("profit_factor", 0)
        cvar    = res.get("cvar_95", 0)
        emoji   = "🟢" if sharpe > 1.5 else ("🟡" if sharpe > 0.7 else "🔴")
        line    = (
            f"{emoji} `{exp['strategy']}/{exp['symbol']}`\n"
            f"  Sharpe={sharpe:+.3f}  Sortino={sortino:+.3f}  Calmar={calmar:+.3f}  Omega={omega:.1f}\n"
            f"  MDD={mdd:+.1%}  CVaR95={cvar:+.2%}  Ret={ret:+.1%}"
        )
        if alpha != 0:
            line += f"  α={alpha:+.3f}"
        if wr > 0:
            line += f"\n  WinRate={wr:.1%}  ProfitFactor={pf:.2f}"
        lines.append(line)

    if failures:
        lines.append("")
        lines.append("*Failures:*")
        for f in failures[:5]:
            lines.append(f"• {f}")

    # Cross-strategy aggregate stats
    if len(successes) >= 3:
        sharpes  = [r["results"].get("sharpe", 0) for r in successes]
        caldmars = [r["results"].get("calmar", 0) for r in successes]
        mdds     = [r["results"].get("max_drawdown", 0) for r in successes]
        omegas   = [r["results"].get("omega", 0) for r in successes if r["results"].get("omega", 0) < 1000]
        lines += [
            "",
            f"*Aggregate across {len(successes)} strategies:*",
            f"  Median Sharpe: `{float(np.median(sharpes)):+.3f}`  |  "
            f"Best Calmar: `{max(caldmars):+.3f}`  |  "
            f"Worst MDD: `{min(mdds):+.1%}`",
            f"  Strategies beating Sharpe 1.0: `{sum(1 for s in sharpes if s > 1.0)}`  |  "
            f"Avg Omega: `{float(np.mean(omegas)):.2f}`" if omegas else "",
        ]

    _post_slack("\n".join(lines))


if __name__ == "__main__":
    main()
