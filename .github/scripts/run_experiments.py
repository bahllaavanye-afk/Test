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
    # New institutional strategies (2 Sigma / Citadel research)
    ("cross_sectional_momentum", "app.strategies.manual.cross_sectional_momentum", "CrossSectionalMomentumStrategy", "SPY", "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("opening_range_breakout",   "app.strategies.manual.opening_range_breakout",   "OpeningRangeBreakoutStrategy",   "SPY", "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("vwap_reversion",           "app.strategies.manual.vwap_reversion",           "VWAPReversionStrategy",          "SPY", "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("vrp_systematic",           "app.strategies.manual.vrp_systematic",           "VRPSystematicStrategy",          "SPY", "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("dispersion_trading",       "app.strategies.manual.dispersion_trading",       "DispersionTradingStrategy",      "SPY", "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("pca_stat_arb",             "app.strategies.manual.pca_stat_arb",             "PCAStatArbStrategy",             "SPY", "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("kalman_pairs",             "app.strategies.manual.kalman_pairs",             "KalmanPairsStrategy",            "SPY", "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("pead_sue",                 "app.strategies.manual.pead_sue",                 "PEADStrategy",                   "SPY", "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    # Options desk
    ("gamma_exposure",           "app.strategies.manual.gamma_exposure",           "GammaExposureStrategy",  "SPY", "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("skew_arb",                 "app.strategies.manual.skew_arb",                 "SkewArbitrageStrategy",  "SPY", "1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    ("vol_term_structure",       "app.strategies.manual.vol_term_structure",       "VolTermStructureStrategy","SPY","1d", "2020-01-01", "2024-01-01", "2025-01-01", {}),
    # Crypto desk
    ("crypto_adaptive_trend",    "app.strategies.manual.crypto_adaptive_trend",    "CryptoAdaptiveTrendStrategy","BTC/USD","1d","2020-01-01","2024-01-01","2025-01-01",{}),
    ("mvrv_zscore_timing",       "app.strategies.manual.mvrv_zscore_timing",       "MVRVZScoreTimingStrategy","BTC/USD","1d","2020-01-01","2024-01-01","2025-01-01",{}),
    ("intraday_seasonality",     "app.strategies.manual.intraday_seasonality",     "IntradaySeasonality",    "BTC/USD","1d","2020-01-01","2024-01-01","2025-01-01",{}),
    ("basis_carry",              "app.strategies.manual.basis_carry",              "BasisCarryStrategy",     "BTC/USD","1d","2020-01-01","2024-01-01","2025-01-01",{}),
    ("btc_eth_stat_arb",         "app.strategies.manual.btc_eth_stat_arb",         "BTCETHStatArb",          "BTC/USD","1d","2020-01-01","2024-01-01","2025-01-01",{}),
    # Macro/FX desk
    ("cross_asset_carry",        "app.strategies.manual.cross_asset_carry",        "CrossAssetCarryStrategy","GLD",  "1d","2020-01-01","2024-01-01","2025-01-01",{}),
    ("intraday_fomc_momentum",   "app.strategies.manual.intraday_fomc_momentum",   "IntradayFOMCMomentumStrategy","SPY","1d","2020-01-01","2024-01-01","2025-01-01",{}),
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
    if STRATEGY_FILTER and STRATEGY_FILTER not in name:
        return {}
    effective_symbol = SYMBOL_FILTER if SYMBOL_FILTER else symbol

    print(f"\n{'─'*60}", flush=True)
    print(f"  {name} | {effective_symbol} | {test_start}→{test_end}", flush=True)

    # Polymarket and similar non-yfinance symbols have no OHLCV data — skip gracefully.
    # The strategy's signal logic runs live via Polymarket API; backtest is not applicable.
    _SKIP_SYMBOLS = {"POLYMARKET", "POLY_DUMMY"}
    if effective_symbol in _SKIP_SYMBOLS:
        print(f"  ↷ skipped (no yfinance data for {effective_symbol} — live-only strategy)", flush=True)
        return {}

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
        print("", flush=True)
        print("╔══════════════════════════════════════════════════════════════════╗", flush=True)
        print("║  ⚠  SLACK SILENT — experiment results were NOT posted to Slack  ║", flush=True)
        print("║                                                                  ║", flush=True)
        print("║  Add SLACK_BOT_TOKEN to repo secrets:                           ║", flush=True)
        print("║  Settings → Secrets and variables → Actions → New secret        ║", flush=True)
        print("╚══════════════════════════════════════════════════════════════════╝", flush=True)
        print("", flush=True)
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


def _post_slack_summary(successes: list[dict], failures: list[str]) -> None:
    """Build and post the full advanced-metrics Slack summary."""
    lines = [f"*QuantEdge ML Experiments* — {date.today().isoformat()}"]
    lines.append(f"✅ {len(successes)} backtests completed  |  ❌ {len(failures)} failed")
    lines.append("")

    successes.sort(key=lambda r: r["results"].get("sharpe", 0), reverse=True)
    lines.append("*Top results by Sharpe (advanced metrics):*")
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
        line = (
            f"{emoji} `{exp['strategy']}/{exp['symbol']}`\n"
            f"  Sharpe={sharpe:+.3f}  Sortino={sortino:+.3f}  Calmar={calmar:+.3f}  Ω={omega:.1f}\n"
            f"  MDD={mdd:+.1%}  CVaR95={cvar:+.2%}  Ret={ret:+.1%}"
        )
        if alpha != 0:
            line += f"  α={alpha:+.3f}"
        if wr > 0:
            line += f"\n  WinRate={wr:.1%}  PF={pf:.2f}"
        lines.append(line)

    if failures:
        lines += ["", "*Failures:*"] + [f"• {f}" for f in failures[:5]]

    if len(successes) >= 3:
        sharpes = [r["results"].get("sharpe", 0) for r in successes]
        calmars = [r["results"].get("calmar", 0) for r in successes]
        mdds    = [r["results"].get("max_drawdown", 0) for r in successes]
        omegas  = [r["results"].get("omega", 0) for r in successes if r["results"].get("omega", 0) < 1000]
        lines += [
            "", f"*Aggregate ({len(successes)} strategies):*",
            f"  Median Sharpe `{float(np.median(sharpes)):+.3f}`  "
            f"Best Calmar `{max(calmars):+.3f}`  Worst MDD `{min(mdds):+.1%}`",
            (f"  Beat Sharpe>1.0: `{sum(1 for s in sharpes if s > 1.0)}`  "
             f"Avg Omega: `{float(np.mean(omegas)):.2f}`") if omegas else "",
        ]

    _post_slack("\n".join(lines))


def main() -> None:
    print(f"QuantEdge Experiment Runner — {datetime.utcnow().isoformat()}Z", flush=True)
    print(f"STRATEGY_FILTER={STRATEGY_FILTER or '(all)'}  SYMBOL_FILTER={SYMBOL_FILTER or '(default)'}", flush=True)

    # Import pipeline tracker (lives in same scripts dir)
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))
    from pipeline_tracker import PipelineTracker, Stage, Status

    successes: list[dict] = []
    failures:  list[str]  = []

    with PipelineTracker("ml_experiments") as tracker:

        # ── Stage 1: data fetch (prefetch SPY for benchmark) ─────────────────
        with tracker.stage(Stage.DATA_FETCH, "Fetch benchmark data", channel="#squad-data"):
            spy = _fetch_spy_returns("2020-01-01", "2026-01-01")
            tracker.set_output(spy_bars=len(spy) if spy is not None else 0)

        # ── Stage 2: cache check ──────────────────────────────────────────────
        with tracker.stage(Stage.CACHE_CHECK, "Cache & data status", channel="#squad-data"):
            n_configs = len(EXPERIMENT_CONFIGS)
            tracker.set_output(n_strategies=n_configs, filter=STRATEGY_FILTER or "all")

        # ── Stage 3: backtesting (parallel via ThreadPoolExecutor) ───────────
        with tracker.stage(Stage.BACKTESTING, "Run backtest signals", channel="#ml-experiments"):
            import concurrent.futures as _cf
            MAX_WORKERS = min(8, len(EXPERIMENT_CONFIGS))

            def _run_cfg(cfg):
                name, module_path, class_name, symbol, interval, train_start, test_start, test_end, params = cfg
                return name, _run_one(name, module_path, class_name, symbol, interval,
                                      train_start, test_start, test_end, params)

            filtered = [
                cfg for cfg in EXPERIMENT_CONFIGS
                if (not STRATEGY_FILTER or STRATEGY_FILTER in cfg[0])
                and (not SYMBOL_FILTER or SYMBOL_FILTER.upper() in cfg[3])
            ]

            with _cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {pool.submit(_run_cfg, cfg): cfg for cfg in filtered}
                for fut in _cf.as_completed(futures):
                    cfg = futures[fut]
                    name = cfg[0]
                    try:
                        _, result = fut.result()
                        if result:
                            path = _save_result(result)
                            print(f"  → saved {path.name}", flush=True)
                            successes.append(result)
                    except Exception as exc:
                        failures.append(f"{name}: {exc}")
                        print(f"  ✗ UNHANDLED {name}: {exc}", flush=True)
            tracker.set_output(
                succeeded=len(successes),
                failed=len(failures),
                best_sharpe=max((r["results"].get("sharpe", 0) for r in successes), default=0),
            )

        # ── Stage 4: evaluation / Slack report ───────────────────────────────
        with tracker.stage(Stage.EVALUATION, "Compute metrics & post report", channel="#ml-experiments"):
            _post_slack_summary(successes, failures)
            tracker.set_output(n_results=len(successes))

    # ── Terminal summary ──────────────────────────────────────────────────────
    print(f"\n{'═'*60}", flush=True)
    print(f"Completed: {len(successes)} succeeded, {len(failures)} failed", flush=True)
    if not successes and not STRATEGY_FILTER:
        print("WARNING: no experiment results — check yfinance connectivity", flush=True)


if __name__ == "__main__":
    main()
