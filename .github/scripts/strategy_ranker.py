"""
Strategy Health Ranker — Sprint 2 of the 60-day plan.

Runs daily via GitHub Actions. For each registered strategy × symbol:
  1. Fetches 2 years of daily OHLCV from yfinance
  2. Computes signals via strategy.backtest_signals()
  3. Runs backtest → Sharpe, drawdown, win rate
  4. Ranks all strategies and posts a leaderboard to Slack

Output:
  - experiments/results/strategy_ranking_YYYYMMDD.json
  - Slack #pnl-daily post with top 10 / bottom 5

Actions trigger:
  - cron daily after market close
  - workflow_dispatch (manual)
"""
from __future__ import annotations
import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import requests
except ImportError as e:
    sys.exit(f"Missing dependency: {e}. Run: pip install numpy pandas yfinance requests")


SLACK_BOT_TOKEN  = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL    = "#pnl-daily"
MIN_SHARPE_PAPER = float(os.environ.get("MIN_SHARPE_PAPER", "0.7"))  # below this → flagged
TOP_N            = int(os.environ.get("TOP_N", "10"))

# Default strategy × symbol pairs to rank
DEFAULT_PAIRS = [
    ("momentum",        "SPY",  "1d"),
    ("mean_reversion",  "SPY",  "1d"),
    ("rsi_macd",        "SPY",  "1d"),
    ("breakout",        "SPY",  "1d"),
    ("pairs_trading",   "SPY",  "1d"),
    ("supertrend",      "QQQ",  "1d"),
    ("low_volatility",  "VTI",  "1d"),
    ("momentum",        "QQQ",  "1d"),
    ("mean_reversion",  "QQQ",  "1d"),
    ("rsi_macd",        "BTC-USD", "1d"),
    ("breakout",        "ETH-USD", "1d"),
]


# ── Simple vectorised backtest ─────────────────────────────────────────────────

def _quick_backtest(signals: pd.Series, prices: pd.Series) -> dict:
    """Minimal backtest — returns sharpe, max_dd, win_rate, total_return."""
    df = pd.DataFrame({"sig": signals, "px": prices}).dropna()
    df["pos"] = df["sig"].replace(0, np.nan).ffill().fillna(0).shift(1).fillna(0)
    df["ret"] = df["pos"] * df["px"].pct_change().fillna(0)
    df["eq"]  = (1 + df["ret"]).cumprod()

    rets = df["ret"].values
    rf_daily = 0.05 / 252
    excess = rets - rf_daily
    std = rets.std()
    sharpe = float(excess.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    peak = np.maximum.accumulate(df["eq"].values)
    dd = (df["eq"].values - peak) / peak
    max_dd = float(dd.min())

    # Trade-level win rate
    trade_rets = df["ret"][df["pos"].diff() != 0].values
    wins = (trade_rets > 0).sum()
    win_rate = float(wins / max(len(trade_rets), 1))

    total_return = float(df["eq"].iloc[-1] - 1.0) if len(df) > 1 else 0.0

    return {
        "sharpe":       round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate":     round(win_rate, 4),
        "total_return": round(total_return, 4),
    }


def _fetch_price(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df is None or len(df) < 60:
            return None
        return df
    except Exception:
        return None


def _run_strategy(strategy_name: str, df: pd.DataFrame) -> pd.Series | None:
    """Try to instantiate the strategy and call backtest_signals()."""
    try:
        from app.strategies import STRATEGY_REGISTRY
        cls = STRATEGY_REGISTRY.get(strategy_name)
        if cls is None:
            return None
        inst = cls(symbol="tmp", account_id="tmp")
        signals = inst.backtest_signals(df)
        if hasattr(signals, "values"):
            return signals
        return pd.Series(signals, index=df.index)
    except Exception as e:
        print(f"  Strategy {strategy_name} failed: {e}", file=sys.stderr)
        return None


def _post_slack(text: str) -> None:
    if not SLACK_BOT_TOKEN:
        print(f"[SLACK dry-run]\n{text}")
        return
    resp = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}", "Content-Type": "application/json"},
        json={"channel": SLACK_CHANNEL, "text": text},
        timeout=10,
    )
    d = resp.json()
    if not d.get("ok"):
        print(f"Slack error: {d.get('error')}", file=sys.stderr)


def main() -> None:
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    results: list[dict] = []

    pairs = DEFAULT_PAIRS
    override = os.environ.get("STRATEGY")
    if override:
        pairs = [(override, "SPY", "1d")]

    print(f"Ranking {len(pairs)} strategy-symbol pairs...")

    for strategy_name, symbol, interval in pairs:
        print(f"  {strategy_name} / {symbol}...", end=" ", flush=True)
        t0 = time.perf_counter()

        df = _fetch_price(symbol, period="2y", interval=interval)
        if df is None:
            print("no data")
            continue

        close = df["Close"]
        if hasattr(close, "squeeze"):
            close = close.squeeze()

        signals = _run_strategy(strategy_name, df)
        if signals is None:
            print("strategy error")
            continue

        # Align
        signals = signals.reindex(close.index).fillna(0)
        metrics = _quick_backtest(signals, close)
        elapsed = round((time.perf_counter() - t0) * 1000)
        print(f"Sharpe={metrics['sharpe']:.2f} ({elapsed}ms)")

        results.append({
            "strategy": strategy_name,
            "symbol": symbol,
            "interval": interval,
            **metrics,
        })

    if not results:
        print("No results — nothing to post")
        return

    # Sort by Sharpe descending
    results.sort(key=lambda r: r["sharpe"], reverse=True)

    # Save JSON
    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"strategy_ranking_{today_str}.json"
    out_path.write_text(json.dumps({"date": today_str, "rankings": results}, indent=2))
    print(f"\nSaved to {out_path}")

    # Build Slack message
    date_str = datetime.now(timezone.utc).strftime("%a %b %-d %Y")
    lines = [f"*Strategy Health Ranking — {date_str}*\n*Top {min(TOP_N, len(results))} strategies:*"]
    for i, r in enumerate(results[:TOP_N], 1):
        flag = " :warning:" if r["sharpe"] < MIN_SHARPE_PAPER else ""
        lines.append(
            f"  {i}. `{r['strategy']} / {r['symbol']}` — "
            f"Sharpe *{r['sharpe']:.2f}*, MaxDD {r['max_drawdown']*100:.1f}%, "
            f"WinRate {r['win_rate']*100:.0f}%{flag}"
        )

    # Flag underperformers
    underperformers = [r for r in results if r["sharpe"] < MIN_SHARPE_PAPER]
    if underperformers:
        lines.append(f"\n:x: *{len(underperformers)} strategies below Sharpe {MIN_SHARPE_PAPER} threshold:*")
        for r in underperformers[-5:]:
            lines.append(f"  • `{r['strategy']} / {r['symbol']}` — Sharpe {r['sharpe']:.2f}")

    _post_slack("\n".join(lines))
    print("Posted to Slack")


if __name__ == "__main__":
    main()
