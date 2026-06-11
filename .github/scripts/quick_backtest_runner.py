"""
Quick Backtest Runner — runs every 15 minutes.
Standalone: no backend deps. Uses yfinance + numpy.
Runs lightweight signal backtests on crypto+equity symbols and posts results to Slack.
"""
from __future__ import annotations
import os, sys, json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests
import numpy as np

def _resolve_key(*names: str) -> str:
    for name in names:
        v = os.environ.get(name, "")
        if v: return v
        if not name[-1].isdigit():
            v = os.environ.get(name + "_1", "")
            if v: return v
    return ""

SLACK_TOKEN     = os.environ.get("SLACK_BOT_TOKEN", "")
ALLOW_PAID_APIS = os.environ.get("ALLOW_PAID_APIS", "False")

if ALLOW_PAID_APIS.lower() == "true":
    sys.exit(1)

STATE_FILE = Path(__file__).resolve().parents[2] / ".github" / "state" / "agent_memory.json"


def fetch_ohlcv(symbol: str, days: int = 365) -> dict | None:
    """Fetch daily OHLCV via yfinance (no auth required)."""
    try:
        import yfinance as yf
        tk = yf.Ticker(symbol)
        df = tk.history(period=f"{days}d", interval="1d")
        if df is None or len(df) < 30:
            return None
        return {
            "close": df["Close"].values.tolist(),
            "high":  df["High"].values.tolist(),
            "low":   df["Low"].values.tolist(),
            "volume": df["Volume"].values.tolist(),
        }
    except Exception as e:
        print(f"yfinance error {symbol}: {e}")
        return None


def fetch_crypto_ohlcv(symbol: str = "BTCUSDT", interval: str = "1d", limit: int = 365) -> dict | None:
    """Fetch Binance daily OHLCV (no auth, public endpoint)."""
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return {
            "close":  [float(c[4]) for c in data],
            "high":   [float(c[2]) for c in data],
            "low":    [float(c[3]) for c in data],
            "volume": [float(c[5]) for c in data],
        }
    except Exception as e:
        print(f"Binance OHLCV error {symbol}: {e}")
        return None


def compute_metrics(returns: list[float]) -> dict:
    """Compute Sharpe, Sortino, max drawdown, total return."""
    r = np.array(returns)
    if len(r) < 5:
        return {}
    total_return = float(np.prod(1 + r) - 1)
    ann_return   = float((1 + total_return) ** (252 / len(r)) - 1)
    vol          = float(np.std(r) * np.sqrt(252))
    sharpe       = float(ann_return / vol) if vol > 0 else 0.0
    downside     = r[r < 0]
    sortino_vol  = float(np.std(downside) * np.sqrt(252)) if len(downside) > 0 else 1e-9
    sortino      = float(ann_return / sortino_vol)
    # Max drawdown
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(dd.min())
    return {
        "total_return_pct": round(total_return * 100, 2),
        "ann_return_pct":   round(ann_return * 100, 2),
        "volatility_pct":   round(vol * 100, 2),
        "sharpe":           round(sharpe, 3),
        "sortino":          round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_days":           len(r),
    }


# ── Strategy implementations (pure numpy, no backend deps) ────────────────────

def backtest_momentum(ohlcv: dict, lookback: int = 20, skip: int = 1) -> list[float]:
    """12-1 momentum: rank by past returns, long if positive."""
    c = np.array(ohlcv["close"])
    signals, returns = [], []
    for i in range(lookback + skip, len(c)):
        past_ret = (c[i - skip] - c[i - lookback - skip]) / c[i - lookback - skip]
        signal = 1 if past_ret > 0 else -1
        daily_ret = (c[i] - c[i - 1]) / c[i - 1]
        returns.append(signal * daily_ret)
    return returns


def backtest_mean_reversion(ohlcv: dict, window: int = 20, z_thresh: float = 1.5) -> list[float]:
    """Bollinger Band mean reversion."""
    c = np.array(ohlcv["close"])
    returns = []
    for i in range(window, len(c)):
        window_c = c[i - window:i]
        mu = np.mean(window_c)
        sigma = np.std(window_c)
        if sigma < 1e-9:
            continue
        z = (c[i - 1] - mu) / sigma
        signal = -1 if z > z_thresh else (1 if z < -z_thresh else 0)
        daily_ret = (c[i] - c[i - 1]) / c[i - 1]
        returns.append(signal * daily_ret)
    return returns


def backtest_breakout(ohlcv: dict, window: int = 52) -> list[float]:
    """52-week high breakout: long when price breaks above recent high."""
    c = np.array(ohlcv["close"])
    h = np.array(ohlcv["high"])
    returns = []
    for i in range(window, len(c)):
        recent_high = np.max(h[i - window:i])
        signal = 1 if c[i - 1] > recent_high * 0.995 else 0
        daily_ret = (c[i] - c[i - 1]) / c[i - 1]
        returns.append(signal * daily_ret)
    return returns


def backtest_rsi(ohlcv: dict, period: int = 14, oversold: float = 30, overbought: float = 70) -> list[float]:
    """RSI mean reversion."""
    c = np.array(ohlcv["close"])
    returns = []
    for i in range(period + 1, len(c)):
        deltas = np.diff(c[i - period - 1:i])
        gains  = deltas[deltas > 0].sum() / period
        losses = -deltas[deltas < 0].sum() / period
        rs     = gains / losses if losses > 0 else 100
        rsi    = 100 - (100 / (1 + rs))
        signal = 1 if rsi < oversold else (-1 if rsi > overbought else 0)
        daily_ret = (c[i] - c[i - 1]) / c[i - 1]
        returns.append(signal * daily_ret)
    return returns


def backtest_funding_rate_fade(symbol_base: str = "BTC") -> list[float]:
    """Simulate funding rate fade: high positive funding → short bias. Uses cached data."""
    try:
        resp = requests.get(
            f"https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": f"{symbol_base}USDT", "limit": 200},
            timeout=10
        )
        if resp.status_code != 200:
            return []
        rates = [float(r["fundingRate"]) for r in resp.json()]
        returns = []
        for i in range(10, len(rates)):
            avg_funding = np.mean(rates[i - 10:i])
            annual_rate = avg_funding * 3 * 365
            if abs(annual_rate) > 0.20:
                signal = -1 if avg_funding > 0 else 1
                # Simulate 8h return as ±0.3% (rough proxy)
                simulated_return = signal * 0.003
                returns.append(simulated_return)
        return returns
    except Exception as e:
        print(f"Funding rate backtest error: {e}")
        return []


# ── Runner ───────────────────────────────────────────────────────────────────

STRATEGIES = {
    "momentum":       (backtest_momentum, "equity"),
    "mean_reversion": (backtest_mean_reversion, "equity"),
    "breakout":       (backtest_breakout, "equity"),
    "rsi":            (backtest_rsi, "equity"),
}

SYMBOLS = {
    "equity": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"],
    "crypto": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
}


def post_slack(channel: str, text: str) -> bool:
    if not SLACK_TOKEN:
        print(f"[Slack #{channel}]: {text[:300]}")
        return False
    try:
        resp = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_TOKEN}", "Content-Type": "application/json"},
            json={"channel": channel, "text": text, "mrkdwn": True},
            timeout=10
        )
        ok = resp.status_code == 200 and resp.json().get("ok")
        if not ok:
            print(f"Slack #{channel} error: {resp.json().get('error', 'unknown')}")
        return ok
    except Exception as e:
        print(f"Slack error: {e}")
        return False


def load_memory() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def save_memory(mem: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    mem["last_updated"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(mem, indent=2))


def main():
    now = datetime.now(timezone.utc)
    print(f"[{now.strftime('%H:%M UTC')}] Quick backtest runner")

    results = []

    # Equity strategies
    for strat_name, (strat_fn, desk) in STRATEGIES.items():
        for symbol in SYMBOLS["equity"][:3]:  # limit to 3 per run to stay under timeout
            ohlcv = fetch_ohlcv(symbol)
            if not ohlcv:
                continue
            try:
                rets = strat_fn(ohlcv)
                if not rets:
                    continue
                metrics = compute_metrics(rets)
                if not metrics:
                    continue
                results.append({
                    "strategy": strat_name,
                    "symbol": symbol,
                    "desk": desk,
                    **metrics
                })
                print(f"  {strat_name}/{symbol}: Sharpe={metrics['sharpe']:.2f} Return={metrics['total_return_pct']:.1f}%")
            except Exception as e:
                print(f"  Error {strat_name}/{symbol}: {e}")

    # Crypto strategies (Binance public OHLCV)
    for symbol in SYMBOLS["crypto"][:2]:
        ohlcv = fetch_crypto_ohlcv(symbol)
        if not ohlcv:
            continue
        for strat_name, (strat_fn, _) in list(STRATEGIES.items())[:2]:
            try:
                rets = strat_fn(ohlcv)
                if not rets:
                    continue
                metrics = compute_metrics(rets)
                if not metrics:
                    continue
                results.append({
                    "strategy": strat_name,
                    "symbol": symbol.replace("USDT", ""),
                    "desk": "crypto",
                    **metrics
                })
                print(f"  {strat_name}/{symbol}: Sharpe={metrics['sharpe']:.2f}")
            except Exception as e:
                print(f"  Error {strat_name}/{symbol}: {e}")

    if not results:
        print("No backtest results — data unavailable")
        return 0

    # Sort by Sharpe
    results.sort(key=lambda x: x.get("sharpe", 0), reverse=True)

    # Save to memory
    mem = load_memory()
    mem.setdefault("backtest_results", [])
    for r in results:
        r["timestamp"] = now.isoformat()
    mem["backtest_results"] = (mem["backtest_results"] + results)[-500:]
    save_memory(mem)

    # Build Slack report
    best = results[:5]
    lines = [f"*Backtest Report — {now.strftime('%H:%M UTC')} | {len(results)} runs across {len(set(r['desk'] for r in results))} desks*\n"]
    lines.append("*Top 5 by Sharpe Ratio*")
    lines.append("```")
    lines.append(f"{'Strategy':<22} {'Symbol':<8} {'Desk':<8} {'Sharpe':>6} {'Return%':>8} {'MaxDD%':>8}")
    lines.append("-" * 64)
    for r in best:
        lines.append(
            f"{r['strategy']:<22} {r['symbol']:<8} {r['desk']:<8} "
            f"{r['sharpe']:>6.2f} {r['total_return_pct']:>8.1f} {r['max_drawdown_pct']:>8.1f}"
        )
    lines.append("```")

    # Highlight any Sharpe > 1.5
    stars = [r for r in results if r.get("sharpe", 0) > 1.5]
    if stars:
        lines.append(f"\n:star: *{len(stars)} strategies with Sharpe > 1.5*: " + ", ".join(f"{r['strategy']}/{r['symbol']}" for r in stars[:5]))

    msg = "\n".join(lines)
    post_slack("signals", msg)

    # Save summary JSON
    summary = {
        "timestamp": now.isoformat(),
        "total_runs": len(results),
        "top_sharpe": results[0]["sharpe"] if results else 0,
        "desks": list(set(r["desk"] for r in results)),
        "strategies": list(set(r["strategy"] for r in results)),
    }
    import json as _json
    with open("/tmp/quick_backtest_summary.json", "w") as f:
        _json.dump(summary, f, indent=2)

    print(f"✓ {len(results)} backtests | top Sharpe: {results[0]['sharpe']:.2f} ({results[0]['strategy']}/{results[0]['symbol']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
