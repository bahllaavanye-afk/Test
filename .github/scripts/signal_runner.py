"""
Signal Runner — every 5 minutes across all desks.
Reads live prices (Binance public API for crypto, yfinance for equity).
Runs all strategy signal logic without executing orders (paper mode).
Posts signals + P&L summary to Slack #signals channel.
"""
from __future__ import annotations
import os, sys, json, importlib.util, glob
from datetime import datetime, timezone
from pathlib import Path
import requests

def _resolve_key(*names: str) -> str:
    for name in names:
        v = os.environ.get(name, "")
        if v: return v
        if not name[-1].isdigit():
            v = os.environ.get(name + "_1", "")
            if v: return v
    return ""

SLACK_TOKEN    = os.environ.get("SLACK_BOT_TOKEN", "")
GROQ_KEY       = _resolve_key("GROQ_API_KEY")
DEEPSEEK_KEYS  = [k for k in [
    _resolve_key("DEEPSEEK_API_KEY"),
    os.environ.get("DEEPSEEK_API_KEY_2", ""),
    os.environ.get("DEEPSEEK_API_KEY_3", ""),
] if k]
SAMBANOVA_KEY  = _resolve_key("SAMBANOVA_API_KEY")
CEREBRAS_KEY   = _resolve_key("CEREBRAS_API_KEY")
HYPERBOLIC_KEY = _resolve_key("HYPERBOLIC_API_KEY")
TOGETHER_KEY   = _resolve_key("TOGETHER_API_KEY")
GEMINI_KEY     = _resolve_key("GEMINI_API_KEY")
ALLOW_PAID_APIS = os.environ.get("ALLOW_PAID_APIS", "False")

if ALLOW_PAID_APIS.lower() == "true":
    sys.exit(1)

STATE_FILE = Path(__file__).resolve().parents[2] / ".github" / "state" / "agent_memory.json"

# ── Price feeds (public, no auth) ─────────────────────────────────────────────

def get_crypto_prices() -> dict[str, float]:
    """Binance public REST — no API key needed."""
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
               "ADAUSDT", "AVAXUSDT", "DOGEUSDT"]
    prices = {}
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbols": json.dumps(symbols)},
            timeout=10
        )
        if resp.status_code == 200:
            for item in resp.json():
                prices[item["symbol"].replace("USDT", "")] = float(item["price"])
    except Exception as e:
        print(f"Binance price fetch error: {e}")
    return prices

def get_equity_prices() -> dict[str, float]:
    """yfinance for equity prices."""
    tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN"]
    prices = {}
    try:
        import yfinance as yf
        data = yf.download(tickers, period="1d", interval="5m", progress=False)
        if "Close" in data.columns:
            last = data["Close"].iloc[-1]
            for t in tickers:
                if t in last.index and not str(last[t]) == "nan":
                    prices[t] = float(last[t])
    except Exception as e:
        print(f"yfinance error (expected if not installed): {e}")
    return prices

def get_funding_rates() -> dict[str, float]:
    """Binance futures funding rates — crypto desk signal."""
    rates = {}
    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            timeout=10
        )
        if resp.status_code == 200:
            for item in resp.json():
                sym = item.get("symbol", "")
                if sym.endswith("USDT") and item.get("lastFundingRate"):
                    rates[sym.replace("USDT", "")] = float(item["lastFundingRate"])
    except Exception as e:
        print(f"Funding rate error: {e}")
    return rates

# ── Simple signal generators (no ML, fast) ────────────────────────────────────

def funding_rate_signal(rates: dict[str, float]) -> list[dict]:
    """High positive funding rate = crowded long = short signal (fade the crowd)."""
    signals = []
    for sym, rate in sorted(rates.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        annual_rate = rate * 3 * 365 * 100  # 8h rate → annualized %
        if abs(annual_rate) > 20:
            direction = "SHORT" if rate > 0 else "LONG"
            signals.append({
                "strategy": "funding_rate_arb",
                "desk": "crypto",
                "symbol": sym,
                "direction": direction,
                "strength": min(100, int(abs(annual_rate) / 2)),
                "reason": f"Funding {annual_rate:+.1f}% annualized → fade via {direction}",
            })
    return signals

def momentum_signal(prices: dict[str, float]) -> list[dict]:
    """Crypto 24h momentum using Binance ticker stats."""
    signals = []
    try:
        resp = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=10)
        if resp.status_code != 200:
            return signals
        for item in resp.json():
            sym = item.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            chg = float(item.get("priceChangePercent", 0))
            vol_usd = float(item.get("quoteVolume", 0))
            if vol_usd < 10_000_000:  # min $10M volume
                continue
            base = sym.replace("USDT", "")
            if abs(chg) > 5:
                signals.append({
                    "strategy": "momentum",
                    "desk": "crypto",
                    "symbol": base,
                    "direction": "LONG" if chg > 0 else "SHORT",
                    "strength": min(100, int(abs(chg) * 5)),
                    "reason": f"{chg:+.1f}% 24h change, ${vol_usd/1e6:.0f}M volume",
                })
    except Exception as e:
        print(f"Momentum signal error: {e}")
    return signals[:3]

def stat_arb_signal() -> list[dict]:
    """BTC-ETH spread signal from Binance."""
    signals = []
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price",
                         params={"symbols": '["BTCUSDT","ETHUSDT"]'}, timeout=8)
        if r.status_code != 200:
            return signals
        prices = {d["symbol"]: float(d["price"]) for d in r.json()}
        btc = prices.get("BTCUSDT", 0)
        eth = prices.get("ETHUSDT", 0)
        if btc and eth:
            ratio = btc / eth
            # Historical average ~15-20
            if ratio > 25:
                signals.append({
                    "strategy": "btc_eth_stat_arb", "desk": "crypto",
                    "symbol": "BTC/ETH", "direction": "SHORT BTC / LONG ETH",
                    "strength": 75, "reason": f"BTC/ETH ratio {ratio:.1f} > historical ~18",
                })
            elif ratio < 12:
                signals.append({
                    "strategy": "btc_eth_stat_arb", "desk": "crypto",
                    "symbol": "BTC/ETH", "direction": "LONG BTC / SHORT ETH",
                    "strength": 75, "reason": f"BTC/ETH ratio {ratio:.1f} < historical ~18",
                })
    except Exception as e:
        print(f"Stat arb error: {e}")
    return signals

def polymarket_arb_signal() -> list[dict]:
    """Fetch Polymarket markets and flag YES+NO < $0.97."""
    signals = []
    try:
        resp = requests.get(
            "https://clob.polymarket.com/markets",
            params={"active": "true", "limit": 50},
            timeout=10
        )
        if resp.status_code != 200:
            return signals
        markets = resp.json().get("data", resp.json() if isinstance(resp.json(), list) else [])
        for mkt in markets[:20]:
            tokens = mkt.get("tokens", [])
            if len(tokens) == 2:
                try:
                    yes_price = float(next(t["price"] for t in tokens if t["outcome"] == "Yes"))
                    no_price  = float(next(t["price"] for t in tokens if t["outcome"] == "No"))
                    total = yes_price + no_price
                    if total < 0.97 and total > 0.5:
                        signals.append({
                            "strategy": "poly_binary_arb", "desk": "polymarket",
                            "symbol": mkt.get("question", "")[:40],
                            "direction": "BUY BOTH",
                            "strength": int((1 - total) * 1000),
                            "reason": f"YES {yes_price:.2f} + NO {no_price:.2f} = {total:.2f} (<0.97)",
                        })
                except (StopIteration, ValueError, KeyError):
                    pass
    except Exception as e:
        print(f"Polymarket error: {e}")
    return signals[:3]

def load_memory() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}

def save_memory(mem: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    mem["last_updated"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(mem, indent=2))

def post_slack(channel: str, text: str) -> bool:
    if not SLACK_TOKEN:
        print(f"[Slack #{channel}]: {text[:200]}")
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

def main():
    now = datetime.now(timezone.utc)
    print(f"[{now.strftime('%H:%M UTC')}] Signal runner — all desks")

    # Gather market data
    crypto_prices = get_crypto_prices()
    equity_prices = get_equity_prices()
    funding_rates = get_funding_rates()

    print(f"  Crypto prices: {len(crypto_prices)} symbols")
    print(f"  Equity prices: {len(equity_prices)} symbols")
    print(f"  Funding rates: {len(funding_rates)} symbols")

    # Generate signals across all desks
    all_signals = []
    all_signals.extend(funding_rate_signal(funding_rates))
    all_signals.extend(momentum_signal(crypto_prices))
    all_signals.extend(stat_arb_signal())
    all_signals.extend(polymarket_arb_signal())

    print(f"  Signals generated: {len(all_signals)}")

    # Save to memory
    mem = load_memory()
    mem.setdefault("signals", [])
    for sig in all_signals:
        sig["timestamp"] = now.isoformat()
    mem["signals"] = (mem["signals"] + all_signals)[-200:]  # keep last 200
    mem["platform_metrics"] = mem.get("platform_metrics", {})
    mem["platform_metrics"]["last_signal_run"] = now.isoformat()
    mem["platform_metrics"]["signal_count_today"] = len([
        s for s in mem["signals"]
        if s.get("timestamp", "")[:10] == now.strftime("%Y-%m-%d")
    ])
    save_memory(mem)

    # Post to Slack
    if all_signals:
        by_desk = {}
        for sig in all_signals:
            desk = sig.get("desk", "unknown")
            by_desk.setdefault(desk, []).append(sig)

        lines = [f"*Signal Report — {now.strftime('%H:%M UTC')} | {len(all_signals)} signals across {len(by_desk)} desks*\n"]
        for desk, sigs in sorted(by_desk.items()):
            lines.append(f"*{desk.upper()} DESK* ({len(sigs)} signals)")
            for s in sigs[:3]:
                strength_bar = "█" * (s["strength"] // 20) + "░" * (5 - s["strength"] // 20)
                lines.append(f"  `{s['strategy']:<25}` {s['symbol']:<15} {s['direction']:<20} [{strength_bar}] {s['strength']}%")
                lines.append(f"    ↳ {s['reason']}")
            lines.append("")

        msg = "\n".join(lines)
        post_slack("signals", msg)
        post_slack("trading", msg[:500] + "..." if len(msg) > 500 else msg)
    else:
        post_slack("signals", f"*{now.strftime('%H:%M UTC')}* — No high-confidence signals across any desk. Markets stable.")

    # Summary
    summary = {
        "timestamp": now.isoformat(),
        "crypto_symbols": len(crypto_prices),
        "equity_symbols": len(equity_prices),
        "signals": len(all_signals),
        "by_desk": {k: len(v) for k, v in {
            s["desk"]: [x for x in all_signals if x["desk"] == s["desk"]]
            for s in all_signals
        }.items()},
    }
    with open("/tmp/signal_runner_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ {len(all_signals)} signals | desks: {list(by_desk.keys()) if all_signals else []}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
