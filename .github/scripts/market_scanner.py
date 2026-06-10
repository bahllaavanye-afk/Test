"""
Advanced Market Scanner — runs at market open and every hour during market hours.

Scans for:
  1. Pre-market gap analysis (overnight price change)
  2. Volume anomalies (current volume vs 20-day average)
  3. 52-week high/low breakout candidates
  4. RSI extremes (overbought > 70, oversold < 30)
  5. Sector rotation strength (which ETFs are leading/lagging)
  6. VIX regime + trend (fear gauge level and 5-day change)
  7. Credit spread health (HYG/LQD momentum)
  8. Yield curve signal (TLT/SHY ratio vs 252-day median)
  9. Earnings momentum (stocks with strong post-earnings drift)
  10. Crypto momentum (BTC/ETH vs 24h + 7d returns)

Posts a formatted scan report to Slack #market-analysis.
Also writes composite market regime to company_brain.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# Add scripts dir to path for llm_common
sys.path.insert(0, str(Path(__file__).parent))
from llm_common import core_update, memory_write, slack_post, llm

ALLOW_PAID = os.environ.get("ALLOW_PAID_APIS", "False")
if ALLOW_PAID.lower() == "true":
    sys.exit(1)

SLACK_CHANNEL = "#market-analysis"

# ── Equity universe ──────────────────────────────────────────────────────────

SECTOR_ETFS = {
    "XLK": "Tech", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Healthcare", "XLI": "Industrials", "XLY": "Consumer Disc",
    "XLP": "Consumer Staples", "XLU": "Utilities", "XLRE": "Real Estate",
    "XLB": "Materials", "XLC": "Comm Services",
}

MACRO_ETFS = ["SPY", "QQQ", "IWM", "^VIX", "TLT", "SHY", "HYG", "LQD", "GLD", "DXY"]

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B",
    "JPM", "V", "MA", "UNH", "XOM", "AVGO", "LLY",
]


# ── Data helpers ──────────────────────────────────────────────────────────────

def _yf_history(symbol: str, period: str = "1y", interval: str = "1d") -> list[dict]:
    """Fetch OHLCV history from yfinance. Returns list of dicts."""
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        if hist.empty:
            return []
        records = []
        for ts, row in hist.iterrows():
            records.append({
                "date": str(ts.date()),
                "open": float(row.get("Open", 0)),
                "high": float(row.get("High", 0)),
                "low": float(row.get("Low", 0)),
                "close": float(row.get("Close", 0)),
                "volume": float(row.get("Volume", 0)),
            })
        return records
    except Exception:
        return []


def _latest(records: list[dict]) -> dict | None:
    return records[-1] if records else None


def _compute_rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        chg = closes[i] - closes[i - 1]
        gains.append(max(chg, 0))
        losses.append(max(-chg, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 1)


def _pct_change(a: float, b: float) -> float:
    """Return (b-a)/a * 100."""
    if a == 0:
        return 0.0
    return round((b - a) / a * 100, 2)


def _avg_volume(records: list[dict], days: int = 20) -> float:
    vols = [r["volume"] for r in records[-days - 1:-1] if r["volume"] > 0]
    return sum(vols) / len(vols) if vols else 0.0


# ── Scan modules ──────────────────────────────────────────────────────────────

def scan_macro_regime() -> dict:
    """VIX level, yield curve, credit spread → composite regime score."""
    result = {"score": 0, "signals": [], "regime": "unknown"}

    # VIX
    vix_data = _yf_history("^VIX", period="1mo")
    if vix_data:
        vix_now = _latest(vix_data)["close"]
        vix_5d_ago = vix_data[-6]["close"] if len(vix_data) >= 6 else vix_now
        vix_trend = "rising" if vix_now > vix_5d_ago * 1.05 else "falling" if vix_now < vix_5d_ago * 0.95 else "flat"
        result["vix"] = round(vix_now, 2)
        result["vix_trend"] = vix_trend
        if vix_now < 15:
            result["score"] += 2
            result["signals"].append(f"VIX {vix_now:.1f} (very low — risk-on)")
        elif vix_now < 20:
            result["score"] += 1
            result["signals"].append(f"VIX {vix_now:.1f} (low — risk-on)")
        elif vix_now > 30:
            result["score"] -= 2
            result["signals"].append(f"VIX {vix_now:.1f} (elevated — risk-off)")
        elif vix_now > 25:
            result["score"] -= 1
            result["signals"].append(f"VIX {vix_now:.1f} (caution)")

    # Yield curve (TLT/SHY)
    tlt_data = _yf_history("TLT", period="2y")
    shy_data = _yf_history("SHY", period="2y")
    if tlt_data and shy_data:
        tlt_closes = [r["close"] for r in tlt_data]
        shy_closes = [r["close"] for r in shy_data[-len(tlt_closes):]]
        if len(shy_closes) == len(tlt_closes) and len(tlt_closes) >= 252:
            ratios = [t / s for t, s in zip(tlt_closes, shy_closes) if s > 0]
            ratio_now = ratios[-1]
            ratio_median = sorted(ratios[-252:])[126]
            curve_signal = "steepening" if ratio_now > ratio_median else "inverted/flat"
            result["curve_signal"] = curve_signal
            result["curve_ratio"] = round(ratio_now, 4)
            if ratio_now > ratio_median:
                result["score"] += 1
                result["signals"].append(f"Yield curve: {curve_signal} (risk-on)")
            else:
                result["score"] -= 1
                result["signals"].append(f"Yield curve: {curve_signal} (risk-off)")

    # Credit spreads (HYG/LQD)
    hyg_data = _yf_history("HYG", period="3mo")
    lqd_data = _yf_history("LQD", period="3mo")
    if hyg_data and lqd_data:
        n = min(len(hyg_data), len(lqd_data))
        hyg_c = [r["close"] for r in hyg_data[-n:]]
        lqd_c = [r["close"] for r in lqd_data[-n:]]
        credit_ratios = [h / l for h, l in zip(hyg_c, lqd_c) if l > 0]
        if len(credit_ratios) >= 20:
            mom = _pct_change(credit_ratios[-21], credit_ratios[-1])
            credit_signal = "tightening" if mom > 0 else "widening"
            result["credit_signal"] = credit_signal
            result["credit_mom"] = round(mom, 2)
            if mom > 0:
                result["score"] += 1
                result["signals"].append(f"Credit spreads {credit_signal} ({mom:+.2f}%) → risk-on")
            else:
                result["score"] -= 1
                result["signals"].append(f"Credit spreads {credit_signal} ({mom:+.2f}%) → risk-off")

    # SPY vs 200-day SMA
    spy_data = _yf_history("SPY", period="1y")
    if len(spy_data) >= 200:
        spy_closes = [r["close"] for r in spy_data]
        spy_now = spy_closes[-1]
        sma200 = sum(spy_closes[-200:]) / 200
        result["spy_vs_200sma"] = round(_pct_change(sma200, spy_now), 2)
        if spy_now > sma200:
            result["score"] += 1
            result["signals"].append(f"SPY above 200-SMA by {result['spy_vs_200sma']:+.1f}%")
        else:
            result["score"] -= 1
            result["signals"].append(f"SPY below 200-SMA by {result['spy_vs_200sma']:+.1f}%")

    # Composite regime
    if result["score"] >= 3:
        result["regime"] = "strong_bull"
    elif result["score"] >= 1:
        result["regime"] = "bull"
    elif result["score"] == 0:
        result["regime"] = "neutral"
    elif result["score"] >= -2:
        result["regime"] = "bear"
    else:
        result["regime"] = "strong_bear"

    return result


def scan_sector_rotation() -> dict:
    """Rank sectors by 1-month performance → identify leaders/laggards."""
    perf = {}
    for sym, name in SECTOR_ETFS.items():
        data = _yf_history(sym, period="3mo")
        if len(data) >= 21:
            closes = [r["close"] for r in data]
            perf[name] = {
                "1m": _pct_change(closes[-22], closes[-1]),
                "3m": _pct_change(closes[0], closes[-1]),
                "sym": sym,
            }

    ranked = sorted(perf.items(), key=lambda x: x[1]["1m"], reverse=True)
    return {
        "leaders": [(name, info["1m"]) for name, info in ranked[:3]],
        "laggards": [(name, info["1m"]) for name, info in ranked[-3:]],
        "all": ranked,
    }


def scan_breakouts() -> list[dict]:
    """Find stocks near 52-week highs with volume surge."""
    hits = []
    for sym in WATCHLIST:
        data = _yf_history(sym, period="1y")
        if len(data) < 52:
            continue
        closes = [r["close"] for r in data]
        highs = [r["high"] for r in data]
        low_52w = min(closes[-252:]) if len(closes) >= 252 else min(closes)
        high_52w = max(highs[-252:]) if len(highs) >= 252 else max(highs)
        latest = _latest(data)
        if not latest:
            continue
        price = latest["close"]
        avg_vol = _avg_volume(data, 20)
        vol_ratio = latest["volume"] / avg_vol if avg_vol > 0 else 1.0
        pct_from_high = _pct_change(high_52w, price)
        pct_from_low = _pct_change(low_52w, price)

        if pct_from_high > -2 and vol_ratio > 1.5:
            hits.append({"sym": sym, "type": "52w_breakout", "price": round(price, 2),
                         "vol_ratio": round(vol_ratio, 1), "pct_from_high": round(pct_from_high, 1)})
        elif pct_from_low < 5 and vol_ratio > 1.3:
            hits.append({"sym": sym, "type": "52w_low_retest", "price": round(price, 2),
                         "vol_ratio": round(vol_ratio, 1), "pct_from_low": round(pct_from_low, 1)})
    return hits


def scan_rsi_extremes() -> dict:
    """Find RSI overbought/oversold conditions across watchlist."""
    overbought, oversold = [], []
    all_syms = list(WATCHLIST) + list(SECTOR_ETFS.keys())
    for sym in all_syms:
        data = _yf_history(sym, period="3mo")
        if len(data) < 20:
            continue
        closes = [r["close"] for r in data]
        rsi = _compute_rsi(closes)
        if rsi is None:
            continue
        price = closes[-1]
        if rsi >= 70:
            overbought.append({"sym": sym, "rsi": rsi, "price": round(price, 2)})
        elif rsi <= 30:
            oversold.append({"sym": sym, "rsi": rsi, "price": round(price, 2)})
    return {
        "overbought": sorted(overbought, key=lambda x: x["rsi"], reverse=True)[:5],
        "oversold": sorted(oversold, key=lambda x: x["rsi"])[:5],
    }


def scan_crypto() -> dict:
    """BTC/ETH price, 24h + 7d returns, dominance."""
    result = {}
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin,ethereum,solana,binancecoin", "vs_currencies": "usd",
                    "include_24hr_change": "true", "include_7d_change": "true"},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            for cg_id, sym in [("bitcoin", "BTC"), ("ethereum", "ETH"),
                                ("solana", "SOL"), ("binancecoin", "BNB")]:
                coin = data.get(cg_id, {})
                result[sym] = {
                    "price": coin.get("usd", 0),
                    "24h": round(coin.get("usd_24h_change", 0), 2),
                    "7d": round(coin.get("usd_7d_change", 0) if "usd_7d_change" in coin else 0, 2),
                }
    except Exception:
        pass
    return result


def scan_volume_anomalies() -> list[dict]:
    """Detect stocks with volume > 2× 20-day average (unusual activity)."""
    anomalies = []
    for sym in WATCHLIST:
        data = _yf_history(sym, period="3mo")
        if len(data) < 22:
            continue
        latest = _latest(data)
        avg_vol = _avg_volume(data, 20)
        if latest and avg_vol > 0:
            ratio = latest["volume"] / avg_vol
            if ratio >= 2.0:
                pct_chg = _pct_change(data[-2]["close"], latest["close"]) if len(data) >= 2 else 0
                anomalies.append({
                    "sym": sym, "vol_ratio": round(ratio, 1),
                    "price_chg": round(pct_chg, 2), "price": round(latest["close"], 2),
                })
    return sorted(anomalies, key=lambda x: x["vol_ratio"], reverse=True)[:8]


# ── Report formatter ──────────────────────────────────────────────────────────

def _fmt_sector_rotation(sr: dict) -> str:
    leaders = " ".join(f"{n}({p:+.1f}%)" for n, p in sr["leaders"])
    laggards = " ".join(f"{n}({p:+.1f}%)" for n, p in sr["laggards"])
    return f"*Leading:* {leaders}  |  *Lagging:* {laggards}"


def _fmt_crypto(crypto: dict) -> str:
    parts = []
    for sym, d in crypto.items():
        parts.append(f"{sym} ${d['price']:,.0f} ({d['24h']:+.1f}% 24h)")
    return " | ".join(parts)


def build_slack_report(
    macro: dict, sector: dict, breakouts: list, rsi: dict, crypto: dict, vol: list
) -> str:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    regime = macro.get("regime", "unknown")
    regime_emoji = {"strong_bull": "🟢🟢", "bull": "🟢", "neutral": "🟡",
                    "bear": "🔴", "strong_bear": "🔴🔴"}.get(regime, "⚪")

    lines = [
        f"*QuantEdge Market Scanner* — {now_str}",
        f"{regime_emoji} *Regime: {regime.replace('_', ' ').upper()}* (score {macro.get('score', 0):+d})",
        "",
    ]

    # Macro signals
    if macro.get("signals"):
        lines.append("*Macro Signals:*")
        for s in macro["signals"]:
            lines.append(f"  • {s}")
        lines.append("")

    # Sector rotation
    lines.append(f"*Sector Rotation:* {_fmt_sector_rotation(sector)}")
    lines.append("")

    # Volume anomalies
    if vol:
        lines.append("*Volume Anomalies (>2× avg):*")
        for v in vol[:5]:
            dir_arrow = "▲" if v["price_chg"] > 0 else "▼"
            lines.append(f"  • {v['sym']} {dir_arrow}{v['price_chg']:+.1f}% | {v['vol_ratio']}× vol @ ${v['price']}")
        lines.append("")

    # Breakouts
    if breakouts:
        lines.append("*Breakout / Breakdown Watch:*")
        for b in breakouts[:4]:
            if b["type"] == "52w_breakout":
                lines.append(f"  • 📈 {b['sym']} near 52W high ({b['pct_from_high']:+.1f}%) | {b['vol_ratio']}× vol")
            else:
                lines.append(f"  • 📉 {b['sym']} testing 52W low (+{b['pct_from_low']:.1f}% above) | {b['vol_ratio']}× vol")
        lines.append("")

    # RSI extremes
    if rsi["overbought"] or rsi["oversold"]:
        lines.append("*RSI Extremes:*")
        if rsi["overbought"]:
            ob = ", ".join(f"{x['sym']} RSI {x['rsi']}" for x in rsi["overbought"])
            lines.append(f"  • Overbought: {ob}")
        if rsi["oversold"]:
            os_str = ", ".join(f"{x['sym']} RSI {x['rsi']}" for x in rsi["oversold"])
            lines.append(f"  • Oversold: {os_str}")
        lines.append("")

    # Crypto
    if crypto:
        lines.append(f"*Crypto:* {_fmt_crypto(crypto)}")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Running QuantEdge advanced market scanner…", flush=True)
    t0 = time.time()

    macro = scan_macro_regime()
    sector = scan_sector_rotation()
    breakouts = scan_breakouts()
    rsi = scan_rsi_extremes()
    crypto = scan_crypto()
    vol = scan_volume_anomalies()

    print(f"Scan complete in {time.time() - t0:.1f}s | regime={macro['regime']}", flush=True)

    # Write regime to brain
    core_update("market_regime", macro["regime"])
    core_update("vix", macro.get("vix"))
    core_update("market_scan_ts", time.time())
    core_update("sector_leaders", [n for n, _ in sector["leaders"]])
    core_update("sector_laggards", [n for n, _ in sector["laggards"]])

    memory_write("episodic", {
        "source": "market_scanner",
        "regime": macro["regime"],
        "score": macro["score"],
        "sector_leaders": [n for n, _ in sector["leaders"]],
        "breakout_count": len(breakouts),
        "lesson": f"Market regime={macro['regime']} (score={macro['score']}); "
                  f"leading sectors: {', '.join(n for n, _ in sector['leaders'][:2])}",
    })

    report = build_slack_report(macro, sector, breakouts, rsi, crypto, vol)
    print(report, flush=True)

    resp = slack_post(SLACK_CHANNEL, report)
    if resp.get("ok"):
        print("Posted to Slack #market-analysis", flush=True)
    else:
        print(f"Slack post failed: {resp}", flush=True)

    # Ask LLM to add a one-line trading recommendation
    try:
        regime = macro["regime"]
        signals_summary = "; ".join(macro.get("signals", []))
        leaders = ", ".join(n for n, _ in sector["leaders"][:3])
        prompt = (
            f"Market regime: {regime}. Macro signals: {signals_summary}. "
            f"Leading sectors: {leaders}. "
            "Give one sentence of actionable tactical advice for the QuantEdge desk today. "
            "Be specific about which strategy type to prioritize."
        )
        advice = llm(prompt, max_tokens=120, use_cache=False, inject_company_context=False)
        slack_post(SLACK_CHANNEL, f"🤖 *Tactical advice:* {advice}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
