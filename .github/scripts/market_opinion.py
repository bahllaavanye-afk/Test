"""
Daily Market Opinion — runs at 09:00 ET (14:00 UTC) on trading days.

Builds a structured daily market view used as a gate by all directional strategies:
  - Bull: full directional allocation allowed
  - Neutral: reduced size, avoid momentum-chasing
  - Bear: skip directional strategies, favour arbitrage only
  - High-vol: only high-confidence signals (conf > 0.80)

Uses the same market scanner data sources (yfinance + CoinGecko, free only).
Writes to company_brain.json:
  core.market_regime     — "bull" | "neutral" | "bear" | "high_vol"
  core.market_opinion    — full structured dict with sub-signals
  core.strategy_gates    — {"directional": bool, "min_confidence": float}

Posts morning brief to #market-analysis.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from llm_common import core_update, memory_write, slack_post, llm

ALLOW_PAID = os.environ.get("ALLOW_PAID_APIS", "False")
if ALLOW_PAID.lower() == "true":
    sys.exit(1)


def _yf_close_series(symbol: str, period: str = "1y") -> list[float]:
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if hist.empty:
            return []
        return [float(c) for c in hist["Close"].dropna().tolist()]
    except Exception:
        return []


def _coingecko_btc() -> dict:
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin,ethereum", "vs_currencies": "usd",
                    "include_24hr_change": "true"},
            timeout=12,
        )
        if resp.status_code == 200:
            d = resp.json()
            return {
                "btc_price": d.get("bitcoin", {}).get("usd", 0),
                "btc_24h": d.get("bitcoin", {}).get("usd_24h_change", 0),
                "eth_24h": d.get("ethereum", {}).get("usd_24h_change", 0),
            }
    except Exception:
        pass
    return {}


def build_opinion() -> dict:
    """Compute multi-factor market opinion. Returns structured dict."""
    opinion: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signals": {},
        "score": 0,
        "regime": "neutral",
        "strategy_gates": {"directional": True, "min_confidence": 0.60},
        "summary": "",
    }

    # 1. VIX regime
    vix_closes = _yf_close_series("^VIX", period="3mo")
    if vix_closes:
        vix = vix_closes[-1]
        vix_5d = vix_closes[-6] if len(vix_closes) >= 6 else vix
        vix_trend = "rising" if vix > vix_5d * 1.08 else "falling" if vix < vix_5d * 0.92 else "flat"
        opinion["signals"]["vix"] = {"value": round(vix, 2), "trend": vix_trend}
        if vix < 15:
            opinion["score"] += 2
        elif vix < 20:
            opinion["score"] += 1
        elif vix > 30:
            opinion["score"] -= 2
        elif vix > 25:
            opinion["score"] -= 1
        if vix_trend == "rising" and vix > 20:
            opinion["score"] -= 1

    # 2. SPY trend (vs 50-day and 200-day SMA)
    spy_closes = _yf_close_series("SPY", period="1y")
    if len(spy_closes) >= 200:
        spy = spy_closes[-1]
        sma50 = sum(spy_closes[-50:]) / 50
        sma200 = sum(spy_closes[-200:]) / 200
        spy_sig = {"price": round(spy, 2), "sma50": round(sma50, 2), "sma200": round(sma200, 2)}
        if spy > sma200:
            opinion["score"] += 1
            spy_sig["trend"] = "above_200sma"
        else:
            opinion["score"] -= 1
            spy_sig["trend"] = "below_200sma"
        if spy > sma50 > sma200:
            opinion["score"] += 1
            spy_sig["trend"] = "strong_uptrend"
        opinion["signals"]["spy"] = spy_sig

    # 3. Yield curve
    tlt_closes = _yf_close_series("TLT", period="2y")
    shy_closes = _yf_close_series("SHY", period="2y")
    if len(tlt_closes) >= 252 and len(shy_closes) >= 252:
        n = min(len(tlt_closes), len(shy_closes))
        ratios = [t / s for t, s in zip(tlt_closes[-n:], shy_closes[-n:]) if s > 0]
        ratio_now = ratios[-1]
        ratio_median = sorted(ratios[-252:])[126]
        opinion["signals"]["yield_curve"] = {
            "ratio": round(ratio_now, 4),
            "median": round(ratio_median, 4),
            "signal": "steepening" if ratio_now > ratio_median else "flat_or_inverted",
        }
        opinion["score"] += 1 if ratio_now > ratio_median else -1

    # 4. Credit spreads (HYG/LQD 20-day momentum)
    hyg_closes = _yf_close_series("HYG", period="3mo")
    lqd_closes = _yf_close_series("LQD", period="3mo")
    if len(hyg_closes) >= 21 and len(lqd_closes) >= 21:
        n = min(len(hyg_closes), len(lqd_closes))
        ratios = [h / l for h, l in zip(hyg_closes[-n:], lqd_closes[-n:]) if l > 0]
        mom = (ratios[-1] - ratios[-21]) / ratios[-21] * 100 if ratios[-21] else 0
        opinion["signals"]["credit"] = {
            "hyg_lqd_momentum_20d": round(mom, 3),
            "signal": "tightening" if mom > 0 else "widening",
        }
        opinion["score"] += 1 if mom > 0 else -1

    # 5. Breadth proxy: sector ETF majority above 20-day SMA
    sector_etfs = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB", "XLC"]
    above_sma20 = 0
    for sym in sector_etfs:
        closes = _yf_close_series(sym, period="3mo")
        if len(closes) >= 20:
            sma20 = sum(closes[-20:]) / 20
            if closes[-1] > sma20:
                above_sma20 += 1
    breadth = above_sma20 / len(sector_etfs)
    opinion["signals"]["breadth"] = {"sectors_above_sma20": above_sma20,
                                     "total_sectors": len(sector_etfs),
                                     "breadth_pct": round(breadth * 100, 1)}
    if breadth >= 0.7:
        opinion["score"] += 1
    elif breadth <= 0.3:
        opinion["score"] -= 1

    # 6. Crypto momentum (BTC as risk proxy)
    crypto = _coingecko_btc()
    if crypto:
        btc_24h = crypto.get("btc_24h", 0)
        opinion["signals"]["btc"] = {"price": crypto.get("btc_price"), "24h_pct": round(btc_24h, 2)}
        if btc_24h > 3:
            opinion["score"] += 1
        elif btc_24h < -5:
            opinion["score"] -= 1

    # ── Regime classification ─────────────────────────────────────────────────
    score = opinion["score"]
    if score >= 5:
        regime = "strong_bull"
        gates = {"directional": True, "min_confidence": 0.55, "size_multiplier": 1.2}
    elif score >= 2:
        regime = "bull"
        gates = {"directional": True, "min_confidence": 0.60, "size_multiplier": 1.0}
    elif score >= -1:
        regime = "neutral"
        gates = {"directional": True, "min_confidence": 0.68, "size_multiplier": 0.75}
    elif score >= -3:
        regime = "bear"
        gates = {"directional": False, "min_confidence": 0.80, "size_multiplier": 0.5}
    else:
        regime = "strong_bear"
        gates = {"directional": False, "min_confidence": 0.90, "size_multiplier": 0.25}

    # Elevate to high_vol if VIX is spiking even when score is moderate
    vix_val = opinion["signals"].get("vix", {}).get("value", 0)
    if vix_val > 28 and regime in ("neutral", "bull"):
        regime = "high_vol"
        gates["min_confidence"] = max(gates["min_confidence"], 0.75)
        gates["size_multiplier"] = min(gates.get("size_multiplier", 1.0), 0.6)

    opinion["regime"] = regime
    opinion["strategy_gates"] = gates

    # ── LLM narrative ─────────────────────────────────────────────────────────
    vix_str = f"VIX {opinion['signals'].get('vix', {}).get('value', '?')}"
    spy_trend = opinion["signals"].get("spy", {}).get("trend", "unknown")
    credit_str = opinion["signals"].get("credit", {}).get("signal", "unknown")
    curve_str = opinion["signals"].get("yield_curve", {}).get("signal", "unknown")
    breadth_str = f"{above_sma20}/{len(sector_etfs)} sectors above 20-SMA"

    prompt = (
        f"Today's market data: regime={regime} (score={score}), {vix_str} "
        f"({opinion['signals'].get('vix', {}).get('trend', '?')} trend), "
        f"SPY trend={spy_trend}, credit spreads={credit_str}, yield curve={curve_str}, "
        f"breadth={breadth_str}. "
        "Write a 2-sentence morning market brief for the QuantEdge trading desk. "
        "Include one tactical implication for which strategy type to favour today."
    )
    try:
        opinion["summary"] = llm(prompt, max_tokens=150, use_cache=False, inject_company_context=False)
    except Exception:
        opinion["summary"] = f"Regime: {regime} (score={score}). {vix_str}. Credit {credit_str}."

    return opinion


def main() -> None:
    print("Building daily market opinion…", flush=True)
    opinion = build_opinion()

    regime = opinion["regime"]
    gates = opinion["strategy_gates"]
    print(f"Regime: {regime} | directional={gates['directional']} | "
          f"min_conf={gates['min_confidence']} | size_mult={gates.get('size_multiplier', 1.0)}", flush=True)

    # Write to brain
    core_update("market_regime", regime)
    core_update("market_opinion", opinion)
    core_update("strategy_gates", gates)
    core_update("opinion_score", opinion["score"])

    memory_write("episodic", {
        "source": "market_opinion",
        "regime": regime,
        "score": opinion["score"],
        "gates": gates,
        "lesson": opinion["summary"],
    })

    # Post to Slack
    regime_emoji = {
        "strong_bull": "🟢🟢", "bull": "🟢", "neutral": "🟡",
        "high_vol": "⚡", "bear": "🔴", "strong_bear": "🔴🔴",
    }.get(regime, "⚪")

    today = datetime.now(timezone.utc).strftime("%A %b %d")
    msg_lines = [
        f"*QuantEdge Morning Brief — {today}*",
        f"{regime_emoji} *Regime: {regime.replace('_', ' ').upper()}* (composite score: {opinion['score']:+d})",
        "",
        opinion["summary"],
        "",
        f"*Strategy Gates:*",
        f"  • Directional strategies: {'✅ ON' if gates['directional'] else '🚫 OFF'}",
        f"  • Min confidence threshold: {gates['min_confidence']:.0%}",
        f"  • Position size multiplier: {gates.get('size_multiplier', 1.0):.2f}×",
    ]

    # Key signals
    vix_info = opinion["signals"].get("vix", {})
    if vix_info:
        msg_lines.append(f"  • VIX: {vix_info.get('value', '?')} ({vix_info.get('trend', '?')})")
    credit_info = opinion["signals"].get("credit", {})
    if credit_info:
        msg_lines.append(f"  • Credit spreads: {credit_info.get('signal', '?')}")
    breadth_info = opinion["signals"].get("breadth", {})
    if breadth_info:
        msg_lines.append(f"  • Market breadth: {breadth_info.get('breadth_pct', '?')}% sectors above 20-SMA")

    slack_post("#market-analysis", "\n".join(msg_lines))
    print("Posted morning brief to #market-analysis", flush=True)

    # Also post regime change alert to #signals if regime is bear/strong_bear
    if regime in ("bear", "strong_bear"):
        slack_post("#signals", (
            f"⚠️ *REGIME ALERT: {regime.replace('_', ' ').upper()}* — "
            f"Directional strategies PAUSED. Only arbitrage strategies active."
        ))


if __name__ == "__main__":
    main()
