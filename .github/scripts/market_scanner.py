"""
Advanced Market Scanner v2 — Regime-adaptive symbol discovery + deep scanning.

Symbol Discovery:
  - Equity universe adapts to detected regime (bull/bear/neutral/high-vol)
  - S&P 500 / NASDAQ candidates organized by factor exposure
  - Crypto: CoinGecko top-30 by volume (with hardcoded fallback)

Scan Modules:
  1. Macro regime (VIX + yield curve + credit spread + SPY trend → score)
  2. Dynamic symbol discovery (regime-adaptive candidate pool → ranked by momentum+vol)
  3. Gap scanner (overnight gaps > ±1.5%)
  4. Relative volume ranking (today/avg20 across full universe)
  5. Multi-timeframe momentum (1d / 5d / 20d composite)
  6. ATR-breakout detection (ATR-normalized distance from 52W high + volume)
  7. RSI extremes across universe (< 30 oversold, > 70 overbought)
  8. Sector rotation heatmap (all 11 GICS sectors ranked by 1M return)
  9. Crypto scanner — top coins by volume, momentum, funding
 10. Earnings proximity flag (yfinance calendar, upcoming 3 days)
 11. Cross-asset correlation alert (rate/dollar risk-off signal)
 12. LLM tactical recommendation seeded with all scan results

All data: yfinance (equity) + CoinGecko free API (crypto).
Posts to #market-analysis. Writes regime + discovery to company_brain.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).parent))
from llm_common import core_update, core_get, memory_write, slack_post, llm

ALLOW_PAID = os.environ.get("ALLOW_PAID_APIS", "False")
if ALLOW_PAID.lower() == "true":
    sys.exit(1)

# ── Universe definitions (regime-adaptive) ────────────────────────────────────

# Hardcoded top-30 crypto symbols ranked by typical market cap (CoinGecko IDs)
_CRYPTO_CG_IDS = [
    "bitcoin", "ethereum", "binancecoin", "solana", "ripple", "cardano",
    "avalanche-2", "dogecoin", "polkadot", "chainlink", "uniswap",
    "litecoin", "bitcoin-cash", "stellar", "cosmos", "near",
    "internet-computer", "filecoin", "vechain", "aave", "maker",
    "compound-governance-token", "curve-dao-token", "synthetix-network-token",
    "matic-network", "arbitrum", "optimism", "aptos", "sui", "injective-protocol",
]
_CRYPTO_SYM = {
    "bitcoin": "BTC", "ethereum": "ETH", "binancecoin": "BNB", "solana": "SOL",
    "ripple": "XRP", "cardano": "ADA", "avalanche-2": "AVAX", "dogecoin": "DOGE",
    "polkadot": "DOT", "chainlink": "LINK", "uniswap": "UNI", "litecoin": "LTC",
    "bitcoin-cash": "BCH", "stellar": "XLM", "cosmos": "ATOM", "near": "NEAR",
    "internet-computer": "ICP", "filecoin": "FIL", "vechain": "VET", "aave": "AAVE",
    "maker": "MKR", "compound-governance-token": "COMP", "curve-dao-token": "CRV",
    "synthetix-network-token": "SNX", "matic-network": "MATIC", "arbitrum": "ARB",
    "optimism": "OP", "aptos": "APT", "sui": "SUI", "injective-protocol": "INJ",
}

# Regime-adaptive equity universes
_UNIVERSE: dict[str, list[str]] = {
    # Strong bull: high-beta growth, semis, tech innovation
    "strong_bull": [
        "NVDA", "AMD", "AVGO", "QCOM", "AMAT", "LRCX", "KLAC", "MRVL",  # Semis
        "MSFT", "AAPL", "GOOGL", "META", "AMZN", "CRM", "SNOW", "PLTR",  # Mega/cloud
        "TSLA", "RIVN", "NIO",                                              # EV
        "IONQ", "RGTI", "QUBT",                                             # Quantum/AI
        "SMH", "SOXX", "XLK", "ARKK", "SKYY",                             # Tech ETFs
        "QQQ", "SPY", "IWM",                                               # Broad
    ],
    # Bull: broad market + growth, avoid deep defensives
    "bull": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B",
        "JPM", "V", "MA", "UNH", "XOM", "AVGO", "LLY", "MRK", "HD",
        "COST", "NFLX", "ADBE", "ORCL", "INTU", "NOW", "PANW", "CRWD",
        "SPY", "QQQ", "IWM", "XLK", "XLF", "XLV", "XLI",
    ],
    # Neutral: balanced core
    "neutral": [
        "SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "SHY",
        "AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "KO", "WMT",
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "GLD",
        "BRK-B", "V", "MA", "UNH", "COST",
    ],
    # High-vol: volatility products + defensives + gold
    "high_vol": [
        "VXX", "UVXY", "VIXY",                                             # VIX ETPs
        "GLD", "GDX", "GDXJ", "SLV",                                      # Safe havens
        "TLT", "IEF", "SHY", "BND",                                       # Bonds
        "XLP", "XLU", "XLRE",                                              # Defensives
        "JNJ", "PG", "KO", "WMT", "PEP", "MCD", "CL",                   # Staples
        "SPY", "QQQ", "SQQQ", "SH",                                       # Broad + inverse
    ],
    # Bear: defensives, short candidates, safe havens
    "bear": [
        "TLT", "IEF", "SHY", "BND", "AGG",                               # Long bonds
        "GLD", "GDX", "SLV", "GOLD",                                      # Gold
        "XLP", "XLU", "XLRE", "XLV",                                      # Defensives
        "JNJ", "PG", "KO", "WMT", "PEP", "MCD", "CL", "CLX",
        "VXX", "UVXY",                                                     # Vol long
        "SQQQ", "SPXS", "SH", "PSQ",                                      # Inverse
        "DXY", "UUP",                                                      # Dollar
    ],
    # Strong bear: maximum defensive
    "strong_bear": [
        "TLT", "IEF", "SHY", "BND",
        "GLD", "GDX", "SLV",
        "XLP", "XLU",
        "VXX", "UVXY",
        "SQQQ", "SPXS",
        "JNJ", "PG", "KO", "WMT",
    ],
}

SECTOR_ETFS = {
    "XLK": "Tech", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Healthcare", "XLI": "Industrials", "XLY": "Cons. Disc.",
    "XLP": "Cons. Staples", "XLU": "Utilities", "XLRE": "Real Estate",
    "XLB": "Materials", "XLC": "Comm.",
}

MACRO_SYMBOLS = ["SPY", "QQQ", "TLT", "SHY", "HYG", "LQD", "GLD", "^VIX", "UUP"]


# ── Data layer ────────────────────────────────────────────────────────────────

def _yf_history(symbol: str, period: str = "1y", interval: str = "1d") -> list[dict]:
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        if hist.empty:
            return []
        out = []
        for ts, row in hist.iterrows():
            out.append({
                "date": str(ts.date()),
                "open": float(row.get("Open", 0) or 0),
                "high": float(row.get("High", 0) or 0),
                "low": float(row.get("Low", 0) or 0),
                "close": float(row.get("Close", 0) or 0),
                "volume": float(row.get("Volume", 0) or 0),
            })
        return out
    except Exception:
        return []


def _closes(records: list[dict]) -> list[float]:
    return [r["close"] for r in records]


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _avg(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def _pct(a: float, b: float) -> float:
    return round((b - a) / a * 100, 3) if a else 0.0


def _atr(records: list[dict], period: int = 14) -> float:
    """Average True Range."""
    if len(records) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(records)):
        h, l, pc = records[i]["high"], records[i]["low"], records[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return _avg(trs[-period:])


def _rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 2:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = _avg(gains[-period:])
    al = _avg(losses[-period:])
    if al == 0:
        return 100.0
    return round(100 - 100 / (1 + ag / al), 1)


def _avg_vol(records: list[dict], days: int = 20) -> float:
    vols = [r["volume"] for r in records[-days - 1:-1] if r["volume"] > 0]
    return _avg(vols)


# ── Crypto discovery ──────────────────────────────────────────────────────────

def _fetch_crypto_universe(n: int = 20) -> list[dict]:
    """Top-N crypto by 24h trading volume from CoinGecko free API."""
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "volume_desc",
                "per_page": n,
                "page": 1,
                "price_change_percentage": "24h,7d",
            },
            headers={"Accept": "application/json"},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    # Fallback: use simple/price endpoint for hardcoded top-20
    fallback_ids = ",".join(_CRYPTO_CG_IDS[:20])
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": fallback_ids,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_24hr_vol": "true",
                "include_7d_change": "true",
            },
            headers={"Accept": "application/json"},
            timeout=15,
        )
        if resp.status_code == 200:
            raw = resp.json()
            result = []
            for cg_id in _CRYPTO_CG_IDS[:20]:
                d = raw.get(cg_id, {})
                if d:
                    result.append({
                        "id": cg_id,
                        "symbol": _CRYPTO_SYM.get(cg_id, cg_id[:4].upper()),
                        "current_price": d.get("usd", 0),
                        "price_change_percentage_24h": d.get("usd_24h_change", 0),
                        "price_change_percentage_7d_in_currency": d.get("usd_7d_change", 0),
                        "total_volume": d.get("usd_24h_vol", 0),
                    })
            return result
    except Exception:
        pass

    return []


# ── Macro regime ──────────────────────────────────────────────────────────────

def scan_macro_regime() -> dict:
    """6-pillar macro regime score: VIX, yield curve, credit spread, SPY trend, dollar, gold."""
    result: dict[str, Any] = {"score": 0, "signals": [], "regime": "neutral", "details": {}}

    def _s(sym: str, period: str = "1y") -> list[float]:
        return _closes(_yf_history(sym, period=period))

    # 1. VIX
    vix_c = _s("^VIX", "3mo")
    if vix_c:
        v = vix_c[-1]
        v5 = vix_c[-6] if len(vix_c) >= 6 else v
        trend = "rising" if v > v5 * 1.08 else "falling" if v < v5 * 0.93 else "flat"
        result["details"]["vix"] = {"value": round(v, 2), "trend": trend}
        if v < 14:
            result["score"] += 2
            result["signals"].append(f"VIX {v:.1f} (very low — complacency/bull)")
        elif v < 20:
            result["score"] += 1
            result["signals"].append(f"VIX {v:.1f} (low — risk-on)")
        elif v > 32:
            result["score"] -= 2
            result["signals"].append(f"VIX {v:.1f} (fear spike — risk-off)")
        elif v > 25:
            result["score"] -= 1
            result["signals"].append(f"VIX {v:.1f} (elevated — caution)")
        if trend == "rising" and v > 18:
            result["score"] -= 1
            result["signals"].append(f"VIX trending {trend} — increasing uncertainty")

    # 2. Yield curve (TLT/SHY ratio vs 252-day median)
    tlt = _s("TLT", "2y")
    shy = _s("SHY", "2y")
    n = min(len(tlt), len(shy))
    if n >= 252:
        ratios = [t / s for t, s in zip(tlt[-n:], shy[-n:]) if s > 0]
        r_now, r_med = ratios[-1], sorted(ratios[-252:])[126]
        curve_pct = _pct(r_med, r_now)
        result["details"]["yield_curve"] = {"ratio": round(r_now, 4), "vs_median_pct": round(curve_pct, 2)}
        if r_now > r_med * 1.01:
            result["score"] += 1
            result["signals"].append(f"Yield curve steepening ({curve_pct:+.1f}% vs median) — risk-on")
        elif r_now < r_med * 0.99:
            result["score"] -= 1
            result["signals"].append(f"Yield curve flat/inverted ({curve_pct:+.1f}% vs median) — risk-off")

    # 3. Credit spreads (HYG/LQD 20-day momentum)
    hyg = _s("HYG", "3mo")
    lqd = _s("LQD", "3mo")
    n2 = min(len(hyg), len(lqd))
    if n2 >= 21:
        crat = [h / l for h, l in zip(hyg[-n2:], lqd[-n2:]) if l > 0]
        mom = _pct(crat[-21], crat[-1])
        result["details"]["credit"] = {"hyg_lqd_mom_20d": round(mom, 3),
                                        "signal": "tightening" if mom > 0 else "widening"}
        if mom > 0.3:
            result["score"] += 1
            result["signals"].append(f"Credit tightening ({mom:+.2f}%) — risk appetite up")
        elif mom < -0.3:
            result["score"] -= 1
            result["signals"].append(f"Credit widening ({mom:+.2f}%) — risk aversion")

    # 4. SPY vs 50-day and 200-day SMA
    spy = _s("SPY", "1y")
    if len(spy) >= 200:
        s, sma50, sma200 = spy[-1], _avg(spy[-50:]), _avg(spy[-200:])
        pct_vs_200 = _pct(sma200, s)
        trend_str = "strong_uptrend" if s > sma50 > sma200 else \
                    "above_200" if s > sma200 else \
                    "below_200" if s < sma200 else "at_200"
        result["details"]["spy"] = {"price": round(s, 2), "vs_200sma_pct": round(pct_vs_200, 2),
                                     "trend": trend_str}
        if s > sma50 > sma200:
            result["score"] += 2
            result["signals"].append(f"SPY strong uptrend (50 > 200 SMA, +{pct_vs_200:.1f}% above 200)")
        elif s > sma200:
            result["score"] += 1
            result["signals"].append(f"SPY above 200-SMA (+{pct_vs_200:.1f}%)")
        else:
            result["score"] -= 1
            result["signals"].append(f"SPY below 200-SMA ({pct_vs_200:.1f}%) — bearish structure")

    # 5. Dollar strength (UUP proxy — dollar bullish = risk-off for equities/crypto)
    uup = _s("UUP", "3mo")
    if len(uup) >= 20:
        d_mom = _pct(uup[-21] if len(uup) >= 21 else uup[0], uup[-1])
        result["details"]["dollar"] = {"uup_20d_mom": round(d_mom, 2)}
        if d_mom > 1.5:
            result["score"] -= 1
            result["signals"].append(f"Dollar strengthening ({d_mom:+.1f}% 20d) — headwind for risk assets")
        elif d_mom < -1.5:
            result["score"] += 1
            result["signals"].append(f"Dollar weakening ({d_mom:+.1f}% 20d) — tailwind for risk assets")

    # 6. Gold trend (GLD > 20-day SMA = flight-to-safety signal)
    gld = _s("GLD", "3mo")
    if len(gld) >= 20:
        gld_sma20 = _avg(gld[-20:])
        gld_pct = _pct(gld_sma20, gld[-1])
        result["details"]["gold"] = {"vs_sma20_pct": round(gld_pct, 2)}
        if gld[-1] > gld_sma20 * 1.02:
            result["score"] -= 1
            result["signals"].append(f"Gold above 20-SMA (+{gld_pct:.1f}%) — safe-haven demand")
        elif gld[-1] < gld_sma20 * 0.98:
            result["score"] += 1
            result["signals"].append(f"Gold below 20-SMA ({gld_pct:.1f}%) — risk-on appetite")

    # Classify regime
    sc = result["score"]
    regime = (
        "strong_bull" if sc >= 5 else
        "bull"         if sc >= 2 else
        "neutral"      if sc >= -1 else
        "bear"         if sc >= -4 else
        "strong_bear"
    )
    # Elevate to high_vol if VIX spiking even in nominal bull
    vix_val = result["details"].get("vix", {}).get("value", 0)
    if vix_val > 28 and regime in ("neutral", "bull"):
        regime = "high_vol"

    result["regime"] = regime
    return result


# ── Dynamic symbol discovery ──────────────────────────────────────────────────

def discover_equity_universe(regime: str, sector_leaders: list[str] | None = None) -> list[str]:
    """
    Select today's equity scanning universe based on regime + leading sectors.
    Returns deduplicated list of symbols optimised for this market environment.
    """
    base = list(_UNIVERSE.get(regime, _UNIVERSE["neutral"]))

    # Add leading sector ETFs and their typical constituents
    if sector_leaders:
        sector_to_stocks = {
            "Tech": ["NVDA", "MSFT", "AAPL", "AVGO", "AMD", "QCOM", "INTC"],
            "Financials": ["JPM", "BAC", "GS", "MS", "WFC", "BLK", "C"],
            "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC"],
            "Healthcare": ["LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "ABT"],
            "Industrials": ["GE", "CAT", "BA", "HON", "UPS", "MMM", "LMT"],
            "Cons. Disc.": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TGT"],
            "Cons. Staples": ["WMT", "PG", "KO", "PEP", "COST", "MO", "CL"],
            "Materials": ["LIN", "APD", "FCX", "NEM", "ALB", "DD", "NUE"],
            "Comm.": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ"],
            "Real Estate": ["AMT", "PLD", "EQIX", "PSA", "SPG", "O", "DLR"],
            "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL"],
        }
        for sector in (sector_leaders or []):
            base.extend(sector_to_stocks.get(sector, []))

    # Deduplicate preserving order
    seen: set[str] = set()
    universe = []
    for sym in base:
        if sym not in seen:
            seen.add(sym)
            universe.append(sym)

    return universe


def rank_universe_by_activity(symbols: list[str], max_n: int = 40) -> list[str]:
    """
    Download recent data for all symbols, rank by composite activity score
    (relative volume × |momentum|) and return top max_n.
    Falls back to original list on data failure.
    """
    scores: dict[str, float] = {}
    for sym in symbols:
        data = _yf_history(sym, period="3mo")
        if len(data) < 22:
            continue
        c = _closes(data)
        avg_vol = _avg_vol(data, 20)
        today_vol = data[-1]["volume"]
        vol_ratio = today_vol / avg_vol if avg_vol > 0 else 1.0
        mom_5d = abs(_pct(c[-6] if len(c) >= 6 else c[0], c[-1]))
        scores[sym] = vol_ratio * (1 + mom_5d / 10)

    if not scores:
        return symbols[:max_n]

    ranked = sorted(scores, key=lambda s: scores[s], reverse=True)
    # Always include the macro anchors in the output
    anchors = [s for s in symbols if s in {"SPY", "QQQ", "TLT", "GLD", "^VIX"} and s not in ranked[:max_n]]
    return ranked[:max_n] + anchors


# ── Scan modules ──────────────────────────────────────────────────────────────

def scan_gap_movers(symbols: list[str]) -> list[dict]:
    """Overnight gap scanner — open today vs close yesterday."""
    gaps = []
    for sym in symbols:
        data = _yf_history(sym, period="5d")
        if len(data) < 2:
            continue
        prev_close = data[-2]["close"]
        today_open = data[-1]["open"]
        if prev_close <= 0 or today_open <= 0:
            continue
        gap_pct = _pct(prev_close, today_open)
        if abs(gap_pct) >= 1.5:
            gaps.append({
                "sym": sym,
                "gap_pct": gap_pct,
                "direction": "gap_up" if gap_pct > 0 else "gap_down",
                "prev_close": round(prev_close, 2),
                "today_open": round(today_open, 2),
                "today_close": round(data[-1]["close"], 2),
                "filled": _pct(today_open, data[-1]["close"]) * (1 if gap_pct > 0 else -1) < 0,
            })
    return sorted(gaps, key=lambda x: abs(x["gap_pct"]), reverse=True)[:8]


def scan_relative_volume(symbols: list[str]) -> list[dict]:
    """Rank symbols by today's volume relative to 20-day average."""
    results = []
    for sym in symbols:
        data = _yf_history(sym, period="3mo")
        if len(data) < 22:
            continue
        avg_vol = _avg_vol(data, 20)
        today_vol = data[-1]["volume"]
        if avg_vol <= 0:
            continue
        ratio = today_vol / avg_vol
        if ratio < 1.5:
            continue
        c = _closes(data)
        price_chg = _pct(c[-2] if len(c) >= 2 else c[0], c[-1])
        results.append({
            "sym": sym,
            "vol_ratio": round(ratio, 2),
            "price_chg_pct": price_chg,
            "today_vol": int(today_vol),
            "avg_vol_20d": int(avg_vol),
        })
    return sorted(results, key=lambda x: x["vol_ratio"], reverse=True)[:10]


def scan_multi_momentum(symbols: list[str]) -> list[dict]:
    """
    Multi-timeframe momentum score (1d / 5d / 20d).
    Composite = 0.2×1d + 0.3×5d + 0.5×20d (longer timeframes weighted more).
    """
    results = []
    for sym in symbols:
        data = _yf_history(sym, period="3mo")
        if len(data) < 25:
            continue
        c = _closes(data)
        m1d  = _pct(c[-2] if len(c) >= 2 else c[-1], c[-1])
        m5d  = _pct(c[-6]  if len(c) >= 6  else c[0], c[-1])
        m20d = _pct(c[-21] if len(c) >= 21 else c[0], c[-1])
        composite = 0.2 * m1d + 0.3 * m5d + 0.5 * m20d
        rsi_val = _rsi(c) or 50.0
        results.append({
            "sym": sym,
            "composite_score": round(composite, 3),
            "m1d": m1d, "m5d": m5d, "m20d": m20d,
            "rsi": rsi_val,
            "price": round(c[-1], 2),
        })
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results[:12]


def scan_atr_breakouts(symbols: list[str]) -> list[dict]:
    """
    ATR-normalized breakout detection.
    A breakout is high-quality when:
      - Price within 1 ATR of 52W high
      - Volume ratio > 1.5×
      - RSI between 55-80 (not overextended)
    Breakdown: price within 1 ATR of 52W low, volume > 1.3×
    """
    hits = []
    for sym in symbols:
        data = _yf_history(sym, period="1y")
        if len(data) < 50:
            continue
        c = _closes(data)
        highs = [r["high"] for r in data]
        lows  = [r["low"]  for r in data]
        price = c[-1]
        high52 = max(highs[-252:] if len(highs) >= 252 else highs)
        low52  = min(lows[-252:]  if len(lows)  >= 252 else lows)
        atr14  = _atr(data, 14)
        if atr14 <= 0:
            continue
        avg_vol = _avg_vol(data, 20)
        vol_ratio = data[-1]["volume"] / avg_vol if avg_vol > 0 else 0
        rsi_val = _rsi(c) or 50.0
        dist_to_high = (high52 - price) / atr14  # in ATR units
        dist_to_low  = (price - low52) / atr14

        if dist_to_high <= 1.0 and vol_ratio >= 1.5 and 50 < rsi_val < 82:
            hits.append({
                "sym": sym, "type": "atr_breakout",
                "price": round(price, 2), "high52": round(high52, 2),
                "dist_atr": round(dist_to_high, 2), "vol_ratio": round(vol_ratio, 1),
                "rsi": rsi_val, "atr": round(atr14, 2),
            })
        elif dist_to_low <= 1.0 and vol_ratio >= 1.3 and rsi_val < 45:
            hits.append({
                "sym": sym, "type": "atr_breakdown",
                "price": round(price, 2), "low52": round(low52, 2),
                "dist_atr": round(dist_to_low, 2), "vol_ratio": round(vol_ratio, 1),
                "rsi": rsi_val, "atr": round(atr14, 2),
            })
    return sorted(hits, key=lambda x: x["vol_ratio"], reverse=True)[:6]


def scan_rsi_extremes(symbols: list[str]) -> dict:
    """RSI extremes with additional confirmation metrics."""
    overbought, oversold = [], []
    for sym in symbols:
        data = _yf_history(sym, period="3mo")
        if len(data) < 20:
            continue
        c = _closes(data)
        rsi_val = _rsi(c)
        if rsi_val is None:
            continue
        m5d = _pct(c[-6] if len(c) >= 6 else c[0], c[-1])
        if rsi_val >= 72:
            overbought.append({"sym": sym, "rsi": rsi_val, "m5d": m5d, "price": round(c[-1], 2)})
        elif rsi_val <= 28:
            oversold.append({"sym": sym, "rsi": rsi_val, "m5d": m5d, "price": round(c[-1], 2)})
    return {
        "overbought": sorted(overbought, key=lambda x: x["rsi"], reverse=True)[:6],
        "oversold":   sorted(oversold,   key=lambda x: x["rsi"])[:6],
    }


def scan_sector_rotation() -> dict:
    """Rank all 11 GICS sectors by 1M and 3M performance."""
    perf: dict[str, dict] = {}
    for sym, name in SECTOR_ETFS.items():
        data = _yf_history(sym, period="6mo")
        if len(data) < 21:
            continue
        c = _closes(data)
        perf[name] = {
            "sym": sym,
            "1m": _pct(c[-22] if len(c) >= 22 else c[0], c[-1]),
            "3m": _pct(c[0], c[-1]),
            "rsi": _rsi(c) or 50.0,
            "above_sma20": c[-1] > _avg(c[-20:]),
        }
    ranked = sorted(perf.items(), key=lambda x: x[1]["1m"], reverse=True)
    return {
        "leaders":  [(n, d) for n, d in ranked[:3]],
        "laggards": [(n, d) for n, d in ranked[-3:]],
        "all":      ranked,
    }


def scan_crypto(n: int = 20) -> dict:
    """
    Scan top-N crypto by volume. Returns movers, leaders, laggards.
    Uses CoinGecko coins/markets (with fallback to simple/price).
    """
    coins = _fetch_crypto_universe(n)
    if not coins:
        return {"error": "coingecko_unavailable", "coins": []}

    result_coins = []
    for coin in coins:
        sym = coin.get("symbol", "?").upper()
        price = coin.get("current_price", 0)
        ch24h = coin.get("price_change_percentage_24h") or 0
        ch7d  = coin.get("price_change_percentage_7d_in_currency") or 0
        vol   = coin.get("total_volume", 0)
        result_coins.append({
            "sym": sym,
            "price": price,
            "ch24h": round(ch24h, 2),
            "ch7d": round(ch7d, 2),
            "vol_usd": vol,
        })

    result_coins.sort(key=lambda x: x["vol_usd"], reverse=True)
    top_gainers  = sorted(result_coins, key=lambda x: x["ch24h"], reverse=True)[:5]
    top_losers   = sorted(result_coins, key=lambda x: x["ch24h"])[:5]
    high_volume  = [c for c in result_coins[:10]]

    return {
        "coins": result_coins,
        "top_gainers": top_gainers,
        "top_losers": top_losers,
        "high_volume": high_volume,
        "total_scanned": len(result_coins),
    }


def scan_earnings_proximity(symbols: list[str]) -> list[dict]:
    """Flag symbols with earnings in next 3 trading days using yfinance calendar."""
    upcoming = []
    today = date.today()
    lookfwd = today + timedelta(days=5)
    for sym in symbols:
        try:
            import yfinance as yf
            cal = yf.Ticker(sym).calendar
            if cal is None or cal.empty:
                continue
            # calendar is a DataFrame; earnings date may be in index or columns
            if hasattr(cal, "iloc"):
                for col in cal.columns:
                    val = cal[col].iloc[0] if len(cal) > 0 else None
                    if val and hasattr(val, "date"):
                        d = val.date()
                        if today <= d <= lookfwd:
                            upcoming.append({"sym": sym, "earnings_date": str(d), "days_out": (d - today).days})
                            break
        except Exception:
            continue
    return upcoming[:8]


def scan_cross_asset_risk() -> dict:
    """
    Cross-asset risk indicators:
    - Bond yield move (TLT 5-day return) — big drop = yield spike = risk-off
    - Dollar 5-day momentum (UUP)
    - Gold vs equity ratio (GLD/SPY 20-day momentum)
    """
    result: dict[str, Any] = {"risk_flags": []}

    tlt = _closes(_yf_history("TLT", "1mo"))
    if len(tlt) >= 6:
        tlt_5d = _pct(tlt[-6], tlt[-1])
        result["tlt_5d"] = round(tlt_5d, 2)
        if tlt_5d < -2:
            result["risk_flags"].append(f"Bond sell-off: TLT {tlt_5d:.1f}% in 5d (yield spike)")

    uup = _closes(_yf_history("UUP", "1mo"))
    if len(uup) >= 6:
        uup_5d = _pct(uup[-6], uup[-1])
        result["uup_5d"] = round(uup_5d, 2)
        if uup_5d > 1.5:
            result["risk_flags"].append(f"Dollar surge: UUP +{uup_5d:.1f}% in 5d (risk-off)")

    gld = _closes(_yf_history("GLD", "3mo"))
    spy = _closes(_yf_history("SPY", "3mo"))
    if len(gld) >= 21 and len(spy) >= 21:
        n = min(len(gld), len(spy))
        ratio_now = gld[-1] / spy[-1]
        ratio_20d = gld[-21] / spy[-21] if spy[-21] else ratio_now
        gld_spy_mom = _pct(ratio_20d, ratio_now)
        result["gld_spy_ratio_20d"] = round(gld_spy_mom, 2)
        if gld_spy_mom > 3:
            result["risk_flags"].append(f"Gold/SPY ratio surging (+{gld_spy_mom:.1f}% 20d) — safe-haven rotation")

    return result


# ── Slack report formatter ────────────────────────────────────────────────────

def _regime_emoji(regime: str) -> str:
    return {
        "strong_bull": "🟢🟢", "bull": "🟢", "neutral": "🟡",
        "high_vol": "⚡", "bear": "🔴", "strong_bear": "🔴🔴",
    }.get(regime, "⚪")


def build_report(
    macro: dict,
    discovered_symbols: list[str],
    gaps: list[dict],
    rel_vol: list[dict],
    momentum: list[dict],
    breakouts: list[dict],
    rsi: dict,
    sectors: dict,
    crypto: dict,
    earnings: list[dict],
    cross_asset: dict,
) -> str:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    regime = macro.get("regime", "neutral")
    score = macro.get("score", 0)
    emoji = _regime_emoji(regime)

    lines = [
        f"*QuantEdge Advanced Market Scanner* — {now_str}",
        f"{emoji} *Regime: {regime.replace('_', ' ').upper()}* (6-pillar score: {score:+d}/+8)",
        f"Scanning {len(discovered_symbols)} regime-adaptive symbols",
        "",
    ]

    # Macro signals
    if macro.get("signals"):
        lines.append("*📊 Macro Signals:*")
        for s in macro["signals"]:
            lines.append(f"  • {s}")
        lines.append("")

    # Cross-asset risk flags
    if cross_asset.get("risk_flags"):
        lines.append("*⚠️ Cross-Asset Risk Flags:*")
        for f in cross_asset["risk_flags"]:
            lines.append(f"  • {f}")
        lines.append("")

    # Sector rotation
    if sectors.get("leaders"):
        lead_str = "  ".join(f"{n} {d['1m']:+.1f}%" for n, d in sectors["leaders"])
        lag_str  = "  ".join(f"{n} {d['1m']:+.1f}%" for n, d in sectors["laggards"])
        lines.append(f"*🔄 Sector Rotation:*  Leading: {lead_str}  |  Lagging: {lag_str}")
        lines.append("")

    # Gap movers
    if gaps:
        lines.append("*📐 Overnight Gaps (≥ ±1.5%):*")
        for g in gaps[:5]:
            dir_str = "▲ Gap Up" if g["gap_pct"] > 0 else "▼ Gap Down"
            fill_str = " (filled)" if g.get("filled") else ""
            lines.append(f"  • {g['sym']} {dir_str} {g['gap_pct']:+.1f}%{fill_str} | Close {g['today_close']}")
        lines.append("")

    # Relative volume leaders
    if rel_vol:
        lines.append("*📈 Relative Volume Leaders (>1.5× avg):*")
        for v in rel_vol[:5]:
            dir_arrow = "▲" if v["price_chg_pct"] > 0 else "▼"
            lines.append(f"  • {v['sym']} {v['vol_ratio']}× avg | {dir_arrow}{v['price_chg_pct']:+.1f}% | ${v['price_chg_pct']}")
        lines.append("")

    # Momentum leaders
    if momentum:
        top_mom = [m for m in momentum if m["composite_score"] > 0][:5]
        bot_mom = [m for m in reversed(momentum) if m["composite_score"] < 0][:3]
        if top_mom:
            lines.append("*🚀 Momentum Leaders (composite 1d/5d/20d):*")
            for m in top_mom:
                lines.append(
                    f"  • {m['sym']} score={m['composite_score']:+.2f} | "
                    f"1d={m['m1d']:+.1f}% 5d={m['m5d']:+.1f}% 20d={m['m20d']:+.1f}% | RSI {m['rsi']}"
                )
        if bot_mom:
            lines.append("*💨 Momentum Laggards:*")
            for m in bot_mom:
                lines.append(f"  • {m['sym']} score={m['composite_score']:+.2f} | 5d={m['m5d']:+.1f}%")
        lines.append("")

    # ATR breakouts
    if breakouts:
        lines.append("*💥 ATR-Normalized Breakouts:*")
        for b in breakouts:
            if b["type"] == "atr_breakout":
                lines.append(f"  • 📈 {b['sym']} near 52W high (dist={b['dist_atr']:.1f}×ATR) | "
                              f"vol={b['vol_ratio']}× | RSI {b['rsi']} | ${b['price']}")
            else:
                lines.append(f"  • 📉 {b['sym']} testing 52W low (dist={b['dist_atr']:.1f}×ATR) | "
                              f"vol={b['vol_ratio']}× | RSI {b['rsi']}")
        lines.append("")

    # RSI extremes
    if rsi["overbought"] or rsi["oversold"]:
        lines.append("*📉📈 RSI Extremes:*")
        if rsi["overbought"]:
            lines.append("  OB (>72): " + "  ".join(f"{x['sym']} {x['rsi']}" for x in rsi["overbought"]))
        if rsi["oversold"]:
            lines.append("  OS (<28): " + "  ".join(f"{x['sym']} {x['rsi']}" for x in rsi["oversold"]))
        lines.append("")

    # Crypto
    if crypto.get("coins"):
        lines.append(f"*🪙 Crypto Top Movers (top {crypto['total_scanned']} by volume):*")
        if crypto.get("top_gainers"):
            gainers = "  ".join(f"{c['sym']} +{c['ch24h']:.1f}%" for c in crypto["top_gainers"][:4])
            lines.append(f"  Gainers: {gainers}")
        if crypto.get("top_losers"):
            losers = "  ".join(f"{c['sym']} {c['ch24h']:.1f}%" for c in crypto["top_losers"][:4])
            lines.append(f"  Losers: {losers}")
        # High volume coins
        if crypto.get("high_volume"):
            vol_str = "  ".join(f"{c['sym']} ${c['price']:,.0f}" for c in crypto["high_volume"][:5])
            lines.append(f"  Top vol: {vol_str}")
        lines.append("")

    # Earnings proximity
    if earnings:
        lines.append("*📅 Earnings Proximity (next 3 days):*")
        for e in earnings:
            lines.append(f"  • {e['sym']} reports in {e['days_out']}d ({e['earnings_date']})")
        lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("QuantEdge Advanced Market Scanner v2 starting…", flush=True)
    t0 = time.time()

    # Step 1: Macro regime (determines universe)
    print("  [1/10] Macro regime…", flush=True)
    macro = scan_macro_regime()
    regime = macro["regime"]
    print(f"    → regime={regime} (score={macro['score']})", flush=True)

    # Step 2: Dynamic symbol discovery
    print(f"  [2/10] Discovering symbols for regime={regime}…", flush=True)
    # Try to read sector leaders from brain to enrich universe
    prev_leaders = core_get("sector_leaders") or []
    candidates = discover_equity_universe(regime, prev_leaders)
    print(f"    → {len(candidates)} candidates, ranking by activity…", flush=True)
    active_syms = rank_universe_by_activity(candidates, max_n=40)
    print(f"    → {len(active_syms)} active symbols selected", flush=True)

    # Step 3: Sector rotation (uses hardcoded ETFs, fast)
    print("  [3/10] Sector rotation…", flush=True)
    sectors = scan_sector_rotation()
    leaders_names = [n for n, _ in sectors.get("leaders", [])]

    # Steps 4-10: Parallel scanning over discovered universe
    print("  [4/10] Gap movers…", flush=True)
    gaps = scan_gap_movers(active_syms)

    print("  [5/10] Relative volume…", flush=True)
    rel_vol = scan_relative_volume(active_syms)

    print("  [6/10] Multi-timeframe momentum…", flush=True)
    momentum = scan_multi_momentum(active_syms)

    print("  [7/10] ATR breakouts…", flush=True)
    breakouts = scan_atr_breakouts(active_syms)

    print("  [8/10] RSI extremes…", flush=True)
    rsi = scan_rsi_extremes(active_syms)

    print("  [9/10] Crypto (top 20 by volume)…", flush=True)
    crypto = scan_crypto(20)

    print("  [10/10] Cross-asset risk + earnings…", flush=True)
    cross_asset = scan_cross_asset_risk()
    earnings = scan_earnings_proximity(active_syms[:20])  # limit to avoid timeout

    elapsed = time.time() - t0
    print(f"\nScan complete in {elapsed:.1f}s", flush=True)

    # Write to brain
    core_update("market_regime", regime)
    core_update("macro_score", macro["score"])
    core_update("vix", macro["details"].get("vix", {}).get("value"))
    core_update("market_scan_ts", time.time())
    core_update("sector_leaders", leaders_names)
    core_update("sector_laggards", [n for n, _ in sectors.get("laggards", [])])
    core_update("active_universe", active_syms[:20])
    core_update("top_momentum", [m["sym"] for m in momentum[:5]])
    core_update("risk_flags", cross_asset.get("risk_flags", []))

    memory_write("episodic", {
        "source": "market_scanner_v2",
        "regime": regime,
        "score": macro["score"],
        "sector_leaders": leaders_names,
        "gap_count": len(gaps),
        "breakout_count": len(breakouts),
        "risk_flags": cross_asset.get("risk_flags", []),
        "lesson": (
            f"Regime={regime} (score={macro['score']}). "
            f"Leading: {', '.join(leaders_names[:2])}. "
            f"Gaps: {len(gaps)}, Breakouts: {len(breakouts)}, "
            f"Risk flags: {len(cross_asset.get('risk_flags', []))}."
        ),
    })

    # Build and post Slack report
    report = build_report(macro, active_syms, gaps, rel_vol, momentum,
                          breakouts, rsi, sectors, crypto, earnings, cross_asset)
    print("\n" + report, flush=True)

    resp = slack_post("#market-analysis", report)
    if resp.get("ok"):
        print("Posted to #market-analysis", flush=True)

    # LLM tactical brief
    try:
        top_mom_syms = ", ".join(m["sym"] for m in momentum[:3])
        gap_summary  = ", ".join(f"{g['sym']} {g['gap_pct']:+.1f}%" for g in gaps[:3]) or "none"
        flag_summary = "; ".join(cross_asset.get("risk_flags", [])) or "none"
        prompt = (
            f"Market regime: {regime} (macro score {macro['score']}). "
            f"Sector leaders: {', '.join(leaders_names[:3])}. "
            f"Top momentum: {top_mom_syms}. "
            f"Gap movers: {gap_summary}. "
            f"Risk flags: {flag_summary}. "
            "In 2 sentences: (1) which strategy type should the QuantEdge desk prioritize today "
            "and (2) one specific setup worth watching based on the data above."
        )
        advice = llm(prompt, max_tokens=150, use_cache=False, inject_company_context=False)
        slack_post("#market-analysis", f"🤖 *Tactical brief:* {advice}")
        print(f"LLM advice: {advice}", flush=True)
    except Exception as e:
        print(f"LLM brief failed: {e}", flush=True)


if __name__ == "__main__":
    main()
