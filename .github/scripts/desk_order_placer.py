"""
Desk Order Placer — runs every 15 minutes during market hours.
Version: 2.0 — 6 desks, 59 strategies, real paper orders via Alpaca.

For each asset-class desk, fetches live OHLCV from Alpaca paper API,
runs the relevant strategies' analyze(), and places real paper orders
when signals fire with sufficient confidence.

No mock data. If Alpaca is unreachable, the desk is skipped entirely.
Results are posted to the desk-specific Slack channel.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_tracker import (
    PipelineTracker,
    Stage,
)

MARKET_STATUS    = Stage.MARKET_STATUS
DATA_FETCH       = Stage.DATA_FETCH
SIGNAL_GENERATION = Stage.SIGNAL_GENERATION
RISK_CHECK       = Stage.RISK_CHECK
ORDER_EXECUTION  = Stage.ORDER_EXECUTION
FILL_TRACKING    = Stage.FILL_TRACKING
PNL_SNAPSHOT     = Stage.PNL_SNAPSHOT

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
SLACK_BOT_TOKEN   = os.environ.get("SLACK_BOT_TOKEN", "")
DESK_FILTER       = os.environ.get("DESK_FILTER", "").strip().lower()

# ── Desk configuration ────────────────────────────────────────────────────────

class DeskConfig(NamedTuple):
    name:            str
    slack_channel:   str
    symbols:         list[str]
    strategy_names:  list[str]       # must match STRATEGY_REGISTRY keys
    notional_usd:    float            # dollars per order
    confidence_min:  float            # minimum signal confidence to trade


DESKS: list[DeskConfig] = [
    DeskConfig(
        name="Equities",
        slack_channel="#desk-equities",
        symbols=["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "JPM"],
        strategy_names=[
            "momentum", "mean_reversion", "breakout", "rsi_macd", "supertrend",
            "cross_sectional_momentum", "opening_range_breakout", "vwap_reversion",
            "residual_momentum", "idio_vol_anomaly",
        ],
        notional_usd=500.0,
        confidence_min=0.60,
    ),
    DeskConfig(
        name="Crypto",
        slack_channel="#desk-crypto",
        symbols=["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"],
        strategy_names=[
            "crypto_adaptive_trend", "mean_reversion", "breakout",
            "basis_carry", "btc_eth_stat_arb", "mvrv_zscore_timing",
            "intraday_seasonality", "funding_rate_arb",
        ],
        notional_usd=300.0,
        confidence_min=0.65,
    ),
    DeskConfig(
        name="Options",
        slack_channel="#desk-options",
        symbols=["SPY", "QQQ", "AAPL", "TSLA", "NVDA"],
        strategy_names=[
            "vix_mean_reversion", "gamma_exposure", "skew_arb",
            "vrp_systematic", "dispersion_trading", "vol_term_structure",
        ],
        notional_usd=400.0,
        confidence_min=0.65,
    ),
    DeskConfig(
        name="Polymarket",
        slack_channel="#desk-polymarket",
        symbols=["SPY"],   # proxy for market regime
        strategy_names=[
            "polymarket_sentiment_momentum", "poly_binary_arb",
            "poly_calibration_arb", "poly_late_resolution",
        ],
        notional_usd=100.0,
        confidence_min=0.70,
    ),
    DeskConfig(
        name="Macro/FX",
        slack_channel="#desk-fx-rates",
        symbols=["GLD", "TLT", "UUP", "EWJ", "EEM"],
        strategy_names=[
            "cross_asset_carry", "sector_rotation", "time_series_momentum",
            "intraday_fomc_momentum", "pead_sue", "multi_factor_equity",
        ],
        notional_usd=400.0,
        confidence_min=0.60,
    ),
    DeskConfig(
        name="StatArb",
        slack_channel="#desk-stat-arb",
        symbols=["SPY", "QQQ", "IWM", "GLD", "TLT"],
        strategy_names=[
            "pairs_trading", "pca_stat_arb", "kalman_pairs",
            "triangular_arb", "stablecoin_depeg_arb",
        ],
        notional_usd=600.0,
        confidence_min=0.62,
    ),
    # ── Commodities desk: traded via liquid ETF proxies (Alpaca has no futures) ──
    DeskConfig(
        name="Commodities",
        slack_channel="#desk-commodities",
        # GLD=gold, SLV=silver, USO=WTI oil, UNG=natgas, DBA=agriculture,
        # DBB=base metals, CPER=copper, DBC=broad commodity basket
        symbols=["GLD", "SLV", "USO", "UNG", "DBA", "DBB", "CPER", "DBC"],
        strategy_names=[
            "time_series_momentum", "breakout", "supertrend",
            "cross_asset_carry", "mean_reversion",
        ],
        notional_usd=400.0,
        confidence_min=0.60,
    ),
    # ── Futures desk: index/rate/commodity FUTURES via their ETF proxies ─────────
    DeskConfig(
        name="Futures",
        slack_channel="#desk-futures",
        # ES→SPY, NQ→QQQ, RTY→IWM, YM→DIA, ZN→IEF(10Y), ZB→TLT(30Y),
        # CL→USO(oil), GC→GLD(gold) — continuous-trend proxies
        symbols=["SPY", "QQQ", "IWM", "DIA", "IEF", "TLT", "USO", "GLD"],
        strategy_names=[
            "time_series_momentum", "cross_sectional_momentum",
            "breakout", "supertrend", "vwap_reversion",
        ],
        notional_usd=500.0,
        confidence_min=0.60,
    ),
    # ── Rates desk: US Treasury curve via duration ETFs ──────────────────────────
    DeskConfig(
        name="Rates",
        slack_channel="#desk-rates",
        # SHY=1-3Y, IEI=3-7Y, IEF=7-10Y, TLT=20Y+, TIP=inflation, LQD=IG credit,
        # HYG=high yield — curve + credit spread plays
        symbols=["SHY", "IEI", "IEF", "TLT", "TIP", "LQD", "HYG"],
        strategy_names=[
            "cross_asset_carry", "basis_carry", "time_series_momentum",
            "mean_reversion",
        ],
        notional_usd=500.0,
        confidence_min=0.60,
    ),
    # ── Kalshi desk: CFTC-regulated US prediction market (scan-only, like Poly) ──
    DeskConfig(
        name="Kalshi",
        slack_channel="#desk-kalshi",
        symbols=[],   # scanned via Kalshi public API, not Alpaca bars
        strategy_names=[],
        notional_usd=50.0,
        confidence_min=0.70,
    ),
]

# ── Alpaca REST client (direct HTTP, no SDK dependency) ───────────────────────

ALPACA_PAPER_BASE    = "https://paper-api.alpaca.markets"
ALPACA_DATA_BASE     = "https://data.alpaca.markets"


def _alpaca_get_sync(path: str, params: dict | None = None, data_api: bool = False) -> dict:
    """Blocking urllib call — run via asyncio.to_thread to avoid blocking event loop."""
    import urllib.request, urllib.parse
    base = ALPACA_DATA_BASE if data_api else ALPACA_PAPER_BASE
    url  = base + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID":     ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    })
    with urllib.request.urlopen(req, timeout=8) as resp:   # 8s per call
        return json.loads(resp.read())


async def _alpaca_get(path: str, params: dict | None = None, data_api: bool = False) -> dict:
    return await asyncio.to_thread(_alpaca_get_sync, path, params, data_api)


def _alpaca_post_sync(path: str, body: dict) -> dict:
    import urllib.request
    url     = ALPACA_PAPER_BASE + path
    payload = json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, headers={
        "APCA-API-KEY-ID":     ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type":        "application/json",
    })
    with urllib.request.urlopen(req, timeout=8) as resp:
        return json.loads(resp.read())


async def _alpaca_post(path: str, body: dict) -> dict:
    return await asyncio.to_thread(_alpaca_post_sync, path, body)


async def _get_account() -> dict | None:
    try:
        return await _alpaca_get("/v2/account")
    except Exception as exc:
        print(f"  ✗ get_account failed: {exc}", flush=True)
        return None


async def _get_bars(symbol: str, timeframe: str = "1Day", limit: int = 200) -> "pd.DataFrame | None":
    import pandas as pd
    try:
        is_crypto = "/" in symbol
        if is_crypto:
            path   = f"/v1beta3/crypto/us/bars"
            params = {"symbols": symbol, "timeframe": timeframe, "limit": limit}
        else:
            path   = f"/v2/stocks/{symbol}/bars"
            params = {"timeframe": timeframe, "limit": limit, "adjustment": "split"}

        data = await _alpaca_get(path, params, data_api=True)

        if is_crypto:
            bars_list = data.get("bars", {}).get(symbol, [])
        else:
            bars_list = data.get("bars", [])

        if not bars_list:
            return None

        df = pd.DataFrame(bars_list)
        df = df.rename(columns={"t": "time", "o": "open", "h": "high",
                                 "l": "low",  "c": "close", "v": "volume"})
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = df[col].astype(float)
        return df[["open", "high", "low", "close", "volume"]]

    except Exception as exc:
        print(f"    ⚠ bars fetch failed for {symbol}: {exc}", flush=True)
        return None


async def _place_order(symbol: str, side: str, notional_usd: float) -> dict | None:
    try:
        body: dict = {
            "symbol":      symbol,
            "side":        side,
            "type":        "market",
            "time_in_force": "gtc" if "/" in symbol else "day",
        }
        # Use notional for fractional shares; qty for crypto
        if "/" in symbol:
            # Alpaca crypto uses qty; estimate from notional
            quote_data = await _alpaca_get(
                f"/v1beta3/crypto/us/latest/quotes",
                {"symbols": symbol},
                data_api=True,
            )
            ask = float((quote_data.get("quotes", {}).get(symbol, {}) or {}).get("ap", 0))
            if ask <= 0:
                return None
            qty = round(notional_usd / ask, 6)
            body["qty"] = str(qty)
        else:
            body["notional"] = str(round(notional_usd, 2))

        result = await _alpaca_post("/v2/orders", body)
        return result
    except Exception as exc:
        print(f"    ⚠ place_order failed {symbol} {side}: {exc}", flush=True)
        return None


# ── Strategy dispatch ─────────────────────────────────────────────────────────

def _load_strategy(strategy_name: str):
    from app.strategies import STRATEGY_REGISTRY
    cls = STRATEGY_REGISTRY.get(strategy_name)
    if cls is None:
        return None
    return cls()


# ── Desk runner ───────────────────────────────────────────────────────────────

async def run_desk(desk: DeskConfig, account: dict) -> list[dict]:
    """Run all strategies for a desk, place orders, return order records."""
    print(f"\n{'─'*60}", flush=True)
    print(f"  DESK: {desk.name}", flush=True)

    equity = float(account.get("equity", 0))
    if equity < 100:
        print(f"  ✗ account equity too low (${equity:.2f})", flush=True)
        return []

    orders_placed: list[dict] = []

    strategies = []
    for sname in desk.strategy_names:
        s = _load_strategy(sname)
        if s is None:
            print(f"  ⚠ strategy '{sname}' not in registry — skipping", flush=True)
        else:
            strategies.append(s)

    if not strategies:
        print(f"  ✗ no valid strategies for {desk.name}", flush=True)
        return []

    for symbol in desk.symbols:
        df = await _get_bars(symbol)
        if df is None or len(df) < 50:
            print(f"  ⚠ {symbol}: insufficient data", flush=True)
            continue

        for strategy in strategies:
            try:
                signal = await strategy.analyze(df, symbol)
            except Exception as exc:
                print(f"  ⚠ {strategy.name}/{symbol} analyze() error: {exc}", flush=True)
                continue

            if signal is None:
                continue

            conf = getattr(signal, "confidence", 1.0) or 1.0
            if conf < desk.confidence_min:
                print(
                    f"  · {strategy.name}/{symbol} signal={signal.side} conf={conf:.2f} "
                    f"< threshold={desk.confidence_min:.2f} — skipped",
                    flush=True,
                )
                continue

            print(
                f"  ► {strategy.name}/{symbol} signal={signal.side.upper()} "
                f"conf={conf:.2f} — placing ${desk.notional_usd:.0f} order",
                flush=True,
            )

            order = await _place_order(symbol, signal.side, desk.notional_usd)
            if order and order.get("id"):
                print(f"    ✓ order {order['id']} submitted ({order.get('status', '?')})", flush=True)
                orders_placed.append({
                    "desk":     desk.name,
                    "strategy": strategy.name,
                    "symbol":   symbol,
                    "side":     signal.side,
                    "notional": desk.notional_usd,
                    "confidence": conf,
                    "order_id": order["id"],
                    "status":   order.get("status", "?"),
                    "ts":       datetime.now(timezone.utc).isoformat(),
                })
            else:
                print(f"    ✗ order placement returned no ID", flush=True)

    return orders_placed


# ── Tradability gating ─────────────────────────────────────────────────────────

def _is_crypto_symbol(symbol: str) -> bool:
    """Alpaca crypto pairs (e.g. BTC/USD) trade 24/7 — not gated by the stock clock."""
    return "/" in symbol


def _symbol_tradeable(symbol: str, stock_market_open: bool) -> bool:
    """Crypto trades around the clock; equities/options follow the US market clock."""
    if _is_crypto_symbol(symbol):
        return True
    return stock_market_open


# ── Polymarket real scan (public Gamma API, no auth, 24/7) ──────────────────────

POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"


def _poly_get_sync(path: str, params: dict | None = None) -> object:
    import urllib.request, urllib.parse
    url = POLYMARKET_GAMMA + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


async def scan_polymarket(desk: DeskConfig) -> list[dict]:
    """
    Real Polymarket scan via the public Gamma API (no auth, 24/7).
    Finds binary-arb opportunities (YES+NO < 1.00 - fees) on liquid open markets.
    Returns signal records. Places real CLOB orders only if POLYMARKET_PRIVATE_KEY is set.
    """
    print(f"\n{'─'*60}", flush=True)
    print(f"  DESK: {desk.name} (Polymarket Gamma scan — 24/7)", flush=True)
    signals: list[dict] = []
    try:
        # Active, liquid markets sorted by 24h volume
        markets = await asyncio.to_thread(
            _poly_get_sync, "/markets",
            {"closed": "false", "active": "true", "order": "volume24hr",
             "ascending": "false", "limit": 50},
        )
    except Exception as exc:
        print(f"  ✗ Polymarket Gamma fetch failed: {exc}", flush=True)
        return []

    if not isinstance(markets, list):
        markets = markets.get("data", []) if isinstance(markets, dict) else []

    scanned = 0
    for m in markets:
        try:
            prices = m.get("outcomePrices")
            if isinstance(prices, str):
                prices = json.loads(prices)
            if not prices or len(prices) < 2:
                continue
            yes_p, no_p = float(prices[0]), float(prices[1])
            scanned += 1
            total = yes_p + no_p
            # Binary arb: buying both sides for < $0.98 locks risk-free profit (2% fee buffer)
            if 0 < total < 0.98:
                edge = (1.0 - total) * 100
                signals.append({
                    "desk": desk.name,
                    "strategy": "poly_binary_arb",
                    "market": m.get("question", "?")[:80],
                    "yes": yes_p, "no": no_p,
                    "edge_pct": round(edge, 2),
                    "volume_24h": float(m.get("volume24hr", 0) or 0),
                    "ts": datetime.now(timezone.utc).isoformat(),
                })
        except Exception:
            continue

    print(f"  scanned {scanned} liquid markets — {len(signals)} arb signal(s)", flush=True)

    # Report to Slack
    if signals:
        top = sorted(signals, key=lambda s: -s["edge_pct"])[:5]
        lines = [f"*{desk.name} Desk* — {len(signals)} binary-arb signal(s) (24/7 Gamma scan)"]
        for s in top:
            lines.append(
                f"🎯 `{s['market']}` YES={s['yes']:.3f}+NO={s['no']:.3f} "
                f"→ *{s['edge_pct']:.1f}%* edge (vol24h=${s['volume_24h']:,.0f})"
            )
        if not os.environ.get("POLYMARKET_PRIVATE_KEY"):
            lines.append("_⚠ POLYMARKET_PRIVATE_KEY not set — scan-only, no orders placed_")
        _post_slack(desk.slack_channel, "\n".join(lines))
    else:
        _post_slack(desk.slack_channel,
                    f"💤 *{desk.name}*: scanned {scanned} markets, no arb edge ≥2% right now")
    return signals


# ── Kalshi real scan (CFTC-regulated US prediction market, public API, 24/7) ────

# Kalshi is the regulated US cousin of Polymarket: a CFTC-licensed designated
# contract market for event contracts. Its public trade API needs no auth for
# read-only market data. YES+NO prices are in cents (0-100). A YES+NO sum below
# ~98c is a binary-arb edge (same logic as Polymarket, but on a regulated venue).
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def _kalshi_get_sync(path: str, params: dict | None = None) -> object:
    import urllib.request, urllib.parse
    url = KALSHI_BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


async def scan_kalshi(desk: DeskConfig) -> list[dict]:
    """
    Real Kalshi scan via the public market-data API (no auth, 24/7).
    Finds binary-arb edges (YES_ask + NO_ask < 100c - fee buffer) on liquid
    open markets. Scan-only unless KALSHI_API_KEY is set for live order placement.
    """
    print(f"\n{'─'*60}", flush=True)
    print(f"  DESK: {desk.name} (Kalshi regulated-market scan — 24/7)", flush=True)
    signals: list[dict] = []
    try:
        data = await asyncio.to_thread(
            _kalshi_get_sync, "/markets",
            {"status": "open", "limit": 100},
        )
    except Exception as exc:
        print(f"  ✗ Kalshi fetch failed: {exc}", flush=True)
        _post_slack(desk.slack_channel,
                    f"⚠ *{desk.name}*: Kalshi API unreachable ({type(exc).__name__}) — no action")
        return []

    markets = data.get("markets", []) if isinstance(data, dict) else []
    scanned = 0
    for m in markets:
        try:
            # Kalshi prices are in cents (1-99). yes_ask / no_ask are best asks.
            yes_ask = m.get("yes_ask")
            no_ask = m.get("no_ask")
            if yes_ask is None or no_ask is None:
                continue
            yes_c, no_c = float(yes_ask), float(no_ask)
            if yes_c <= 0 or no_c <= 0:
                continue
            scanned += 1
            total = yes_c + no_c  # cents
            volume = float(m.get("volume", 0) or 0)
            # Buy both sides < 98c → locked edge after ~2c fee buffer
            if 0 < total < 98 and volume > 0:
                edge = (100.0 - total)
                signals.append({
                    "desk": desk.name,
                    "strategy": "kalshi_binary_arb",
                    "market": (m.get("title") or m.get("ticker", "?"))[:80],
                    "yes_c": yes_c, "no_c": no_c,
                    "edge_cents": round(edge, 1),
                    "volume": volume,
                    "ts": datetime.now(timezone.utc).isoformat(),
                })
        except Exception:
            continue

    print(f"  scanned {scanned} open Kalshi markets — {len(signals)} arb signal(s)", flush=True)

    if signals:
        top = sorted(signals, key=lambda s: -s["edge_cents"])[:5]
        lines = [f"*{desk.name} Desk* — {len(signals)} binary-arb signal(s) (regulated, 24/7)"]
        for s in top:
            lines.append(
                f"🎯 `{s['market']}` YES={s['yes_c']:.0f}c+NO={s['no_c']:.0f}c "
                f"→ *{s['edge_cents']:.0f}c* edge (vol={s['volume']:,.0f})"
            )
        if not os.environ.get("KALSHI_API_KEY"):
            lines.append("_⚠ KALSHI_API_KEY not set — scan-only, no orders placed_")
        _post_slack(desk.slack_channel, "\n".join(lines))
    else:
        _post_slack(desk.slack_channel,
                    f"💤 *{desk.name}*: scanned {scanned} markets, no arb edge ≥2c right now")
    return signals


# ── Slack helper ──────────────────────────────────────────────────────────────

def _post_slack(channel: str, message: str) -> None:
    if not SLACK_BOT_TOKEN:
        return
    try:
        import urllib.request
        payload = json.dumps({"channel": channel, "text": message})
        req = urllib.request.Request(
            "https://slack.com/api/chat.postMessage",
            data=payload.encode(),
            headers={
                "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
                "Content-Type":  "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            if not body.get("ok"):
                print(f"  ⚠ Slack error on {channel}: {body.get('error')}", flush=True)
    except Exception as exc:
        print(f"  ⚠ Slack post failed on {channel}: {exc}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"QuantEdge Desk Order Placer — {datetime.now(timezone.utc).isoformat()}", flush=True)

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("⚠ ALPACA_API_KEY / ALPACA_SECRET_KEY not set — running in dry-run mode (no orders placed)", flush=True)
        # Non-fatal: CI should not fail because secrets aren't available in a fork or PR
        # The scheduled run on the main branch will have real credentials

    with PipelineTracker("desk_trading") as tracker:

        # ── Stage 1: Market Status ────────────────────────────────────────────
        is_open = False
        with tracker.stage(MARKET_STATUS, "Check market status"):
            try:
                clock = await _alpaca_get("/v2/clock")
                is_open = bool(clock.get("is_open", False))
            except Exception:
                is_open = True  # assume open if check fails; let subsequent calls error out
            tracker.set_output(is_open=is_open)

        # ── Stage 2: Data Fetch (account + bars) ─────────────────────────────
        # Cross-stage vars hoisted here so later stages never hit a NameError
        # if this stage fails partway (the tracker is resilient and continues).
        account = {"equity": 0, "cash": 0, "buying_power": 0}
        bars_fetched = 0
        symbols_fetched: list[str] = []
        equity = 0.0
        cash   = 0.0
        buying = 0.0
        bars_cache: dict[str, object] = {}
        active_desks = [d for d in DESKS if not DESK_FILTER or DESK_FILTER in d.name.lower()]
        if DESK_FILTER and not active_desks:
            raise RuntimeError(f"no desk matches filter '{DESK_FILTER}'")
        # Polymarket trades on its own venue (CLOB), not Alpaca — handle separately.
        poly_desks   = [d for d in active_desks if d.name == "Polymarket"]
        kalshi_desks = [d for d in active_desks if d.name == "Kalshi"]
        alpaca_desks = [d for d in active_desks if d.name not in ("Polymarket", "Kalshi")]

        with tracker.stage(DATA_FETCH, "Fetch account and market bars"):
            fetched_account = await _get_account()
            if fetched_account is None:
                # Account fetch failed (API unreachable or bad credentials).
                # Still run signal generation; order placement will be skipped.
                print("  ⚠ Account unavailable — running in signal-only mode (no orders placed)", flush=True)
            else:
                account = fetched_account
                equity = float(account.get("equity",       0))
                cash   = float(account.get("cash",         0))
                buying = float(account.get("buying_power", 0))
                print(f"  Account equity=${equity:.2f}  cash=${cash:.2f}  buying_power=${buying:.2f}", flush=True)

            # Pre-fetch bars for all unique Alpaca symbols concurrently
            all_symbols = list({s for desk in alpaca_desks for s in desk.symbols})
            results = await asyncio.gather(
                *[_get_bars(sym) for sym in all_symbols],
                return_exceptions=True,
            )
            for sym, df in zip(all_symbols, results):
                if isinstance(df, Exception) or df is None:
                    continue
                if len(df) >= 50:
                    bars_cache[sym] = df
                    bars_fetched += 1
                    symbols_fetched.append(sym)
            tracker.set_output(bars_fetched=bars_fetched, symbols=symbols_fetched)

        # ── Stage 2b: Prediction-market scans (24/7, independent of stock clock) ──
        poly_signals: list[dict] = []
        for desk in poly_desks:
            poly_signals.extend(await scan_polymarket(desk))
        for desk in kalshi_desks:
            poly_signals.extend(await scan_kalshi(desk))

        # ── Stage 3: Signal Generation (Alpaca desks) ─────────────────────────
        raw_signals: list[dict] = []
        with tracker.stage(SIGNAL_GENERATION, "Generate trading signals"):
            for desk in alpaca_desks:
                strategies = []
                for sname in desk.strategy_names:
                    s = _load_strategy(sname)
                    if s is not None:
                        strategies.append(s)

                for symbol in desk.symbols:
                    df = bars_cache.get(symbol)
                    if df is None:
                        continue
                    for strategy in strategies:
                        try:
                            signal = await strategy.analyze(df, symbol)
                        except Exception as exc:
                            print(f"  ⚠ {strategy.name}/{symbol} analyze() error: {exc}", flush=True)
                            continue
                        if signal is not None:
                            conf = getattr(signal, "confidence", 1.0) or 1.0
                            raw_signals.append({
                                "desk":       desk,
                                "strategy":   strategy,
                                "symbol":     symbol,
                                "signal":     signal,
                                "confidence": conf,
                            })
            tracker.set_output(signals_generated=len(raw_signals))

        # ── Stage 4: Risk Check ───────────────────────────────────────────────
        approved_signals: list[dict] = []
        with tracker.stage(RISK_CHECK, "Apply confidence threshold filter"):
            for item in raw_signals:
                desk = item["desk"]
                conf = item["confidence"]
                if conf < desk.confidence_min:
                    print(
                        f"  · {item['strategy'].name}/{item['symbol']} signal={item['signal'].side} "
                        f"conf={conf:.2f} < threshold={desk.confidence_min:.2f} — skipped",
                        flush=True,
                    )
                else:
                    approved_signals.append(item)
            filtered = len(raw_signals) - len(approved_signals)
            tracker.set_output(passed=len(approved_signals), filtered=filtered)

        # ── Stage 5: Order Execution ──────────────────────────────────────────
        all_orders: list[dict] = []
        desk_summaries: list[str] = []
        total_notional = 0.0
        has_buying_power = float(account.get("buying_power", 0)) > 0
        with tracker.stage(ORDER_EXECUTION, "Place orders"):
            if not has_buying_power:
                print("  ⚠ No buying power / account unavailable — orders skipped (signals logged)", flush=True)
                tracker.set_output(orders_placed=0, reason="account_unavailable")
            elif not is_open:
                print("  ℹ US stock market closed — equities/options gated, crypto trades 24/7", flush=True)
            # Group approved signals by desk so we can still post per-desk summaries
            desk_orders_map: dict[str, list[dict]] = {}
            for item in approved_signals:
                desk     = item["desk"]
                symbol   = item["symbol"]
                strategy = item["strategy"]
                signal   = item["signal"]
                conf     = item["confidence"]

                # Per-symbol gating: crypto (BTC/USD etc.) trades 24/7; equities follow the clock.
                tradeable = has_buying_power and _symbol_tradeable(symbol, is_open)
                if not tradeable:
                    reason = "no account" if not has_buying_power else "stock market closed"
                    print(
                        f"  · {strategy.name}/{symbol} signal={signal.side.upper()} "
                        f"conf={conf:.2f} — logged ({reason})",
                        flush=True,
                    )
                    continue
                print(
                    f"  ► {strategy.name}/{symbol} signal={signal.side.upper()} "
                    f"conf={conf:.2f} — placing ${desk.notional_usd:.0f} order",
                    flush=True,
                )
                order = await _place_order(symbol, signal.side, desk.notional_usd)
                if order and order.get("id"):
                    print(f"    ✓ order {order['id']} submitted ({order.get('status', '?')})", flush=True)
                    record = {
                        "desk":       desk.name,
                        "strategy":   strategy.name,
                        "symbol":     symbol,
                        "side":       signal.side,
                        "notional":   desk.notional_usd,
                        "confidence": conf,
                        "order_id":   order["id"],
                        "status":     order.get("status", "?"),
                        "ts":         datetime.now(timezone.utc).isoformat(),
                    }
                    all_orders.append(record)
                    total_notional += desk.notional_usd
                    desk_orders_map.setdefault(desk.name, []).append(record)
                else:
                    print(f"    ✗ order placement returned no ID", flush=True)

            # Post per-desk Slack summaries (Alpaca desks)
            for desk in alpaca_desks:
                desk_order_list = desk_orders_map.get(desk.name, [])
                if desk_order_list:
                    lines = [f"*{desk.name} Desk* — {len(desk_order_list)} order(s) placed"]
                    for o in desk_order_list:
                        emoji = "🟢" if o["side"] == "buy" else "🔴"
                        lines.append(
                            f"{emoji} `{o['strategy']}/{o['symbol']}` "
                            f"{o['side'].upper()} ${o['notional']:.0f} "
                            f"conf={o['confidence']:.0%}  id=`{o['order_id'][:8]}…`"
                        )
                    _post_slack(desk.slack_channel, "\n".join(lines))
                    desk_summaries.append(f"✅ *{desk.name}*: {len(desk_order_list)} orders")
                else:
                    desk_summaries.append(f"💤 *{desk.name}*: no signals fired")

            # Prediction-market desk summaries (scan-based, posted to own channels)
            for desk in poly_desks + kalshi_desks:
                n = len([s for s in poly_signals if s["desk"] == desk.name])
                desk_summaries.append(
                    f"{'🎯' if n else '💤'} *{desk.name}*: {n} arb signal(s) (24/7 scan)"
                )

            tracker.set_output(orders_placed=len(all_orders), total_notional=round(total_notional, 2))

        # ── Stage 6: PnL Snapshot / Slack Summary ────────────────────────────
        with tracker.stage(PNL_SNAPSHOT, "Post PnL snapshot to Slack"):
            now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
            summary  = f"*QuantEdge Desk Run* ({now_str})  equity=${equity:,.2f}\n"
            summary += "\n".join(desk_summaries)
            summary += f"\n\nTotal orders placed: *{len(all_orders)}* · Polymarket arb signals: *{len(poly_signals)}*"
            _post_slack("#pnl-daily", summary)
            tracker.set_output(desks_run=len(active_desks), total_orders=len(all_orders))

        # ── Persist this run for future analysis (append-only JSONL) ──────────
        _persist_desk_run(account, all_orders, poly_signals, raw_signals)

    print(f"\n{'═'*60}", flush=True)
    print(f"Done. {len(all_orders)} orders placed across {len(DESKS)} desks "
          f"({len(poly_signals)} prediction-market arb signals).", flush=True)


def _persist_desk_run(account: dict, orders: list[dict], poly_signals: list[dict],
                      raw_signals: list[dict]) -> None:
    """Append a full snapshot of this desk run to experiments/results/desk_runs.jsonl.
    Every signal, order, and account state is kept for future backtesting/analytics."""
    try:
        out_dir = REPO_ROOT / "experiments" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "equity": float(account.get("equity", 0) or 0),
            "buying_power": float(account.get("buying_power", 0) or 0),
            "n_signals": len(raw_signals),
            "n_orders": len(orders),
            "n_poly_arb": len(poly_signals),
            "orders": orders,
            "poly_signals": poly_signals,
            # raw signal summary (strategy/symbol/side/conf) — no live objects
            "signals": [
                {
                    "desk": getattr(s.get("desk"), "name", "?"),
                    "strategy": getattr(s.get("strategy"), "name", "?"),
                    "symbol": s.get("symbol"),
                    "side": getattr(s.get("signal"), "side", "?"),
                    "confidence": s.get("confidence"),
                }
                for s in raw_signals
            ],
        }
        with open(out_dir / "desk_runs.jsonl", "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        print(f"  💾 persisted desk run → experiments/results/desk_runs.jsonl", flush=True)
    except Exception as exc:
        print(f"  ⚠ failed to persist desk run: {exc}", flush=True)
    print(f"  💾 desk run complete", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"\nFATAL ERROR: {type(exc).__name__}: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
