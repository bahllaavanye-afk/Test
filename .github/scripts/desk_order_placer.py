"""
Desk Order Placer — runs every 15 minutes during market hours.

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
]

# ── Alpaca REST client (direct HTTP, no SDK dependency) ───────────────────────

ALPACA_PAPER_BASE    = "https://paper-api.alpaca.markets"
ALPACA_DATA_BASE     = "https://data.alpaca.markets"


async def _alpaca_get(path: str, params: dict | None = None, data_api: bool = False) -> dict:
    import urllib.request, urllib.parse
    base = ALPACA_DATA_BASE if data_api else ALPACA_PAPER_BASE
    url  = base + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID":     ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


async def _alpaca_post(path: str, body: dict) -> dict:
    import urllib.request
    url     = ALPACA_PAPER_BASE + path
    payload = json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, headers={
        "APCA-API-KEY-ID":     ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type":        "application/json",
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


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
        print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY not set", flush=True)
        sys.exit(1)

    async with PipelineTracker("desk_trading") as tracker:

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
        account = None
        bars_fetched = 0
        symbols_fetched: list[str] = []
        with tracker.stage(DATA_FETCH, "Fetch account and market bars"):
            account = await _get_account()
            if account is None:
                raise RuntimeError("could not reach Alpaca paper account")

            equity = float(account.get("equity",       0))
            cash   = float(account.get("cash",         0))
            buying = float(account.get("buying_power", 0))
            print(f"  Account equity=${equity:.2f}  cash=${cash:.2f}  buying_power=${buying:.2f}", flush=True)

            active_desks = [d for d in DESKS if not DESK_FILTER or DESK_FILTER in d.name.lower()]
            if DESK_FILTER and not active_desks:
                raise RuntimeError(f"no desk matches filter '{DESK_FILTER}'")

            # Pre-fetch bars for all unique symbols across all active desks
            all_symbols = list({s for desk in active_desks for s in desk.symbols})
            bars_cache: dict[str, object] = {}
            for sym in all_symbols:
                df = await _get_bars(sym)
                if df is not None and len(df) >= 50:
                    bars_cache[sym] = df
                    bars_fetched += 1
                    symbols_fetched.append(sym)
            tracker.set_output(bars_fetched=bars_fetched, symbols=symbols_fetched)

        # ── Stage 3: Signal Generation ────────────────────────────────────────
        raw_signals: list[dict] = []
        with tracker.stage(SIGNAL_GENERATION, "Generate trading signals"):
            for desk in active_desks:
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
        with tracker.stage(ORDER_EXECUTION, "Place orders"):
            # Group approved signals by desk so we can still post per-desk summaries
            desk_orders_map: dict[str, list[dict]] = {}
            for item in approved_signals:
                desk     = item["desk"]
                symbol   = item["symbol"]
                strategy = item["strategy"]
                signal   = item["signal"]
                conf     = item["confidence"]

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

            # Post per-desk Slack summaries
            for desk in active_desks:
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

            tracker.set_output(orders_placed=len(all_orders), total_notional=round(total_notional, 2))

        # ── Stage 6: PnL Snapshot / Slack Summary ────────────────────────────
        with tracker.stage(PNL_SNAPSHOT, "Post PnL snapshot to Slack"):
            now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
            summary  = f"*QuantEdge Desk Run* ({now_str})  equity=${equity:,.2f}\n"
            summary += "\n".join(desk_summaries)
            summary += f"\n\nTotal orders placed: *{len(all_orders)}*"
            _post_slack("#pnl-daily", summary)
            tracker.set_output(desks_run=len(active_desks), total_orders=len(all_orders))

    print(f"\n{'═'*60}", flush=True)
    print(f"Done. {len(all_orders)} orders placed across {len(DESKS)} desks.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
