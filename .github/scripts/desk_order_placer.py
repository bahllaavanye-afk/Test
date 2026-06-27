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
import time
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
            "realized_vol_asymmetry", "analyst_revision_momentum",
        ],
        notional_usd=500.0,
        confidence_min=0.68,
    ),
    DeskConfig(
        name="Crypto",
        slack_channel="#desk-crypto",
        # The ONLY genuinely 24/7 venue here (Alpaca crypto). Widened so the desk
        # has real opportunity on weekends when every equity desk is closed.
        symbols=["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LTC/USD",
                 "LINK/USD", "UNI/USD", "AAVE/USD", "DOT/USD", "BCH/USD"],
        strategy_names=[
            "crypto_adaptive_trend", "mean_reversion", "breakout",
            "basis_carry", "btc_eth_stat_arb", "mvrv_zscore_timing",
            "intraday_seasonality", "funding_rate_arb",
            "on_chain_exchange_netflow", "vol_of_vol_timing",
        ],
        notional_usd=300.0,
        # Lowered 0.70→0.58: at 0.70 the 24/7 crypto desk almost never fired, so
        # there were no weekend trades at all. Paper mode — more activity is safe.
        confidence_min=0.58,
    ),
    DeskConfig(
        name="Options",
        slack_channel="#desk-options",
        symbols=["SPY", "QQQ", "AAPL", "TSLA", "NVDA"],
        strategy_names=[
            "vix_mean_reversion", "gamma_exposure", "skew_arb",
            "vrp_systematic", "dispersion_trading", "vol_term_structure",
            "vol_of_vol_timing",
        ],
        notional_usd=400.0,
        confidence_min=0.70,
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
        confidence_min=0.75,
    ),
    DeskConfig(
        name="Macro/FX",
        slack_channel="#desk-fx-rates",
        symbols=["GLD", "TLT", "UUP", "EWJ", "EEM"],
        strategy_names=[
            "cross_asset_carry", "sector_rotation", "time_series_momentum",
            "intraday_fomc_momentum", "pead_sue", "multi_factor_equity",
            "analyst_revision_momentum",
        ],
        notional_usd=400.0,
        confidence_min=0.68,
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

# ── Auto-tuned thresholds (written nightly by strategy_auto_tuner.py) ─────────

_TUNED_THRESHOLDS: dict[str, float] = {}
_TUNED_FILE = REPO_ROOT / "backend" / "performance_log" / "tuned_thresholds.json"
try:
    if _TUNED_FILE.exists():
        _data = json.loads(_TUNED_FILE.read_text())
        _TUNED_THRESHOLDS = {k: float(v) for k, v in _data.get("thresholds", {}).items()}
        if _TUNED_THRESHOLDS:
            print(f"✓ Loaded {len(_TUNED_THRESHOLDS)} auto-tuned thresholds", flush=True)
except Exception:
    pass

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


def _kelly_notional(equity: float, confidence: float, max_pct: float = 0.03) -> float:
    """Half-Kelly sizing: confidence score → win probability → Kelly fraction, capped at max_pct."""
    p = min(max(0.50 + (confidence - 0.60) * 1.25, 0.35), 0.75)
    b = 1.25  # avg_win / avg_loss
    q = 1.0 - p
    kelly_f   = max((p * b - q) / b, 0.0)
    half_kelly = kelly_f * 0.5
    capped     = min(half_kelly, max_pct)
    return max(equity * capped, 50.0)


async def _place_order(
    symbol: str,
    side: str,
    notional_usd: float,
    limit_price: float | None = None,
    client_order_id: str | None = None,
) -> dict | None:
    try:
        is_crypto = "/" in symbol
        body: dict = {
            "symbol":        symbol,
            "side":          side,
            "time_in_force": "gtc" if is_crypto else "day",
        }
        if client_order_id:
            body["client_order_id"] = client_order_id[:48]

        if limit_price and limit_price > 0:
            # Limit-first: post limit slightly through the market to maximise fill probability
            lp  = round(limit_price * (1.001 if side == "buy" else 0.999), 2)
            qty = round(notional_usd / lp, 6 if is_crypto else 2)
            body["type"]        = "limit"
            body["limit_price"] = str(lp)
            body["qty"]         = str(qty)
        elif is_crypto:
            quote_data = await _alpaca_get(
                "/v1beta3/crypto/us/latest/quotes",
                {"symbols": symbol},
                data_api=True,
            )
            ask = float((quote_data.get("quotes", {}).get(symbol, {}) or {}).get("ap", 0))
            if ask <= 0:
                return None
            body["type"] = "market"
            body["qty"]  = str(round(notional_usd / ask, 6))
        else:
            body["type"]     = "market"
            body["notional"] = str(round(notional_usd, 2))

        return await _alpaca_post("/v2/orders", body)
    except Exception as exc:
        print(f"    ⚠ place_order failed {symbol} {side}: {exc}", flush=True)
        return None


# ── Regime detection (SPY-based heuristic, no Redis dependency) ───────────────
# Mirrors the logic in backend/app/tasks/regime_monitor.py for use in CI.
# 0 = bear, 1 = sideways, 2 = bull

_STRATEGY_REGIME_MAP: dict[str, list[int]] = {
    "momentum":                  [2],
    "cross_sectional_momentum":  [2],
    "mean_reversion":            [1],
    "vwap_reversion":            [1],
    "rsi_macd":                  [1, 2],
    "breakout":                  [2],
    "supertrend":                [2],
    "pairs_trading":             [0, 1, 2],
    "btc_eth_stat_arb":          [0, 1, 2],
    "triangular_arb":            [0, 1, 2],
    "poly_binary_arb":           [0, 1, 2],
    "funding_rate_arb":          [0, 1, 2],
    "basis_carry":               [0, 1, 2],
    "vix_mean_reversion":        [0, 1],
    "liquidation_cascade_fade":  [0],
    "realized_vol_asymmetry":    [0, 1, 2],
    "analyst_revision_momentum": [1, 2],
    "on_chain_exchange_netflow": [0, 1, 2],
    "vol_of_vol_timing":         [0, 1, 2],
    # Equities (intraday / all-regime)
    "opening_range_breakout":    [1, 2],   # intraday breakout — avoid bear chop
    "residual_momentum":         [1, 2],   # factor momentum — avoid bear
    "idio_vol_anomaly":          [0, 1, 2],
    # Crypto
    "crypto_adaptive_trend":     [1, 2],   # trend strategy
    "mvrv_zscore_timing":        [0, 1, 2],
    "intraday_seasonality":      [0, 1, 2],
    # Options / vol strategies (run in all regimes; vol strategies especially useful in bear)
    "gamma_exposure":            [0, 1, 2],
    "skew_arb":                  [0, 1, 2],
    "vrp_systematic":            [0, 1, 2],
    "dispersion_trading":        [0, 1, 2],
    "vol_term_structure":        [0, 1, 2],
    # Polymarket
    "polymarket_sentiment_momentum": [1, 2],
    "poly_calibration_arb":      [0, 1, 2],
    "poly_late_resolution":      [0, 1, 2],
    # Macro/FX
    "cross_asset_carry":         [0, 1, 2],
    "sector_rotation":           [1, 2],
    "time_series_momentum":      [1, 2],
    "intraday_fomc_momentum":    [0, 1, 2],
    "pead_sue":                  [1, 2],
    "multi_factor_equity":       [1, 2],
    # StatArb
    "pca_stat_arb":              [0, 1, 2],
    "kalman_pairs":              [0, 1, 2],
    "stablecoin_depeg_arb":      [0, 1, 2],
}
_DEFAULT_REGIMES = [0, 1, 2]


def _detect_regime_from_bars(spy_df) -> int:
    """
    Compute market regime from SPY price data.
    Uses recent return + vol ratio heuristic matching regime_monitor.py fallback.
    Returns 0=bear, 1=sideways, 2=bull.
    """
    import numpy as np
    try:
        close = spy_df["close"].astype(float).values
        if len(close) < 40:
            return 1
        log_rets = np.diff(np.log(close))
        recent_ret = float(np.mean(log_rets[-20:]))
        recent_vol = float(np.std(log_rets[-20:]))
        long_vol   = float(np.std(log_rets[-min(252, len(log_rets)):]))
        vol_ratio  = recent_vol / max(long_vol, 1e-8)
        if recent_ret < -0.002 and vol_ratio > 1.3:
            return 0  # bear: negative drift + elevated vol
        if recent_ret > 0.001 and vol_ratio < 1.2:
            return 2  # bull: positive drift + calm vol
        return 1      # sideways
    except Exception:
        return 1


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
        account = None
        bars_fetched = 0
        symbols_fetched: list[str] = []
        equity = 0.0
        cash   = 0.0
        buying = 0.0
        with tracker.stage(DATA_FETCH, "Fetch account and market bars"):
            account = await _get_account()
            if account is None:
                # Account fetch failed (API unreachable or bad credentials).
                # Still run signal generation; order placement will be skipped.
                print("  ⚠ Account unavailable — running in signal-only mode (no orders placed)", flush=True)
                account = {"equity": 0, "cash": 0, "buying_power": 0}
            else:
                equity = float(account.get("equity",       0))
                cash   = float(account.get("cash",         0))
                buying = float(account.get("buying_power", 0))
                print(f"  Account equity=${equity:.2f}  cash=${cash:.2f}  buying_power=${buying:.2f}", flush=True)

            active_desks = [d for d in DESKS if not DESK_FILTER or DESK_FILTER in d.name.lower()]
            if DESK_FILTER and not active_desks:
                raise RuntimeError(f"no desk matches filter '{DESK_FILTER}'")

            # Pre-fetch bars for all unique symbols concurrently
            all_symbols = list({s for desk in active_desks for s in desk.symbols})
            bars_cache: dict[str, object] = {}
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

        # Detect market regime from SPY bars (0=bear, 1=sideways, 2=bull)
        _REGIME_NAMES = {0: "bear", 1: "sideways", 2: "bull"}
        spy_df = bars_cache.get("SPY")
        current_regime: int = _detect_regime_from_bars(spy_df) if spy_df is not None else 1
        print(f"  Market regime: {_REGIME_NAMES[current_regime]} ({current_regime})", flush=True)

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
        with tracker.stage(RISK_CHECK, "Apply confidence threshold + top-K filter"):
            for item in raw_signals:
                desk  = item["desk"]
                conf  = item["confidence"]
                sname = item["strategy"].name

                # Regime gate: skip strategies not allowed in current regime
                allowed_regimes = _STRATEGY_REGIME_MAP.get(sname, _DEFAULT_REGIMES)
                if current_regime not in allowed_regimes:
                    print(f"  · {sname}/{item['symbol']} skipped — regime {_REGIME_NAMES[current_regime]} not in {[_REGIME_NAMES[r] for r in allowed_regimes]}", flush=True)
                    continue

                # Use auto-tuned threshold if available, floored at desk minimum
                threshold = max(_TUNED_THRESHOLDS.get(sname, desk.confidence_min), desk.confidence_min)
                if conf < threshold:
                    print(f"  · {sname}/{item['symbol']} conf={conf:.2f} < {threshold:.2f} — skipped", flush=True)
                else:
                    approved_signals.append(item)

            # Top-K per desk: keep at most 3 highest-confidence signals per desk
            _TOP_K = 3
            desk_groups: dict[str, list[dict]] = {}
            for item in approved_signals:
                desk_groups.setdefault(item["desk"].name, []).append(item)
            top_k_signals: list[dict] = []
            for dname, items in desk_groups.items():
                ranked = sorted(items, key=lambda x: x["confidence"], reverse=True)
                top_k_signals.extend(ranked[:_TOP_K])
                dropped = len(ranked) - min(len(ranked), _TOP_K)
                if dropped:
                    print(f"  · top-K[{dname}]: dropped {dropped} lower-confidence signals", flush=True)
            approved_signals = top_k_signals
            filtered = len(raw_signals) - len(approved_signals)
            tracker.set_output(passed=len(approved_signals), filtered=filtered)

        # ── Stage 5: Order Execution ──────────────────────────────────────────
        all_orders: list[dict] = []
        desk_summaries: list[str] = []
        total_notional = 0.0
        _can_trade = is_open and float(account.get("buying_power", 0)) > 0
        with tracker.stage(ORDER_EXECUTION, "Place orders"):
            if not is_open:
                print("  ⚠ Market is closed — skipping order placement", flush=True)
                tracker.set_output(orders_placed=0, reason="market_closed")
            elif not _can_trade:
                print("  ⚠ Skipping order placement (no buying power / account unavailable)", flush=True)
                tracker.set_output(orders_placed=0, reason="account_unavailable")
            # Group approved signals by desk so we can still post per-desk summaries
            desk_orders_map: dict[str, list[dict]] = {}
            for item in approved_signals:
                desk     = item["desk"]
                symbol   = item["symbol"]
                strategy = item["strategy"]
                signal   = item["signal"]
                conf     = item["confidence"]

                if not _can_trade:
                    print(
                        f"  · {strategy.name}/{symbol} signal={signal.side.upper()} "
                        f"conf={conf:.2f} — logged (no account)",
                        flush=True,
                    )
                    continue
                kelly_notional = _kelly_notional(equity, conf)
                coid = f"qe-{strategy.name[:10]}-{symbol[:4].replace('/', '')}-{int(time.time())}"
                limit_price: float | None = None
                _df = bars_cache.get(symbol)
                if _df is not None and len(_df) > 0:
                    limit_price = float(_df["close"].iloc[-1])
                print(
                    f"  ► {strategy.name}/{symbol} signal={signal.side.upper()} "
                    f"conf={conf:.2f} — placing ${kelly_notional:.0f} limit-first order",
                    flush=True,
                )
                order = await _place_order(symbol, signal.side, kelly_notional,
                                           limit_price=limit_price, client_order_id=coid)
                if order and order.get("id"):
                    print(f"    ✓ order {order['id']} submitted ({order.get('status', '?')})", flush=True)
                    record = {
                        "desk":            desk.name,
                        "strategy":        strategy.name,
                        "symbol":          symbol,
                        "side":            signal.side,
                        "notional":        kelly_notional,
                        "confidence":      conf,
                        "order_id":        order["id"],
                        "client_order_id": coid,
                        "order_type":      order.get("type", "limit"),
                        "status":          order.get("status", "?"),
                        "ts":              datetime.now(timezone.utc).isoformat(),
                    }
                    all_orders.append(record)
                    total_notional += kelly_notional
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
    try:
        asyncio.run(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"\nFATAL ERROR: {type(exc).__name__}: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
