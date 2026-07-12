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
    # True for venues that never close (Alpaca crypto trades 24/7). The equity
    # clock gated ALL desks for weeks — the "Crypto 24/7" workflow ran nights
    # and weekends only to print "market closed" and exit with 0 orders.
    always_open:     bool = False


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
            # Advanced strategies that already exist in the library but were never
            # deployed (verified to run on Alpaca daily bars): HMM regime detection,
            # Larry Connors RSI-2 pullback, Donchian breakout, low-vol anomaly,
            # overnight-drift anomaly.
            "hmm_regime", "rsi2_pullback", "donchian_breakout",
            "low_volatility", "overnight_return",
        ],
        notional_usd=500.0,
        confidence_min=0.60,
    ),
    DeskConfig(
        name="Crypto",
        slack_channel="#desk-crypto",
        # Full Alpaca US crypto universe (majors + liquid alts) — the desk was
        # stuck on 4 pairs while Alpaca supports ~20. More symbols = more
        # independent shots at a ≥conf_min setup every 24/7 run; per-desk
        # top-K + Kelly caps keep total exposure unchanged.
        symbols=[
            "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD",
            "LTC/USD", "DOGE/USD", "LINK/USD", "UNI/USD",
            "AAVE/USD", "BCH/USD", "DOT/USD", "XRP/USD",
        ],
        # basis_carry / funding_rate_arb (Binance 451 geo-block from US
        # runners) and mvrv_zscore_timing (CoinGecko now requires an API key)
        # can NEVER fetch their data here — they errored on every run. Removed
        # until their data is rerouted (see agent-fix-needed issue); keeping
        # them was pure log noise masquerading as desk coverage.
        strategy_names=[
            "crypto_adaptive_trend", "mean_reversion", "breakout",
            "btc_eth_stat_arb", "intraday_seasonality",
            "on_chain_exchange_netflow", "vol_of_vol_timing",
            # Advanced additions verified to run on Alpaca daily bars:
            # Avellaneda-Stoikov market-making, Donchian breakout, RSI-2 pullback.
            "avellaneda_stoikov_mm", "donchian_breakout", "rsi2_pullback",
        ],
        notional_usd=300.0,
        confidence_min=0.60,
        always_open=True,
    ),
    DeskConfig(
        name="Options",
        slack_channel="#desk-options",
        symbols=["SPY", "QQQ", "AAPL", "TSLA", "NVDA"],
        # Income structures (wheel/condor/credit-spread — the Options Alpha core,
        # see docs/research/OPTIONS_ALPHA_DEEP_2026.md §3) trade the underlying as
        # a directional proxy until real multi-leg routing lands. They need
        # iv_rank injected into bars (done after the bars fetch below) — without
        # it their analyze() returns None on every call. covered_call is NOT
        # wired: it requires existing share inventory the desk doesn't track.
        strategy_names=[
            "vix_mean_reversion", "gamma_exposure", "skew_arb",
            "vrp_systematic", "dispersion_trading", "vol_term_structure",
            "vol_of_vol_timing",
            "wheel", "iron_condor", "credit_spread_income",
            # Short-vol carry — a genuine "profit from no movement" income
            # strategy (verified to fire on daily bars), complementing the
            # premium-selling income structures above.
            "vol_carry_short",
        ],
        notional_usd=400.0,
        confidence_min=0.60,
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
        confidence_min=0.60,
    ),
    DeskConfig(
        name="Macro/FX",
        slack_channel="#desk-fx-rates",
        # INDA/EPI/SMIN = India sleeve (docs/research/INDIA_GLOBAL_SOTA_2026.md §1):
        # NSE is the largest derivatives market globally and Indian equities carry a
        # documented momentum premium — tradable today as US ETFs on Alpaca, no new broker.
        symbols=["GLD", "TLT", "UUP", "EWJ", "EEM", "INDA", "EPI", "SMIN"],
        strategy_names=[
            "cross_asset_carry", "sector_rotation", "time_series_momentum",
            "intraday_fomc_momentum", "pead_sue", "multi_factor_equity",
            "analyst_revision_momentum",
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
            # ETF statistical-arbitrage (verified to run) — market-neutral,
            # fits the stat-arb desk directly.
            "stat_arb_etf",
        ],
        notional_usd=600.0,
        confidence_min=0.60,
    ),
    DeskConfig(
        name="International",
        slack_channel="#desk-equities",
        # Country-ETF rotation (docs/research/COUNTRY_DESKS_2026.md): documented
        # country momentum/reversal premia, tradable on Alpaca US-listed ETFs.
        symbols=["EWJ", "FXI", "EWY", "EWT", "EWZ", "EWW", "EWC",
                 "EWU", "EWG", "EWQ", "EZA", "EIDO", "VNM"],
        strategy_names=[
            "cross_sectional_momentum", "time_series_momentum",
            "mean_reversion", "low_volatility", "sector_rotation",
        ],
        notional_usd=400.0,
        confidence_min=0.60,
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
    """Blocking urllib call — run via asyncio.to_thread to avoid blocking event loop.

    Retries HTTP 429 with backoff. The free data tier throttles aggressively, and
    firing every symbol's bars request at once used to 429 nearly all of them
    (bars_fetched=2/12 → signals_generated=0 → no trades). Callers now batch
    symbols into one request each, and this handles any residual rate-limiting."""
    import urllib.request, urllib.parse, urllib.error
    import time as _time
    base = ALPACA_DATA_BASE if data_api else ALPACA_PAPER_BASE
    url  = base + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID":     ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:   # 8s per call
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 2:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    delay = float(retry_after) if retry_after else 1.5 * (attempt + 1)
                except (TypeError, ValueError):
                    delay = 1.5 * (attempt + 1)
                _time.sleep(min(delay, 5.0))
                continue
            raise
    raise RuntimeError("unreachable")  # loop either returns or raises


async def _alpaca_get(path: str, params: dict | None = None, data_api: bool = False) -> dict:
    return await asyncio.to_thread(_alpaca_get_sync, path, params, data_api)


def _alpaca_post_sync(path: str, body: dict) -> dict:
    import urllib.error
    import urllib.request
    url     = ALPACA_PAPER_BASE + path
    payload = json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, headers={
        "APCA-API-KEY-ID":     ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type":        "application/json",
        # No UA = Python-urllib default, which Cloudflare-fronted APIs 403.
        # Same bug class already hit the LLM cascade (error 1010) and Discord.
        "User-Agent":          "Mozilla/5.0 (X11; Linux x86_64) QuantEdge-Desk/1.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        # Surface Alpaca's actual refusal reason — a bare "HTTP Error 403"
        # hid whether it was Cloudflare, account permissions, or the payload.
        detail = ""
        try:
            detail = exc.read().decode(errors="replace")[:300]
        except Exception:  # noqa: BLE001
            pass
        print(f"    ⚠ alpaca POST {path} → {exc.code}: {detail}", flush=True)
        raise


async def _alpaca_post(path: str, body: dict) -> dict:
    return await asyncio.to_thread(_alpaca_post_sync, path, body)


def _alpaca_delete_sync(path: str) -> dict:
    import urllib.error
    import urllib.request
    req = urllib.request.Request(ALPACA_PAPER_BASE + path, method="DELETE", headers={
        "APCA-API-KEY-ID":     ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "User-Agent":          "Mozilla/5.0 (X11; Linux x86_64) QuantEdge-Desk/1.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return {"status": resp.status, "body": json.loads(resp.read() or "[]")}
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode(errors="replace")[:200]
        except Exception:  # noqa: BLE001
            pass
        print(f"    ⚠ alpaca DELETE {path} → {exc.code}: {detail}", flush=True)
        raise


AUTO_FLATTEN_ON_NEGATIVE_CASH = os.environ.get("AUTO_FLATTEN_ON_NEGATIVE_CASH", "1") == "1"


async def recover_negative_cash(account: dict) -> bool:
    """Auto-recovery for the account state that blocked the first trade for
    days: cash deeply negative with $0 available (orphaned notional buys that
    nothing tracks — Alpaca then 403s every crypto order with
    'insufficient balance for USD'). Flatten the orphaned PAPER book once:
    cancel all open orders + close all positions, freeing the cash.

    Triple-guarded: paper URL asserted, TRADING_MODE must not be live, and
    AUTO_FLATTEN_ON_NEGATIVE_CASH=0 disables it entirely."""
    if not AUTO_FLATTEN_ON_NEGATIVE_CASH:
        return False
    cash = float(account.get("cash", 0) or 0)
    nmbp = float(account.get("non_marginable_buying_power", 0) or 0)
    if cash >= 0 or nmbp > 0:
        return False
    if "paper" not in ALPACA_PAPER_BASE or os.environ.get("TRADING_MODE", "paper") == "live":
        print("  🛑 negative cash but NOT a paper endpoint — refusing to auto-flatten", flush=True)
        return False
    print(f"  🚑 RECOVERY: cash={cash:.2f}, available=0 — flattening orphaned paper book "
          f"(cancel all orders, close all positions) to restore trading cash", flush=True)
    try:
        await asyncio.to_thread(_alpaca_delete_sync, "/v2/orders")
        out = await asyncio.to_thread(_alpaca_delete_sync, "/v2/positions?cancel_orders=true")
        closed = out.get("body") or []
        print(f"  🚑 RECOVERY: close-all accepted for {len(closed)} position(s) — "
              f"cash frees as closes fill; next run trades normally", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  🚑 RECOVERY failed: {str(exc)[:120]}", flush=True)
        return False


async def _get_account() -> dict | None:
    try:
        return await _alpaca_get("/v2/account")
    except Exception as exc:
        print(f"  ✗ get_account failed: {exc}", flush=True)
        return None


def _bars_list_to_df(bars_list: list) -> "pd.DataFrame | None":
    """Alpaca bar dicts → normalized OHLCV DataFrame (None if empty)."""
    import pandas as pd
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


# 420 calendar days ≈ 290 trading days. Was 300d (~200 trading days), which is
# BELOW the 252-row (1y) minimum that range/premium strategies like iron_condor
# require — so they returned None on every desk run regardless of setup. Without
# an explicit start Alpaca returns only the current partial day, which failed the
# >=50-row minimum and left the bars cache empty on every run.
def _bars_start() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=420)).strftime("%Y-%m-%dT%H:%M:%SZ")


async def _get_bars(symbol: str, timeframe: str = "1Day", limit: int = 300) -> "pd.DataFrame | None":
    try:
        is_crypto = "/" in symbol
        if is_crypto:
            path   = "/v1beta3/crypto/us/bars"
            params = {"symbols": symbol, "timeframe": timeframe, "limit": limit, "start": _bars_start()}
        else:
            path   = f"/v2/stocks/{symbol}/bars"
            params = {"timeframe": timeframe, "limit": limit, "adjustment": "split", "start": _bars_start()}

        data = await _alpaca_get(path, params, data_api=True)
        bars_list = data.get("bars", {}).get(symbol, []) if is_crypto else data.get("bars", [])
        return _bars_list_to_df(bars_list)

    except Exception as exc:
        print(f"    ⚠ bars fetch failed for {symbol}: {exc}", flush=True)
        return None


async def _get_bars_batch(symbols: list[str], timeframe: str = "1Day",
                          limit: int = 300) -> "dict[str, pd.DataFrame]":
    """Fetch bars for many symbols in as few requests as possible.

    Alpaca's data API takes a comma-separated ``symbols=`` for both crypto
    (/v1beta3/crypto/us/bars) and stocks (/v2/stocks/bars), returning
    ``{"bars": {symbol: [...]}}``. Collapsing ~12 concurrent single-symbol
    calls into one request per asset class is what stops the free-tier 429s
    that were zeroing out bars_fetched — and therefore signals and trades —
    on every run. Paginates on next_page_token and chunks to keep URLs sane."""
    import pandas as pd
    out: dict[str, pd.DataFrame] = {}
    crypto = [s for s in symbols if "/" in s]
    stocks = [s for s in symbols if "/" not in s]

    async def _fetch(path: str, syms: list[str], extra: dict) -> None:
        CHUNK = 20
        for i in range(0, len(syms), CHUNK):
            chunk = syms[i:i + CHUNK]
            page_token: str | None = None
            while True:
                params = {"symbols": ",".join(chunk), "timeframe": timeframe,
                          "limit": limit, "start": _bars_start(), **extra}
                if page_token:
                    params["page_token"] = page_token
                try:
                    data = await _alpaca_get(path, params, data_api=True)
                except Exception as exc:
                    print(f"    ⚠ batch bars fetch failed for {chunk}: {exc}", flush=True)
                    break
                for sym, blist in (data.get("bars") or {}).items():
                    df = _bars_list_to_df(blist)
                    if df is None:
                        continue
                    out[sym] = pd.concat([out[sym], df]).sort_index() if sym in out else df
                page_token = data.get("next_page_token")
                if not page_token:
                    break
            await asyncio.sleep(0.3)  # gentle spacing between chunks

    if crypto:
        await _fetch("/v1beta3/crypto/us/bars", crypto, {})
    if stocks:
        # feed=iex is the free-tier default; explicit keeps it working without a
        # paid SIP subscription.
        await _fetch("/v2/stocks/bars", stocks, {"adjustment": "split", "feed": "iex"})
    return out


# Vol targeting (Moreira-Muir 2017): scale size toward a constant risk budget.
# High realized vol → smaller size, calm markets → larger, clamped so it can
# only halve or double the Kelly base. The single most robust documented
# Sharpe improvement across asset classes.
_TARGET_ANNUAL_VOL = 0.20

# Daily loss circuit breaker: if account equity is down more than this vs the
# broker's prior-close equity (Alpaca `last_equity`), the run places NO new
# orders. Stateless — no local ledger to drift or lose.
DAILY_LOSS_CAP_PCT = float(os.environ.get("DAILY_LOSS_CAP_PCT", "0.02"))


def daily_loss_cap_hit(equity: float, last_equity: float,
                       cap: float = None) -> bool:
    cap = DAILY_LOSS_CAP_PCT if cap is None else cap
    if equity <= 0 or last_equity <= 0:
        return False   # unknown baseline — don't false-trigger
    return equity < last_equity * (1.0 - cap)


def _vol_scalar(bars) -> float:
    """target/realized annualized vol from 20d closes, clamped [0.5, 2.0].
    1.0 when bars are absent/short/degenerate — sizing then falls back to
    pure Kelly, never crashes."""
    try:
        import numpy as np
        closes = bars["close"].astype(float)
        if len(closes) < 21:
            return 1.0
        rets = np.log(closes / closes.shift(1)).dropna().tail(20)
        realized = float(rets.std() * np.sqrt(252))
        if not (realized > 0):
            return 1.0
        return float(min(max(_TARGET_ANNUAL_VOL / realized, 0.5), 2.0))
    except Exception:  # noqa: BLE001 — sizing must never take the desk down
        return 1.0


def _kelly_notional(equity: float, confidence: float, max_pct: float = 0.03,
                    bars=None) -> float:
    """Half-Kelly sizing scaled to a constant vol target, capped at max_pct."""
    p = min(max(0.50 + (confidence - 0.60) * 1.25, 0.35), 0.75)
    b = 1.25  # avg_win / avg_loss
    q = 1.0 - p
    kelly_f   = max((p * b - q) / b, 0.0)
    half_kelly = kelly_f * 0.5
    capped     = min(half_kelly, max_pct)
    scalar     = _vol_scalar(bars) if bars is not None else 1.0
    return max(equity * capped * scalar, 50.0)


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
    # Newly-deployed advanced strategies
    "donchian_breakout":         [2],         # breakout — trending only
    "rsi2_pullback":             [1],         # Connors RSI-2 — mean-revert / range
    "vol_carry_short":           [1],         # short vol — profits from calm
    "hmm_regime":                [0, 1, 2],   # adapts to regime internally
    "stat_arb_etf":              [0, 1, 2],   # market-neutral
    "avellaneda_stoikov_mm":     [0, 1, 2],   # market-making
    "low_volatility":            [0, 1, 2],   # defensive factor
    "overnight_return":          [0, 1, 2],   # overnight-drift anomaly
    # Options / vol strategies (run in all regimes; vol strategies especially useful in bear)
    "gamma_exposure":            [0, 1, 2],
    "skew_arb":                  [0, 1, 2],
    "vrp_systematic":            [0, 1, 2],
    "dispersion_trading":        [0, 1, 2],
    "vol_term_structure":        [0, 1, 2],
    # Options income (premium selling — avoid strong bear trends)
    "wheel":                     [1, 2],
    "iron_condor":               [1],      # range-bound only
    "credit_spread_income":      [1, 2],
    "covered_call":              [1, 2],
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


def _inject_iv_rank(df) -> None:
    """Attach an HV-based iv_rank proxy to a bars DataFrame (via df.attrs).

    The options income strategies (wheel / iron_condor / credit_spread_income)
    gate on data.attrs['iv_rank'] and silently return None without it — wiring
    them into the desk without this injection would be pure log noise. Proxy:
    percentile-rank of 20-day realized vol within the fetched history (~200d).
    Same construction the strategies use internally for backtests.
    """
    import numpy as np
    try:
        log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
        if len(log_ret) < 60:
            return
        hv20 = log_ret.rolling(20).std() * np.sqrt(252)
        hv20 = hv20.dropna()
        cur, lo, hi = float(hv20.iloc[-1]), float(hv20.min()), float(hv20.max())
        df.attrs["iv_rank"] = (cur - lo) / max(hi - lo, 1e-6) * 100.0
    except Exception:
        pass  # strategies just skip when iv_rank is absent


# ── Performance-weighted sizing (the self-scaling loop) ───────────────────────
# Read the platform's live per-strategy P&L ranking and scale winners up /
# losers down (Marshall Wace TOPS-style, see docs/research/GLOBAL_QUANT_FIRMS_2026.md).
# Uses the public demo token exactly like the website does; best-effort — any
# failure returns {} and every strategy sizes at 1.0x. Bounds keep it sane.

_WEIGHT_MAX = 1.3
_WEIGHT_MIN = 0.6
_API_BASE = os.environ.get("QUANTEDGE_API_URL", "https://quantedge-api-agb8.onrender.com").rstrip("/")


def _fetch_performance_weights() -> dict[str, float]:
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{_API_BASE}/api/v1/auth/demo", data=b"", method="POST",
            headers={"User-Agent": "QuantEdge-Desk/1.0", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            token = json.loads(r.read()).get("access_token", "")
        if not token:
            return {}
        req = urllib.request.Request(
            f"{_API_BASE}/api/v1/leaderboard/live",
            headers={"User-Agent": "QuantEdge-Desk/1.0", "Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            strategies = json.loads(r.read()).get("strategies", [])
    except Exception as exc:
        print(f"  · performance weights unavailable ({exc}) — sizing all at 1.0x", flush=True)
        return {}

    weights: dict[str, float] = {}
    for s in strategies:
        n = int(s.get("trades") or 0)
        if n < 5:
            continue  # not enough evidence to re-weight
        pnl = float(s.get("total_pnl") or 0)
        sharpe = s.get("pnl_sharpe")
        if pnl > 0 and (sharpe or 0) > 0:
            w = min(_WEIGHT_MAX, 1.0 + min(float(sharpe) * 0.15, 0.3))
        elif pnl < 0:
            w = _WEIGHT_MIN
        else:
            w = 1.0
        weights[s.get("strategy", "")] = round(w, 3)
    if weights:
        print(f"✓ Performance weights active for {len(weights)} strategies", flush=True)
    return weights


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


def _trimmed_strategies() -> set:
    """Names retired by strategy_trimmer.py — they must NOT trade until recovered."""
    import json
    from pathlib import Path
    f = Path(__file__).resolve().parent.parent / "state" / "strategy_trims.json"
    if not f.exists():
        return set()
    try:
        return set(json.loads(f.read_text()).keys())
    except Exception:
        return set()


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

    trimmed = _trimmed_strategies()
    strategies = []
    for sname in desk.strategy_names:
        if sname in trimmed:
            print(f"  ✂ strategy '{sname}' retired by trimmer — skipping", flush=True)
            continue
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
                signal = await asyncio.wait_for(strategy.analyze(df, symbol), timeout=10.0)
            except asyncio.TimeoutError:
                print(f"  ⚠ {strategy.name}/{symbol} analyze() timed out (>10s) — skipped", flush=True)
                continue
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
                # Daily loss circuit breaker (stateless — uses the broker's own
                # prior-close equity): if the account is down more than the cap
                # since yesterday's close, place NO new orders this run. Signals
                # are still generated and logged for the record.
                last_equity = float(account.get("last_equity", 0) or 0)
                _loss_cap_hit = daily_loss_cap_hit(equity, last_equity)
                await recover_negative_cash(account)
                if _loss_cap_hit:
                    print(f"  🛑 DAILY LOSS CAP: equity down {1.0 - equity / last_equity:.2%} vs prior "
                          f"close (cap {DAILY_LOSS_CAP_PCT:.0%}) — no new orders this run", flush=True)

            active_desks = [d for d in DESKS if not DESK_FILTER or DESK_FILTER in d.name.lower()]
            if DESK_FILTER and not active_desks:
                raise RuntimeError(f"no desk matches filter '{DESK_FILTER}'")

            # Pre-fetch bars for all unique symbols in ONE request per asset
            # class (crypto + stocks). Firing every symbol concurrently used to
            # 429 nearly all of them on the free data tier.
            all_symbols = list({s for desk in active_desks for s in desk.symbols})
            bars_cache: dict[str, object] = {}
            fetched = await _get_bars_batch(all_symbols)
            for sym in all_symbols:
                df = fetched.get(sym)
                if df is None:
                    print(f"    ⚠ {sym}: no bars returned", flush=True)
                    continue
                if len(df) >= 50:
                    _inject_iv_rank(df)  # options income strategies gate on this
                    bars_cache[sym] = df
                    bars_fetched += 1
                    symbols_fetched.append(sym)
                else:
                    # A silent drop here hid the missing-start bug for weeks.
                    print(f"    ⚠ {sym}: only {len(df)} bars (<50) — dropped", flush=True)
            tracker.set_output(bars_fetched=bars_fetched, symbols=symbols_fetched)

        # Detect market regime from SPY bars (0=bear, 1=sideways, 2=bull)
        _REGIME_NAMES = {0: "bear", 1: "sideways", 2: "bull"}
        spy_df = bars_cache.get("SPY")
        current_regime: int = _detect_regime_from_bars(spy_df) if spy_df is not None else 1
        print(f"  Market regime: {_REGIME_NAMES[current_regime]} ({current_regime})", flush=True)

        # Live P&L-based sizing weights (self-scaling loop; {} on any failure)
        _perf_weights = await asyncio.to_thread(_fetch_performance_weights)

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
                            # Hard per-strategy timeout: 18 registry strategies
                            # do network I/O in analyze() (contract-test audit);
                            # an unbounded await let one slow fetch stall the
                            # WHOLE desk run for minutes. 10s bounds them all.
                            signal = await asyncio.wait_for(strategy.analyze(df, symbol), timeout=10.0)
                        except asyncio.TimeoutError:
                            print(f"  ⚠ {strategy.name}/{symbol} analyze() timed out (>10s) — skipped", flush=True)
                            continue
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
        # ── Cross-strategy signal ensembling (per desk+symbol) ───────────────
        # Multiple independent strategies agreeing is stronger evidence than any
        # single one (stacked/ensemble-signal literature): combine agreeing
        # confidences as 1-prod(1-ci). Conflicting directions on the same desk
        # symbol = no edge -> stand aside entirely.
        from collections import defaultdict
        _groups: dict = defaultdict(list)
        for _it in raw_signals:
            _side = str(getattr(_it["signal"], "side", "")).lower()
            _groups[(_it["desk"].name, _it["symbol"], _side)].append(_it)
        _ensembled: list[dict] = []
        for (_dn, _sym, _side), _items in _groups.items():
            _opp = "sell" if _side == "buy" else "buy"
            if (_dn, _sym, _opp) in _groups:
                print(f"  · ensemble[{_dn}]: {_sym} {_side}/{_opp} conflict — stand aside", flush=True)
                continue
            if len(_items) == 1:
                _ensembled.append(_items[0]); continue
            _keep = dict(max(_items, key=lambda x: x["confidence"]))
            _p = 1.0
            for _x in _items:
                _p *= (1.0 - min(max(float(_x["confidence"]), 0.0), 1.0))
            _keep["confidence"] = round(1.0 - _p, 4)
            try:  # keep the Signal object consistent for downstream sizing
                _keep["signal"].confidence = _keep["confidence"]
            except Exception:  # noqa: BLE001 — frozen dataclass etc.
                pass
            _names = "+".join(getattr(_x["strategy"], "name", "?") for _x in _items)
            print(f"  · ensemble[{_dn}]: {_sym} {_side} x{len(_items)} ({_names}) -> conf={_keep['confidence']:.2f}", flush=True)
            _ensembled.append(_keep)
        raw_signals = _ensembled

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
        _account_ok = (float(account.get("buying_power", 0)) > 0
                       and not locals().get("_loss_cap_hit", False))
        with tracker.stage(ORDER_EXECUTION, "Place orders"):
            # The market clock only gates equity-hours desks — always-open desks
            # (crypto) trade through nights and weekends.
            if not is_open:
                print("  ⚠ Equity market closed — only always-open desks may trade", flush=True)
            if not _account_ok:
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

                desk_open = is_open or desk.always_open
                if not desk_open:
                    print(
                        f"  · {strategy.name}/{symbol} signal={signal.side.upper()} "
                        f"conf={conf:.2f} — logged ({desk.name} closed)",
                        flush=True,
                    )
                    continue
                if not _account_ok:
                    print(
                        f"  · {strategy.name}/{symbol} signal={signal.side.upper()} "
                        f"conf={conf:.2f} — logged (no account)",
                        flush=True,
                    )
                    continue
                kelly_notional = _kelly_notional(equity, conf, bars=bars_cache.get(symbol))
                # Self-scaling: winners (by live realized P&L) size up, losers down.
                perf_w = _perf_weights.get(strategy.name, 1.0)
                if perf_w != 1.0:
                    print(f"    · perf weight {perf_w:.2f}x for {strategy.name}", flush=True)
                    kelly_notional *= perf_w
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
