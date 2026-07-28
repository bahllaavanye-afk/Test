"""
Desk Order Placer — runs every 15 minutes during market hours.
Version: 3.0 — 9 desks (incl. TV Indicators + Commodities), ~100 wired strategies, real paper orders via Alpaca.

For each asset-class desk, fetches live OHLCV from Alpaca paper API,
runs the relevant strategies' analyze(), and places real paper orders
when signals fire with sufficient confidence.

No mock data. If Alpaca is unreachable, the desk is skipped entirely.
Results are posted to the desk-specific Discord channel.
"""
from __future__ import annotations

import asyncio
import json
import math
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

# yfinance-backed strategies run their fetch under a hard budget (see
# app/strategies/_failsoft.py); on a live desk give them more room than the
# 3.5s contract-test default — the desk's own 10s wait_for is the outer cap.
os.environ.setdefault("STRATEGY_ANALYZE_BUDGET_S", "8")

ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
CHAT_ENABLED = bool(os.environ.get("DISCORD_BOT_TOKEN", "").strip()
                    or os.environ.get("DISCORD_WEBHOOK_URL", "").strip())
DESK_FILTER       = os.environ.get("DESK_FILTER", "").strip().lower()

# ── Desk configuration ────────────────────────────────────────────────────────

class DeskConfig(NamedTuple):
    name:            str
    chat_channel:   str
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
        chat_channel="#desk-equities",
        # Scaled 10 -> 30 (2026-07-15): megacaps + liquid large-caps across
        # sectors. Bars come batched (comma-separated symbols=, chunk 20), so
        # more symbols costs one extra API call, not 20.
        symbols=["SPY", "QQQ", "IWM", "DIA",
                 "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "JPM",
                 "AMD", "AVGO", "NFLX", "CRM", "COST", "UNH", "XOM", "CVX",
                 "JNJ", "PG", "HD", "BAC", "WMT", "DIS", "V", "MA", "LLY", "ORCL"],
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
            # 2026-07-15 scale-up: contract-passing strategies that were in the
            # registry but wired to NO desk (60 found by the audit).
            "cci_reversion", "fifty_two_week_high", "triple_barrier_momentum",
            "ml_momentum", "ml_mean_reversion", "ml_breakout", "ensemble",
            "event_driven_gap", "open_close_revert",
            # 2026-07-18: documented calendar/reversion premia (see modules)
            "turn_of_month", "gap_fill_fade", "double_seven",
        ],
        notional_usd=500.0,
        confidence_min=0.60,
    ),
    DeskConfig(
        name="Crypto",
        chat_channel="#desk-crypto",
        # Full Alpaca US crypto universe (majors + liquid alts) — the desk was
        # stuck on 4 pairs while Alpaca supports ~20. More symbols = more
        # independent shots at a ≥conf_min setup every 24/7 run; per-desk
        # top-K + Kelly caps keep total exposure unchanged.
        symbols=[
            "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD",
            "LTC/USD", "DOGE/USD", "LINK/USD", "UNI/USD",
            "AAVE/USD", "BCH/USD", "DOT/USD", "XRP/USD",
            # 2026-07-15 scale-up to the rest of Alpaca's US pairs; an unlisted
            # pair simply returns no bars and is skipped (fail-soft, logged).
            "SHIB/USD", "SUSHI/USD", "YFI/USD", "GRT/USD",
            "CRV/USD", "MKR/USD", "XTZ/USD", "BAT/USD",
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
            # 2026-07-15 scale-up (previously unwired; all fail-soft guarded —
            # a missing data source degrades to None, never noise or a crash):
            "crypto_whale_momentum", "liquidation_cascade_fade",
            "funding_settlement_timer", "mvrv_zscore_timing",
        ],
        notional_usd=300.0,
        confidence_min=0.60,
        always_open=True,
    ),
    DeskConfig(
        name="Options",
        chat_channel="#desk-options",
        symbols=["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "IWM", "AMD", "META"],
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
            # 2026-07-15 scale-up: sentiment/flow + income structures that were
            # never desk-wired. covered_call stays out (needs share inventory).
            "options_pcr_reversal", "put_call_ratio_contrarian",
            "earnings_iv_crush", "options_gamma_scalp",
            "long_call_momentum", "cash_secured_put",
        ],
        notional_usd=400.0,
        confidence_min=0.60,
    ),
    DeskConfig(
        name="Polymarket",
        chat_channel="#desk-polymarket",
        symbols=["SPY"],   # proxy for market regime
        strategy_names=[
            "polymarket_sentiment_momentum", "poly_binary_arb",
            "poly_calibration_arb", "poly_late_resolution",
            # 2026-07-15 scale-up (previously unwired):
            "poly_market_maker", "poly_liquidity_provision",
            "poly_time_value_fade", "poly_cross_market_hedge",
        ],
        notional_usd=100.0,
        confidence_min=0.60,
    ),
    DeskConfig(
        name="Macro/FX",
        chat_channel="#desk-fx-rates",
        # INDA/EPI/SMIN = India sleeve (docs/research/INDIA_GLOBAL_SOTA_2026.md §1):
        # NSE is the largest derivatives market globally and Indian equities carry a
        # documented momentum premium — tradable today as US ETFs on Alpaca, no new broker.
        symbols=["GLD", "TLT", "UUP", "EWJ", "EEM", "INDA", "EPI", "SMIN",
                 # 2026-07-15 scale-up: full rates/credit/inflation ETF complex
                 "IEF", "SHY", "LQD", "HYG", "TIP", "DBC"],
        strategy_names=[
            "cross_asset_carry", "sector_rotation", "time_series_momentum",
            "intraday_fomc_momentum", "pead_sue", "multi_factor_equity",
            "analyst_revision_momentum",
            # 2026-07-15 scale-up: the whole macro/rates family was unwired.
            # All hard-budget guarded (yfinance) — slow/absent data -> None.
            "bond_equity_rotation", "central_bank_window", "macro_risk_barometer",
            "breakeven_inflation", "duration_momentum", "pmi_sector_rotation",
            "yield_curve_momentum", "yield_spread_reversion", "tlt_spy_rotation",
        ],
        notional_usd=400.0,
        confidence_min=0.60,
    ),
    DeskConfig(
        name="StatArb",
        chat_channel="#desk-stat-arb",
        symbols=["SPY", "QQQ", "IWM", "GLD", "TLT",
                 # 2026-07-15 scale-up: sector/region ETFs widen the pair pool
                 "XLF", "XLK", "XLE", "EFA", "EEM", "DIA", "MDY"],
        strategy_names=[
            "pairs_trading", "pca_stat_arb", "kalman_pairs",
            "triangular_arb", "stablecoin_depeg_arb",
            # ETF statistical-arbitrage (verified to run) — market-neutral,
            # fits the stat-arb desk directly.
            "stat_arb_etf",
            # 2026-07-15 scale-up (previously unwired):
            "ml_pca_arb", "lorentzian_knn",
        ],
        notional_usd=600.0,
        confidence_min=0.60,
    ),
    DeskConfig(
        name="Commodities",
        chat_channel="#desk-commodities",
        # Commodity exposure via US-listed ETF proxies (no futures account needed):
        # gold, silver, oil, natgas, agriculture, broad basket, miners, copper.
        # Time-series momentum is THE documented commodity premium (Moskowitz-
        # Ooi-Pedersen); Donchian breakout is the classic trend system for these.
        symbols=["GLD", "SLV", "USO", "UNG", "DBA", "PDBC", "GDX", "CPER",
                 # 2026-07-15 scale-up: platinum, uranium, grains
                 "PPLT", "URA", "CORN", "WEAT"],
        strategy_names=[
            "time_series_momentum", "cross_sectional_momentum",
            "donchian_breakout", "mean_reversion", "supertrend",
            "rsi2_pullback",
            # 2026-07-15 scale-up: the dedicated commodity family was unwired.
            "commodity_momentum", "commodity_reversion", "commodity_trend",
            "basis_carry",
        ],
        notional_usd=400.0,
        confidence_min=0.60,
    ),
    DeskConfig(
        name="TV Indicators",
        chat_channel="#desk-tv-indicators",
        # The 12 TradingView-community indicator strategies (maintained by the
        # tv-indicator-improvement agent) finally get a venue — they were in
        # the registry, contract-tested, SOTA-upgraded on a schedule... and
        # wired to no desk. Liquid megacap subset; bars are shared with the
        # Equities desk fetch (same batch), so this desk adds no API cost.
        symbols=["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "AMZN"],
        strategy_names=[
            "ema_stack_tv", "squeeze_pro_tv", "wave_trend_tv", "hull_suite_tv",
            "supertrend_rsi_tv", "kama_roc_tv", "vwap_bands_tv",
            "ichimoku_cloud_tv", "macd_divergence_tv", "adx_dmi_tv",
            "stoch_rsi_macd_tv", "elliott_wave_proxy_tv",
        ],
        notional_usd=300.0,
        confidence_min=0.60,
    ),
    DeskConfig(
        name="International",
        chat_channel="#desk-equities",
        # Country-ETF rotation (docs/research/COUNTRY_DESKS_2026.md): documented
        # country momentum/reversal premia, tradable on Alpaca US-listed ETFs.
        symbols=["EWJ", "FXI", "EWY", "EWT", "EWZ", "EWW", "EWC",
                 "EWU", "EWG", "EWQ", "EZA", "EIDO", "VNM",
                 # 2026-07-15 scale-up: AU, SG, TH, PL, AR, MY
                 "EWA", "EWS", "THD", "EPOL", "ARGT", "EWM"],
        strategy_names=[
            "cross_sectional_momentum", "time_series_momentum",
            "mean_reversion", "low_volatility", "sector_rotation",
            # 2026-07-15 scale-up:
            "ichimoku_cloud_tv", "triple_barrier_momentum", "fifty_two_week_high",
            "turn_of_month",
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


_tradable_crypto_cache: "set[str] | None" = None


async def _tradable_crypto_symbols() -> "set[str] | None":
    """Set of Alpaca-tradable crypto symbols ('BTC/USD' form), cached per process.

    Returns None when the lookup fails — callers must then NOT filter (fail-soft:
    a lookup blip must never shrink the universe). Fixes the wasted-signal class
    where a DELISTED pair (e.g. MKR/USD, 422 'asset not active') still had bars,
    generated a signal, and only failed at order time every single run.
    """
    global _tradable_crypto_cache
    if _tradable_crypto_cache is not None:
        return _tradable_crypto_cache
    try:
        assets = await _alpaca_get("/v2/assets", {"asset_class": "crypto", "status": "active"})
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(assets, list):
        return None
    out = {a["symbol"] for a in assets
           if isinstance(a, dict) and a.get("symbol") and a.get("tradable", True)}
    _tradable_crypto_cache = out
    return out or None


def _filter_tradable_crypto(symbols: list[str], tradable: "set[str] | None") -> "tuple[list[str], list[str]]":
    """(kept, dropped). Non-crypto symbols (no '/') always pass through. Crypto
    pairs are dropped only when `tradable` is present AND clearly in the desk's
    'SYM/USD' format (the majors are in it) — otherwise keep everything, so a
    format mismatch or empty/failed lookup can never nuke the universe."""
    if not tradable or not ({"BTC/USD", "ETH/USD"} & tradable):
        return list(symbols), []
    kept = [s for s in symbols if "/" not in s or s in tradable]
    dropped = [s for s in symbols if "/" in s and s not in tradable]
    return (kept or list(symbols)), dropped


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
    bp = float(account.get("buying_power", 0) or 0)
    if cash >= 0 or nmbp > 0:
        return False
    # A MARGIN DEBIT IS NOT DISTRESS. Buying marginable equities drives cash
    # negative and non-marginable buying power to zero *by construction*, so
    # `cash < 0 and nmbp <= 0` alone is just "this account used margin" — it
    # matches every healthy long book the equity desks open.
    #
    # Measured 2026-07-27: the equity desks placed 13 orders at 17:49 (cash
    # -$10,829.50, bp $25,001.91 — healthy). At 18:42 the crypto 24x7 run hit
    # this function with cash -$14,972.80 and bp $17,247.12 and flattened the
    # entire 25-position book. The realised losses took equity down 2.28% vs
    # prior close, which tripped the daily loss cap, which — with the book now
    # empty and only risk-REDUCING orders allowed — blocked everything until
    # the next session rollover. Buy on margin, get liquidated, get frozen.
    #
    # The pathology this function is actually for is "$0 available": orphaned
    # notional buys that leave the account unable to place ANY order. That
    # state has no buying power left. Healthy margin use does.
    if bp > 0:
        print(f"  · negative cash ${cash:,.2f} is a MARGIN DEBIT, not orphaned "
              f"notional (buying power ${bp:,.2f} > 0) — not flattening", flush=True)
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


# Pagination retry budget. 1+2+4+8 = 15s of backoff per page, well inside the
# workflow's 15-minute timeout, and only spent when Alpaca actually says 429.
_BARS_MAX_RETRIES = 4
_BARS_BACKOFF_S = 1.0


def _sane_confidence(raw: object) -> float:
    """A strategy's confidence, forced into [0, 1] with anything malformed at 0.

    Three consumers trust this range — Kelly sizing, the desk confidence gate,
    and cross-strategy conflict resolution — and none of them were protected.
    The previous expression, `getattr(signal, "confidence", 1.0) or 1.0`, failed
    OPEN in three separate ways:

        NaN     every comparison against NaN is False, so `conf < threshold`
                did not skip it and the signal was approved and Kelly-sized.
                Worse, the `min(0.90, nan)` idiom several strategies clamp with
                returns 0.90 — a bad bar became MAXIMUM conviction.
        0.0     `or 1.0` treats a legitimate zero-conviction signal as falsy
                and promotes it to 1.0, the largest possible size.
        >1 / junk  passed straight through to sizing.

    A malformed number means "no conviction", never "total conviction" — the
    same direction the scanner normaliser was fixed in on 2026-07-28.

    NOTE: the right home for this is `Signal.__post_init__`, which would cover
    the backend bot path too, but backend/app/strategies/CLAUDE.md says base.py
    must never be modified. This covers the desk path only.
    """
    try:
        c = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if c != c or c in (float("inf"), float("-inf")):
        return 0.0
    return max(0.0, min(1.0, c))


def _is_rate_limited(exc: object) -> bool:
    """True for Alpaca's free-tier throttle, which is worth waiting out.

    Deliberately narrow: a 404/422 is a real answer and retrying it just burns
    the job's time budget. Only the throttle earns a retry.
    """
    text = str(exc)
    return "429" in text or "Too Many Requests" in text


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
            attempt = 0
            while True:
                params = {"symbols": ",".join(chunk), "timeframe": timeframe,
                          "limit": limit, "start": _bars_start(), **extra}
                if page_token:
                    params["page_token"] = page_token
                try:
                    data = await _alpaca_get(path, params, data_api=True)
                except Exception as exc:
                    # A 429 mid-pagination used to `break`, silently discarding
                    # every symbol Alpaca had not yet returned. Alpaca paginates
                    # in SYMBOL ORDER, so that truncation is alphabetical and
                    # deterministic — not random loss.
                    #
                    # Measured 2026-07-28 across three runs, the symbols kept
                    # were EXACTLY the alphabetically-first 4, 5 and 11 of the
                    # 20-symbol crypto universe. The back half — SHIB, SOL,
                    # SUSHI, UNI, XRP, XTZ, YFI, MKR, LTC — never received bars
                    # at all, so no ensemble ever voted on them.
                    #
                    # Retry the SAME page with backoff instead of abandoning it.
                    attempt += 1
                    if _is_rate_limited(exc) and attempt <= _BARS_MAX_RETRIES:
                        await asyncio.sleep(_BARS_BACKOFF_S * (2 ** (attempt - 1)))
                        continue
                    missing = [s for s in chunk if s not in out]
                    print(f"    ⚠ batch bars TRUNCATED on {path}: {exc} — "
                          f"{len(missing)} of {len(chunk)} symbols have NO bars "
                          f"this run: {missing}", flush=True)
                    break
                attempt = 0
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

# Cross-strategy conflict resolution. The old rule stood aside on ANY opposing
# signal regardless of strength, so a lone low-confidence dissent vetoed a
# multi-strategy consensus. Measured 2026-07-28 over 76 conflicts:
#
#   Crypto  crypto_adaptive_trend was the ONLY sell voice on all 16 conflicts,
#           at 0.16-0.52, blocking buy consensus of 0.61-0.97. SHIB/USD was
#           avellaneda_stoikov_mm(0.90) vetoed by a single 0.16.
#
# Sides are now combined with the same 1-prod(1-ci) used for agreement, and the
# dominant side trades at the NET confidence (dissent subtracted, not ignored)
# only when that net clears this bar — after which the desk's own
# confidence_min and any per-strategy tuned threshold still apply. Default
# matches the desks' confidence_min so nothing trades on weaker evidence than
# an unopposed signal would need.
#
# Set ENSEMBLE_NET_MIN > 1.0 to restore the previous always-stand-aside rule
# without a code change.
_ENSEMBLE_NET_MIN = float(os.environ.get("ENSEMBLE_NET_MIN", "0.60"))


MIN_ORDER_USD = 25.0


def _price_precision(price: float, is_crypto: bool) -> int:
    """Decimal places that keep `price` non-zero and meaningful.

    Equities quote in cents, so 2 is right. Crypto spans nine orders of
    magnitude — BTC at 60,000 and SHIB at 0.00001 cannot share a precision.
    A flat 2dp rounded SHIB to 0.0 and the caller then divided by it.
    """
    if not is_crypto:
        return 2
    p = abs(float(price))
    if p >= 1:
        return 2
    if p >= 0.01:
        return 4
    if p >= 0.0001:
        return 6
    return 8


def cash_capped_notional(notional: float, symbol: str, account: dict) -> float:
    """Cap an order's notional at what the account can actually pay (95% of
    the relevant buying power; crypto needs non-marginable cash). Sizing used
    to ignore this and Alpaca 403'd 'insufficient balance' — with $215
    available we requested $378 instead of placing a $205 order. Returns 0
    when even MIN_ORDER_USD isn't affordable (caller skips honestly)."""
    is_crypto = "/" in symbol
    field = "non_marginable_buying_power" if is_crypto else "buying_power"
    avail = float(account.get(field, 0) or 0) * 0.95
    if avail < MIN_ORDER_USD:
        return 0.0
    return min(float(notional), avail)


def daily_loss_cap_hit(equity: float, last_equity: float,
                       cap: float = None) -> bool:
    cap = DAILY_LOSS_CAP_PCT if cap is None else cap
    if equity <= 0 or last_equity <= 0:
        return False   # unknown baseline — don't false-trigger
    return equity < last_equity * (1.0 - cap)


def is_risk_reducing(side: str, position_qty: float) -> bool:
    """True when an order REDUCES an existing position (sell against a long,
    buy against a short). 2026-07-21 lesson: the loss cap blocked ALL orders for
    an entire session — including exits — trapping a losing book (couldn't add,
    couldn't de-risk). Under the cap, reducing orders must always stay allowed;
    only exposure-INCREASING orders are blocked.
    """
    return (side == "sell" and position_qty > 0) or (side == "buy" and position_qty < 0)


async def _alpaca_position_map() -> dict[str, float]:
    """{symbol: signed_qty} of open Alpaca positions. Fail-soft {} — when the
    fetch fails while the loss cap is active, callers treat every order as
    non-reducing (i.e. the cap blocks everything, the pre-2026-07-21 behavior).
    """
    try:
        positions = await _alpaca_get("/v2/positions")
        out: dict[str, float] = {}
        for p in positions or []:
            qty = float(p.get("qty") or 0)
            if p.get("side") == "short":
                qty = -abs(qty)
            sym = p.get("symbol", "")
            if sym and qty:
                out[sym] = qty
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ position fetch failed ({str(exc)[:80]}) — cap stays strict", flush=True)
        return {}


def _order_origin(client_order_id: str | None) -> str:
    """Who placed an order, from its client_order_id.

    Every order this script places is tagged `qe-<strategy>-<sym>-<ts>` (see
    the `coid = ...` line in the placement loop). Anything else came from
    somewhere we are not looking at: the backend's PositionMonitor exit loop,
    Alpaca's own auto-liquidation, or a hand-placed order.
    """
    if client_order_id and client_order_id.startswith("qe-"):
        return "this desk placer"
    return "EXTERNAL (backend exit loop / broker / manual)"


async def _report_recent_closes(limit: int = 8) -> None:
    """Print who most recently closed positions. Fail-soft and diagnostic only.

    Written after a book of 13 orders (+$9,634 notional, placed 2026-07-27
    17:49) was fully flat by 23:44 having realised enough loss to trip the
    daily cap — and NOTHING in this repo could say what closed it. The orders
    carried no bracket legs, `recover_negative_cash` never fired, and no other
    Actions script closes positions. The backend DB is on its sqlite fallback
    and had zero order rows, so it held no evidence either.

    `client_order_id` is the one discriminator that survives all of that, and
    it lives at the broker rather than in any of our storage. Reading it back
    turns "something flattened the book" into a name.
    """
    try:
        orders = await _alpaca_get("/v2/orders", {"status": "closed",
                                                  "limit": limit,
                                                  "direction": "desc"})
    except Exception as exc:  # noqa: BLE001
        print(f"     (recent-close lookup failed: {str(exc)[:70]})", flush=True)
        return
    if not orders:
        print("     no closed orders on record — the book was never filled", flush=True)
        return
    print(f"     last {len(orders)} closed order(s), newest first:", flush=True)
    for o in orders:
        print(f"       {o.get('filled_at') or o.get('updated_at')} "
              f"{o.get('symbol')} {o.get('side')} qty={o.get('filled_qty')} "
              f"[{o.get('status')}] ← {_order_origin(o.get('client_order_id'))}",
              flush=True)


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


async def _equity_short_safe_qty(
    symbol: str, side: str, qty: float, is_crypto: bool
) -> float | None:
    """Whole-share the quantity when an equity SELL would open a short.

    Alpaca permits fractional shares for LONGS but rejects them on the short
    side. Every run wasted signals on it — measured 2026-07-28, 3 of 14
    attempted orders:

        422 {"code":42210000,"message":"fractional orders cannot be sold short"}
        place_order failed EIDO sell / ORCL sell / UNG sell

    Those signals cleared data, ensembling, the confidence gate, Kelly sizing
    and the risk manager, then died at the broker — the most expensive possible
    place to discover it.

    A sell is NOT always a short, though: under the daily loss cap only
    risk-REDUCING orders pass, and those are closes. Flooring those would strand
    a sub-1-share long forever. So the held position decides:

        held >= qty   closing a long  -> fractional is legal, leave it alone
        otherwise     opens a short   -> floor to whole shares, skip if < 1

    Returns None to mean "do not place this order".
    """
    if is_crypto or side != "sell":
        return qty
    held = (await _cached_position_map()).get(symbol, 0.0)
    if held >= qty:
        return qty  # a close, not a short
    whole = math.floor(qty)
    if whole < 1:
        print(f"    · {symbol} sell {qty} would be a fractional SHORT "
              f"(held {held}) — Alpaca rejects those, and flooring gives 0. "
              f"Skipping instead of failing at the broker.", flush=True)
        return None
    if whole != qty:
        print(f"    · {symbol} sell {qty} -> {whole} whole shares "
              f"(fractional shorts are rejected)", flush=True)
    return float(whole)


async def _equity_last_price(symbol: str) -> float | None:
    """Latest equity trade price, for converting a notional into whole shares.

    Fail-soft None: the caller then falls back to a notional order, which is
    what it did before this existed — worse for shorts, but never worse than
    not placing the order at all.
    """
    try:
        data = await _alpaca_get(
            "/v2/stocks/trades/latest", {"symbols": symbol, "feed": "iex"},
            data_api=True,
        )
        px = float(((data.get("trades") or {}).get(symbol) or {}).get("p", 0) or 0)
        return px or None
    except Exception:  # noqa: BLE001 — pricing is best-effort here
        return None


_position_map_cache: "dict[str, float] | None" = None


async def _cached_position_map() -> dict[str, float]:
    """`_alpaca_position_map()` memoised for the process.

    Called per sell order, so an uncached fetch would be one broker round trip
    per signal.
    """
    global _position_map_cache
    if _position_map_cache is None:
        _position_map_cache = await _alpaca_position_map()
    return _position_map_cache


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
            raw_lp = limit_price * (1.001 if side == "buy" else 0.999)
            # Rounding to 2dp destroys sub-cent crypto: SHIB at ~$0.00001 became
            # 0.0, and the next line divided by it — "float division by zero",
            # which the caller swallowed as a generic place_order failure. Scale
            # the precision to the price instead.
            lp = round(raw_lp, _price_precision(raw_lp, is_crypto))
            if lp <= 0:
                print(
                    f"    ⚠ {symbol}: limit price rounds to zero "
                    f"(raw={raw_lp!r}) — skipping rather than dividing by it",
                    flush=True,
                )
                return None
            qty = round(notional_usd / lp, 6 if is_crypto else 2)
            qty = await _equity_short_safe_qty(symbol, side, qty, is_crypto)
            if qty is None:
                return None
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
            body["type"] = "market"
            # A NOTIONAL market order lets Alpaca derive the share count, and it
            # derives a fractional one — which it then rejects on the short side.
            # The limit path above is already whole-shared, but `_ensure_filled`
            # replaces an unfilled limit with a market order through here, so the
            # 422 came back on exactly that route:
            #
            #   · UNG sell 37.27 -> 37 whole shares
            #   ↻ limit unfilled after 20s — replaced with market
            #   ⚠ 422 {"code":42210000,"message":"fractional orders cannot be
            #        sold short"}
            #
            # So a short-side equity market order has to carry an explicit whole
            # `qty` instead of a notional.
            qty_for_short = None
            if side == "sell":
                px = await _equity_last_price(symbol)
                if px and px > 0:
                    qty_for_short = await _equity_short_safe_qty(
                        symbol, side, round(notional_usd / px, 2), is_crypto=False
                    )
                    if qty_for_short is None:
                        return None
            if qty_for_short is not None:
                body["qty"] = str(qty_for_short)
            else:
                body["notional"] = str(round(notional_usd, 2))

        return await _alpaca_post("/v2/orders", body)
    except Exception as exc:
        print(f"    ⚠ place_order failed {symbol} {side}: {exc}", flush=True)
        return None


FILL_WAIT_S = 20   # give a limit this long to fill before replacing
FILL_POLL_S = 5


async def _ensure_filled(order: dict | None, symbol: str, side: str,
                         notional_usd: float) -> dict | None:
    """Cancel-replace execution (SOTA queue): limit-first orders that don't
    fill within FILL_WAIT_S are cancelled and replaced with a market order,
    so signals stop dying as stale unfilled limits. Double-fill safe: if the
    cancel races a fill, the fill wins and NO replacement is sent."""
    oid = (order or {}).get("id")
    if not oid or (order or {}).get("type") != "limit":
        return order
    waited = 0
    while waited < FILL_WAIT_S:
        await asyncio.sleep(FILL_POLL_S)
        waited += FILL_POLL_S
        try:
            cur = await _alpaca_get(f"/v2/orders/{oid}")
        except Exception:  # noqa: BLE001 — can't poll: keep the original order
            return order
        st = str(cur.get("status", ""))
        if st == "filled":
            print(f"    ✓ limit filled after {waited}s", flush=True)
            return cur
        if st in ("canceled", "expired", "rejected", "done_for_day"):
            print(f"    ✗ limit terminal ({st}) — no fill", flush=True)
            return cur
    try:
        await asyncio.to_thread(_alpaca_delete_sync, f"/v2/orders/{oid}")
    except Exception:  # noqa: BLE001 — cancel may race a fill
        try:
            cur = await _alpaca_get(f"/v2/orders/{oid}")
            if str(cur.get("status")) == "filled":
                print("    ✓ filled during cancel race — keeping fill", flush=True)
                return cur
        except Exception:  # noqa: BLE001
            pass
        print("    ⚠ cancel state unknown — NOT sending replacement (double-fill guard)", flush=True)
        return order
    print(f"    ↻ limit unfilled after {FILL_WAIT_S}s — replaced with market", flush=True)
    replacement = await _place_order(symbol, side, notional_usd)
    return replacement or order


# ── Real multi-leg options routing (Options desk income structures) ──────────
# IMPROVEMENTS 2026-07-15: wheel/condor/credit-spread traded the UNDERLYING as
# a directional proxy. Alpaca paper supports real mleg orders on the same keys
# (proven by the bot engine + backend tests) — so income signals now place
# actual defined-risk spreads. Strikes are picked by MONEYNESS vs spot (no
# greeks needed → one /v2/options/contracts call per option type). Everything
# fails soft back to the underlying proxy: a spread that can't be fully
# resolved places NO partial legs.

MLEG_DTE_TARGET = 35
MLEG_DTE_WINDOW = (25, 50)

# structure spec: list of (option_type, side, moneyness, target_delta). The
# moneyness is the fallback strike; target_delta is the real-greek strike when a
# Tradier feed is available (short legs ~0.30Δ, long protective legs ~0.15Δ —
# standard OA-style defined-risk deltas). side_hint picks the put or call wing.
_MLEG_STRUCTURES: dict[str, dict] = {
    "credit_spread_income": {
        "buy":  [("put", "sell", 0.96, 0.30), ("put", "buy", 0.92, 0.15)],      # bull put
        "sell": [("call", "sell", 1.04, 0.30), ("call", "buy", 1.08, 0.15)],    # bear call
    },
    "iron_condor": {
        "any": [("put", "sell", 0.95, 0.30), ("put", "buy", 0.91, 0.15),
                ("call", "sell", 1.05, 0.30), ("call", "buy", 1.09, 0.15)],
    },
    "vol_carry_short": {
        "any": [("put", "sell", 0.95, 0.30), ("put", "buy", 0.91, 0.15),
                ("call", "sell", 1.05, 0.30), ("call", "buy", 1.09, 0.15)],
    },
    "cash_secured_put": {
        "buy": [("put", "sell", 0.95, 0.30), ("put", "buy", 0.90, 0.12)],       # defined-risk CSP
    },
    "wheel": {
        "buy": [("put", "sell", 0.95, 0.30), ("put", "buy", 0.90, 0.12)],       # wheel entry leg
    },
}


def _income_leg_spec(strategy_name: str, side: str) -> list[tuple[str, str, float]] | None:
    """Leg spec for an income structure signal, or None if not mleg-routed."""
    struct = _MLEG_STRUCTURES.get(strategy_name)
    if not struct:
        return None
    return struct.get(side) or struct.get("any")


def _pick_contract(contracts: list[dict], target_strike: float, target_expiry: str) -> dict | None:
    """Nearest strike, then nearest expiry. Pure (unit-tested)."""
    best, best_key = None, None
    for c in contracts:
        try:
            strike = float(c["strike_price"])
            exp = str(c.get("expiration_date", ""))
        except (KeyError, TypeError, ValueError):
            continue
        key = (abs(strike - target_strike), abs((datetime.fromisoformat(exp)
               - datetime.fromisoformat(target_expiry)).days) if exp else 999)
        if best_key is None or key < best_key:
            best, best_key = c, key
    return best


def _tradier_target_strikes(underlying: str, spec: list[tuple]) -> dict[tuple[str, float], float]:
    """{(opt_type, moneyness): real_strike} from Tradier's actual deltas, or {}.

    One expirations + one chain call for the whole spread (all legs share the
    nearest ~35-DTE expiration). Fail-soft: any miss just omits that leg's key so
    the caller falls back to the moneyness strike. Real deltas replace the OTM
    guess with OA-style delta-selected strikes.
    """
    try:
        import tradier_data
    except Exception:  # noqa: BLE001
        return {}
    if not tradier_data.available():
        return {}
    exp = tradier_data.nearest_expiration(underlying, MLEG_DTE_TARGET)
    if not exp:
        return {}
    tchain = tradier_data.chain(underlying, exp, greeks=True)
    if not tchain:
        return {}
    out: dict[tuple[str, float], float] = {}
    for opt_type, _side, moneyness, target_delta in spec:
        cands = [o for o in tchain
                 if o.get("option_type") == opt_type
                 and o.get("strike") is not None
                 and (o.get("greeks") or {}).get("delta") is not None]
        if not cands:
            continue
        best = min(cands, key=lambda o: abs(abs(float(o["greeks"]["delta"])) - abs(target_delta)))
        out[(opt_type, moneyness)] = float(best["strike"])
    return out


async def _resolve_income_legs(underlying: str, spot: float,
                               spec: list[tuple]) -> list[dict] | None:
    """Resolve every leg to a real OCC contract, or None (never partial).

    Target strike per leg = Tradier's real-delta strike when a Tradier feed is
    configured, else the moneyness proxy (spot * moneyness). Orders still route
    through Alpaca, so Tradier only refines the target; Alpaca resolves the
    tradable contract nearest that strike.
    """
    today = datetime.now(timezone.utc).date()
    exp_gte = (today + timedelta(days=MLEG_DTE_WINDOW[0])).isoformat()
    exp_lte = (today + timedelta(days=MLEG_DTE_WINDOW[1])).isoformat()
    target_exp = (today + timedelta(days=MLEG_DTE_TARGET)).isoformat()

    # Real-delta strikes (one Tradier chain fetch), off the event loop; {} if no feed.
    real = await asyncio.to_thread(_tradier_target_strikes, underlying, spec)

    def _target(opt_type: str, moneyness: float) -> float:
        return real.get((opt_type, moneyness), spot * moneyness)

    by_type: dict[str, list[dict]] = {}
    for opt_type in {t for t, *_ in spec}:
        strikes = [_target(t, m) for t, _s, m, _d in spec if t == opt_type]
        try:
            resp = await _alpaca_get("/v2/options/contracts", {
                "underlying_symbols": underlying, "status": "active", "type": opt_type,
                "expiration_date_gte": exp_gte, "expiration_date_lte": exp_lte,
                "strike_price_gte": f"{min(strikes) * 0.97:.2f}",
                "strike_price_lte": f"{max(strikes) * 1.03:.2f}",
                "limit": "300",
            })
            by_type[opt_type] = resp.get("option_contracts") or []
        except Exception as exc:  # noqa: BLE001 — no contracts -> proxy fallback
            print(f"    ⚠ contracts fetch failed for {underlying} {opt_type}: {str(exc)[:80]}", flush=True)
            return None

    legs: list[dict] = []
    for opt_type, side, moneyness, _target_delta in spec:
        tgt = _target(opt_type, moneyness)
        pick = _pick_contract(by_type.get(opt_type, []), tgt, target_exp)
        if not pick or not pick.get("symbol"):
            print(f"    ⚠ no {opt_type} contract near {tgt:.0f} — proxy fallback", flush=True)
            return None
        legs.append({
            "symbol": pick["symbol"],
            "ratio_qty": "1",
            "side": side,
            "position_intent": "sell_to_open" if side == "sell" else "buy_to_open",
        })
    return legs


async def _place_income_spread(underlying: str, strategy_name: str, side: str,
                               spot: float) -> dict | None:
    """Place a real defined-risk spread for an income signal. None -> caller
    falls back to the underlying proxy. Fixed 1 contract (paper-first sizing;
    a spread's risk is its width, not Kelly notional)."""
    spec = _income_leg_spec(strategy_name, side)
    if not spec or spot <= 0:
        return None
    legs = await _resolve_income_legs(underlying, spot, spec)
    if not legs:
        return None
    body = {
        "order_class": "mleg",
        "qty": "1",
        "type": "market",
        "time_in_force": "day",
        "legs": legs,
    }
    try:
        out = await _alpaca_post("/v2/orders", body)
        return out if out and out.get("id") else None
    except Exception as exc:  # noqa: BLE001 — rejection reasons print upstream
        print(f"    ⚠ mleg order failed: {str(exc)[:100]}", flush=True)
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
    "ml_pca_arb":                [1],         # residual reversion — range
    "lorentzian_knn":            [1, 2],
    # 2026-07-15 scale-up — TV indicator suite
    "ema_stack_tv":              [2],
    "hull_suite_tv":             [2],
    "ichimoku_cloud_tv":         [2],
    "supertrend_rsi_tv":         [2],
    "adx_dmi_tv":                [2],
    "kama_roc_tv":               [2],
    "squeeze_pro_tv":            [2],         # volatility breakout
    "wave_trend_tv":             [1, 2],
    "macd_divergence_tv":        [1],
    "stoch_rsi_macd_tv":         [1],
    "vwap_bands_tv":             [1],
    "elliott_wave_proxy_tv":     [1, 2],
    # 2026-07-15 scale-up — equities
    "cci_reversion":             [1],
    "fifty_two_week_high":       [2],         # 52w-high momentum anomaly
    "triple_barrier_momentum":   [2],
    "ml_momentum":               [2],
    "ml_mean_reversion":         [1],
    "ml_breakout":               [2],
    "ensemble":                  [0, 1, 2],   # blends members per regime
    "event_driven_gap":          [1, 2],
    "open_close_revert":         [1],
    # 2026-07-15 scale-up — crypto
    "crypto_whale_momentum":     [2],
    "funding_settlement_timer":  [0, 1, 2],
    # 2026-07-15 scale-up — options flow/sentiment
    "options_pcr_reversal":      [0, 1],      # contrarian at fear extremes
    "put_call_ratio_contrarian": [0, 1],
    "earnings_iv_crush":         [0, 1, 2],   # event-driven, regime-agnostic
    "options_gamma_scalp":       [1],         # range harvesting
    "long_call_momentum":        [2],
    "cash_secured_put":          [1, 2],      # premium selling — not in bear
    # 2026-07-15 scale-up — polymarket
    "poly_market_maker":         [0, 1, 2],
    "poly_liquidity_provision":  [0, 1, 2],
    "poly_time_value_fade":      [0, 1, 2],
    "poly_cross_market_hedge":   [0, 1, 2],
    # 2026-07-15 scale-up — macro/rates
    "bond_equity_rotation":      [1, 2],
    "central_bank_window":       [0, 1, 2],
    "macro_risk_barometer":      [0, 1, 2],
    "breakeven_inflation":       [1, 2],
    "duration_momentum":         [2],
    "pmi_sector_rotation":       [1, 2],
    "yield_curve_momentum":      [2],
    "yield_spread_reversion":    [1],
    "tlt_spy_rotation":          [1, 2],
    # 2026-07-15 scale-up — commodities
    "commodity_momentum":        [2],
    "commodity_reversion":       [1],
    "commodity_trend":           [2],
    # 2026-07-18 — calendar/reversion premia
    "turn_of_month":             [1, 2],       # flow anomaly — not in bear panic
    "gap_fill_fade":             [1],          # mean reversion — range regime
    "double_seven":              [1, 2],       # trend-filtered pullback buyer
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
# Auto-pruning (IMPROVEMENTS 2026-07-15): with REAL adverse evidence a strategy
# stops trading entirely (weight 0.0) instead of limping at 0.6x — the flip
# side of "always add strategies" is "always cut proven losers". Evidence bar
# is deliberately high; a pruned strategy revives automatically if its live
# stats recover (weights re-fetch every run).
_PRUNE_MIN_TRADES = 20
_PRUNE_SHARPE_BELOW = -0.5
# Hit-rate rule (TV-desk item, applied to ALL strategies): with a big sample,
# a sub-45% win rate on roughly symmetric trades is a coin toss minus costs.
_PRUNE_HITRATE_MIN_TRADES = 100
_PRUNE_HITRATE_BELOW = 0.45
# `or` not a get() default — an env var set to "" would otherwise produce a
# relative URL and fail every leaderboard fetch silently.
_API_BASE = (os.environ.get("QUANTEDGE_API_URL") or "https://quantedge-api-9jz0.onrender.com").rstrip("/")


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
    pruned: list[str] = []
    for s in strategies:
        n = int(s.get("trades") or 0)
        if n < 5:
            continue  # not enough evidence to re-weight
        pnl = float(s.get("total_pnl") or 0)
        sharpe = s.get("pnl_sharpe")
        win_rate = s.get("win_rate")
        if (n >= _PRUNE_MIN_TRADES and pnl < 0
                and float(sharpe if sharpe is not None else 0) < _PRUNE_SHARPE_BELOW):
            w = 0.0
            pruned.append(s.get("strategy", ""))
        elif (n >= _PRUNE_HITRATE_MIN_TRADES and pnl < 0
                and win_rate is not None and float(win_rate) < _PRUNE_HITRATE_BELOW):
            # hit-rate rule: large sample, losing money, sub-coin-toss accuracy
            w = 0.0
            pruned.append(s.get("strategy", ""))
        elif pnl > 0 and (sharpe or 0) > 0:
            w = min(_WEIGHT_MAX, 1.0 + min(float(sharpe) * 0.15, 0.3))
        elif pnl < 0:
            w = _WEIGHT_MIN
        else:
            w = 1.0
        weights[s.get("strategy", "")] = round(w, 3)
    if weights:
        print(f"✓ Performance weights active for {len(weights)} strategies", flush=True)
    if pruned:
        print(f"✂ Auto-pruned (≥{_PRUNE_MIN_TRADES} trades, negative P&L, "
              f"sharpe<{_PRUNE_SHARPE_BELOW}): {', '.join(sorted(pruned))}", flush=True)
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


def _detect_vol_regime_from_bars(spy_df) -> str:
    """Volatility regime from SPY bars: 'stressed' when recent realized vol is
    elevated vs its own history, else 'calm'. Free (no new feed), fail-soft."""
    import numpy as np
    try:
        close = spy_df["close"].astype(float).values
        if len(close) < 25:
            return "calm"
        log_rets = np.diff(np.log(close))
        recent_vol = float(np.std(log_rets[-20:]))
        long_vol = float(np.std(log_rets[-min(252, len(log_rets)):]))
        return "stressed" if recent_vol / max(long_vol, 1e-8) >= 1.25 else "calm"
    except Exception:
        return "calm"


# Premium-selling income structures: selling option premium is most rewarded when
# implied/realized vol is elevated. In CALM vol these face a higher confidence bar
# (defined below) rather than a hard block, so the income desk still trades but
# leans into stressed regimes — matching the documented 0DTE variance-risk-premium.
_PREMIUM_SELLERS: frozenset[str] = frozenset(_MLEG_STRUCTURES.keys()) | {"vol_carry"}
_CALM_PREMIUM_THRESHOLD_BUMP = 0.08


def _vol_adjusted_threshold(strategy_name: str, base_threshold: float, vol_regime: str) -> float:
    """Raise the bar for premium sellers in calm vol; unchanged otherwise."""
    if vol_regime == "calm" and strategy_name in _PREMIUM_SELLERS:
        return base_threshold + _CALM_PREMIUM_THRESHOLD_BUMP
    return base_threshold


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
    unfunded = 0          # signals dropped because buying power was exhausted
    rejected = 0          # signals the broker refused

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

    # Drop crypto pairs Alpaca no longer lists as tradable (fail-soft: keeps all
    # on any lookup issue). Non-crypto desks pass through untouched.
    run_symbols, dropped = _filter_tradable_crypto(desk.symbols, await _tradable_crypto_symbols())
    if dropped:
        print(f"  ⓘ skipping {len(dropped)} non-tradable pair(s): {', '.join(dropped)}", flush=True)

    for symbol in run_symbols:
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

            conf = _sane_confidence(getattr(signal, "confidence", None))
            if conf < desk.confidence_min:
                print(
                    f"  · {strategy.name}/{symbol} signal={signal.side} conf={conf:.2f} "
                    f"< threshold={desk.confidence_min:.2f} — skipped",
                    flush=True,
                )
                continue

            # Size to what the account can actually pay. cash_capped_notional
            # has existed for exactly this since the last 403 round, but this
            # path passed desk.notional_usd RAW — so every crypto desk run
            # asked Alpaca for $135 against $6.71 of buying power and got
            # "403 insufficient balance" on essentially every order. Nine desks
            # generated signals and placed nothing.
            sized = cash_capped_notional(desk.notional_usd, symbol, account)
            if sized <= 0:
                print(
                    f"  ⏭ {strategy.name}/{symbol} {signal.side.upper()} — SKIPPED, "
                    f"account cannot fund even ${MIN_ORDER_USD:.0f} "
                    f"(buying power exhausted)",
                    flush=True,
                )
                unfunded += 1
                continue

            print(
                f"  ► {strategy.name}/{symbol} signal={signal.side.upper()} "
                f"conf={conf:.2f} — placing ${sized:.0f} order"
                + (f" (capped from ${desk.notional_usd:.0f})"
                   if sized < desk.notional_usd else ""),
                flush=True,
            )

            order = await _place_order(symbol, signal.side, sized)
            if order and order.get("id"):
                print(f"    ✓ order {order['id']} submitted ({order.get('status', '?')})", flush=True)
                orders_placed.append({
                    "desk":     desk.name,
                    "strategy": strategy.name,
                    "symbol":   symbol,
                    "side":     signal.side,
                    "notional": sized,
                    "confidence": conf,
                    "order_id": order["id"],
                    "status":   order.get("status", "?"),
                    "ts":       datetime.now(timezone.utc).isoformat(),
                })
            else:
                rejected += 1
                print(f"    ✗ order placement returned no ID", flush=True)

    # A desk that produced signals and placed nothing is NOT a healthy desk.
    # It used to end with a tidy "✓ Place orders" and no further comment, which
    # is how nine desks ran for weeks while the account sat at -$8,287 with
    # $6.71 available and every single order 403'd.
    if not orders_placed and (unfunded or rejected):
        print(
            f"  🚨 DESK {desk.name} PLACED NOTHING — {unfunded} signal(s) unfunded, "
            f"{rejected} rejected by the broker. Buying power: "
            f"${float(account.get('buying_power', 0) or 0):.2f}, "
            f"non-marginable: ${float(account.get('non_marginable_buying_power', 0) or 0):.2f}",
            flush=True,
        )

    return orders_placed


# ── Discord helper ──────────────────────────────────────────────────────────────

def _post_chat(channel: str, message: str) -> None:
    """Post to Discord via the shared notifier (Slack removed 2026-07-25)."""
    import notify
    notify.post(channel, message)


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

            # ── Polymarket REAL data (2026-07-18): the desk ran on an SPY
            # proxy; Gamma+CLOB market data is public. Feed the top markets'
            # hourly price bars and point the desk at them. Fail-soft: any
            # feed problem keeps the SPY proxy (never an empty desk).
            for _i, _d in enumerate(active_desks):
                if _d.name != "Polymarket":
                    continue
                try:
                    import polymarket_data
                    _pm = await asyncio.to_thread(polymarket_data.desk_feed, 6)
                    if _pm:
                        bars_cache.update(_pm)
                        active_desks[_i] = _d._replace(symbols=list(_pm.keys()))
                        print(f"  ✓ Polymarket desk on REAL data: {len(_pm)} live markets", flush=True)
                    else:
                        print("  · Polymarket feed empty — desk stays on SPY proxy", flush=True)
                except Exception as _exc:  # noqa: BLE001
                    print(f"  · Polymarket feed failed ({str(_exc)[:80]}) — SPY proxy kept", flush=True)

        # Detect market regime from SPY bars (0=bear, 1=sideways, 2=bull)
        _REGIME_NAMES = {0: "bear", 1: "sideways", 2: "bull"}
        spy_df = bars_cache.get("SPY")
        current_regime: int = _detect_regime_from_bars(spy_df) if spy_df is not None else 1
        # Volatility axis (calm|stressed): the dimension that decides whether
        # selling option premium is worth it. Premium sellers face a HIGHER
        # confidence bar in calm vol (thin premium — 0DTE VRP evidence), rather
        # than a hard block that would starve the income desk.
        vol_regime: str = _detect_vol_regime_from_bars(spy_df)
        print(f"  Market regime: {_REGIME_NAMES[current_regime]} ({current_regime}) | vol: {vol_regime}", flush=True)

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
                            conf = _sane_confidence(getattr(signal, "confidence", None))
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

        def _opposite(side: str) -> str:
            return "sell" if side == "buy" else "buy"

        # Report each conflicted (desk, symbol) ONCE, naming who is on each
        # side. Previously the line was printed from inside the per-side loop,
        # so a single disagreement logged twice — once from the buy group and
        # once from the sell group — and the mirrored `buy/sell` + `sell/buy`
        # pair read as two separate events. Measured 2026-07-28: 34 lines for
        # 17 symbols, on both the starved and the full-universe runs.
        #
        # It also named neither strategy, which is the thing actually worth
        # knowing: whether one pair disagrees on nearly everything (a
        # systematic mismatch worth fixing) or the disagreements are spread
        # around (genuinely no edge, and standing aside is correct). The
        # stand-aside behaviour is unchanged here — this is instrumentation to
        # answer that question, not a change to what trades.
        def _combined(items: list) -> float:
            """1 - prod(1-ci) — the same rule already used for agreement."""
            _p = 1.0
            for _x in items:
                _p *= (1.0 - min(max(float(_x["confidence"]), 0.0), 1.0))
            return 1.0 - _p

        _conflicted = {
            (_dn, _sym)
            for (_dn, _sym, _side) in _groups
            if (_dn, _sym, _opposite(_side)) in _groups
        }
        # (desk, symbol) -> (winning_side, net_confidence)
        _overrides: dict = {}
        for _dn, _sym in sorted(_conflicted):
            _sides = []
            for _s in ("buy", "sell", "neutral", "none", ""):
                _grp = _groups.get((_dn, _sym, _s))
                if _grp:
                    _who = ", ".join(
                        f"{getattr(_x['strategy'], 'name', '?')}({_x['confidence']:.2f})"
                        for _x in _grp
                    )
                    _sides.append(f"{_s or 'unset'}: {_who}")

            _b = _groups.get((_dn, _sym, "buy")) or []
            _s_ = _groups.get((_dn, _sym, "sell")) or []
            if not (_b and _s_):
                # Not a true buy-vs-sell disagreement (e.g. a neutral against a
                # buy). Prior behaviour, unchanged.
                print(f"  · ensemble[{_dn}]: {_sym} CONFLICT — stand aside | "
                      + " | ".join(_sides), flush=True)
                continue

            _bc, _sc = _combined(_b), _combined(_s_)
            _net = abs(_bc - _sc)
            _win = "buy" if _bc > _sc else "sell"
            _verdict = (f"net {_win} {_net:.2f} ≥ {_ENSEMBLE_NET_MIN:.2f} — trading dominant side"
                        if _net >= _ENSEMBLE_NET_MIN else
                        f"net {_net:.2f} < {_ENSEMBLE_NET_MIN:.2f} — stand aside")
            print(f"  · ensemble[{_dn}]: {_sym} CONFLICT — {_verdict} "
                  f"(buy {_bc:.2f} vs sell {_sc:.2f}) | " + " | ".join(_sides), flush=True)
            if _net >= _ENSEMBLE_NET_MIN:
                _overrides[(_dn, _sym)] = (_win, _net)

        _ensembled: list[dict] = []
        for (_dn, _sym, _side), _items in _groups.items():
            if (_dn, _sym, _opposite(_side)) in _groups:
                _ov = _overrides.get((_dn, _sym))
                if not (_ov and _ov[0] == _side):
                    continue
                # Dominant side survives, carrying the NET confidence — the
                # dissent is subtracted, not ignored. It still has to clear the
                # desk's own confidence_min and any per-strategy tuned
                # threshold downstream, so this widens the funnel, it does not
                # bypass the gate.
                _keep = dict(max(_items, key=lambda x: x["confidence"]))
                _keep["confidence"] = round(_ov[1], 4)
                try:
                    _keep["signal"].confidence = _keep["confidence"]
                except Exception:  # noqa: BLE001 — frozen dataclass etc.
                    pass
                _ensembled.append(_keep)
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

                # Use auto-tuned threshold if available, floored at desk minimum,
                # then raise the bar for premium sellers in calm vol.
                threshold = max(_TUNED_THRESHOLDS.get(sname, desk.confidence_min), desk.confidence_min)
                threshold = _vol_adjusted_threshold(sname, threshold, vol_regime)
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

            # ── Exploration allocation ────────────────────────────────────────
            # Without live fills the ≥20-trade pruning/promotion loop can never
            # judge ~95% of the book (fills concentrate in a couple of winners).
            # One MIN-notional clip per desk per run for a strategy the gates
            # filtered, rotating daily. Regime gate stays enforced (checked
            # above), pruned strategies stay excluded (perf-weight 0 check in
            # the execution loop), and a 0.45 noise floor keeps it sane.
            chosen = {(i["strategy"].name, i["symbol"]) for i in approved_signals}
            explore_pool: dict[str, list[dict]] = {}
            for item in raw_signals:
                sname = item["strategy"].name
                if (sname, item["symbol"]) in chosen or sname in {n for n, _ in chosen}:
                    continue
                if current_regime not in _STRATEGY_REGIME_MAP.get(sname, _DEFAULT_REGIMES):
                    continue
                if item["confidence"] < 0.45:
                    continue
                explore_pool.setdefault(item["desk"].name, []).append(item)
            _day_seed = int(datetime.now(timezone.utc).strftime("%Y%m%d"))
            explored = 0
            for dname, items in sorted(explore_pool.items()):
                items.sort(key=lambda x: (x["strategy"].name, x["symbol"]))
                pick = items[(_day_seed + len(dname)) % len(items)]
                pick["explore"] = True
                approved_signals.append(pick)
                explored += 1
                print(f"  · explore[{dname}]: {pick['strategy'].name}/{pick['symbol']} "
                      f"conf={pick['confidence']:.2f} — min-notional evidence clip", flush=True)

            filtered = len(raw_signals) - len(approved_signals)
            tracker.set_output(passed=len(approved_signals), filtered=filtered,
                               explored=explored)

        # ── Stage 5: Order Execution ──────────────────────────────────────────
        all_orders: list[dict] = []
        desk_summaries: list[str] = []
        total_notional = 0.0
        _account_ok = float(account.get("buying_power", 0)) > 0
        # Loss cap no longer folds into _account_ok: it blocked EVERYTHING for a
        # whole session (2026-07-20: weekend crypto drift vs Friday's last_equity
        # tripped it at the open) — including exits, trapping a losing book.
        # Under the cap, risk-REDUCING orders stay allowed; only new exposure is
        # blocked. Positions fetched once; fetch failure → cap stays strict.
        _cap_active = bool(locals().get("_loss_cap_hit", False)) and _account_ok
        _cap_positions: dict[str, float] = (await _alpaca_position_map()) if _cap_active else {}
        with tracker.stage(ORDER_EXECUTION, "Place orders"):
            # The market clock only gates equity-hours desks — always-open desks
            # (crypto) trade through nights and weekends.
            if not is_open:
                print("  ⚠ Equity market closed — only always-open desks may trade", flush=True)
            if _cap_active:
                # The equity numbers go here too, not only in the Discord
                # summary. The DAILY LOSS CAP line that carries them is printed
                # hundreds of lines earlier (during the account fetch, before
                # the per-symbol ensemble output), so reading it back through
                # the Actions API means paging through the whole run. Repeating
                # them at the point of blocking makes the diagnosis reachable
                # in a short tail — which is how anyone actually reads this.
                #
                # Shipped after hitting exactly that: the numbers were added to
                # Discord last run, and the next investigation still could not
                # see them from the log.
                _eq = locals().get("equity") or 0.0
                _last_eq = locals().get("last_equity") or 0.0
                # SIGNED RETURN, not the drawdown magnitude. The first cut of
                # this computed `1 - eq/last_eq` (positive on a loss) and then
                # formatted it `{:+.2%}`, so the 2026-07-28 run reported a
                # 2.28% LOSS as "+2.28%" — a diagnostic that states the
                # opposite of what happened. Read as a gain, it makes an
                # active loss cap look like a bug; it wasn't.
                _ret = (_eq / _last_eq - 1.0) if _last_eq else 0.0
                print(f"  🛑 Loss cap ACTIVE — only risk-reducing orders allowed "
                      f"({len(_cap_positions)} open positions eligible to reduce)", flush=True)
                print(f"     equity ${_eq:,.2f} vs prior close ${_last_eq:,.2f} "
                      f"({_ret:+.2%}, cap -{DAILY_LOSS_CAP_PCT:.0%})"
                      + ("  ⚠️ 0 reducible → NOTHING can pass" if not _cap_positions else ""),
                      flush=True)
                if not _cap_positions:
                    # A flat book under an active cap means the loss was
                    # REALISED — something closed the positions. Name it.
                    await _report_recent_closes()
                if _last_eq and abs(_eq - _last_eq) < 1e-9:
                    # equity == last_equity cannot trip a drawdown cap. If this
                    # ever prints, the cap is firing on stale/mismatched inputs
                    # rather than a real loss.
                    print("     ⚠️ equity EQUALS prior close — cap should not be "
                          "active; suspect a stale last_equity", flush=True)
            if not _account_ok:
                print("  ⚠ Skipping order placement (no buying power / account unavailable)", flush=True)
                tracker.set_output(orders_placed=0, reason="account_unavailable")
            # Group approved signals by desk so we can still post per-desk summaries
            desk_orders_map: dict[str, list[dict]] = {}
            # Why a surviving signal never became an order. The Discord summary
            # used to report only "N generated → M survived → 0 placed" and then
            # "💤 no signals fired" for every desk — so 15 signals could survive
            # the gate, all be dropped, and the message state neither that it
            # happened nor why. Every reason below is counted and published.
            from collections import Counter

            drops: "Counter[str]" = Counter()
            desk_signals: "Counter[str]" = Counter()
            desk_drops: dict[str, Counter] = {}

            def _drop(reason: str, desk_name: str) -> None:
                drops[reason] += 1
                desk_drops.setdefault(desk_name, Counter())[reason] += 1

            for item in approved_signals:
                desk     = item["desk"]
                symbol   = item["symbol"]
                strategy = item["strategy"]
                signal   = item["signal"]
                conf     = item["confidence"]
                desk_signals[desk.name] += 1

                desk_open = is_open or desk.always_open
                if not desk_open:
                    print(
                        f"  · {strategy.name}/{symbol} signal={signal.side.upper()} "
                        f"conf={conf:.2f} — logged ({desk.name} closed)",
                        flush=True,
                    )
                    _drop("market closed", desk.name)
                    continue
                if not _account_ok:
                    print(
                        f"  · {strategy.name}/{symbol} signal={signal.side.upper()} "
                        f"conf={conf:.2f} — logged (no account)",
                        flush=True,
                    )
                    _drop("no account", desk.name)
                    continue
                if _cap_active and not is_risk_reducing(
                    signal.side, _cap_positions.get(symbol.replace("/", ""), _cap_positions.get(symbol, 0.0))
                ):
                    print(
                        f"  🛑 {strategy.name}/{symbol} {signal.side.upper()} — blocked by loss cap "
                        f"(would increase exposure)", flush=True,
                    )
                    _drop("loss cap", desk.name)
                    continue
                kelly_notional = _kelly_notional(equity, conf, bars=bars_cache.get(symbol))
                # Self-scaling: winners (by live realized P&L) size up, losers down,
                # proven losers pruned outright (weight 0 → no order at all).
                perf_w = _perf_weights.get(strategy.name, 1.0)
                if perf_w == 0.0:
                    print(f"  ✂ {strategy.name}/{symbol} pruned by attribution "
                          f"(sustained negative live P&L) — no order", flush=True)
                    _drop("pruned by attribution", desk.name)
                    continue
                if perf_w != 1.0:
                    print(f"    · perf weight {perf_w:.2f}x for {strategy.name}", flush=True)
                    kelly_notional *= perf_w
                if item.get("explore"):
                    # Evidence clip: smallest placeable size — exploration buys
                    # information, not exposure.
                    kelly_notional = min(kelly_notional, MIN_ORDER_USD) or MIN_ORDER_USD
                kelly_notional = cash_capped_notional(kelly_notional, symbol, account)
                if kelly_notional <= 0:
                    print(f"  · {strategy.name}/{symbol} skipped — insufficient available cash "
                          f"(< ${MIN_ORDER_USD:.0f}; frees as pending closes fill)", flush=True)
                    _drop("insufficient cash", desk.name)
                    continue
                coid = f"qe-{strategy.name[:10]}-{symbol[:4].replace('/', '')}-{int(time.time())}"
                limit_price: float | None = None
                _df = bars_cache.get(symbol)
                if _df is not None and len(_df) > 0:
                    limit_price = float(_df["close"].iloc[-1])
                # Polymarket symbols are REAL market data but execution needs
                # py-clob-client signing (queued) — log the signal, never send
                # a "PM:..." symbol to Alpaca.
                if symbol.startswith("PM:"):
                    print(f"  ◆ {strategy.name} on {symbol[:44]!r}: {signal.side.upper()} "
                          f"conf={conf:.2f} — REAL Polymarket signal (execution pending py-clob)",
                          flush=True)
                    continue
                # Options-desk income structures place REAL defined-risk
                # spreads (order_class=mleg); anything unresolvable falls back
                # to the underlying proxy below — never partial legs.
                if desk.name == "Options" and strategy.name in _MLEG_STRUCTURES and limit_price:
                    spread = await _place_income_spread(symbol, strategy.name,
                                                        signal.side, limit_price)
                    if spread:
                        print(f"  ► {strategy.name}/{symbol} — REAL {len(spread.get('legs', []) or [])}-leg "
                              f"spread {spread['id']} ({spread.get('status', '?')})", flush=True)
                        record = {
                            "desk": desk.name, "strategy": strategy.name,
                            "symbol": symbol, "side": signal.side,
                            "notional": 0.0, "confidence": conf,
                            "order_id": spread["id"], "client_order_id": coid,
                            "order_type": "mleg", "status": spread.get("status", "?"),
                            "ts": datetime.now(timezone.utc).isoformat(),
                        }
                        all_orders.append(record)
                        desk_orders_map.setdefault(desk.name, []).append(record)
                        continue
                    print(f"    · mleg unavailable for {strategy.name}/{symbol} — using underlying proxy", flush=True)
                print(
                    f"  ► {strategy.name}/{symbol} signal={signal.side.upper()} "
                    f"conf={conf:.2f} — placing ${kelly_notional:.0f} limit-first order",
                    flush=True,
                )
                order = await _place_order(symbol, signal.side, kelly_notional,
                                           limit_price=limit_price, client_order_id=coid)
                order = await _ensure_filled(order, symbol, signal.side, kelly_notional)
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

            # Post per-desk Discord summaries
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
                    _post_chat(desk.chat_channel, "\n".join(lines))
                    desk_summaries.append(f"✅ *{desk.name}*: {len(desk_order_list)} orders")
                else:
                    # "no signals fired" was printed whenever no ORDER was placed,
                    # which conflates "this desk had nothing to say" with "this
                    # desk fired signals and every one was dropped". The second is
                    # a problem; the first is a quiet market. They must not read
                    # identically.
                    n_sig = desk_signals.get(desk.name, 0)
                    if n_sig:
                        why = ", ".join(
                            f"{n} {reason}"
                            for reason, n in desk_drops.get(desk.name, Counter()).most_common(3)
                        ) or "no reason recorded"
                        desk_summaries.append(
                            f"⚠️ *{desk.name}*: {n_sig} signal(s) fired, **0 placed** — {why}"
                        )
                    else:
                        desk_summaries.append(f"💤 *{desk.name}*: no signals fired")

            tracker.set_output(orders_placed=len(all_orders), total_notional=round(total_notional, 2))

        # ── Stage 6: PnL Snapshot / Discord Summary ────────────────────────────
        with tracker.stage(PNL_SNAPSHOT, "Post PnL snapshot to Discord"):
            now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
            # Funnel telemetry: makes "why so few trades?" visible at a glance —
            # generated → gated/topK survivors → exploration clips → placed —
            # plus the active regime that gated strategy selection.
            _gen = len(raw_signals)
            _survivors = len(approved_signals)
            _explored = locals().get("explored", 0)
            _regime_lbl = _REGIME_NAMES.get(current_regime, str(current_regime))
            summary  = f"*QuantEdge Desk Run* ({now_str})  equity=${equity:,.2f}  regime={_regime_lbl}/{vol_regime}\n"
            summary += (f"funnel: {_gen} generated → {_survivors} survived gate+topK "
                        f"({_explored} exploration) → {len(all_orders)} placed\n")
            # The gap between "survived" and "placed" is where every silent
            # failure lives. It used to be unexplained: 15 survived, 0 placed,
            # no reason anywhere in the message. Publish the breakdown.
            _drops = locals().get("drops") or Counter()
            if _drops:
                _total_dropped = sum(_drops.values())
                _why = " · ".join(f"{n} {reason}" for reason, n in _drops.most_common())
                summary += f"⚠️ {_total_dropped} dropped before placement — {_why}\n"
            if locals().get("_cap_active"):
                # WITH THE NUMBERS. This used to say only "loss cap ACTIVE", and
                # the drawdown figure went to the CI log — which nobody reads —
                # so from Discord you could not tell a one-off 3% day from a
                # structural halt. Measured 2026-07-28: the desks had placed
                # ZERO orders for 18 days (last trade 2026-07-10) with every
                # signal blocked here, and answering "why" needed a dig through
                # Actions logs.
                #
                # `0 positions eligible to reduce` is the state worth seeing at
                # a glance: under the cap only risk-REDUCING orders pass, so
                # with no open positions nothing can pass at all. That is
                # correct as a daily cooling-off rule and pathological if it
                # persists, and these two numbers are what distinguish them.
                _eq = locals().get("equity") or 0.0
                _last_eq = locals().get("last_equity") or 0.0
                # Signed return — see the note at the console twin above: the
                # drawdown magnitude formatted `{:+.2%}` printed a loss as a
                # gain, which is worse than printing nothing.
                _ret = (_eq / _last_eq - 1.0) if _last_eq else 0.0
                _n_reducible = len(locals().get("_cap_positions") or {})
                summary += (
                    f"🛑 loss cap ACTIVE — new exposure blocked, risk-reducing only. "
                    f"equity ${_eq:,.2f} vs prior close ${_last_eq:,.2f} "
                    f"({_ret:+.2%}, cap -{DAILY_LOSS_CAP_PCT:.0%}) · "
                    f"{_n_reducible} position(s) eligible to reduce"
                    + ("  ⚠️ nothing can pass while this is 0" if _n_reducible == 0 else "")
                    + "\n"
                )
            summary += "\n".join(desk_summaries)
            summary += f"\n\nTotal orders placed: *{len(all_orders)}*"
            _post_chat("#pnl-daily", summary)

            # Graphical companion (user: "show me more graphical"): orders-per-desk
            # bar chart. Only when something was placed, so we don't spam empty runs.
            if all_orders:
                try:
                    from notify import discord_post_chart
                    per_desk: dict[str, int] = {}
                    for _o in all_orders:
                        per_desk[_o.get("desk", "?")] = per_desk.get(_o.get("desk", "?"), 0) + 1
                    if per_desk:
                        discord_post_chart(
                            "#pnl-daily",
                            title=f"Orders by desk — {now_str}",
                            labels=list(per_desk.keys()),
                            series={"orders": list(per_desk.values())},
                            kind="bar",
                            description=f"{len(all_orders)} orders | regime {_regime_lbl}/{vol_regime}",
                            username="QuantEdge Desk",
                        )
                except Exception as _exc:  # noqa: BLE001
                    print(f"  (desk chart skipped: {_exc})", flush=True)
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
