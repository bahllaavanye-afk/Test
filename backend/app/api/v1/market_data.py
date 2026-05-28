"""Market data endpoints: quotes, historical OHLCV, news, earnings, IV Rank."""
from fastapi import APIRouter, Depends, Query, HTTPException
from app.api.deps import get_current_user
from app.models.user import User
from app.config import settings
import asyncio
import httpx
import math
from datetime import datetime, timezone, timedelta

router = APIRouter(prefix="/market-data", tags=["market_data"])

ALPACA_DATA_URL = "https://data.alpaca.markets"

# Crypto base symbols — any symbol containing these is routed to the crypto endpoint
_CRYPTO_BASES = {"BTC", "ETH", "SOL", "ADA", "AVAX", "DOT", "MATIC", "LINK",
                 "UNI", "LTC", "XRP", "DOGE", "SHIB", "BCH"}


def _alpaca_headers() -> dict:
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }


def _is_crypto(symbol: str) -> bool:
    """Return True if symbol looks like a crypto asset."""
    sym = symbol.upper()
    if "/" in sym:
        return True
    for base in _CRYPTO_BASES:
        if sym.startswith(base) or sym.endswith(base):
            return True
    return False


def _interval_to_alpaca(interval: str) -> str:
    mapping = {
        "1m": "1Min",
        "5m": "5Min",
        "15m": "15Min",
        "1h": "1Hour",
        "4h": "4Hour",
        "1d": "1Day",
        "1w": "1Week",
    }
    return mapping.get(interval, "1Day")


def _period_to_start(period: str) -> str:
    days_map = {
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
    }
    days = days_map.get(period, 365)
    start_dt = datetime.now(timezone.utc) - timedelta(days=days)
    return start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


async def _fetch_alpaca_bars(symbol: str, timeframe: str, start: str, limit: int = 1000) -> list[dict]:
    """
    Fetch OHLCV bars from Alpaca for stocks or crypto.
    Returns list of dicts with keys: time, open, high, low, close, volume, vwap
    """
    sym_upper = symbol.upper()
    headers = _alpaca_headers()

    async with httpx.AsyncClient(timeout=15.0) as client:
        if _is_crypto(sym_upper):
            url = f"{ALPACA_DATA_URL}/v1beta3/crypto/us/bars"
            params = {"symbols": sym_upper, "timeframe": timeframe, "start": start, "limit": limit}
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Alpaca crypto bars error: {resp.status_code} {resp.text[:200]}")
            data = resp.json()
            bars_by_sym = data.get("bars", {})
            raw_bars = bars_by_sym.get(sym_upper, [])
        else:
            url = f"{ALPACA_DATA_URL}/v2/stocks/{sym_upper}/bars"
            params = {"timeframe": timeframe, "start": start, "limit": limit}
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Alpaca bars error: {resp.status_code} {resp.text[:200]}")
            data = resp.json()
            raw_bars = data.get("bars", [])

    records = []
    for bar in raw_bars:
        records.append({
            "time": bar.get("t", ""),
            "open": float(bar.get("o", 0)),
            "high": float(bar.get("h", 0)),
            "low": float(bar.get("l", 0)),
            "close": float(bar.get("c", 0)),
            "volume": float(bar.get("v", 0)),
            "vwap": float(bar.get("vw", 0)),
        })
    return records


def _sentiment_score(text: str) -> float:
    """Simple keyword-based sentiment scoring clamped to [-1, 1]."""
    positive_words = ["beat", "surge", "rally", "record", "strong", "upgrade", "buy", "profit", "growth", "bullish"]
    negative_words = ["miss", "drop", "fall", "cut", "weak", "downgrade", "sell", "loss", "decline", "bearish"]
    lower = text.lower()
    pos = sum(lower.count(w) for w in positive_words)
    neg = sum(lower.count(w) for w in negative_words)
    denom = max(pos + neg, 1)
    raw = (pos - neg) / denom
    return max(-1.0, min(1.0, raw))


# ─── Existing endpoints ───────────────────────────────────────────────────────

@router.get("/quote/{symbol}")
async def get_quote(
    symbol: str,
    current_user: User = Depends(get_current_user),
):
    """Get live quote for a single symbol via Alpaca."""
    sym_upper = symbol.upper()
    # Try Alpaca stocks quote
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{ALPACA_DATA_URL}/v2/stocks/{sym_upper}/quotes/latest",
                headers=_alpaca_headers(),
            )
            if resp.status_code == 200:
                data = resp.json()
                q = data.get("quote", {})
                bid = q.get("bp") or q.get("bid_price")
                ask = q.get("ap") or q.get("ask_price")
                bid_size = q.get("bs") or q.get("bid_size")
                ask_size = q.get("as") or q.get("ask_size")
                ts = q.get("t") or q.get("timestamp")
                mid = round((float(bid) + float(ask)) / 2, 4) if bid and ask else None
                return {
                    "symbol": sym_upper,
                    "bid_price": float(bid) if bid else None,
                    "ask_price": float(ask) if ask else None,
                    "bid_size": int(bid_size) if bid_size else None,
                    "ask_size": int(ask_size) if ask_size else None,
                    "mid_price": mid,
                    "last": mid,
                    "timestamp": ts,
                    "source": "alpaca",
                }
    except Exception:
        pass

    # Fallback: try latest bar price from Alpaca
    try:
        bars = await _fetch_alpaca_bars(sym_upper, "1Day", _period_to_start("1mo"), limit=1)
        if bars:
            last = bars[-1]["close"]
            return {
                "symbol": sym_upper,
                "bid_price": None,
                "ask_price": None,
                "bid_size": None,
                "ask_size": None,
                "mid_price": last,
                "last": last,
                "timestamp": bars[-1]["time"],
                "source": "alpaca_bars",
            }
    except Exception:
        pass

    return {
        "symbol": sym_upper,
        "bid_price": None,
        "ask_price": None,
        "bid_size": None,
        "ask_size": None,
        "mid_price": None,
        "last": None,
        "timestamp": None,
        "source": "unavailable",
    }


@router.get("/quotes")
async def get_quotes_batch(
    symbols: str = Query(..., description="Comma-separated symbols, e.g. AAPL,MSFT,NVDA"),
    current_user: User = Depends(get_current_user),
):
    """Get live quotes for multiple symbols at once."""
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not sym_list:
        return []

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{ALPACA_DATA_URL}/v2/stocks/quotes/latest",
                params={"symbols": ",".join(sym_list)},
                headers=_alpaca_headers(),
            )
            if resp.status_code == 200:
                data = resp.json()
                quotes = data.get("quotes", {})
                result = []
                for sym, q in quotes.items():
                    bid = q.get("bp") or q.get("bid_price")
                    ask = q.get("ap") or q.get("ask_price")
                    bid_size = q.get("bs") or q.get("bid_size")
                    ask_size = q.get("as") or q.get("ask_size")
                    ts = q.get("t") or q.get("timestamp")
                    mid = round((float(bid) + float(ask)) / 2, 4) if bid and ask else None
                    result.append({
                        "symbol": sym,
                        "bid_price": float(bid) if bid else None,
                        "ask_price": float(ask) if ask else None,
                        "bid_size": int(bid_size) if bid_size else None,
                        "ask_size": int(ask_size) if ask_size else None,
                        "mid_price": mid,
                        "last": mid,
                        "timestamp": ts,
                    })
                return result
    except Exception:
        pass

    return []


@router.get("/history/{symbol}")
async def get_history(
    symbol: str,
    interval: str = Query("1d", enum=["1m", "5m", "15m", "1h", "4h", "1d", "1w"]),
    period: str = Query("1y", enum=["1mo", "3mo", "6mo", "1y", "2y", "5y"]),
    current_user: User = Depends(get_current_user),
):
    """OHLCV history via Alpaca bars API (7+ years free, official API)."""
    tf = _interval_to_alpaca(interval)
    start = _period_to_start(period)
    return await _fetch_alpaca_bars(symbol, tf, start)


@router.get("/bars/{symbol}")
async def get_bars(
    symbol: str,
    interval: str = Query("1d", enum=["1m", "5m", "15m", "1h", "4h", "1d", "1w"]),
    period: str = Query("1y", enum=["1mo", "3mo", "6mo", "1y", "2y", "5y"]),
    current_user: User = Depends(get_current_user),
):
    """Alias for /history/{symbol} — used by backtester."""
    tf = _interval_to_alpaca(interval)
    start = _period_to_start(period)
    return await _fetch_alpaca_bars(symbol, tf, start)


# ─── IV Rank / IV Percentile ─────────────────────────────────────────────────

async def _compute_iv_rank(symbol: str) -> dict:
    """
    Core IV Rank / IV Percentile calculation.

    Approach:
    1. Fetch 52 weeks of daily OHLCV from Alpaca
    2. Estimate HV as annualised rolling 20-day std of log returns × sqrt(252)
    3. Current IV = ATM option snapshot from Alpaca if available, else current HV × 1.1
    4. IV Rank = (current_iv - min_hv_52w) / (max_hv_52w - min_hv_52w) * 100
    5. IV Percentile = % of days where HV < current_iv
    """
    import pandas as pd

    sym_upper = symbol.upper()

    # 1. Fetch 52 weeks of daily bars
    start = (datetime.now(timezone.utc) - timedelta(days=370)).strftime("%Y-%m-%dT%H:%M:%SZ")
    bars = await _fetch_alpaca_bars(sym_upper, "1Day", start, limit=1000)
    if len(bars) < 25:
        raise HTTPException(status_code=422, detail=f"Insufficient data for {sym_upper} IV Rank calculation (got {len(bars)} bars, need 25+)")

    # 2. Build close series and compute rolling 20-day HV
    closes = pd.Series([b["close"] for b in bars])
    log_returns = closes.pct_change().apply(lambda r: math.log(1 + r) if r > -1 else float("nan"))
    hv_series = log_returns.rolling(20).std() * math.sqrt(252)
    hv_series = hv_series.dropna()

    if len(hv_series) < 5:
        raise HTTPException(status_code=422, detail=f"Insufficient HV data for {sym_upper}")

    hv_min_52w = float(hv_series.min())
    hv_max_52w = float(hv_series.max())
    current_hv = float(hv_series.iloc[-1])

    # HV 30-day realized volatility
    hv_30 = float(log_returns.tail(30).std() * math.sqrt(252)) if len(log_returns) >= 30 else current_hv

    # 3. Try to get current IV from Alpaca options snapshot (ATM contract)
    current_iv = None
    source = "alpaca_hv_proxy"
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{ALPACA_DATA_URL}/v1beta1/options/snapshots/{sym_upper}",
                params={"limit": 10, "type": "call"},
                headers=_alpaca_headers(),
            )
            if resp.status_code == 200:
                snap_data = resp.json()
                snapshots = snap_data.get("snapshots", {})
                ivs = []
                for contract_sym, snap in snapshots.items():
                    greeks = snap.get("greeks", {})
                    iv = greeks.get("iv") or snap.get("impliedVolatility")
                    if iv and float(iv) > 0.01:
                        ivs.append(float(iv))
                if ivs:
                    current_iv = sum(ivs) / len(ivs)
                    source = "alpaca_options"
    except Exception:
        pass

    # Fall back to HV × 1.1 proxy
    if current_iv is None or current_iv <= 0:
        current_iv = current_hv * 1.1
        source = "alpaca_hv_proxy"

    # 4. IV Rank
    denom = max(hv_max_52w - hv_min_52w, 0.001)
    iv_rank = max(0.0, min(100.0, (current_iv - hv_min_52w) / denom * 100))

    # 5. IV Percentile
    days_below = int((hv_series < current_iv).sum())
    total_days = len(hv_series)
    iv_percentile = round(days_below / total_days * 100, 2)

    # 6. HV/IV ratio and term structure proxy
    hv_iv_ratio = round(current_iv / max(hv_30, 0.001), 4)

    # Rough term structure: compare current HV (near) vs 60-day HV (far)
    hv_far = float(log_returns.tail(60).std() * math.sqrt(252)) if len(log_returns) >= 60 else current_hv
    term_structure = "contango" if current_hv < hv_far else "backwardation"

    # 7. Regime and trade signal
    if iv_rank >= 50:
        regime = "high_iv"
        trade_signal = "sell_premium"
    elif iv_rank <= 20:
        regime = "low_iv"
        trade_signal = "buy_premium"
    else:
        regime = "normal"
        trade_signal = "neutral"

    return {
        "symbol": sym_upper,
        "current_iv": round(current_iv, 6),
        "iv_rank": round(iv_rank, 2),
        "iv_percentile": iv_percentile,
        "hv_30": round(hv_30, 6),
        "hv_iv_ratio": hv_iv_ratio,
        "iv_52w_high": round(hv_max_52w, 6),
        "iv_52w_low": round(hv_min_52w, 6),
        "regime": regime,
        "trade_signal": trade_signal,
        "term_structure": term_structure,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
    }


@router.get("/iv-rank/{symbol}")
async def get_iv_rank(
    symbol: str,
    current_user: User = Depends(get_current_user),
):
    """
    Calculate IV Rank and IV Percentile for a symbol using 52 weeks of
    historical implied volatility estimated from ATM option contracts.

    IV Rank = (current_IV - min_IV_52w) / (max_IV_52w - min_IV_52w) * 100
    IV Percentile = % of days in past year where IV was below current IV

    These are the core Options Alpha entry filters:
    - IV Rank > 50: premium selling environment (sell options)
    - IV Rank < 20: premium buying environment (buy options)
    """
    return await _compute_iv_rank(symbol)


@router.get("/iv-rank-scan")
async def get_iv_rank_scan(
    symbols: str = Query("AAPL,MSFT,NVDA,SPY,QQQ,TSLA,AMZN,META",
                         description="Comma-separated symbols to scan"),
    current_user: User = Depends(get_current_user),
):
    """
    Batch IV Rank scanner — calls iv-rank for each symbol concurrently.
    Returns results sorted by iv_rank descending (highest premium-selling
    opportunities first). Powers the Options Alpha-style scanner.
    """
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not sym_list:
        return []

    async def _safe_iv_rank(sym: str) -> dict | None:
        try:
            return await _compute_iv_rank(sym)
        except Exception as exc:
            return {"symbol": sym, "error": str(exc), "iv_rank": -1}

    results = await asyncio.gather(*[_safe_iv_rank(s) for s in sym_list])
    valid = [r for r in results if r is not None]
    valid.sort(key=lambda x: x.get("iv_rank", -1), reverse=True)
    return valid


# ─── News ─────────────────────────────────────────────────────────────────────

@router.get("/news")
async def get_news(
    symbols: str = Query("SPY", description="Comma-separated symbols"),
    limit: int = Query(20, ge=1, le=50),
    current_user: User = Depends(get_current_user),
):
    """Proxy Alpaca News API and add keyword-based sentiment scoring."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{ALPACA_DATA_URL}/v1beta1/news",
                params={"symbols": symbols, "limit": limit},
                headers=_alpaca_headers(),
            )
            if resp.status_code != 200:
                return {"news": [], "error": f"Alpaca returned {resp.status_code}", "data_source": "unavailable"}

            raw = resp.json()
            articles = raw if isinstance(raw, list) else raw.get("news", [])

            result = []
            for item in articles:
                headline = item.get("headline", "")
                summary = item.get("summary", "")
                score = _sentiment_score(f"{headline} {summary}")
                result.append({
                    "id": item.get("id"),
                    "headline": headline,
                    "summary": summary,
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "author": item.get("author", ""),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "symbols": item.get("symbols", []),
                    "sentiment_score": round(score, 3),
                })
            return {"news": result, "data_source": "alpaca"}

    except httpx.TimeoutException:
        return {"news": [], "error": "Request timed out", "data_source": "unavailable"}
    except Exception as exc:
        return {"news": [], "error": str(exc), "data_source": "unavailable"}


# ─── Earnings Calendar ────────────────────────────────────────────────────────

@router.get("/earnings")
async def get_earnings(
    symbols: str = Query("AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA,SPY"),
    current_user: User = Depends(get_current_user),
):
    """Proxy Alpaca corporate actions earnings data. Returns empty if not available (premium)."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{ALPACA_DATA_URL}/v1beta1/corporate-actions",
                params={"types": "earnings", "symbols": symbols},
                headers=_alpaca_headers(),
            )
            if resp.status_code == 403 or resp.status_code == 402:
                return {"earnings": [], "data_source": "unavailable", "reason": "premium_required"}
            if resp.status_code != 200:
                return {"earnings": [], "data_source": "unavailable", "reason": f"status_{resp.status_code}"}

            raw = resp.json()
            items = raw if isinstance(raw, list) else raw.get("earnings", raw.get("corporate_actions", []))

            result = []
            for item in items:
                eps_est = item.get("estimate_eps") or item.get("eps_estimate")
                eps_act = item.get("reported_eps") or item.get("eps_actual")
                surprise = None
                if eps_est is not None and eps_act is not None and eps_est != 0:
                    surprise = round((float(eps_act) - float(eps_est)) / abs(float(eps_est)) * 100, 2)
                result.append({
                    "symbol": item.get("symbol", ""),
                    "report_date": item.get("report_date") or item.get("date"),
                    "fiscal_year": item.get("fiscal_year") or item.get("fiscal_year_ending"),
                    "fiscal_quarter": item.get("fiscal_quarter") or item.get("quarter"),
                    "estimate_eps": float(eps_est) if eps_est is not None else None,
                    "reported_eps": float(eps_act) if eps_act is not None else None,
                    "surprise_pct": surprise,
                })
            return {"earnings": result, "data_source": "alpaca" if result else "unavailable"}

    except httpx.TimeoutException:
        return {"earnings": [], "data_source": "unavailable", "reason": "timeout"}
    except Exception as exc:
        return {"earnings": [], "data_source": "unavailable", "reason": str(exc)}
