"""Market data endpoints: quotes, historical OHLCV, news, earnings, IV Rank, PCR."""
from fastapi import APIRouter, Depends, Query, HTTPException
from app.api.deps import get_current_user
from app.models.user import User
from app.config import settings
import asyncio
import httpx
import math
from datetime import datetime, timezone, timedelta
from app.utils.logging import logger

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
    except Exception as exc:
        logger.debug("quote fetch (stocks) failed", symbol=sym_upper, error=str(exc))

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
    except Exception as exc:
        logger.debug("quote fetch (bars fallback) failed", symbol=sym_upper, error=str(exc))

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
    except Exception as exc:
        logger.warning("quotes_batch fetch failed", error=str(exc))

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


@router.get("/bars")
async def get_bars_query(
    symbol: str = Query(..., description="Ticker symbol, e.g. SPY"),
    timeframe: str = Query("1Day", description="Alpaca timeframe string, e.g. 1Day, 1Hour, 5Min"),
    limit: int = Query(200, ge=1, le=10000, description="Number of bars to return"),
    current_user: User = Depends(get_current_user),
):
    """OHLCV bars using query parameters — used by the frontend chart widget.

    Accepts Alpaca-style timeframe strings (1Day, 1Hour, 5Min) directly.
    The start date is calculated to cover the requested number of bars.
    """
    # Map common timeframe strings to Alpaca API format
    tf_map = {
        "1min": "1Min", "1Min": "1Min",
        "5min": "5Min", "5Min": "5Min",
        "15min": "15Min", "15Min": "15Min",
        "1hour": "1Hour", "1Hour": "1Hour",
        "4hour": "4Hour", "4Hour": "4Hour",
        "1day": "1Day", "1Day": "1Day",
        "1week": "1Week", "1Week": "1Week",
        # shorthand aliases
        "1m": "1Min", "5m": "5Min", "15m": "15Min",
        "1h": "1Hour", "4h": "4Hour", "1d": "1Day", "1w": "1Week",
    }
    tf = tf_map.get(timeframe, timeframe)

    # Estimate start date: assume worst case 2 bars/day for intraday, 1/day for daily+
    intraday = tf in ("1Min", "5Min", "15Min", "1Hour", "4Hour")
    lookback_days = max(int(limit / 6.5) + 5, 30) if intraday else limit + 30
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")

    bars = await _fetch_alpaca_bars(symbol, tf, start, limit=limit)
    return bars


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


# ─── Polymarket ───────────────────────────────────────────────────────────────

_POLYMARKET_API_URL = "https://clob.polymarket.com"


@router.get("/polymarket")
async def get_polymarket_markets(
    filter: str = Query("", description="Optional keyword filter for market question"),
    sort: str = Query("volume", description="Sort field: volume, end_date, created_at"),
    limit: int = Query(50, ge=1, le=200, description="Number of markets to return"),
    current_user: User = Depends(get_current_user),
):
    """Return active Polymarket prediction markets.

    Fetches from the Polymarket CLOB API (no auth required for reads).
    Returns an empty list if the upstream call fails rather than 404.
    """
    try:
        params: dict = {"limit": limit, "active": "true", "closed": "false"}
        if filter:
            params["search"] = filter

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{_POLYMARKET_API_URL}/markets", params=params)
            if resp.status_code != 200:
                logger.warning("Polymarket API returned non-200", status=resp.status_code)
                return []

            data = resp.json()
            markets = data if isinstance(data, list) else data.get("data", [])

            result = []
            for m in markets:
                question = m.get("question", "") or m.get("description", "")
                if filter and filter.lower() not in question.lower():
                    continue
                # Determine category from tags or market group
                tags = m.get("tags", []) or []
                tag_names = [t.get("slug", t) if isinstance(t, dict) else str(t) for t in tags]
                category = "other"
                for tag in tag_names:
                    tl = tag.lower()
                    if any(k in tl for k in ("politi", "election", "govern")):
                        category = "politics"
                        break
                    if any(k in tl for k in ("crypto", "bitcoin", "ethereum", "btc", "eth")):
                        category = "crypto"
                        break
                    if any(k in tl for k in ("sport", "nba", "nfl", "soccer", "football", "baseball")):
                        category = "sports"
                        break
                    if any(k in tl for k in ("econ", "fed", "rate", "inflation", "gdp", "market")):
                        category = "economics"
                        break
                result.append({
                    "id": m.get("condition_id") or m.get("id", ""),
                    "title": question,
                    "end_date": m.get("end_date_iso") or m.get("end_date") or "",
                    "yes_price": float(m.get("outcomePrices", ["0"])[0]) if m.get("outcomePrices") else 0.0,
                    "no_price": float(m.get("outcomePrices", ["0", "0"])[1]) if len(m.get("outcomePrices", [])) > 1 else 0.0,
                    "volume_24h": float(m.get("volume24hr", 0) or m.get("volume", 0) or 0),
                    "liquidity": float(m.get("liquidity", 0) or 0),
                    "category": category,
                    "active": m.get("active", True),
                    "closed": m.get("closed", False),
                })

            # Sort results
            sort_key = {"volume": "volume", "end_date": "end_date", "created_at": "end_date"}.get(sort, "volume")
            result.sort(key=lambda x: x.get(sort_key) or "", reverse=(sort_key == "volume"))
            return result

    except Exception as exc:
        logger.warning("polymarket endpoint failed", error=str(exc))
        return []


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
    except Exception as exc:
        logger.debug("IV rank options fetch failed", symbol=sym_upper, error=str(exc))

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


# ─── Put/Call Ratio (PCR) ─────────────────────────────────────────────────────

@router.get("/pcr")
async def get_pcr(
    symbol: str = Query("SPY", description="Underlying symbol, e.g. SPY, QQQ, AAPL"),
    current_user: User = Depends(get_current_user),
):
    """
    Compute the live Put/Call Ratio for the given symbol using Alpaca options
    snapshots. Returns PCR value, directional signal, and confidence score.

    PCR > 1.2  → bullish (contrarian — excessive bearishness signals reversal)
    PCR 0.8-1.2 → neutral
    PCR < 0.8  → bearish (contrarian — excessive bullishness signals reversal)

    Implements the same logic as OptionsPCRReversalStrategy._fetch_pcr().
    """
    sym_upper = symbol.upper()

    PCR_HIGH = 1.20
    PCR_LOW  = 0.80  # wider neutral band than strategy's 0.55 for the dashboard

    if not (settings.alpaca_api_key and settings.alpaca_secret_key):
        return {
            "symbol": sym_upper,
            "pcr": None,
            "put_volume": None,
            "call_volume": None,
            "signal": "unavailable",
            "confidence": None,
            "regime": "unavailable",
            "source": "no_credentials",
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{ALPACA_DATA_URL}/v1beta1/options/snapshots/{sym_upper}",
                headers=headers,
                params={"feed": "indicative"},
            )
            if resp.status_code != 200:
                return {
                    "symbol": sym_upper,
                    "pcr": None,
                    "put_volume": None,
                    "call_volume": None,
                    "signal": "unavailable",
                    "confidence": None,
                    "regime": "unavailable",
                    "source": f"alpaca_error_{resp.status_code}",
                    "computed_at": datetime.now(timezone.utc).isoformat(),
                }

            data = resp.json()
            snapshots = data.get("snapshots", {})

            put_vol = 0.0
            call_vol = 0.0
            for occ_sym, snap in snapshots.items():
                if len(occ_sym) < 16:
                    continue
                cp_flag = occ_sym[-9]
                daily = snap.get("dailyBar") or snap.get("minuteBar") or {}
                vol = float(daily.get("v") or 0)
                if cp_flag == "P":
                    put_vol += vol
                elif cp_flag == "C":
                    call_vol += vol

            if call_vol < 1:
                return {
                    "symbol": sym_upper,
                    "pcr": None,
                    "put_volume": put_vol,
                    "call_volume": call_vol,
                    "signal": "unavailable",
                    "confidence": None,
                    "regime": "unavailable",
                    "source": "no_call_volume",
                    "computed_at": datetime.now(timezone.utc).isoformat(),
                }

            pcr = round(put_vol / call_vol, 4)

            # Determine signal and confidence
            if pcr >= PCR_HIGH:
                signal = "buy"
                regime = "bullish"
                confidence = round(min(0.90, 0.55 + (pcr - PCR_HIGH) * 0.5), 4)
            elif pcr <= PCR_LOW:
                signal = "sell"
                regime = "bearish"
                confidence = round(min(0.90, 0.55 + (PCR_LOW - pcr) * 0.8), 4)
            else:
                signal = "neutral"
                regime = "neutral"
                confidence = round(0.30 + abs(pcr - 1.0) * 0.2, 4)

            return {
                "symbol": sym_upper,
                "pcr": pcr,
                "put_volume": put_vol,
                "call_volume": call_vol,
                "signal": signal,
                "confidence": confidence,
                "regime": regime,
                "pcr_high_threshold": PCR_HIGH,
                "pcr_low_threshold": PCR_LOW,
                "source": "alpaca_options",
                "computed_at": datetime.now(timezone.utc).isoformat(),
            }

    except httpx.TimeoutException:
        return {
            "symbol": sym_upper,
            "pcr": None,
            "put_volume": None,
            "call_volume": None,
            "signal": "unavailable",
            "confidence": None,
            "regime": "unavailable",
            "source": "timeout",
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.warning("PCR endpoint failed", symbol=sym_upper, error=str(exc))
        return {
            "symbol": sym_upper,
            "pcr": None,
            "put_volume": None,
            "call_volume": None,
            "signal": "unavailable",
            "confidence": None,
            "regime": "unavailable",
            "source": str(exc),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }


# ─── Sector Heatmap ──────────────────────────────────────────────────────────

SECTOR_ETFS = {
    "Technology":       "XLK",
    "Healthcare":       "XLV",
    "Financials":       "XLF",
    "Consumer Discr.":  "XLY",
    "Industrials":      "XLI",
    "Communication":    "XLC",
    "Consumer Staples": "XLP",
    "Energy":           "XLE",
    "Utilities":        "XLU",
    "Real Estate":      "XLRE",
    "Materials":        "XLB",
}


@router.get("/sector-heatmap")
async def get_sector_heatmap(
    current_user: User = Depends(get_current_user),
):
    """Return % change for each S&P 500 sector ETF for the heatmap widget."""
    symbols = list(SECTOR_ETFS.values())

    # Fetch today's bar and yesterday's close for each ETF concurrently
    async def _pct_change(sym: str) -> dict:
        try:
            start = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
            bars = await _fetch_alpaca_bars(sym, "1Day", start, limit=5)
            if len(bars) >= 2:
                prev_close = bars[-2]["close"]
                cur_close  = bars[-1]["close"]
                chg = (cur_close - prev_close) / prev_close * 100 if prev_close else 0.0
                return {"symbol": sym, "close": cur_close, "change_pct": round(chg, 4)}
            elif len(bars) == 1:
                return {"symbol": sym, "close": bars[-1]["close"], "change_pct": 0.0}
        except Exception as exc:
            logger.debug("sector heatmap bar fetch failed", symbol=sym, error=str(exc))
        return {"symbol": sym, "close": None, "change_pct": 0.0}

    results = await asyncio.gather(*[_pct_change(sym) for sym in symbols])
    by_sym = {r["symbol"]: r for r in results}

    return [
        {
            "sector": sector,
            "symbol": etf_sym,
            "change_pct": by_sym.get(etf_sym, {}).get("change_pct", 0.0),
            "close": by_sym.get(etf_sym, {}).get("close"),
        }
        for sector, etf_sym in SECTOR_ETFS.items()
    ]


# ─── Economic Calendar ────────────────────────────────────────────────────────

# Free economic data via FRED API (no auth needed for most series)
_FRED_BASE = "https://api.stlouisfed.org/fred"

# Curated list of high-impact macro events with their FRED series IDs
_MACRO_SERIES: list[dict] = [
    {"id": "UNRATE",   "name": "Unemployment Rate",        "country": "US", "flag": "🇺🇸", "impact": "high"},
    {"id": "CPIAUCSL", "name": "CPI (YoY)",                "country": "US", "flag": "🇺🇸", "impact": "high"},
    {"id": "PPIACO",   "name": "PPI",                      "country": "US", "flag": "🇺🇸", "impact": "medium"},
    {"id": "GDP",      "name": "GDP Growth (QoQ)",         "country": "US", "flag": "🇺🇸", "impact": "high"},
    {"id": "PAYEMS",   "name": "Nonfarm Payrolls",         "country": "US", "flag": "🇺🇸", "impact": "high"},
    {"id": "FEDFUNDS", "name": "Fed Funds Rate",           "country": "US", "flag": "🇺🇸", "impact": "high"},
    {"id": "T10Y2Y",   "name": "10Y-2Y Yield Spread",     "country": "US", "flag": "🇺🇸", "impact": "medium"},
    {"id": "DCOILWTICO","name": "WTI Crude Oil Price",     "country": "US", "flag": "🌍",  "impact": "medium"},
]


@router.get("/economic-calendar")
async def get_economic_calendar(
    current_user: User = Depends(get_current_user),
):
    """
    Return recent macro data releases from FRED public API.

    Uses the St. Louis Fed FRED API (no API key required for most endpoints).
    Returns the most recent observation plus the previous one as 'forecast proxy'.
    """
    fred_api_key = getattr(settings, "fred_api_key", None) or "abcdefghijklmnopqrstuvwxyz123456"

    async def _fetch_series(meta: dict) -> dict | None:
        series_id = meta["id"]
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(
                    f"{_FRED_BASE}/series/observations",
                    params={
                        "series_id": series_id,
                        "api_key": fred_api_key,
                        "file_type": "json",
                        "limit": 3,
                        "sort_order": "desc",
                        "observation_start": (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d"),
                    },
                )
                if resp.status_code != 200:
                    return None
                data = resp.json()
                obs = data.get("observations", [])
                if not obs:
                    return None

                latest = obs[0]
                previous = obs[1] if len(obs) > 1 else None

                # Parse release date as scheduled_at
                release_date = latest.get("date", "")
                try:
                    scheduled_at = datetime.strptime(release_date, "%Y-%m-%d").replace(
                        hour=8, minute=30, tzinfo=timezone.utc
                    ).isoformat()
                except ValueError:
                    scheduled_at = datetime.now(timezone.utc).isoformat()

                actual_val = latest.get("value")
                if actual_val in (".", "", None):
                    actual_val = None

                prev_val = previous.get("value") if previous else None
                if prev_val in (".", "", None):
                    prev_val = None

                return {
                    "id": series_id,
                    "name": meta["name"],
                    "country": meta["country"],
                    "country_flag": meta["flag"],
                    "impact": meta["impact"],
                    "scheduled_at": scheduled_at,
                    "actual": actual_val,
                    "forecast": None,   # FRED doesn't publish forecasts; use prev as proxy
                    "previous": prev_val,
                }
        except Exception as exc:
            logger.debug("FRED series fetch failed", series=series_id, error=str(exc))
            return None

    results = await asyncio.gather(*[_fetch_series(m) for m in _MACRO_SERIES])
    events = [r for r in results if r is not None]
    # Sort: unreleased first, then by date descending
    events.sort(key=lambda e: (e["actual"] is not None, e["scheduled_at"]))
    return events


# ─── Underscore-prefix alias ──────────────────────────────────────────────────
# The frontend calls /market_data/* (underscore) while the primary router uses
# /market-data/* (hyphen).  Mount a second router at /market_data so both work.

router_underscore = APIRouter(prefix="/market_data", tags=["market_data"])


@router_underscore.get("/polymarket")
async def get_polymarket_markets_alias(
    filter: str = Query("", description="Optional keyword filter for market question"),
    sort: str = Query("volume", description="Sort field: volume, end_date, created_at"),
    limit: int = Query(50, ge=1, le=200, description="Number of markets to return"),
    current_user: User = Depends(get_current_user),
):
    """Underscore-prefix alias for /market-data/polymarket."""
    return await get_polymarket_markets(filter=filter, sort=sort, limit=limit, current_user=current_user)


@router_underscore.get("/bars")
async def get_bars_query_alias(
    symbol: str = Query(..., description="Ticker symbol, e.g. SPY"),
    timeframe: str = Query("1Day", description="Alpaca timeframe string, e.g. 1Day, 1Hour, 5Min"),
    limit: int = Query(200, ge=1, le=10000, description="Number of bars to return"),
    current_user: User = Depends(get_current_user),
):
    """Underscore-prefix alias for /market-data/bars (query-param form)."""
    return await get_bars_query(symbol=symbol, timeframe=timeframe, limit=limit, current_user=current_user)
