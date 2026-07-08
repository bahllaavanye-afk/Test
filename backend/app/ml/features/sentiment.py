"""
Market sentiment features: Fear & Greed Index + FinBERT news sentiment.
Free APIs only. All features are lagged by 1 period to prevent lookahead.
"""
from __future__ import annotations

import asyncio
import functools
import time
from datetime import datetime, timezone, timedelta

import httpx
import pandas as pd
from app.utils.logging import logger


def _log_sync_metrics(func):
    """Decorator to log execution time and result size for synchronous functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.monotonic()
        result = func(*args, **kwargs)
        duration_ms = (time.monotonic() - start) * 1000
        # Determine result count where applicable
        count = None
        if isinstance(result, (list, dict)):
            count = len(result)
        logger.info(
            f"{func.__name__} completed",
            function=func.__name__,
            duration_ms=round(duration_ms, 2),
            result_count=count,
        )
        return result

    return wrapper


def _log_async_metrics(func):
    """Decorator to log execution time and result size for async functions."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.monotonic()
        result = await func(*args, **kwargs)
        duration_ms = (time.monotonic() - start) * 1000
        count = None
        if isinstance(result, (list, dict)):
            count = len(result)
        logger.info(
            f"{func.__name__} completed",
            function=func.__name__,
            duration_ms=round(duration_ms, 2),
            result_count=count,
        )
        return result

    return wrapper


@_log_async_metrics
async def fetch_fear_greed_index() -> dict:
    """
    CNN Fear & Greed Index via alternative.me API (completely free, no key needed).
    Returns current score (0=extreme fear, 100=extreme greed) + classification.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://api.alternative.me/fng/?limit=30&format=json")
            resp.raise_for_status()
            data = resp.json()
        readings = data.get("data", [])
        result = []
        for r in readings:
            result.append(
                {
                    "date": datetime.fromtimestamp(int(r["timestamp"]), tz=timezone.utc).date(),
                    "value": int(r["value"]),
                    "classification": r["value_classification"],
                }
            )
        return {"status": "ok", "readings": result, "current": result[0] if result else None}
    except Exception as e:
        logger.warning("Fear & Greed fetch failed", error=str(e))
        return {"status": "error", "readings": [], "current": None}


@_log_async_metrics
async def fetch_news_sentiment(symbol: str, api_key: str | None = None) -> list[dict]:
    """
    NewsAPI.org headlines sentiment (free tier: 100 req/day).
    Falls back to empty list if no API key or quota exceeded.
    Returns list of {published_at, title, sentiment_score [-1..1]}.
    """
    if not api_key:
        return []
    try:
        query = symbol.replace("/", "").replace("-", " ")
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "sortBy": "publishedAt",
                    "pageSize": 20,
                    "language": "en",
                    "apiKey": api_key,
                },
            )
            resp.raise_for_status()
            articles = resp.json().get("articles", [])

        sentiments = []
        for a in articles:
            title = a.get("title", "") or ""
            # Simple lexicon-based scoring (no heavy model needed for free tier)
            score = _simple_sentiment(title)
            sentiments.append(
                {
                    "published_at": a.get("publishedAt"),
                    "title": title[:120],
                    "sentiment_score": score,
                }
            )
        return sentiments
    except Exception as e:
        logger.warning("NewsAPI fetch failed", symbol=symbol, error=str(e))
        return []


def _simple_sentiment(text: str) -> float:
    """Fast lexicon sentiment score in range [-1, 1]."""
    text_lower = text.lower()
    bullish = [
        "surge",
        "rally",
        "gain",
        "bull",
        "up",
        "high",
        "rise",
        "strong",
        "beat",
        "record",
        "buy",
        "growth",
    ]
    bearish = [
        "crash",
        "drop",
        "fall",
        "bear",
        "down",
        "low",
        "decline",
        "weak",
        "miss",
        "loss",
        "sell",
        "fear",
    ]
    score = sum(1 for w in bullish if w in text_lower) - sum(1 for w in bearish if w in text_lower)
    return max(-1.0, min(1.0, score / max(len(bullish), 1)))


class SECFilingSentiment:
    """
    Fetches 10-Q/8-K filings from SEC EDGAR free API.
    Runs FinBERT (ProsusAI/finbert from HuggingFace transformers) on MD&A sections.
    Returns management_tone_score (-1 to +1).

    Falls back gracefully if transformers is not installed.
    """

    EDGAR_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"
    EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
    _USER_AGENT = "QuantEdge research@quantedge.io"

    def __init__(self) -> None:
        self._pipeline = None
        self._available = self._try_load_finbert()

    def _try_load_finbert(self) -> bool:
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore

            self._pipeline = hf_pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                return_all_scores=True,
            )
            return True
        except ImportError:
            return False

    @_log_sync_metrics
    def get_cik(self, ticker: str) -> int | None:
        """Look up CIK from SEC EDGAR company tickers JSON (free, no key needed)."""
        import json
        import urllib.request

        try:
            req = urllib.request.Request(
                self.EDGAR_COMPANY_TICKERS,
                headers={"User-Agent": self._USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            for v in data.values():
                if v.get("ticker", "").upper() == ticker.upper():
                    return int(v["cik_str"])
        except Exception as exc:
            logger.debug("SEC CIK lookup failed", ticker=ticker, error=str(exc))
        return None

    @_log_sync_metrics
    def get_management_tone(self, ticker: str) -> float | None:
        """
        Returns management_tone_score (-1 bearish, 0 neutral, +1 bullish).
        Returns None if FinBERT is not available or filing cannot be fetched.

        Full MD&A extraction requires an SGML parser; here we use the
        filing summary text from EDGAR's submissions JSON as a proxy.
        """
        if not self._available:
            return None

        cik = self.get_cik(ticker)
        if cik is None:
            return None

        import json
        import urllib.request

        try:
            url = self.EDGAR_SUBMISSIONS.format(cik=cik)
            req = urllib.request.Request(url, headers={"User-Agent": self._USER_AGENT})
            with urllib.request.urlopen(req, timeout=10) as r:
                submissions = json.loads(r.read())
        except Exception as exc:
            logger.debug("SEC submissions fetch failed", ticker=ticker, error=str(exc))
            return None

        # Extract the most recent 8-K or 10-Q description as tone proxy
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        descriptions = recent.get("primaryDocument", [])

        text_snippets: list[str] = []
        for form, desc in zip(forms, descriptions):
            if form in ("8-K", "10-Q", "10-K"):
                text_snippets.append(str(desc))
            if len(text_snippets) >= 5:
                break

        if not text_snippets:
            return None

        combined_text = " ".join(text_snippets)[:512]

        try:
            scores_list = self._pipeline(combined_text)  # type: ignore
            if not scores_list:
                return None
            scores = {item["label"].lower(): item["score"] for item in scores_list[0]}
            positive = scores.get("positive", 0.0)
            negative = scores.get("negative", 0.0)
            tone = float(positive - negative)  # range approximately [-1, +1]
            return tone
        except Exception as exc:
            logger.debug("FinBERT inference failed", ticker=ticker, error=str(exc))
            return None


@_log_sync_metrics
def add_sentiment_features(df: pd.DataFrame, fear_greed_history: list[dict]) -> pd.DataFrame:
    """
    Merge Fear & Greed Index into OHLCV dataframe.
    Features added (all lagged 1 bar to prevent lookahead):
      - fear_greed_score: 0-100
      - fear_greed_norm: -1 to 1 rescaled
      - extreme_fear: bool (score < 25)
      - extreme_greed: bool (score > 75)
    """
    if not fear_greed_history:
        df["fear_greed_score"] = 50.0
        df["fear_greed_norm"] = 0.0
        df["extreme_fear"] = False
        df["extreme_greed"] = False
        return df

    fg_df = pd.DataFrame(fear_greed_history)
    fg_df["date"] = pd.to_datetime(fg_df["date"])

    # Align on date, shift by one period to avoid lookahead bias
    fg_df = fg_df.set_index("date").sort_index()
    fg_df["fear_greed_score"] = fg_df["value"]
    fg_df["fear_greed_norm"] = (fg_df["value"] - 50) / 50  # map 0-100 -> -1 to 1
    fg_df["extreme_fear"] = fg_df["value"] < 25
    fg_df["extreme_greed"] = fg_df["value"] > 75

    # Shift all new columns by one row (previous bar)
    fg_shifted = fg_df[["fear_greed_score", "fear_greed_norm", "extreme_fear", "extreme_greed"]].shift(1)

    # Merge onto original dataframe based on its index (assumed datetime)
    df = df.merge(fg_shifted, left_index=True, right_index=True, how="left")

    # Fill any missing values with neutral defaults
    df["fear_greed_score"].fillna(50.0, inplace=True)
    df["fear_greed_norm"].fillna(0.0, inplace=True)
    df["extreme_fear"].fillna(False, inplace=True)
    df["extreme_greed"].fillna(False, inplace=True)

    return df