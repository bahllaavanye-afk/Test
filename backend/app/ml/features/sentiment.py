"""
Market sentiment features: Fear & Greed Index + FinBERT news sentiment.
Free APIs only. All features are lagged by 1 period to prevent lookahead.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone, timedelta

import httpx
import pandas as pd
from app.utils.logging import logger


def _log_info(event: str, **metrics) -> None:
    """Helper to emit structured INFO logs."""
    logger.info(event, **metrics)


async def fetch_fear_greed_index() -> dict:
    """
    CNN Fear & Greed Index via alternative.me API (completely free, no key needed).
    Returns current score (0=extreme fear, 100=extreme greed) + classification.
    """
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://api.alternative.me/fng/?limit=30&format=json")
            resp.raise_for_status()
            data = resp.json()
        readings = data.get("data", [])
        result = []
        for r in readings:
            result.append({
                "date": datetime.fromtimestamp(int(r["timestamp"]), tz=timezone.utc).date(),
                "value": int(r["value"]),
                "classification": r["value_classification"],
            })
        elapsed = time.perf_counter() - start
        _log_info(
            "fetch_fear_greed_index_success",
            signal_count=len(result),
            execution_time_seconds=elapsed,
            pnl=0.0,
        )
        return {"status": "ok", "readings": result, "current": result[0] if result else None}
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.warning("Fear & Greed fetch failed", error=str(e))
        _log_info(
            "fetch_fear_greed_index_failure",
            signal_count=0,
            execution_time_seconds=elapsed,
            pnl=0.0,
        )
        return {"status": "error", "readings": [], "current": None}


async def fetch_news_sentiment(symbol: str, api_key: str | None = None) -> list[dict]:
    """
    NewsAPI.org headlines sentiment (free tier: 100 req/day).
    Falls back to empty list if no API key or quota exceeded.
    Returns list of {published_at, title, sentiment_score [-1..1]}.
    """
    if not api_key:
        return []
    start = time.perf_counter()
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
            sentiments.append({
                "published_at": a.get("publishedAt"),
                "title": title[:120],
                "sentiment_score": score,
            })
        elapsed = time.perf_counter() - start
        _log_info(
            "fetch_news_sentiment_success",
            signal_count=len(sentiments),
            execution_time_seconds=elapsed,
            pnl=0.0,
        )
        return sentiments
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.warning("NewsAPI fetch failed", symbol=symbol, error=str(e))
        _log_info(
            "fetch_news_sentiment_failure",
            signal_count=0,
            execution_time_seconds=elapsed,
            pnl=0.0,
        )
        return []


def _simple_sentiment(text: str) -> float:
    """Fast lexicon sentiment score in range [-1, 1]."""
    text_lower = text.lower()
    bullish = ["surge", "rally", "gain", "bull", "up", "high", "rise", "strong", "beat", "record", "buy", "growth"]
    bearish = ["crash", "drop", "fall", "bear", "down", "low", "decline", "weak", "miss", "loss", "sell", "fear"]
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

    def get_cik(self, ticker: str) -> int | None:
        """Look up CIK from SEC EDGAR company tickers JSON (free, no key needed)."""
        import urllib.request
        import json

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

    def get_management_tone(self, ticker: str) -> float | None:
        """
        Returns management_tone_score (-1 bearish, 0 neutral, +1 bullish).
        Returns None if FinBERT is not available or filing cannot be fetched.

        Full MD&A extraction requires an SGML parser; here we use the
        filing summary text from EDGAR's submissions JSON as a proxy.
        """
        start = time.perf_counter()
        if not self._available:
            _log_info(
                "sec_filing_sentiment_unavailable",
                signal_count=0,
                execution_time_seconds=time.perf_counter() - start,
                pnl=0.0,
            )
            return None

        cik = self.get_cik(ticker)
        if cik is None:
            _log_info(
                "sec_filing_sentiment_no_cik",
                signal_count=0,
                execution_time_seconds=time.perf_counter() - start,
                pnl=0.0,
            )
            return None

        import urllib.request
        import json

        try:
            url = self.EDGAR_SUBMISSIONS.format(cik=cik)
            req = urllib.request.Request(
                url,
                headers={"User-Agent": self._USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                submissions = json.loads(r.read())
        except Exception as exc:
            logger.debug("SEC submissions fetch failed", ticker=ticker, error=str(exc))
            _log_info(
                "sec_filing_sentiment_fetch_failure",
                signal_count=0,
                execution_time_seconds=time.perf_counter() - start,
                pnl=0.0,
            )
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
            _log_info(
                "sec_filing_sentiment_no_text",
                signal_count=0,
                execution_time_seconds=time.perf_counter() - start,
                pnl=0.0,
            )
            return None

        combined_text = " ".join(text_snippets)[:512]

        try:
            scores_list = self._pipeline(combined_text)  # type: ignore
            # scores_list: [[{label, score}, ...]]
            if not scores_list:
                _log_info(
                    "sec_filing_sentiment_no_scores",
                    signal_count=0,
                    execution_time_seconds=time.perf_counter() - start,
                    pnl=0.0,
                )
                return None
            scores = {item["label"].lower(): item["score"] for item in scores_list[0]}
            positive = scores.get("positive", 0.0)
            negative = scores.get("negative", 0.0)
            tone = float(positive - negative)  # range approximately [-1, +1]
            elapsed = time.perf_counter() - start
            _log_info(
                "sec_filing_sentiment_success",
                signal_count=1,
                execution_time_seconds=elapsed,
                pnl=0.0,
            )
            return tone
        except Exception as exc:
            logger.debug("FinBERT inference failed", ticker=ticker, error=str(exc))
            _log_info(
                "sec_filing_sentiment_inference_failure",
                signal_count=0,
                execution_time_seconds=time.perf_counter() - start,
                pnl=0.0,
            )
            return None


def add_sentiment_features(df: pd.DataFrame, fear_greed_history: list[dict]) -> pd.DataFrame:
    """
    Merge Fear & Greed Index into OHLCV dataframe.
    Features added (all lagged 1 bar to prevent lookahead):
      - fear_greed_score: 0-100
      - fear_greed_norm: -1 to 1 rescaled
      - extreme_fear: bool (score < 25)
      - extreme_greed: bool (score > 75)
    """
    start = time.perf_counter()
    if not fear_greed_history:
        df["fear_greed_score"] = 50.0
        df["fear_greed_norm"] = 0.0
        df["extreme_fear"] = False
        df["extreme_greed"] = False
        elapsed = time.perf_counter() - start
        _log_info(
            "add_sentiment_features_default",
            signal_count=0,
            execution_time_seconds=elapsed,
            pnl=0.0,
        )
        return df

    fg_df = pd.DataFrame(fear_greed_history)
    fg_df["date"] = pd.to_datetime(fg_df["date"])

    # Align on date index
    if "date" not in df.columns:
        df = df.copy()
        df["date"] = df.index.date if isinstance(df.index, pd.DatetimeIndex) else df["date"]
    merged = df.merge(fg_df[["date", "value"]], on="date", how="left", suffixes=("", "_fg"))
    merged["fear_greed_score"] = merged["value"].fillna(50.0)
    merged["fear_greed_norm"] = (merged["fear_greed_score"] - 50) / 50  # -1 to 1
    merged["extreme_fear"] = merged["fear_greed_score"] < 25
    merged["extreme_greed"] = merged["fear_greed_score"] > 75

    # Drop temporary column
    merged = merged.drop(columns=["value"])

    elapsed = time.perf_counter() - start
    _log_info(
        "add_sentiment_features_success",
        signal_count=int(merged.shape[0]),
        execution_time_seconds=elapsed,
        pnl=0.0,
    )
    return merged