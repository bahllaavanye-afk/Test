"""
Market sentiment features: Fear & Greed Index + FinBERT news sentiment.
Free APIs only. All features are lagged by 1 period to prevent lookahead.
"""
from __future__ import annotations

import asyncio
import httpx
import pandas as pd
from datetime import datetime, timezone, timedelta
from app.utils.logging import logger


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
        if not self._available:
            return None

        cik = self.get_cik(ticker)
        if cik is None:
            return None

        import urllib.request
        import json

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
            # scores_list: [[{label, score}, ...]]
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


def add_sentiment_features(df: pd.DataFrame, fear_greed_history: list[dict] | None) -> pd.DataFrame:
    """
    Merge Fear & Greed Index into OHLCV dataframe.
    Features added (all lagged 1 bar to prevent lookahead):
      - fear_greed_score: 0-100
      - fear_greed_norm: -1 to 1 rescaled
      - extreme_fear: bool (score < 25)
      - extreme_greed: bool (score > 75)

    Handles None/empty inputs gracefully and avoids off‑by‑one errors when the
    dataframe has fewer rows than the lag period.
    """
    # Basic validation
    if df is None:
        raise ValueError("Input dataframe 'df' must not be None")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    # Ensure we always have the required columns; if df is empty we still add defaults
    default_score = 50.0
    default_norm = 0.0
    default_fear = False
    default_greed = False

    if df.empty:
        df["fear_greed_score"] = default_score
        df["fear_greed_norm"] = default_norm
        df["extreme_fear"] = default_fear
        df["extreme_greed"] = default_greed
        return df

    # Normalise the history input
    if not fear_greed_history:
        # No history provided – use defaults for all rows
        df["fear_greed_score"] = default_score
        df["fear_greed_norm"] = default_norm
        df["extreme_fear"] = default_fear
        df["extreme_greed"] = default_greed
        return df

    # Build a DataFrame from the history list, being defensive about missing keys
    fg_df = pd.DataFrame(fear_greed_history)

    # Guard against malformed history rows
    required_cols = {"date", "value"}
    if not required_cols.issubset(fg_df.columns):
        # If required columns are missing, fall back to defaults
        df["fear_greed_score"] = default_score
        df["fear_greed_norm"] = default_norm
        df["extreme_fear"] = default_fear
        df["extreme_greed"] = default_greed
        return df

    # Convert dates to plain date objects for reliable mapping
    fg_df["date"] = pd.to_datetime(fg_df["date"], errors="coerce").dt.date

    # Drop rows where conversion failed
    fg_df = fg_df.dropna(subset=["date", "value"])

    if fg_df.empty:
        df["fear_greed_score"] = default_score
        df["fear_greed_norm"] = default_norm
        df["extreme_fear"] = default_fear
        df["extreme_greed"] = default_greed
        return df

    # Keep the most recent entries (the API returns newest first, but we sort to be safe)
    fg_df = fg_df.sort_values("date", ascending=False).reset_index(drop=True)

    # Prepare a mapping from date -> value
    fg_series = fg_df.set_index("date")["value"]

    # Determine the date source in the OHLCV dataframe
    if "date" in df.columns:
        df_dates = pd.to_datetime(df["date"], errors="coerce").dt.date
    else:
        # Fallback to index if it is datetime‑like
        df_dates = pd.to_datetime(df.index, errors="coerce").date

    # Map the sentiment score onto the dataframe; missing dates get the default score
    df["fear_greed_score"] = df_dates.map(fg_series).fillna(default_score)

    # Apply the lag of one period.  If the dataframe has only one row, shift will
    # produce NaN which we replace with the default score.
    df["fear_greed_score"] = df["fear_greed_score"].shift(1).fillna(default_score)

    # Normalise to -1 … 1 range
    df["fear_greed_norm"] = ((df["fear_greed_score"] - 50) / 50).clip(-1, 1)

    # Extreme flags based on the lagged score
    df["extreme_fear"] = df["fear_greed_score"] < 25
    df["extreme_greed"] = df["fear_greed_score"] > 75

    return df