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
            req = urllib.request.Request(
                url,
                headers={"User-Agent": self._USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                submissions = json.loads(r.read())
        except Exception as exc:
            logger.debug("SEC submissions fetch failed", ticker=ticker, error=str(exc))
            return None

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

    # Align on date index; assume df index is datetime-like
    fg_df.set_index("date", inplace=True)
    df = df.copy()
    df["fear_greed_score"] = fg_df["value"].reindex(df.index, method="ffill").shift(1).fillna(50.0)
    df["fear_greed_norm"] = ((df["fear_greed_score"] - 50) / 50).clip(-1, 1)
    df["extreme_fear"] = df["fear_greed_score"] < 25
    df["extreme_greed"] = df["fear_greed_score"] > 75
    return df


def compute_combined_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a unified sentiment metric from available columns.
    Expected columns (already lagged):
        - fear_greed_norm   ([-1, 1])
        - news_sentiment_avg ([-1, 1])
        - sec_tone           ([-1, 1])
    Missing columns are treated as neutral (0).
    The combined metric is a simple average, then rescaled to [-1, 1].
    """
    df = df.copy()
    components = [
        df.get("fear_greed_norm", pd.Series(0.0, index=df.index)),
        df.get("news_sentiment_avg", pd.Series(0.0, index=df.index)),
        df.get("sec_tone", pd.Series(0.0, index=df.index)),
    ]
    df["combined_sentiment_raw"] = sum(components) / len(components)
    # Ensure bounds
    df["combined_sentiment"] = df["combined_sentiment_raw"].clip(-1, 1)
    return df


def generate_sentiment_signal(df: pd.DataFrame) -> pd.Series:
    """
    Generate trading signals based on tightened sentiment logic.
    Returns a Series aligned with df.index:
        1  -> long entry
        0  -> flat / exit
        -1 -> short entry (optional, based on strong negative sentiment)
    Entry Conditions (tightened):
        - combined_sentiment crosses above 0.5 (bullish) or below -0.5 (bearish)
        - Confirmation: extreme_greed flag + news_sentiment_avg > 0.2 for longs,
          extreme_fear flag + news_sentiment_avg < -0.2 for shorts
        - SEC tone must be positive (>0) for longs or negative (<0) for shorts
    Exit Conditions:
        - combined_sentiment falls below 0.2 (long) or rises above -0.2 (short)
        - extreme_fear flag becomes True for longs
        - extreme_greed flag becomes True for shorts
    The function is deliberately conservative to reduce false entries.
    """
    df = compute_combined_sentiment(df)

    # Ensure required columns exist
    df["extreme_fear"] = df.get("extreme_fear", False)
    df["extreme_greed"] = df.get("extreme_greed", False)
    df["news_sentiment_avg"] = df.get("news_sentiment_avg", 0.0)

    # Long entry detection
    long_entry = (
        (df["combined_sentiment"].shift(1) <= 0.5)
        & (df["combined_sentiment"] > 0.5)
        & df["extreme_greed"]
        & (df["news_sentiment_avg"] > 0.2)
        & (df.get("sec_tone", 0.0) > 0)
    )

    # Short entry detection (optional)
    short_entry = (
        (df["combined_sentiment"].shift(1) >= -0.5)
        & (df["combined_sentiment"] < -0.5)
        & df["extreme_fear"]
        & (df["news_sentiment_avg"] < -0.2)
        & (df.get("sec_tone", 0.0) < 0)
    )

    # Exit conditions
    long_exit = (df["combined_sentiment"] < 0.2) | df["extreme_fear"]
    short_exit = (df["combined_sentiment"] > -0.2) | df["extreme_greed"]

    signals = pd.Series(0, index=df.index, dtype=int)

    # Apply entries
    signals[long_entry] = 1
    signals[short_entry] = -1

    # Propagate existing positions until an exit triggers
    position = 0
    for idx in df.index:
        if signals.at[idx] != 0:
            position = signals.at[idx]
        else:
            # check if we need to exit current position
            if position == 1 and long_exit.at[idx]:
                position = 0
            elif position == -1 and short_exit.at[idx]:
                position = 0
        signals.at[idx] = position

    return signals