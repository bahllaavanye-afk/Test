"""
Market sentiment features: Fear & Greed Index + FinBERT news sentiment.
Free APIs only. All features are lagged by 1 period to prevent lookahead.
"""
from __future__ import annotations

import asyncio
import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timezone
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
    score = sum(1 for w in bullish if w in text_lower) - sum(
        1 for w in bearish if w in text_lower
    )
    # Normalise by the larger of the two vocab sizes to keep the output in [-1, 1]
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


def _normalize_fear_greed(value: int) -> float:
    """Rescale 0‑100 Fear & Greed score to a -1 to +1 range."""
    return (value - 50) / 50.0


def add_sentiment_features(df: pd.DataFrame, fear_greed_history: list[dict]) -> pd.DataFrame:
    """
    Merge Fear & Greed Index into OHLCV dataframe.

    Features added (all lagged 1 bar to prevent lookahead):
      - fear_greed_score: raw 0‑100 value
      - fear_greed_norm: -1 to +1 rescaled version
      - extreme_fear: bool (score < 25)
      - extreme_greed: bool (score > 75)
      - fear_greed_norm_ma3: 3‑period rolling mean (lagged)
    """
    if not fear_greed_history:
        df["fear_greed_score"] = 50.0
        df["fear_greed_norm"] = 0.0
        df["extreme_fear"] = False
        df["extreme_greed"] = False
        df["fear_greed_norm_ma3"] = 0.0
        return df

    fg_df = pd.DataFrame(fear_greed_history)
    fg_df["date"] = pd.to_datetime(fg_df["date"])
    fg_df = fg_df.set_index("date").sort_index()

    # Ensure the main dataframe has a datetime index named 'date' for alignment
    if "date" in df.columns:
        df = df.set_index(pd.to_datetime(df["date"])).sort_index()
        df = df.drop(columns=["date"])

    # Align on date, forward‑fill missing days (e.g., weekends) then shift for lag
    merged = df.join(fg_df[["value", "classification"]], how="left")
    merged["value"] = merged["value"].ffill()
    merged["classification"] = merged["classification"].ffill()

    # Raw score
    merged["fear_greed_score"] = merged["value"]

    # Normalised score
    merged["fear_greed_norm"] = merged["fear_greed_score"].apply(_normalize_fear_greed)

    # Extreme flags
    merged["extreme_fear"] = merged["fear_greed_score"] < 25
    merged["extreme_greed"] = merged["fear_greed_score"] > 75

    # Rolling mean (3‑period) – lagged to avoid look‑ahead
    merged["fear_greed_norm_ma3"] = (
        merged["fear_greed_norm"].rolling(window=3, min_periods=1).mean().shift(1)
    )

    # Apply a universal 1‑bar lag to all new features
    for col in ["fear_greed_score", "fear_greed_norm", "extreme_fear", "extreme_greed", "fear_greed_norm_ma3"]:
        merged[col] = merged[col].shift(1)

    # Restore original column order (excluding the temporary 'value' and 'classification')
    merged = merged.drop(columns=["value", "classification"])
    merged = merged.sort_index()
    return merged


def generate_sentiment_signal(df: pd.DataFrame) -> pd.Series:
    """
    Produce a binary trading signal based on enriched sentiment features.

    Entry (long) conditions (tightened):
      • extreme_greed is True
      • fear_greed_norm_ma3 is positive (up‑trend in sentiment)
      • optional confirmation: price close above its 10‑period SMA

    Exit conditions:
      • extreme_fear becomes True
      • or fear_greed_norm drops below zero (sentiment turning bearish)

    The function returns a Series aligned with ``df`` where:
        1  → long position
        0  → flat / exit
    """
    # Ensure required columns exist
    required = {"extreme_greed", "extreme_fear", "fear_greed_norm_ma3", "close"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("Sentiment signal generation missing columns", missing=list(missing))
        return pd.Series(0, index=df.index)

    # 10‑period simple moving average of price for confirmation
    price_sma10 = df["close"].rolling(window=10, min_periods=1).mean()

    # Entry signal
    entry = (
        df["extreme_greed"]
        & (df["fear_greed_norm_ma3"] > 0)
        & (df["close"] > price_sma10)
    )

    # Exit signal
    exit_signal = df["extreme_fear"] | (df["fear_greed_norm_ma3"] < 0)

    # Build cumulative position: start flat, go long on entry, flat on exit
    signal = pd.Series(0, index=df.index, dtype=int)
    in_position = False
    for idx in df.index:
        if not in_position and entry.loc[idx]:
            signal.loc[idx] = 1
            in_position = True
        elif in_position and exit_signal.loc[idx]:
            signal.loc[idx] = 0
            in_position = False
        elif in_position:
            signal.loc[idx] = 1
    return signal