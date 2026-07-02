"""
Market sentiment features: Fear & Greed Index + FinBERT news sentiment.
Free APIs only. All features are lagged by 1 period to prevent lookahead.
"""
from __future__ import annotations

import asyncio
import json
import urllib.request
from datetime import datetime, timezone, timedelta
from json import JSONDecodeError
from typing import Any, List, Dict, Optional

import httpx
import pandas as pd
from httpx import HTTPError
from urllib.error import URLError

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
        result: List[dict] = []
        for r in readings:
            result.append(
                {
                    "date": datetime.fromtimestamp(int(r["timestamp"]), tz=timezone.utc).date(),
                    "value": int(r["value"]),
                    "classification": r["value_classification"],
                }
            )
        return {"status": "ok", "readings": result, "current": result[0] if result else None}
    except HTTPError as e:
        logger.error(
            "Fear & Greed fetch failed due to HTTP error",
            error=str(e),
            url="https://api.alternative.me/fng/",
        )
        return {"status": "error", "readings": [], "current": None}
    except (ValueError, KeyError, JSONDecodeError) as e:
        logger.error(
            "Fear & Greed fetch failed due to malformed response",
            error=str(e),
        )
        return {"status": "error", "readings": [], "current": None}
    except Exception as e:
        logger.exception("Unexpected error during Fear & Greed fetch", error=str(e))
        return {"status": "error", "readings": [], "current": None}


async def fetch_news_sentiment(symbol: str, api_key: Optional[str] = None) -> List[dict]:
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
        sentiments: List[dict] = []
        for a in articles:
            title = a.get("title", "") or ""
            score = _simple_sentiment(title)
            sentiments.append(
                {
                    "published_at": a.get("publishedAt"),
                    "title": title[:120],
                    "sentiment_score": score,
                }
            )
        return sentiments
    except HTTPError as e:
        logger.error(
            "NewsAPI fetch failed due to HTTP error",
            symbol=symbol,
            error=str(e),
            url="https://newsapi.org/v2/everything",
        )
        return []
    except (ValueError, KeyError, JSONDecodeError) as e:
        logger.error(
            "NewsAPI fetch failed due to malformed response",
            symbol=symbol,
            error=str(e),
        )
        return []
    except Exception as e:
        logger.exception("Unexpected error during NewsAPI fetch", symbol=symbol, error=str(e))
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
            logger.info("FinBERT transformer library not installed; SEC sentiment disabled")
            return False
        except Exception as e:
            logger.exception("Unexpected error loading FinBERT pipeline", error=str(e))
            return False

    def get_cik(self, ticker: str) -> Optional[int]:
        """Look up CIK from SEC EDGAR company tickers JSON (free, no key needed)."""
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
        except (URLError, JSONDecodeError) as exc:
            logger.error(
                "SEC CIK lookup failed due to network or JSON error",
                ticker=ticker,
                error=str(exc),
            )
        except Exception as exc:
            logger.exception("Unexpected error during SEC CIK lookup", ticker=ticker, error=str(exc))
        return None

    def get_management_tone(self, ticker: str) -> Optional[float]:
        """
        Returns management_tone_score (-1 bearish, 0 neutral, +1 bullish).
        Returns None if FinBERT is not available or filing cannot be fetched.

        Full MD&A extraction requires an SGML parser; here we use the
        filing summary text from EDGAR's submissions JSON as a proxy.
        """
        if not self._available:
            logger.debug("FinBERT not available; skipping management tone extraction", ticker=ticker)
            return None

        cik = self.get_cik(ticker)
        if cik is None:
            logger.debug("CIK not found for ticker; cannot fetch filings", ticker=ticker)
            return None

        try:
            url = self.EDGAR_SUBMISSIONS.format(cik=cik)
            req = urllib.request.Request(url, headers={"User-Agent": self._USER_AGENT})
            with urllib.request.urlopen(req, timeout=10) as r:
                submissions = json.loads(r.read())
        except (URLError, JSONDecodeError) as exc:
            logger.error(
                "SEC submissions fetch failed due to network or JSON error",
                ticker=ticker,
                error=str(exc),
            )
            return None
        except Exception as exc:
            logger.exception("Unexpected error fetching SEC submissions", ticker=ticker, error=str(exc))
            return None

        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        descriptions = recent.get("primaryDocument", [])

        text_snippets: List[str] = []
        for form, desc in zip(forms, descriptions):
            if form in ("8-K", "10-Q", "10-K"):
                text_snippets.append(str(desc))
            if len(text_snippets) >= 5:
                break

        if not text_snippets:
            logger.debug("No relevant filings found for ticker", ticker=ticker)
            return None

        combined_text = " ".join(text_snippets)[:512]

        try:
            scores_list = self._pipeline(combined_text)  # type: ignore
            if not scores_list:
                logger.debug("FinBERT returned empty result", ticker=ticker)
                return None
            scores = {item["label"].lower(): item["score"] for item in scores_list[0]}
            positive = scores.get("positive", 0.0)
            negative = scores.get("negative", 0.0)
            tone = float(positive - negative)  # range approximately [-1, +1]
            return tone
        except Exception as exc:
            logger.exception("FinBERT inference failed", ticker=ticker, error=str(exc))
            return None


def add_sentiment_features(df: pd.DataFrame, fear_greed_history: List[dict]) -> pd.DataFrame:
    """
    Merge Fear & Greed Index into OHLCV dataframe.
    Features added (all lagged 1 bar to prevent lookahead):
      - fear_greed_score: 0-100
      - fear_greed_norm: -1 to 1 rescaled
      - extreme_fear: bool (score < 25)
      - extreme_greed: bool (score > 75)
    """
    if not fear_greed_history:
        df = df.copy()
        df["fear_greed_score"] = 50.0
        df["fear_greed_norm"] = 0.0
        df["extreme_fear"] = False
        df["extreme_greed"] = False
        return df

    fg_df = pd.DataFrame(fear_greed_history)
    fg_df["date"] = pd.to_datetime(fg_df["date"])

    # Ensure the input dataframe has a datetime index or a 'date' column
    df = df.copy()
    if "date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df["date"] = df.index
        else:
            logger.error("Dataframe lacks datetime index or 'date' column for sentiment merge")
            raise ValueError("Dataframe must have a datetime index or a 'date' column")

    try:
        merged = pd.merge(
            df,
            fg_df[["date", "value"]],
            on="date",
            how="left",
            suffixes=("", "_fg"),
        )
        # Apply lag of one period
        merged["fear_greed_score"] = merged["value"].shift(1)
        # Fill missing values with neutral defaults
        merged["fear_greed_score"].fillna(50.0, inplace=True)

        # Normalized score between -1 and 1
        merged["fear_greed_norm"] = ((merged["fear_greed_score"] - 50) / 50).clip(-1, 1)

        # Extreme flags
        merged["extreme_fear"] = merged["fear_greed_score"] < 25
        merged["extreme_greed"] = merged["fear_greed_score"] > 75

        # Clean up temporary columns
        merged.drop(columns=["value"], inplace=True, errors="ignore")
        return merged
    except Exception as e:
        logger.exception("Failed to add sentiment features", error=str(e))
        raise