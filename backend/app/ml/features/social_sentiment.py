"""
Social sentiment features for crypto ML models.
Free sources only — no API keys required for basic endpoints.

Sources:
1. Reddit (pushshift-style via reddit JSON API — no auth needed)
   GET https://www.reddit.com/r/CryptoCurrency/search.json?q={symbol}&sort=new&limit=25&t=day
2. Fear & Greed Index (free, no auth)
   GET https://api.alternative.me/fng/?limit=7
3. CoinGecko public API (free tier, no auth needed for basic endpoints)
   GET https://api.coingecko.com/api/v3/search/trending
   GET https://api.coingecko.com/api/v3/coins/{id}/market_chart?vs_currency=usd&days=7
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


class SocialSentimentFeatures:
    FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=7"
    REDDIT_BASE = "https://www.reddit.com"
    COINGECKO_BASE = "https://api.coingecko.com/api/v3"

    # Map common crypto symbols to CoinGecko IDs
    SYMBOL_TO_CG: Dict[str, str] = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "BNB": "binancecoin",
        "XRP": "ripple",
        "ADA": "cardano",
    }

    _POSITIVE_WORDS = frozenset(["moon", "bullish", "buy", "pump", "breakout"])
    _NEGATIVE_WORDS = frozenset(["crash", "dump", "bear", "sell", "fud", "scam"])

    async def _fetch_json(self, url: str, *, params: Dict[str, Any] | None = None, timeout: int = 10) -> Any:
        """Utility wrapper for GET requests returning JSON. Logs and swallows errors."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, params=params, headers={"User-Agent": "QuantEdge/1.0"})
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:  # pragma: no cover
            logger.debug("Failed to fetch %s: %s", url, exc)
            return None

    async def get_fear_greed(self) -> Dict[str, Any]:
        """Returns a dict with recent Fear & Greed readings and the current value."""
        data = await self._fetch_json(self.FEAR_GREED_URL)
        if not data:
            return {"readings": [], "current": None}

        readings = [
            {
                "value": int(r.get("value", 0)),
                "classification": r.get("value_classification", ""),
                "timestamp": r.get("timestamp", ""),
            }
            for r in data.get("data", [])
        ]

        return {"readings": readings, "current": readings[0] if readings else None}

    async def get_reddit_sentiment(self, symbol: str, limit: int = 25) -> Dict[str, Any]:
        """
        Fetch recent Reddit posts mentioning *symbol* from r/CryptoCurrency.
        Returns a summary dict with mention count, average score, positive ratio and top title.
        """
        url = f"{self.REDDIT_BASE}/r/CryptoCurrency/search.json"
        params = {"q": symbol, "sort": "new", "limit": limit, "t": "day"}

        data = await self._fetch_json(url, params=params, timeout=15)
        if not data:
            return {
                "mention_count": 0,
                "avg_score": 0.0,
                "positive_ratio": 0.5,
                "top_title": "",
            }

        posts = data.get("data", {}).get("children", [])
        if not posts:
            return {
                "mention_count": 0,
                "avg_score": 0.0,
                "positive_ratio": 0.5,
                "top_title": "",
            }

        scores: List[int] = []
        positive_count = 0
        top_title = ""
        top_score = -1

        for post in posts:
            pd_data = post.get("data", {})
            title = pd_data.get("title", "") or ""
            score = int(pd_data.get("score", 0) or 0)
            scores.append(score)

            title_lower = title.lower()
            pos_hits = sum(1 for w in self._POSITIVE_WORDS if w in title_lower)
            neg_hits = sum(1 for w in self._NEGATIVE_WORDS if w in title_lower)
            if pos_hits > neg_hits:
                positive_count += 1

            if score > top_score:
                top_score = score
                top_title = title

        mention_count = len(posts)
        avg_score = sum(scores) / mention_count if mention_count else 0.0
        positive_ratio = positive_count / mention_count if mention_count else 0.5

        return {
            "mention_count": mention_count,
            "avg_score": round(avg_score, 2),
            "positive_ratio": round(positive_ratio, 4),
            "top_title": top_title[:200],
        }

    async def get_trending_coins(self) -> List[str]:
        """Return a list of trending coin symbols from CoinGecko /search/trending."""
        url = f"{self.COINGECKO_BASE}/search/trending"
        data = await self._fetch_json(url)
        if not data:
            return []

        symbols = [
            coin.get("item", {}).get("symbol", "").upper()
            for coin in data.get("coins", [])
            if coin.get("item", {}).get("symbol")
        ]
        return symbols

    async def compute_features(self, symbol: str) -> Dict[str, Any]:
        """
        Compute all sentiment features for *symbol*.
        Returns a dict with:
            - fear_greed_value: int (0‑100, current)
            - fear_greed_7d_avg: float
            - fear_greed_change: float (today - 7‑day avg)
            - reddit_mentions: int (last 24 h)
            - reddit_positive_ratio: float (0‑1)
            - reddit_avg_score: float
            - is_trending: bool (CoinGecko top‑7)
            - sentiment_composite: float (0‑1 weighted score)
        An empty dict is returned on failure.
        """
        try:
            fg_task = asyncio.create_task(self.get_fear_greed())
            reddit_task = asyncio.create_task(self.get_reddit_sentiment(symbol))
            trending_task = asyncio.create_task(self.get_trending_coins())

            fg_data, reddit_data, trending_coins = await asyncio.gather(
                fg_task, reddit_task, trending_task
            )

            # ---- Fear & Greed ----
            readings = fg_data.get("readings", [])
            if readings:
                current_fg = readings[0]["value"]
                fg_7d_avg = sum(r["value"] for r in readings) / len(readings)
            else:
                current_fg = 50
                fg_7d_avg = 50.0
            fg_change = float(current_fg) - fg_7d_avg

            # ---- Reddit ----
            reddit_mentions = int(reddit_data.get("mention_count", 0))
            reddit_positive_ratio = float(reddit_data.get("positive_ratio", 0.5))
            reddit_avg_score = float(reddit_data.get("avg_score", 0.0))

            # ---- Trending ----
            symbol_up = symbol.upper()
            is_trending = symbol_up in trending_coins

            # ---- Composite sentiment ----
            # We tighten the entry logic by raising the weight of confidence signals.
            fg_norm = float(current_fg) / 100.0
            trending_bonus = 1.0 if is_trending else 0.5
            sentiment_composite = round(
                0.40 * fg_norm
                + 0.40 * reddit_positive_ratio
                + 0.20 * trending_bonus,
                4,
            )

            return {
                "fear_greed_value": int(current_fg),
                "fear_greed_7d_avg": round(fg_7d_avg, 2),
                "fear_greed_change": round(fg_change, 2),
                "reddit_mentions": reddit_mentions,
                "reddit_positive_ratio": reddit_positive_ratio,
                "reddit_avg_score": round(reddit_avg_score, 2),
                "is_trending": is_trending,
                "sentiment_composite": sentiment_composite,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:  # pragma: no cover
            logger.debug("Error computing sentiment features for %s: %s", symbol, exc)
            return {}

    async def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Derive a simple trading signal from the sentiment features.
        Entry criteria (tightened):
            * sentiment_composite >= 0.70
            * fear_greed_change > 0  (up‑trend in market sentiment)
            * reddit_positive_ratio >= 0.60
            * is_trending == True
        Exit criteria:
            * sentiment_composite drops below 0.50 **or**
            * fear_greed_change becomes negative.
        Returns a dict:
            {
                "signal": "long" | "flat",
                "reason": str,
                "features": <dict from compute_features>
            }
        """
        features = await self.compute_features(symbol)
        if not features:
            return {"signal": "flat", "reason": "feature_error", "features": {}}

        composite = features.get("sentiment_composite", 0.0)
        fg_change = features.get("fear_greed_change", 0.0)
        reddit_ratio = features.get("reddit_positive_ratio", 0.0)
        trending = features.get("is_trending", False)

        # Entry check – all conditions must be satisfied
        if (
            composite >= 0.70
            and fg_change > 0
            and reddit_ratio >= 0.60
            and trending
        ):
            reason = "entry_conditions_met"
            signal = "long"
        # Exit check – any condition that weakens sentiment forces flat position
        elif composite < 0.50 or fg_change < 0:
            reason = "exit_conditions_met"
            signal = "flat"
        else:
            reason = "hold"
            signal = "flat"

        return {"signal": signal, "reason": reason, "features": features}