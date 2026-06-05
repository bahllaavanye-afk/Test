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

import httpx
import pandas as pd
from datetime import datetime, timezone


class SocialSentimentFeatures:
    FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=7"
    REDDIT_BASE = "https://www.reddit.com"
    COINGECKO_BASE = "https://api.coingecko.com/api/v3"

    # Map common crypto symbols to CoinGecko IDs
    SYMBOL_TO_CG = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "BNB": "binancecoin",
        "XRP": "ripple",
        "ADA": "cardano",
    }

    _POSITIVE_WORDS = frozenset(["moon", "bullish", "buy", "pump", "breakout"])
    _NEGATIVE_WORDS = frozenset(["crash", "dump", "bear", "sell", "fud", "scam"])

    async def get_fear_greed(self) -> dict:
        """Returns {value: int, classification: str, timestamp: str} for today and 6 prior days."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(self.FEAR_GREED_URL)
                resp.raise_for_status()
                data = resp.json()
            readings = data.get("data", [])
            result = []
            for r in readings:
                result.append({
                    "value": int(r["value"]),
                    "classification": r["value_classification"],
                    "timestamp": r.get("timestamp", ""),
                })
            return {"readings": result, "current": result[0] if result else None}
        except Exception:
            return {"readings": [], "current": None}

    async def get_reddit_sentiment(self, symbol: str, limit: int = 25) -> dict:
        """
        Fetch recent Reddit posts mentioning symbol from r/CryptoCurrency.
        Returns {mention_count, avg_score, positive_ratio, top_title}.
        Uses simple keyword-based sentiment (positive words: moon, bullish, buy, pump, breakout;
        negative: crash, dump, bear, sell, FUD, scam).
        No auth needed — uses reddit public JSON API with User-Agent header.
        """
        url = f"{self.REDDIT_BASE}/r/CryptoCurrency/search.json"
        params = {
            "q": symbol,
            "sort": "new",
            "limit": limit,
            "t": "day",
        }
        headers = {
            "User-Agent": "QuantEdge/1.0 (crypto sentiment feature collector)",
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            posts = data.get("data", {}).get("children", [])
            if not posts:
                return {
                    "mention_count": 0,
                    "avg_score": 0.0,
                    "positive_ratio": 0.5,
                    "top_title": "",
                }

            scores = []
            positive_count = 0
            top_title = ""
            top_score = -1

            for post in posts:
                pd_data = post.get("data", {})
                title = pd_data.get("title", "") or ""
                score = pd_data.get("score", 0) or 0
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
            avg_score = sum(scores) / mention_count if mention_count > 0 else 0.0
            positive_ratio = positive_count / mention_count if mention_count > 0 else 0.5

            return {
                "mention_count": mention_count,
                "avg_score": avg_score,
                "positive_ratio": positive_ratio,
                "top_title": top_title[:200],
            }
        except Exception:
            return {
                "mention_count": 0,
                "avg_score": 0.0,
                "positive_ratio": 0.5,
                "top_title": "",
            }

    async def get_trending_coins(self) -> list[str]:
        """Return list of trending coin symbols from CoinGecko /search/trending."""
        url = f"{self.COINGECKO_BASE}/search/trending"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()

            coins = data.get("coins", [])
            symbols = []
            for coin in coins:
                item = coin.get("item", {})
                symbol = item.get("symbol", "")
                if symbol:
                    symbols.append(symbol.upper())
            return symbols
        except Exception:
            return []

    async def compute_features(self, symbol: str) -> dict:
        """
        Compute all sentiment features for a symbol. Returns dict with:
        - fear_greed_value: int (0-100, current)
        - fear_greed_7d_avg: float
        - fear_greed_change: float (today - 7d_avg, momentum)
        - reddit_mentions: int (last 24h)
        - reddit_positive_ratio: float (0-1)
        - reddit_avg_score: float
        - is_trending: bool (in CoinGecko top 7)
        - sentiment_composite: float (weighted combination, 0-1)
        Returns {} on any error — callers must handle empty dict.
        """
        try:
            import asyncio

            fg_task = asyncio.create_task(self.get_fear_greed())
            reddit_task = asyncio.create_task(self.get_reddit_sentiment(symbol))
            trending_task = asyncio.create_task(self.get_trending_coins())

            fg_data, reddit_data, trending_coins = await asyncio.gather(
                fg_task, reddit_task, trending_task
            )

            # Fear & Greed features
            readings = fg_data.get("readings", [])
            current_fg = readings[0]["value"] if readings else 50
            fear_greed_7d_avg = (
                sum(r["value"] for r in readings) / len(readings)
                if readings else 50.0
            )
            fear_greed_change = float(current_fg) - fear_greed_7d_avg

            # Reddit features
            reddit_mentions: int = reddit_data.get("mention_count", 0)
            reddit_positive_ratio: float = reddit_data.get("positive_ratio", 0.5)
            reddit_avg_score: float = float(reddit_data.get("avg_score", 0.0))

            # Trending check — symbol itself or its CoinGecko ID symbol
            symbol_upper = symbol.upper()
            is_trending = symbol_upper in trending_coins

            # Sentiment composite (weighted combination, 0-1):
            #   40% Fear & Greed (normalised)
            #   40% Reddit positive ratio
            #   20% trending bonus
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
                "fear_greed_7d_avg": round(fear_greed_7d_avg, 2),
                "fear_greed_change": round(fear_greed_change, 2),
                "reddit_mentions": reddit_mentions,
                "reddit_positive_ratio": round(reddit_positive_ratio, 4),
                "reddit_avg_score": round(reddit_avg_score, 2),
                "is_trending": is_trending,
                "sentiment_composite": sentiment_composite,
            }
        except Exception:
            return {}

    def to_dataframe_row(self, features: dict) -> pd.Series:
        """Convert features dict to a pandas Series for ML feature matrix."""
        defaults = {
            "fear_greed_value": 50,
            "fear_greed_7d_avg": 50.0,
            "fear_greed_change": 0.0,
            "reddit_mentions": 0,
            "reddit_positive_ratio": 0.5,
            "reddit_avg_score": 0.0,
            "is_trending": False,
            "sentiment_composite": 0.5,
        }
        merged = {**defaults, **features}
        return pd.Series(merged)
