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

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

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

    # --------------------------------------------------------------------- #
    # Helper / API calls
    # --------------------------------------------------------------------- #

    async def _fetch_json(self, url: str, *, params: Optional[Dict[str, Any]] = None,
                          headers: Optional[Dict[str, str]] = None,
                          timeout: int = 10) -> Any:
        """Utility wrapper for async GET requests with basic error handling."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.debug("Failed to fetch %s: %s", url, exc)
            return None

    # --------------------------------------------------------------------- #
    # Data sources
    # --------------------------------------------------------------------- #

    async def get_fear_greed(self) -> Dict[str, Any]:
        """Returns a dict with today's and the past 6 days Fear & Greed readings."""
        data = await self._fetch_json(self.FEAR_GREED_URL)
        if not data:
            return {"readings": [], "current": None}

        readings = data.get("data", [])
        processed = [
            {
                "value": int(r["value"]),
                "classification": r["value_classification"],
                "timestamp": r.get("timestamp", ""),
            }
            for r in readings
        ]
        return {"readings": processed, "current": processed[0] if processed else None}

    async def get_reddit_sentiment(self, symbol: str, limit: int = 25) -> Dict[str, Any]:
        """
        Fetch recent Reddit posts mentioning *symbol* from r/CryptoCurrency.
        Returns a dict with mention count, average score, positive ratio and the most
        up‑voted title.
        """
        url = f"{self.REDDIT_BASE}/r/CryptoCurrency/search.json"
        params = {"q": symbol, "sort": "new", "limit": limit, "t": "day"}
        headers = {"User-Agent": "QuantEdge/1.0 (crypto sentiment feature collector)"}

        data = await self._fetch_json(url, params=params, headers=headers, timeout=15)
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
        avg_score = sum(scores) / mention_count if mention_count else 0.0
        positive_ratio = positive_count / mention_count if mention_count else 0.5

        return {
            "mention_count": mention_count,
            "avg_score": avg_score,
            "positive_ratio": positive_ratio,
            "top_title": top_title[:200],
        }

    async def get_trending_coins(self) -> List[str]:
        """Return a list of trending coin symbols from CoinGecko /search/trending."""
        url = f"{self.COINGECKO_BASE}/search/trending"
        data = await self._fetch_json(url)
        if not data:
            return []

        coins = data.get("coins", [])
        symbols = [
            item.get("symbol", "").upper()
            for coin in coins
            if (item := coin.get("item", {})) and item.get("symbol")
        ]
        return symbols

    async def get_price_momentum(self, symbol: str) -> Optional[float]:
        """
        Fetch 7‑day price chart for *symbol* (via CoinGecko) and return the
        percentage change from the oldest to the newest price point.
        Returns None on failure.
        """
        cg_id = self.SYMBOL_TO_CG.get(symbol.upper())
        if not cg_id:
            logger.debug("No CoinGecko ID mapping for symbol %s", symbol)
            return None

        url = f"{self.COINGECKO_BASE}/coins/{cg_id}/market_chart"
        params = {"vs_currency": "usd", "days": "7"}
        data = await _fetch_json_static(url, params=params)  # static helper to avoid self recursion
        if not data:
            return None

        prices = data.get("prices", [])
        if len(prices) < 2:
            return None

        # CoinGecko returns [timestamp, price]; take first and last price.
        oldest_price = prices[0][1]
        newest_price = prices[-1][1]
        if oldest_price == 0:
            return None

        change_pct = ((newest_price - oldest_price) / oldest_price) * 100
        return round(change_pct, 2)

    # --------------------------------------------------------------------- #
    # Feature aggregation
    # --------------------------------------------------------------------- #

    async def compute_features(self, symbol: str) -> Dict[str, Any]:
        """
        Compute sentiment‑related features for *symbol*.

        Returned dictionary includes:
            - fear_greed_value: int (0‑100, today)
            - fear_greed_7d_avg: float
            - fear_greed_change: float (today – 7d avg)
            - reddit_mentions: int (last 24 h)
            - reddit_positive_ratio: float (0‑1)
            - reddit_avg_score: float
            - is_trending: bool (CoinGecko top‑7)
            - price_7d_change_pct: float | None
            - sentiment_composite: float (0‑1)
            - bullish_signal: bool
            - exit_signal: bool
        """
        import asyncio

        # Parallelise I/O bound calls
        fg_task = asyncio.create_task(self.get_fear_greed())
        reddit_task = asyncio.create_task(self.get_reddit_sentiment(symbol))
        trending_task = asyncio.create_task(self.get_trending_coins())
        price_task = asyncio.create_task(self.get_price_momentum(symbol))

        fg_data, reddit_data, trending_coins, price_change = await asyncio.gather(
            fg_task, reddit_task, trending_task, price_task
        )

        # -----------------------------------------------------------------
        # Fear & Greed calculations
        # -----------------------------------------------------------------
        readings = fg_data.get("readings", [])
        if readings:
            current_fg = readings[0]["value"]
            fg_7d_avg = sum(r["value"] for r in readings) / len(readings)
            fg_change = current_fg - fg_7d_avg
        else:
            current_fg = 50
            fg_7d_avg = 50.0
            fg_change = 0.0

        fg_norm = current_fg / 100.0  # normalised 0‑1

        # -----------------------------------------------------------------
        # Reddit calculations
        # -----------------------------------------------------------------
        reddit_mentions = reddit_data.get("mention_count", 0)
        reddit_positive_ratio = reddit_data.get("positive_ratio", 0.5)
        reddit_avg_score = float(reddit_data.get("avg_score", 0.0))

        # -----------------------------------------------------------------
        # Trending / momentum
        # -----------------------------------------------------------------
        symbol_upper = symbol.upper()
        is_trending = symbol_upper in trending_coins
        price_change_pct = price_change  # may be None

        # -----------------------------------------------------------------
        # Composite signal
        # -----------------------------------------------------------------
        # We apply tighter entry conditions:
        #   • Base composite from Fear & Greed (40%) and Reddit positive ratio (40%)
        #   • Add a momentum bonus (20%) only if both trending and price is positive.
        trending_bonus = 0.2 if is_trending and (price_change_pct or 0) > 0 else 0.0

        sentiment_composite = round(
            0.40 * fg_norm + 0.40 * reddit_positive_ratio + trending_bonus,
            4,
        )

        # -----------------------------------------------------------------
        # Signal flags
        # -----------------------------------------------------------------
        # Entry (bullish) signal requires:
        #   – Composite >= 0.6
        #   – Fear & Greed change positive (bullish momentum)
        #   – Reddit positive ratio > 0.6
        #   – If trending, price momentum must be > 0%
        bullish_signal = (
            sentiment_composite >= 0.6
            and fg_change > 0
            and reddit_positive_ratio > 0.6
            and (not is_trending or (price_change_pct or 0) > 0)
        )

        # Exit signal when sentiment deteriorates:
        #   – Composite falls below 0.4 OR
        #   – Fear & Greed change negative OR
        #   – Reddit positive ratio drops below 0.4
        exit_signal = (
            sentiment_composite < 0.4
            or fg_change < 0
            or reddit_positive_ratio < 0.4
        )

        return {
            "fear_greed_value": int(current_fg),
            "fear_greed_7d_avg": round(fg_7d_avg, 2),
            "fear_greed_change": round(fg_change, 2),
            "reddit_mentions": reddit_mentions,
            "reddit_positive_ratio": round(reddit_positive_ratio, 4),
            "reddit_avg_score": round(reddit_avg_score, 2),
            "is_trending": is_trending,
            "price_7d_change_pct": price_change_pct,
            "sentiment_composite": sentiment_composite,
            "bullish_signal": bullish_signal,
            "exit_signal": exit_signal,
        }


# --------------------------------------------------------------------- #
# Static helper to avoid recursion inside the class when fetching price.
# --------------------------------------------------------------------- #

async def _fetch_json_static(url: str, *, params: Optional[Dict[str, Any]] = None,
                            headers: Optional[Dict[str, str]] = None,
                            timeout: int = 10) -> Any:
    """Standalone async GET wrapper used by get_price_momentum."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.debug("Static fetch failed for %s: %s", url, exc)
        return None