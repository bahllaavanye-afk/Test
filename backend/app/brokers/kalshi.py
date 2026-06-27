"""Kalshi read-only public client (no API key for market data)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from app.utils.logging import logger

BASE = "https://demo-api.kalshi.co/trade-api/v2"


class KalshiPublicClient:
    """Read-only Kalshi client — uses demo API, no auth needed for reads."""

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        url = f"{BASE}{path}"
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{qs}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as http_err:
            logger.error(
                "KalshiPublicClient HTTP error",
                url=url,
                status=http_err.code,
                reason=str(http_err.reason),
            )
            return {}
        except urllib.error.URLError as url_err:
            logger.error(
                "KalshiPublicClient URL error",
                url=url,
                error=str(url_err.reason),
            )
            return {}
        except json.JSONDecodeError as json_err:
            logger.error(
                "KalshiPublicClient JSON decode error",
                url=url,
                error=str(json_err),
            )
            return {}
        except Exception as exc:
            logger.error(
                "KalshiPublicClient unexpected error",
                url=url,
                error=str(exc),
            )
            return {}

    def get_events(self, status: str = "open", limit: int = 25) -> list[dict]:
        """Return open prediction events."""
        if not isinstance(status, str):
            raise ValueError("Status must be a string")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("Limit must be a positive integer")
        data = self._get("/events", {"status": status, "limit": str(limit)})
        if isinstance(data, dict):
            return data.get("events", []) or []
        return []

    def get_markets(self, event_ticker: str) -> list[dict]:
        """Return markets for a specific event."""
        if not isinstance(event_ticker, str):
            raise ValueError("Event ticker must be a string")
        data = self._get("/markets", {"event_ticker": event_ticker})
        if isinstance(data, dict):
            return data.get("markets", []) or []
        return []

    def get_market(self, ticker: str) -> dict:
        """Return a single market's detail."""
        if not isinstance(ticker, str):
            raise ValueError("Ticker must be a string")
        data = self._get(f"/markets/{ticker}")
        if isinstance(data, dict):
            return data.get("market", data)
        return {}