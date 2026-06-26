"""TradeStation REST API broker with OAuth2 client credentials."""
from datetime import UTC, datetime, timedelta
from typing import List, Dict

import httpx
import functools

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.utils.logging import logger


def _log_metrics(func):
    """Async decorator to log execution time and key metrics for broker methods."""
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        start = datetime.now(UTC)
        result = await func(self, *args, **kwargs)
        end = datetime.now(UTC)
        duration = (end - start).total_seconds()
        metrics = {
            "method": func.__name__,
            "execution_time_s": duration,
        }

        # Add method‑specific metrics
        if func.__name__ == "place_order" and isinstance(result, OrderResult):
            metrics.update(
                {
                    "order_id": result.broker_order_id,
                    "status": result.status,
                    "filled_qty": result.filled_qty,
                }
            )
        elif func.__name__ == "cancel_order":
            metrics["canceled"] = result
        elif func.__name__ == "get_order":
            metrics.update(result or {})
        elif func.__name__ == "get_positions":
            metrics["position_count"] = len(result)
            total_pnl = sum(p.get("unrealized_pnl", 0) for p in result)
            metrics["total_unrealized_pnl"] = total_pnl
        elif func.__name__ == "get_account":
            metrics.update(result or {})
        elif func.__name__ == "get_quote":
            metrics.update(
                {
                    "symbol": result.symbol,
                    "bid": result.bid,
                    "ask": result.ask,
                    "last": result.last,
                    "volume": result.volume,
                }
            )
        elif func.__name__ == "get_historical":
            metrics["bars_returned"] = len(result)

        logger.info("TradeStation broker operation", **metrics)
        return result

    return wrapper


class TradeStationBroker(AbstractBroker):
    INTERVAL_MAP: Dict[str, str] = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "1h": "60",
        "4h": "240",
        "1d": "1440",
    }

    def __init__(self, client_id: str, client_secret: str, account_id: str, paper: bool = True):
        self.client_id = client_id
        self.client_secret = client_secret
        self.account_id = account_id
        self.paper = paper
        self.base_url = "https://sim.api.tradestation.com/v3" if paper else "https://api.tradestation.com/v3"
        self._access_token: str | None = None
        self._token_expires_at: datetime = datetime.min.replace(tzinfo=UTC)

    async def _get_token(self) -> str:
        if self._access_token and datetime.now(UTC) < self._token_expires_at:
            return self._access_token
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://signin.tradestation.com/oauth/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "audience": "https://api.tradestation.com",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            self._access_token = data["access_token"]
            self._token_expires_at = datetime.now(UTC) + timedelta(
                seconds=data.get("expires_in", 1200) - 60
            )
        return self._access_token

    async def _headers(self) -> dict:
        token = await self._get_token()
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    @_log_metrics
    async def place_order(self, request: OrderRequest) -> OrderResult:
        body = {
            "AccountID": self.account_id,
            "Symbol": request.symbol,
            "Quantity": str(int(request.quantity)),
            "OrderType": "Market" if request.order_type == "market" else "Limit",
            "TradeAction": "BUY" if request.side == "buy" else "SELL",
            "TimeInForce": {"Duration": "DAY"},
            "Route": "Intelligent",
        }
        if request.order_type == "limit" and request.limit_price:
            body["LimitPrice"] = str(request.limit_price)

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/orderexecution/orders",
                json=body,
                headers=await self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        order_id = data.get("OrderID", "unknown")
        status = data.get("Message", "queued").lower()
        filled_qty = float(data.get("FilledQuantity", 0))
        avg_fill = float(data.get("AveragePrice", 0)) or None

        logger.info("TradeStation order placed", order_id=order_id, status=status)
        return OrderResult(
            broker_order_id=order_id,
            status=status,
            filled_qty=filled_qty,
            avg_fill_price=avg_fill,
        )

    @_log_metrics
    async def cancel_order(self, broker_order_id: str) -> bool:
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"{self.base_url}/orderexecution/orders/{broker_order_id}",
                headers=await self._headers(),
            )
        return resp.status_code == 200

    @_log_metrics
    async def get_order(self, broker_order_id: str) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/brokerage/accounts/{self.account_id}/orders/{broker_order_id}",
                headers=await self._headers(),
            )
            resp.raise_for_status()
        data = resp.json()
        orders = data.get("Orders", [{}])
        o = orders[0] if orders else {}
        return {
            "status": o.get("Status", "unknown").lower(),
            "filled_qty": float(o.get("FilledQuantity", 0)),
        }

    @_log_metrics
    async def get_positions(self) -> List[dict]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/brokerage/accounts/{self.account_id}/positions",
                headers=await self._headers(),
            )
            resp.raise_for_status()
        data = resp.json()
        positions = []
        for p in data.get("Positions", []):
            positions.append(
                {
                    "symbol": p.get("Symbol"),
                    "qty": float(p.get("Quantity", 0)),
                    "market_value": float(p.get("MarketValue", 0)),
                    "avg_entry_price": float(p.get("AveragePrice", 0)),
                    "unrealized_pnl": float(p.get("UnrealizedProfitLoss", 0)),
                    "side": "long" if float(p.get("Quantity", 0)) > 0 else "short",
                }
            )
        return positions

    @_log_metrics
    async def get_account(self) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/brokerage/accounts/{self.account_id}/balances",
                headers=await self._headers(),
            )
            resp.raise_for_status()
        data = resp.json()
        bal = data.get("Balances", [{}])[0] if data.get("Balances") else {}
        return {
            "equity": float(bal.get("Equity", 0)),
            "cash": float(bal.get("CashBalance", 0)),
            "buying_power": float(bal.get("BuyingPower", 0)),
            "day_trade_count": 0,
        }

    @_log_metrics
    async def get_quote(self, symbol: str) -> QuoteResult:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/marketdata/quotes/{symbol}",
                headers=await self._headers(),
            )
            resp.raise_for_status()
        data = resp.json()
        quotes = data.get("Quotes", [{}])
        q = quotes[0] if quotes else {}
        return QuoteResult(
            symbol=symbol,
            bid=float(q.get("Bid", 0)),
            ask=float(q.get("Ask", 0)),
            last=float(q.get("Last", 0)),
            volume=int(q.get("Volume", 0)),
        )

    @_log_metrics
    async def get_historical(self, symbol: str, interval: str, start: datetime, end: datetime) -> List[dict]:
        """
        Retrieve historical bar data for a symbol.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        interval : str
            One of the supported intervals (e.g., "1m", "5m", "1h", "1d").
        start : datetime
            Start time (unused by TradeStation API; kept for interface compatibility).
        end : datetime
            End time (unused by TradeStation API; kept for interface compatibility).

        Returns
        -------
        List[dict]
            List of bar dictionaries with keys: ts, open, high, low, close, volume.
        """
        params = self._build_historical_params(interval)
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/marketdata/barcharts/{symbol}",
                params=params,
                headers=await self._headers(),
            )
            resp.raise_for_status()
        data = resp.json()
        return self._parse_historical_data(data)

    def _build_historical_params(self, interval: str, bars_back: int = 500) -> dict:
        """
        Construct query parameters for the historical request.
        """
        unit = "Minute" if interval != "1d" else "Daily"
        interval_value = self.INTERVAL_MAP.get(interval, "1")
        return {
            "unit": unit,
            "interval": interval_value,
            "barsBack": str(bars_back),
        }

    def _parse_historical_data(self, data: dict) -> List[dict]:
        """
        Parse raw historical data into a list of dictionaries.
        """
        bars = data.get("Bars", [])
        parsed = []
        for bar in bars:
            parsed.append(
                {
                    "ts": datetime.fromtimestamp(bar.get("Timestamp", 0), tz=UTC),
                    "open": float(bar.get("Open", 0)),
                    "high": float(bar.get("High", 0)),
                    "low": float(bar.get("Low", 0)),
                    "close": float(bar.get("Close", 0)),
                    "volume": int(bar.get("Volume", 0)),
                }
            )
        return parsed