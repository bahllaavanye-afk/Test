"""TradeStation REST API broker with OAuth2 client credentials.

Options support
---------------
Option symbols use TradeStation's symbology: ``{ROOT} {YYMMDD}{C|P}{STRIKE}``
e.g. ``SPY 240119C447.5`` (SPY 19-Jan-2024 $447.5 call). Multi-leg orders
(spreads, condors, straddles) POST to the same ``/orderexecution/orders``
endpoint with a ``Legs`` array; each leg carries its own opening/closing
``TradeAction`` (BUYTOOPEN / SELLTOOPEN / BUYTOCLOSE / SELLTOCLOSE).

The request-building helpers below (``build_option_symbol``,
``build_option_order_body``) are pure functions with no network or auth, so
they are unit-testable without live TradeStation credentials.
"""
import httpx
from datetime import date, datetime, timezone, timedelta
from typing import List, Dict, Any

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.utils.logging import logger


class TradeStationBroker(AbstractBroker):
    def __init__(self, client_id: str, client_secret: str, account_id: str, paper: bool = True):
        if not isinstance(client_id, str) or not client_id:
            raise ValueError("client_id must be a non‑empty string")
        if not isinstance(client_secret, str) or not client_secret:
            raise ValueError("client_secret must be a non‑empty string")
        if not isinstance(account_id, str) or not account_id:
            raise ValueError("account_id must be a non‑empty string")
        if not isinstance(paper, bool):
            raise ValueError("paper must be a boolean")
        self.client_id = client_id
        self.client_secret = client_secret
        self.account_id = account_id
        self.paper = paper
        self.base_url = "https://sim.api.tradestation.com/v3" if paper else "https://api.tradestation.com/v3"
        self._access_token: str | None = None
        self._token_expires_at: datetime = datetime.min.replace(tzinfo=timezone.utc)

    async def _get_token(self) -> str:
        if self._access_token and datetime.now(timezone.utc) < self._token_expires_at:
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
            self._token_expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=data.get("expires_in", 1200) - 60
            )
        return self._access_token

    async def _headers(self) -> dict:
        token = await self._get_token()
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async def place_order(self, request: OrderRequest) -> OrderResult:
        if not isinstance(request, OrderRequest):
            raise ValueError("request must be an OrderRequest instance")
        if not isinstance(request.symbol, str) or not request.symbol:
            raise ValueError("order symbol must be a non‑empty string")
        if not isinstance(request.quantity, (int, float)) or request.quantity <= 0:
            raise ValueError("order quantity must be a positive number")
        if request.order_type not in {"market", "limit"}:
            raise ValueError("order_type must be either 'market' or 'limit'")
        if request.side not in {"buy", "sell"}:
            raise ValueError("side must be either 'buy' or 'sell'")
        if request.order_type == "limit":
            if request.limit_price is None:
                raise ValueError("limit_price must be provided for limit orders")
            if not isinstance(request.limit_price, (int, float)) or request.limit_price <= 0:
                raise ValueError("limit_price must be a positive number")

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
            resp = await client.post(f"{self.base_url}/orderexecution/orders", json=body, headers=await self._headers())
            resp.raise_for_status()
            data = resp.json()

        order_id = data.get("OrderID", "unknown")
        status = data.get("Message", "queued").lower()
        filled_qty = float(data.get("FilledQuantity", 0))
        avg_fill = float(data.get("AveragePrice", 0)) or None

        logger.info("TradeStation order placed", order_id=order_id, status=status)
        return OrderResult(broker_order_id=order_id, status=status, filled_qty=filled_qty, avg_fill_price=avg_fill)

    async def cancel_order(self, broker_order_id: str) -> bool:
        if not isinstance(broker_order_id, str) or not broker_order_id:
            raise ValueError("broker_order_id must be a non‑empty string")
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"{self.base_url}/orderexecution/orders/{broker_order_id}",
                headers=await self._headers(),
            )
        return resp.status_code == 200

    async def get_order(self, broker_order_id: str) -> dict:
        if not isinstance(broker_order_id, str) or not broker_order_id:
            raise ValueError("broker_order_id must be a non‑empty string")
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

    async def get_positions(self) -> list[dict]:
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

    async def get_quote(self, symbol: str) -> QuoteResult:
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non‑empty string")
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

    # ------------------------------------------------------------------ #
    # Options                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_option_symbol(
        underlying: str, expiration: date, strike: float, option_type: str
    ) -> str:
        """Build a TradeStation option symbol: ``SPY 240119C447.5``.

        Pure function — no network/auth. ``option_type`` is ``call``/``put``
        (or ``c``/``p``). Whole-number strikes drop the trailing ``.0``.
        """
        if not isinstance(underlying, str) or not underlying:
            raise ValueError("underlying must be a non‑empty string")
        if not isinstance(expiration, date):
            raise ValueError("expiration must be a datetime.date instance")
        if not isinstance(strike, (int, float)) or strike <= 0:
            raise ValueError("strike must be a positive number")
        if not isinstance(option_type, str) or not option_type:
            raise ValueError("option_type must be a non‑empty string")
        cp = "C" if option_type.lower().startswith("c") else "P"
        ymd = expiration.strftime("%y%m%d")
        strike_str = f"{strike:g}"  # 447.5 -> "447.5", 150.0 -> "150"
        return f"{underlying.upper()} {ymd}{cp}{strike_str}"

    @staticmethod
    def build_option_order_body(
        account_id: str,
        legs: List[Dict[str, Any]],
        quantity: int = 1,
        order_type: str = "market",
        limit_price: float | None = None,
        *,
        opening: bool = True,
        route: str = "Intelligent",
        duration: str = "DAY",
    ) -> dict:
        """Build a TradeStation multi-leg options order body. Pure function."""
        if not isinstance(account_id, str) or not account_id:
            raise ValueError("account_id must be a non‑empty string")
        if not isinstance(legs, list) or not legs:
            raise ValueError("legs must be a non‑empty list of dicts")
        for i, leg in enumerate(legs):
            if not isinstance(leg, dict):
                raise ValueError(f"leg at index {i} must be a dict")
            if "Symbol" not in leg or not leg["Symbol"]:
                raise ValueError(f"leg at index {i} missing required 'Symbol'")
            if "TradeAction" not in leg or not leg["TradeAction"]:
                raise ValueError(f"leg at index {i} missing required 'TradeAction'")
        if not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("quantity must be a positive integer")
        if order_type not in {"market", "limit"}:
            raise ValueError("order_type must be either 'market' or 'limit'")
        if order_type == "limit":
            if limit_price is None:
                raise ValueError("limit_price must be provided for limit orders")
            if not isinstance(limit_price, (int, float)) or limit_price <= 0:
                raise ValueError("limit_price must be a positive number")
        if not isinstance(opening, bool):
            raise ValueError("opening must be a boolean")
        if not isinstance(route, str) or not route:
            raise ValueError("route must be a non‑empty string")
        if not isinstance(duration, str) or not duration:
            raise ValueError("duration must be a non‑empty string")

        body: dict[str, Any] = {
            "AccountID": account_id,
            "Legs": legs,
            "Quantity": quantity,
            "OrderType": "Market" if order_type == "market" else "Limit",
            "TradeAction": "BUY" if opening else "SELL",
            "Route": route,
            "TimeInForce": {"Duration": duration},
        }
        if order_type == "limit" and limit_price is not None:
            body["LimitPrice"] = limit_price
        return body