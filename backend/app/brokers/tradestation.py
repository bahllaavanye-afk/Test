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
from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.utils.logging import logger


class TradeStationBroker(AbstractBroker):
    def __init__(self, client_id: str, client_secret: str, account_id: str, paper: bool = True):
        self.client_id = client_id
        self.client_secret = client_secret
        self.account_id = account_id
        self.paper = paper
        self.base_url = (
            "https://sim.api.tradestation.com/v3" if paper else "https://api.tradestation.com/v3"
        )
        self._access_token: str | None = None
        self._token_expires_at: datetime = datetime.min.replace(tzinfo=timezone.utc)
        self._auth_headers: dict | None = None
        # Reuse a single AsyncClient for the lifetime of the broker
        self._client = httpx.AsyncClient()

    async def _close_client(self):
        await self._client.aclose()

    async def _get_token(self) -> str:
        """Retrieve a cached token or request a new one if expired."""
        now = datetime.now(timezone.utc)
        if self._access_token and now < self._token_expires_at:
            return self._access_token

        resp = await self._client.post(
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
        # Refresh a bit earlier than actual expiry to avoid edge cases
        self._token_expires_at = now + timedelta(seconds=data.get("expires_in", 1200) - 60)
        # Update cached headers
        self._auth_headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }
        return self._access_token

    async def _headers(self) -> dict:
        """Return authentication headers, refreshing token if needed."""
        if self._auth_headers is None or datetime.now(timezone.utc) >= self._token_expires_at:
            await self._get_token()
        return self._auth_headers  # type: ignore[return-value]

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

        resp = await self._client.post(
            f"{self.base_url}/orderexecution/orders", json=body, headers=await self._headers()
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

    async def cancel_order(self, broker_order_id: str) -> bool:
        resp = await self._client.delete(
            f"{self.base_url}/orderexecution/orders/{broker_order_id}",
            headers=await self._headers(),
        )
        return resp.status_code == 200

    async def get_order(self, broker_order_id: str) -> dict:
        resp = await self._client.get(
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
        resp = await self._client.get(
            f"{self.base_url}/brokerage/accounts/{self.account_id}/positions",
            headers=await self._headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "symbol": p.get("Symbol"),
                "qty": float(p.get("Quantity", 0)),
                "market_value": float(p.get("MarketValue", 0)),
                "avg_entry_price": float(p.get("AveragePrice", 0)),
                "unrealized_pnl": float(p.get("UnrealizedProfitLoss", 0)),
                "side": "long" if float(p.get("Quantity", 0)) > 0 else "short",
            }
            for p in data.get("Positions", [])
        ]

    async def get_account(self) -> dict:
        resp = await self._client.get(
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
        resp = await self._client.get(
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
        cp = "C" if str(option_type).lower().startswith("c") else "P"
        ymd = expiration.strftime("%y%m%d")
        strike_str = f"{strike:g}"  # 447.5 -> "447.5", 150.0 -> "150"
        return f"{underlying.upper()} {ymd}{cp}{strike_str}"

    @staticmethod
    def build_option_order_body(
        account_id: str,
        legs: list[dict],
        quantity: int = 1,
        order_type: str = "market",
        limit_price: float | None = None,
        *,
        opening: bool = True,
        route: str = "Intelligent",
        duration: str = "DAY",
    ) -> dict:
        """Build a TradeStation multi-leg options order body. Pure function.

        Parameters
        ----------
        account_id : str
            TradeStation account identifier.
        legs : list[dict]
            List of leg dictionaries as expected by TradeStation API.
        quantity : int, default 1
            Number of contracts for the multi-leg order.
        order_type : str, default "market"
            ``"market"`` or ``"limit"``.
        limit_price : float | None
            Required when ``order_type`` is ``"limit"``.
        opening : bool, default True
            ``True`` for opening positions, ``False`` for closing.
        route : str, default "Intelligent"
            Execution routing option.
        duration : str, default "DAY"
            Time in force duration.

        Returns
        -------
        dict
            JSON‑serialisable payload for the ``/orderexecution/orders`` endpoint.
        """
        trade_action = "BUYTOOPEN" if opening else "SELLTOCLOSE"
        body = {
            "AccountID": account_id,
            "OrderType": "Market" if order_type == "market" else "Limit",
            "Route": route,
            "TimeInForce": {"Duration": duration},
            "Quantity": str(quantity),
            "TradeAction": trade_action,
            "Legs": legs,
        }
        if order_type == "limit" and limit_price is not None:
            body["LimitPrice"] = str(limit_price)
        return body

    # Ensure the async client is properly closed when the broker is garbage‑collected
    def __del__(self):
        # Schedule closure but ignore if event loop is already closed
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._close_client())
        except Exception:
            pass