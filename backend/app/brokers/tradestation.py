"""TradeStation REST API broker with OAuth2 client credentials.

Options support
---------------
Option symbols use TradeStation's symbology: ``{ROOT} {YYMMDD}{C|P}{STRIKE}``
e.g. ``SPY 240119C447.5`` (SPY 19‑Jan‑2024 $447.5 call). Multi‑leg orders
(spreads, condors, straddles) POST to the same ``/orderexecution/orders``
endpoint with a ``Legs`` array; each leg carries its own opening/closing
``TradeAction`` (BUYTOOPEN / SELLTOOPEN / BUYTOCLOSE / SELLTOCLOSE).

The request‑building helpers below (``build_option_symbol``,
``build_option_order_body``) are pure functions with no network or auth, so
they are unit‑testable without live TradeStation credentials.
"""
import httpx
from datetime import date, datetime, timezone, timedelta
from typing import Any, List, Mapping, Optional

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.utils.logging import logger


class TradeStationBroker(AbstractBroker):
    """Concrete broker implementation for TradeStation's REST API."""

    def __init__(self, client_id: str, client_secret: str, account_id: str, paper: bool = True):
        self.client_id = client_id
        self.client_secret = client_secret
        self.account_id = account_id
        self.paper = paper
        self.base_url = "https://sim.api.tradestation.com/v3" if paper else "https://api.tradestation.com/v3"
        self._access_token: Optional[str] = None
        self._token_expires_at: datetime = datetime.min.replace(tzinfo=timezone.utc)

    async def _get_token(self) -> str:
        """Retrieve a valid OAuth2 token, refreshing it if necessary."""
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
            # Subtract a safety margin of 60 seconds.
            self._token_expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=data.get("expires_in", 1200) - 60
            )
        return self._access_token

    async def _headers(self) -> Mapping[str, str]:
        """Return request headers containing the bearer token."""
        token = await self._get_token()
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place a market or limit order via TradeStation."""
        body: dict[str, Any] = {
            "AccountID": self.account_id,
            "Symbol": request.symbol,
            "Quantity": str(int(request.quantity)),
            "OrderType": "Market" if request.order_type == "market" else "Limit",
            "TradeAction": "BUY" if request.side == "buy" else "SELL",
            "TimeInForce": {"Duration": "DAY"},
            "Route": "Intelligent",
        }

        if request.order_type == "limit":
            if request.limit_price is None:
                raise ValueError("limit_price must be provided for limit orders")
            body["LimitPrice"] = str(request.limit_price)

        async with httpx.AsyncClient() as client:
            resp = await client.post(
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
            broker_order_id=order_id, status=status, filled_qty=filled_qty, avg_fill_price=avg_fill
        )

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an existing order."""
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"{self.base_url}/orderexecution/orders/{broker_order_id}",
                headers=await self._headers(),
            )
        return resp.status_code == 200

    async def get_order(self, broker_order_id: str) -> dict:
        """Retrieve order status and fill information."""
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

    async def get_positions(self) -> List[dict]:
        """Return a list of current positions for the account."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/brokerage/accounts/{self.account_id}/positions",
                headers=await self._headers(),
            )
            resp.raise_for_status()
        data = resp.json()
        positions: List[dict] = []
        for p in data.get("Positions", []):
            qty = float(p.get("Quantity", 0))
            positions.append(
                {
                    "symbol": p.get("Symbol"),
                    "qty": qty,
                    "market_value": float(p.get("MarketValue", 0)),
                    "avg_entry_price": float(p.get("AveragePrice", 0)),
                    "unrealized_pnl": float(p.get("UnrealizedProfitLoss", 0)),
                    "side": "long" if qty > 0 else "short",
                }
            )
        return positions

    async def get_account(self) -> dict:
        """Fetch basic account balance information."""
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
        """Retrieve the latest quote for a given symbol."""
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
        """Return a TradeStation option symbol.

        Example
        -------
        ``build_option_symbol("SPY", date(2024, 1, 19), 447.5, "call")`` → ``"SPY 240119C447.5"``

        Parameters
        ----------
        underlying: str
            Underlying ticker (e.g., ``"SPY"``). Case‑insensitive.
        expiration: date
            Expiration date of the option.
        strike: float
            Strike price. Whole numbers lose the trailing ``.0``.
        option_type: str
            ``"call"``/``"c"`` for calls, ``"put"``/``"p"`` for puts (case‑insensitive).

        Returns
        -------
        str
            Formatted option symbol.
        """
        cp = "C" if str(option_type).lower().startswith("c") else "P"
        ymd = expiration.strftime("%y%m%d")
        strike_str = f"{strike:g}"  # 447.5 -> "447.5", 150.0 -> "150"
        return f"{underlying.upper()} {ymd}{cp}{strike_str}"

    @staticmethod
    def build_option_order_body(
        account_id: str,
        legs: List[dict],
        quantity: int = 1,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        *,
        opening: bool = True,
        route: str = "Intelligent",
        duration: str = "DAY",
    ) -> dict:
        """Construct a multi‑leg options order payload.

        This pure function validates the supplied arguments and produces a
        dictionary ready for JSON‑encoding and transmission to the
        ``/orderexecution/orders`` endpoint.

        Parameters
        ----------
        account_id: str
            The TradeStation account identifier.
        legs: list[dict]
            Each leg must contain at least ``Symbol`` and ``TradeAction``.
        quantity: int, default 1
            Number of contracts for the whole multi‑leg order.
        order_type: {"market", "limit"}, default "market"
            Determines whether ``LimitPrice`` is required.
        limit_price: float, optional
            Required when ``order_type`` is ``"limit"``.
        opening: bool, default True
            ``True`` → opening order (BUYTOOPEN/SELLTOOPEN), ``False`` → closing.
        route: str, default "Intelligent"
            Execution route; TradeStation validates accepted strings.
        duration: str, default "DAY"
            Order duration (e.g., ``"DAY"``, ``"GTC"``).

        Returns
        -------
        dict
            A payload compatible with TradeStation's order API.
        """
        if not account_id:
            raise ValueError("account_id must be a non‑empty string")
        if not legs:
            raise ValueError("legs list cannot be empty")
        if quantity <= 0:
            raise ValueError("quantity must be a positive integer")
        order_type = order_type.lower()
        if order_type not in {"market", "limit"}:
            raise ValueError('order_type must be either "market" or "limit"')
        if order_type == "limit" and limit_price is None:
            raise ValueError("limit_price is required for limit orders")
        if duration.upper() not in {"DAY", "GTC", "FOK", "IOC"}:
            raise ValueError(f'Unsupported duration: {duration}')

        # Ensure each leg has the required fields and normalise the TradeAction.
        normalized_legs = []
        for leg in legs:
            if "Symbol" not in leg or "TradeAction" not in leg:
                raise ValueError("Each leg must contain 'Symbol' and 'TradeAction'")
            action = leg["TradeAction"].upper()
            if opening:
                # Convert generic BUY/SELL to the appropriate opening action.
                if action == "BUY":
                    action = "BUYTOOPEN"
                elif action == "SELL":
                    action = "SELLTOOPEN"
            else:
                if action == "BUY":
                    action = "BUYTOCLOSE"
                elif action == "SELL":
                    action = "SELLTOCLOSE"
            normalized_leg = {**leg, "TradeAction": action}
            normalized_legs.append(normalized_leg)

        body: dict[str, Any] = {
            "AccountID": account_id,
            "OrderType": order_type.capitalize(),
            "Route": route,
            "Duration": duration.upper(),
            "Quantity": quantity,
            "Legs": normalized_legs,
        }

        if order_type == "limit":
            body["LimitPrice"] = str(limit_price)

        return body

# End of file