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
            # Subtract a minute to avoid edge‑case expiry.
            self._token_expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=data.get("expires_in", 1200) - 60
            )
        return self._access_token

    async def _headers(self) -> dict:
        token = await self._get_token()
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # ------------------------------------------------------------------ #
    # Order handling                                                    #
    # ------------------------------------------------------------------ #
    async def place_order(self, request: OrderRequest) -> OrderResult:
        body = self._build_order_body(request)
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/orderexecution/orders", json=body, headers=await self._headers()
            )
            resp.raise_for_status()
            data = resp.json()
        return self._parse_order_response(data)

    def _build_order_body(self, request: OrderRequest) -> dict:
        """Construct the JSON payload for a standard (equity/ETF) order."""
        body = {
            "AccountID": self.account_id,
            "Symbol": request.symbol,
            "Quantity": str(int(request.quantity)),
            "OrderType": "Market" if request.order_type == "market" else "Limit",
            "TradeAction": "BUY" if request.side == "buy" else "SELL",
            "TimeInForce": {"Duration": "DAY"},
            "Route": "Intelligent",
        }
        if request.order_type == "limit" and request.limit_price is not None:
            body["LimitPrice"] = str(request.limit_price)
        return body

    def _parse_order_response(self, data: dict) -> OrderResult:
        """Translate TradeStation's order response into an ``OrderResult``."""
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
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"{self.base_url}/orderexecution/orders/{broker_order_id}",
                headers=await self._headers(),
            )
        return resp.status_code == 200

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

    # ------------------------------------------------------------------ #
    # Position & account queries                                         #
    # ------------------------------------------------------------------ #
    async def get_positions(self) -> list[dict]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
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
    def build_option_symbol(underlying: str, expiration: date, strike: float, option_type: str) -> str:
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

        The function translates a user‑friendly ``legs`` description into the
        exact JSON structure expected by TradeStation. Each leg dictionary must
        contain at least ``symbol`` and ``action`` keys. ``action`` should be one
        of ``BUYTOOPEN``, ``SELLTOOPEN``, ``BUYTOCLOSE`` or ``SELLTOCLOSE``.
        ``quantity`` applies to the whole spread, while individual leg ratios can
        be expressed via an optional ``ratio`` key (default is ``1``).

        Parameters
        ----------
        account_id: str
            The TradeStation account identifier.
        legs: list[dict]
            List of leg specifications.
        quantity: int, default 1
            Number of spreads to trade.
        order_type: str, default "market"
            ``"market"`` or ``"limit"``.
        limit_price: float | None, default None
            Required when ``order_type`` is ``"limit"``.
        opening: bool, default True
            ``True`` for opening trades, ``False`` for closing.
        route: str, default "Intelligent"
            Execution routing preference.
        duration: str, default "DAY"
            Time‑in‑force duration.

        Returns
        -------
        dict
            JSON payload ready for ``/orderexecution/orders``.
        """
        # Helper to normalise a single leg.
        def _normalize_leg(leg: dict) -> dict:
            symbol = leg["symbol"]
            action = leg["action"]
            ratio = leg.get("ratio", 1)
            # TradeStation expects each leg to contain its own TradeAction.
            return {
                "Symbol": symbol,
                "TradeAction": action,
                "Quantity": str(ratio),
            }

        normalized_legs = [_normalize_leg(leg) for leg in legs]

        body: dict = {
            "AccountID": account_id,
            "Legs": normalized_legs,
            "Quantity": str(quantity),
            "OrderType": "Market" if order_type == "market" else "Limit",
            "TradeAction": "BUY" if opening else "SELL",
            "TimeInForce": {"Duration": duration},
            "Route": route,
        }

        if order_type == "limit":
            if limit_price is None:
                raise ValueError("limit_price must be provided for limit orders")
            body["LimitPrice"] = str(limit_price)

        return body