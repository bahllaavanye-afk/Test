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
            self._token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=data.get("expires_in", 1200) - 60)
        return self._access_token

    async def _headers(self) -> dict:
        token = await self._get_token()
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

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
            positions.append({
                "symbol": p.get("Symbol"),
                "qty": float(p.get("Quantity", 0)),
                "market_value": float(p.get("MarketValue", 0)),
                "avg_entry_price": float(p.get("AveragePrice", 0)),
                "unrealized_pnl": float(p.get("UnrealizedProfitLoss", 0)),
                "side": "long" if float(p.get("Quantity", 0)) > 0 else "short",
            })
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

        Parameters
        ----------
        account_id : str
            TradeStation account identifier.
        legs : list[dict]
            Each leg dict must contain at least ``Symbol`` and ``TradeAction``.
        quantity : int, default 1
            Number of contracts (or shares) to trade.
        order_type : str, default "market"
            ``"market"`` or ``"limit"``.
        limit_price : float | None, optional
            Required for limit orders; omitted for market orders.
        opening : bool, default True
            If True the leg ``TradeAction`` values are interpreted as opening
            positions (``BUYTOOPEN``/``SELLTOOPEN``); otherwise they are closing
            actions (``BUYTOCLOSE``/``SELLTOCLOSE``).
        route : str, default "Intelligent"
            Execution route.
        duration : str, default "DAY"
            Time in force duration.

        Returns
        -------
        dict
            JSON‑serialisable body suitable for the ``/orderexecution/orders``
            endpoint.
        """
        body = {
            "AccountID": account_id,
            "Quantity": quantity,
            "OrderType": "Market" if order_type == "market" else "Limit",
            "Route": route,
            "TimeInForce": {"Duration": duration},
            "Legs": legs,
        }
        if order_type == "limit":
            if limit_price is None:
                raise ValueError("limit_price must be provided for limit orders")
            body["LimitPrice"] = limit_price
        # ``opening`` flag does not modify the payload directly; it is expected that
        # callers provide the correct ``TradeAction`` values in each leg.
        return body


# ----------------------------------------------------------------------
# Unit tests for edge cases (pure functions)
# ----------------------------------------------------------------------
import pytest
from datetime import date

def test_build_option_symbol_basic():
    sym = TradeStationBroker.build_option_symbol('spy', date(2024, 1, 19), 447.5, 'call')
    assert sym == 'SPY 240119C447.5'

def test_build_option_symbol_edge_cases():
    # Integer strike, lower‑case underlying, abbreviated put type
    sym = TradeStationBroker.build_option_symbol('aapl', date(2025, 12, 31), 150.0, 'p')
    assert sym == 'AAPL 251231P150'
    # Zero strike and mixed‑case option type
    sym_zero = TradeStationBroker.build_option_symbol('msft', date(2023, 6, 15), 0, 'CALL')
    assert sym_zero == 'MSFT 230615C0'

def test_build_option_order_body_limit_without_price():
    # Limit order without limit_price should raise a ValueError
    with pytest.raises(ValueError):
        TradeStationBroker.build_option_order_body(
            account_id='ACC123',
            legs=[{'Symbol': 'SPY 240119C447.5', 'TradeAction': 'BUYTOOPEN'}],
            quantity=2,
            order_type='limit',
            limit_price=None,
        )

def test_build_option_order_body_limit_with_price_and_empty_legs():
    body = TradeStationBroker.build_option_order_body(
        account_id='ACC123',
        legs=[],
        quantity=1,
        order_type='limit',
        limit_price=10.5,
        opening=False,
    )
    assert body['AccountID'] == 'ACC123'
    assert body['Quantity'] == 1
    assert body['OrderType'] == 'Limit'
    assert body['LimitPrice'] == 10.5
    assert body['Legs'] == []
    # Ensure optional fields are present with default values
    assert body['Route'] == 'Intelligent'
    assert body['TimeInForce']['Duration'] == 'DAY'

def test_build_option_order_body_market_defaults():
    body = TradeStationBroker.build_option_order_body(
        account_id='ACC123',
        legs=[{'Symbol': 'SPY 240119C447.5', 'TradeAction': 'BUYTOOPEN'}],
    )
    assert body['OrderType'] == 'Market'
    assert 'LimitPrice' not in body
    assert body['Legs'][0]['Symbol'] == 'SPY 240119C447.5'