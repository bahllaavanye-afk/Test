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
            Each leg dict must contain ``Symbol`` and ``TradeAction`` fields.
        quantity : int, default 1
            Number of contracts for the entire multi‑leg order.
        order_type : str, default "market"
            Either ``"market"`` or ``"limit"``. ``"limit"`` requires ``limit_price``.
        limit_price : float | None
            Required when ``order_type`` is ``"limit"``; otherwise ignored.
        opening : bool, default True
            If ``True`` the order opens a position; ``False`` closes.
        route : str, default "Intelligent"
        duration : str, default "DAY"
        """
        # Basic validation to surface edge‑case failures early
        if not legs:
            raise ValueError("At least one leg must be provided")
        if order_type == "limit" and limit_price is None:
            raise ValueError("limit_price must be set for limit orders")

        body = {
            "AccountID": account_id,
            "Quantity": str(quantity),
            "OrderType": "Market" if order_type == "market" else "Limit",
            "Route": route,
            "TimeInForce": {"Duration": duration},
            "Legs": legs,
        }

        if order_type == "limit":
            body["LimitPrice"] = str(limit_price)

        # Adjust TradeAction based on opening/closing intent
        for leg in body["Legs"]:
            action = leg.get("TradeAction", "").upper()
            if opening and action.startswith("SELL"):
                leg["TradeAction"] = "SELLTOOPEN"
            elif opening and action.startswith("BUY"):
                leg["TradeAction"] = "BUYTOOPEN"
            elif not opening and action.startswith("SELL"):
                leg["TradeAction"] = "SELLTOCLOSE"
            elif not opening and action.startswith("BUY"):
                leg["TradeAction"] = "BUYTOCLOSE"
        return body


# ----------------------------------------------------------------------
# Unit tests for the pure helper functions
# ----------------------------------------------------------------------
import unittest
from datetime import date


class TestTradeStationHelperFunctions(unittest.TestCase):
    def test_build_option_symbol_basic(self):
        # Standard call option
        sym = TradeStationBroker.build_option_symbol(
            underlying="spy",
            expiration=date(2024, 1, 19),
            strike=447.5,
            option_type="call",
        )
        self.assertEqual(sym, "SPY 240119C447.5")

    def test_build_option_symbol_strike_whole_number_and_put(self):
        # Whole-number strike should drop trailing .0 and be a put
        sym = TradeStationBroker.build_option_symbol(
            underlying="aapl",
            expiration=date(2025, 12, 31),
            strike=150.0,
            option_type="p",
        )
        self.assertEqual(sym, "AAPL 251231P150")

    def test_build_option_symbol_edge_cases(self):
        # Edge case: leap year date, mixed‑case option_type, and zero strike
        sym = TradeStationBroker.build_option_symbol(
            underlying="msft",
            expiration=date(2024, 2, 29),  # valid leap day
            strike=0,
            option_type="CaLl",
        )
        # Zero strike should be represented as "0"
        self.assertEqual(sym, "MSFT 240229C0")

    def test_build_option_order_body_validation_no_legs(self):
        # An empty legs list should raise a ValueError
        with self.assertRaises(ValueError):
            TradeStationBroker.build_option_order_body(
                account_id="ACC123",
                legs=[],
                quantity=1,
                order_type="market",
            )

    def test_build_option_order_body_limit_without_price(self):
        # Limit order without limit_price should raise a ValueError
        with self.assertRaises(ValueError):
            TradeStationBroker.build_option_order_body(
                account_id="ACC123",
                legs=[{"Symbol": "SPY 240119C447.5", "TradeAction": "BUY"}],
                quantity=2,
                order_type="limit",
                limit_price=None,
            )

    def test_build_option_order_body_trade_action_mapping(self):
        # Verify that TradeAction strings are correctly transformed based on opening flag
        legs = [
            {"Symbol": "SPY 240119C447.5", "TradeAction": "buy"},
            {"Symbol": "SPY 240119P447.5", "TradeAction": "sell"},
        ]
        body_open = TradeStationBroker.build_option_order_body(
            account_id="ACC123",
            legs=legs.copy(),
            quantity=1,
            order_type="market",
            opening=True,
        )
        self.assertEqual(body_open["Legs"][0]["TradeAction"], "BUYTOOPEN")
        self.assertEqual(body_open["Legs"][1]["TradeAction"], "SELLTOOPEN")

        body_close = TradeStationBroker.build_option_order_body(
            account_id="ACC123",
            legs=legs.copy(),
            quantity=1,
            order_type="market",
            opening=False,
        )
        self.assertEqual(body_close["Legs"][0]["TradeAction"], "BUYTOCLOSE")
        self.assertEqual(body_close["Legs"][1]["TradeAction"], "SELLTOCLOSE")


if __name__ == "__main__":
    unittest.main()