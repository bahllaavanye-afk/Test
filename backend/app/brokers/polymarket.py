"""
Polymarket CLOB broker integration via py-clob-client.
Supports YES/NO binary market trading and arbitrage scanning.
"""

import asyncio
from typing import List, Dict

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.utils.exceptions import BrokerError
from app.utils.logging import logger
from app.config import settings

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    # Specific exception types from the client library (fallback to generic Exception if not present)
    from py_clob_client.exceptions import CLOBError  # type: ignore
    POLY_AVAILABLE = True
except ImportError:
    # py-clob-client is optional; the broker will raise a clear error upon instantiation
    POLY_AVAILABLE = False
except Exception:  # pragma: no cover
    # Any other import‑time issue should be surfaced early
    POLY_AVAILABLE = False


class PolymarketBroker(AbstractBroker):
    """Broker implementation for Polymarket's CLOB using the py‑clob‑client library."""

    def __init__(self, private_key: str, chain_id: int = 137):
        if not POLY_AVAILABLE:
            raise ImportError("py-clob-client required to use PolymarketBroker")
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
        )

    async def get_markets(self, min_open_interest: float = 10000) -> List[Dict]:
        """Auto‑discover active markets with sufficient liquidity."""
        try:
            markets = await asyncio.to_thread(self.client.get_markets)
            return [
                m
                for m in markets
                if float(m.get("openInterest", 0)) >= min_open_interest
            ]
        except (CLOBError, ValueError) as e:
            logger.error(
                "Polymarket market fetch failed",
                error=str(e),
                min_open_interest=min_open_interest,
            )
            return []
        except Exception as e:  # pragma: no cover
            logger.exception("Unexpected error while fetching Polymarket markets", error=str(e))
            return []

    async def get_order_book(self, token_id: str) -> Dict:
        """Retrieve the order book for a specific market token."""
        try:
            return await asyncio.to_thread(self.client.get_order_book, token_id)
        except (CLOBError, ValueError) as e:
            logger.error(
                "Polymarket get_order_book failed",
                token_id=token_id,
                error=str(e),
            )
            raise BrokerError(f"Failed to fetch order book for token {token_id}: {e}") from e
        except Exception as e:  # pragma: no cover
            logger.exception("Unexpected error in get_order_book", token_id=token_id, error=str(e))
            raise BrokerError(f"Unexpected error fetching order book for token {token_id}") from e

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Create and submit an order to Polymarket."""
        try:
            args = OrderArgs(
                token_id=request.symbol,
                price=request.limit_price or 0.5,
                size=request.quantity,
                side=request.side.upper(),
            )
            order = await asyncio.to_thread(self.client.create_and_post_order, args)
            return OrderResult(
                broker_order_id=str(order.get("orderID", "")),
                status=order.get("status", "pending"),
                raw_payload=order,
            )
        except (CLOBError, ValueError) as e:
            logger.error(
                "Polymarket place_order failed",
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                error=str(e),
            )
            raise BrokerError(f"Polymarket: {e}") from e
        except Exception as e:  # pragma: no cover
            logger.exception(
                "Unexpected error during Polymarket place_order",
                request=request.dict() if hasattr(request, "dict") else str(request),
                error=str(e),
            )
            raise BrokerError(f"Polymarket unexpected error: {e}") from e

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            await asyncio.to_thread(self.client.cancel, broker_order_id)
            return True
        except (CLOBError, ValueError) as e:
            logger.warning(
                "Polymarket cancel_order failed",
                order_id=broker_order_id,
                error=str(e),
            )
            return False
        except Exception as e:  # pragma: no cover
            logger.exception(
                "Unexpected error in Polymarket cancel_order",
                order_id=broker_order_id,
                error=str(e),
            )
            return False

    async def get_order(self, broker_order_id: str) -> Dict:
        """Fetch details of a specific order."""
        try:
            return await asyncio.to_thread(self.client.get_order, broker_order_id)
        except (CLOBError, ValueError) as e:
            logger.error(
                "Polymarket get_order failed",
                order_id=broker_order_id,
                error=str(e),
            )
            raise BrokerError(f"Failed to retrieve order {broker_order_id}: {e}") from e
        except Exception as e:  # pragma: no cover
            logger.exception(
                "Unexpected error in Polymarket get_order",
                order_id=broker_order_id,
                error=str(e),
            )
            raise BrokerError(f"Unexpected error retrieving order {broker_order_id}") from e

    async def get_positions(self) -> List[Dict]:
        """Polymarket does not expose positions via the CLOB client; return empty list."""
        return []

    async def get_account(self) -> Dict:
        """Polymarket does not expose account balances via the CLOB client; return empty dict."""
        return {}

    async def get_quote(self, symbol: str) -> QuoteResult:
        """Generate a quote using the best bid/ask from the order book."""
        ob = await self.get_order_book(symbol)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0
        return QuoteResult(
            symbol=symbol,
            bid=best_bid,
            ask=best_ask,
            last=(best_bid + best_ask) / 2,
        )

    async def get_historical(self, symbol: str, interval: str = "1d", limit: int = 500) -> List[Dict]:
        """Polymarket does not provide traditional OHLCV data; return empty list."""
        return []