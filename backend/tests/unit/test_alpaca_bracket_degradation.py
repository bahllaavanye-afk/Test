from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .base import BaseBroker, OrderRequest, OrderResponse

log = logging.getLogger(__name__)

# Stubbed imports for CI environment; real SDK symbols are monkey‑patched in tests.
try:
    from alpaca_trade_api.rest import (
        MarketOrderRequest,
        LimitOrderRequest,
        StopOrderRequest,
        TakeProfitRequest,
        StopLossRequest,
        OrderSide,
        TimeInForce,
        OrderClass,
    )
    ALPACA_AVAILABLE = True
    ALPACA_BRACKET_AVAILABLE = True
except Exception:  # pragma: no cover
    # The test suite monkey‑patches these symbols, so we provide harmless fallbacks.
    ALPACA_AVAILABLE = False
    ALPACA_BRACKET_AVAILABLE = False

    class _Placeholder:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    MarketOrderRequest = LimitOrderRequest = StopOrderRequest = _Placeholder
    TakeProfitRequest = StopLossRequest = _Placeholder
    OrderSide = type("OrderSide", (), {"BUY": "buy", "SELL": "sell"})
    TimeInForce = type("TimeInForce", (), {"GTC": "gtc", "DAY": "day"})
    OrderClass = type("OrderClass", (), {"BRACKET": "bracket"})


class AlpacaBroker(BaseBroker):
    """Broker implementation for Alpaca with graceful degradation of failed brackets."""

    def __init__(self, trading_client: Any) -> None:
        """
        Initialise the broker.

        Parameters
        ----------
        trading_client: Any
            The Alpaca trading client exposing a ``submit_order`` method.
        """
        if trading_client is None:
            raise ValueError("trading_client must not be None")
        self.trading = trading_client
        self._calls: List[str] = []

    async def place_order(self, request: Optional[OrderRequest]) -> OrderResponse:
        """
        Place an order, handling the special case where a bracket order is rejected.

        Parameters
        ----------
        request: OrderRequest | None
            The order specification. ``None`` is treated as an error.

        Returns
        -------
        OrderResponse
            The response from Alpaca, possibly decorated with ``bracket_degraded``.
        """
        if request is None:
            raise TypeError("OrderRequest cannot be None")

        # Defensive copy to avoid mutating the caller's object.
        req = OrderRequest(**request.dict()) if hasattr(request, "dict") else request

        # Build the primary order payload.
        order_data = self._build_order_data(req)

        # Determine if we are attempting a bracket order.
        is_bracket = getattr(order_data, "order_class", None) is not None

        # Attempt the order; if a bracket fails we fall back to a plain order.
        try:
            raw_order = await self._call(order_data)
        except RuntimeError as exc:
            # Only degrade if the failure originates from a bracket request.
            if is_bracket:
                log.warning("Bracket order rejected (%s); degrading to plain order.", exc)
                # Re‑build a plain order without the bracket class.
                plain_order_data = self._build_plain_order_data(req)
                raw_order = await self._call(plain_order_data)
                # Decorate the response to indicate degradation.
                payload = self._decorate_degraded_payload(
                    raw_order,
                    unapplied_stop=req.stop_loss,
                    unapplied_tp=req.take_profit,
                )
                return OrderResponse.from_raw(payload)
            raise  # Re‑raise unexpected errors.

        # Successful path – no degradation needed.
        return OrderResponse.from_raw(self._extract_payload(raw_order))

    def _build_order_data(self, req: OrderRequest) -> Any:
        """
        Construct the SDK order object, including bracket legs when appropriate.

        Handles empty or ``None`` collections gracefully.
        """
        # Primary order class selection.
        order_cls = MarketOrderRequest if req.order_type == "market" else LimitOrderRequest

        # Basic order fields.
        base_kwargs: Dict[str, Any] = {
            "symbol": req.symbol,
            "qty": req.quantity,
            "side": OrderSide.BUY if req.side == "buy" else OrderSide.SELL,
            "time_in_force": TimeInForce.GTC,
        }

        # Attach limit price if required.
        if req.order_type == "limit":
            if req.limit_price is None:
                raise ValueError("limit_price must be provided for limit orders")
            base_kwargs["limit_price"] = req.limit_price

        # Bracket handling – only if both legs are supplied.
        if req.stop_loss is not None or req.take_profit is not None:
            # Ensure we have a list to iterate; empty list means no bracket.
            legs: List[Any] = []

            if req.take_profit is not None:
                legs.append(TakeProfitRequest(limit_price=req.take_profit))
            if req.stop_loss is not None:
                legs.append(StopLossRequest(stop_price=req.stop_loss))

            # Guard against off‑by‑one errors: the SDK expects exactly two legs.
            if len(legs) != 2:
                # If only one leg is present, treat as a plain order.
                return order_cls(**base_kwargs)

            base_kwargs.update(
                {
                    "order_class": OrderClass.BRACKET,
                    "take_profit": legs[0],
                    "stop_loss": legs[1],
                }
            )
        return order_cls(**base_kwargs)

    def _build_plain_order_data(self, req: OrderRequest) -> Any:
        """
        Construct a plain (non‑bracket) order payload.
        """
        order_cls = MarketOrderRequest if req.order_type == "market" else LimitOrderRequest
        kwargs: Dict[str, Any] = {
            "symbol": req.symbol,
            "qty": req.quantity,
            "side": OrderSide.BUY if req.side == "buy" else OrderSide.SELL,
            "time_in_force": TimeInForce.GTC,
        }
        if req.order_type == "limit":
            if req.limit_price is None:
                raise ValueError("limit_price must be provided for limit orders")
            kwargs["limit_price"] = req.limit_price
        return order_cls(**kwargs)

    async def _call(self, order_data: Any) -> Any:
        """
        Submit the order to Alpaca, recording whether it was a bracket or plain order.

        This method is deliberately thin so tests can monkey‑patch it.
        """
        is_bracket = getattr(order_data, "order_class", None) is not None
        self._calls.append("bracket" if is_bracket else "plain")
        # The real SDK call would be asynchronous; we simulate with ``await``.
        return await self.trading.submit_order(order_data=order_data)

    @staticmethod
    def _extract_payload(raw_order: Any) -> Dict[str, Any]:
        """
        Pull the useful fields from the raw SDK order object.
        """
        payload = {
            "id": getattr(raw_order, "id", None),
            "status": getattr(raw_order, "status", None),
            "filled_qty": getattr(raw_order, "filled_qty", 0),
            "filled_avg_price": getattr(raw_order, "filled_avg_price", None),
        }
        return payload

    @staticmethod
    def _decorate_degraded_payload(
        raw_order: Any,
        unapplied_stop: Optional[float],
        unapplied_tp: Optional[float],
    ) -> Dict[str, Any]:
        """
        Add ``bracket_degraded`` information to the payload when a bracket fails.

        Parameters
        ----------
        raw_order: Any
            The raw order object returned from Alpaca.
        unapplied_stop: float | None
            The stop‑loss value that was not applied.
        unapplied_tp: float | None
            The take‑profit value that was not applied.
        """
        payload = AlpacaBroker._extract_payload(raw_order)
        payload["bracket_degraded"] = True
        if unapplied_stop is not None:
            payload["unapplied_stop_loss"] = unapplied_stop
        if unapplied_tp is not None:
            payload["unapplied_take_profit"] = unapplied_tp
        return payload

    # --------------------------------------------------------------------- #
    # Compatibility shim for existing test suite expectations.
    # --------------------------------------------------------------------- #
    @property
    def _calls(self) -> List[str]:
        """Expose the internal call log for test introspection."""
        return getattr(self, "__calls", [])

    @_calls.setter
    def _calls(self, value: List[str]) -> None:
        setattr(self, "__calls", value)