"""A bracket that loses its protective legs must not look like a clean fill.

Three ways `BracketOrder.execute` could hand back an ordinary-looking success for
a position with no stop-loss:

  1. TP/SL prices computed nonsensically (tp <= sl) — protection never submitted
  2. the OCO submission raising — entry filled, legs never placed
  3. the OCO timing out — both legs cancelled 8h later, position left bare

...plus the confirmation filter, which never rejected anything: it built
`OrderResult(reason=...)`, a kwarg the dataclass does not have, so every
rejection raised TypeError into a `log and continue` handler and the entry was
submitted anyway.

NOTE: these assert on log output via `capsys`, not `caplog` — structlog does not
propagate to stdlib logging, so caplog would pass for the wrong reason.
"""
from __future__ import annotations

import pytest

from app.brokers.base import OrderRequest, OrderResult, QuoteResult
from app.execution.advanced_orders import (
    BracketOrder,
    BracketOrderConfig,
    OCOOrder,
)


class _Broker:
    """Minimal broker double. Knobs control each failure mode under test."""

    def __init__(self, *, fill_price=100.0, quote_last=100.0, quote_raises=False,
                 fail_leg=None, cancel_raises=False, leg_status="filled"):
        self.fill_price = fill_price
        self.quote_last = quote_last
        self.quote_raises = quote_raises
        self.fail_leg = fail_leg          # index into placed[] (1 = TP, 2 = SL)
        self.cancel_raises = cancel_raises
        # "filled" resolves the OCO on the first poll. "open" never resolves it —
        # only use that with max_wait_seconds=0, or the poll loop runs for the
        # full 8h window.
        self.leg_status = leg_status
        self.placed: list[OrderRequest] = []
        self.cancelled: list[str] = []

    async def get_quote(self, symbol):
        if self.quote_raises:
            raise RuntimeError("quote feed down")
        return QuoteResult(symbol=symbol, bid=self.quote_last, ask=self.quote_last,
                           last=self.quote_last)

    async def place_order(self, request):
        self.placed.append(request)
        if self.fail_leg is not None and len(self.placed) - 1 == self.fail_leg:
            raise RuntimeError("broker rejected leg")
        return OrderResult(
            broker_order_id=f"ord-{len(self.placed)}",
            status="filled",
            filled_qty=request.quantity,
            avg_fill_price=self.fill_price,
        )

    async def get_order(self, broker_order_id):
        return {"status": self.leg_status}

    async def cancel_order(self, broker_order_id):
        if self.cancel_raises:
            raise RuntimeError("cancel rejected")
        self.cancelled.append(broker_order_id)
        return True


def _entry(order_type="market", limit_price=None):
    return OrderRequest(
        account_id="test", symbol="AAPL", side="buy", order_type=order_type,
        quantity=10, limit_price=limit_price, stop_price=None,
        time_in_force="GTC", execution_algo="market", risk_bucket="directional",
    )


# ── the confirmation filter ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_price_tolerance_breach_rejects_instead_of_submitting():
    """The guard used to crash on OrderResult(reason=...) and submit anyway."""
    broker = _Broker(quote_last=100.0)
    config = BracketOrderConfig(
        entry=_entry("limit", limit_price=130.0),   # 30% above market
        take_profit_pct=0.05, stop_loss_pct=0.02, price_tolerance=0.02,
    )

    res = await BracketOrder(broker).execute(config)

    assert broker.placed == [], "a rejected entry must never reach the broker"
    assert res.status == "rejected"
    assert res.raw_payload.get("reason") == "price_tolerance_exceeded"


@pytest.mark.asyncio
async def test_configured_tolerance_is_actually_honoured():
    """price_tolerance was read off OrderRequest (no such field) → always 0.02."""
    broker = _Broker(quote_last=100.0)
    config = BracketOrderConfig(
        entry=_entry("limit", limit_price=105.0),   # 5% out: over 2%, under 10%
        take_profit_pct=0.05, stop_loss_pct=0.02, price_tolerance=0.10,
    )

    res = await BracketOrder(broker).execute(config)

    assert res.status != "rejected", "a 10% tolerance must permit a 5% deviation"
    assert broker.placed, "entry should have been submitted"


@pytest.mark.asyncio
async def test_market_entry_skips_the_price_check():
    broker = _Broker(quote_last=100.0)
    config = BracketOrderConfig(entry=_entry("market"), take_profit_pct=0.05,
                                stop_loss_pct=0.02, price_tolerance=0.0)
    res = await BracketOrder(broker).execute(config)
    assert res.status != "rejected"


@pytest.mark.asyncio
async def test_dead_quote_feed_does_not_block_the_entry():
    """No quote means nothing to validate against — fail open, but say so."""
    broker = _Broker(quote_raises=True)
    config = BracketOrderConfig(entry=_entry("limit", limit_price=100.0),
                                take_profit_pct=0.05, stop_loss_pct=0.02)
    res = await BracketOrder(broker).execute(config)
    assert res.status != "rejected"
    assert broker.placed, "entry should still be submitted when the quote is missing"


# ── unprotected positions ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_invalid_tp_sl_is_flagged_unprotected(capsys):
    """tp <= sl → entry fills, no legs submitted. Must not look like success."""
    broker = _Broker(fill_price=100.0)
    config = BracketOrderConfig(
        entry=_entry("market"),
        take_profit_pct=-0.05,   # TP below entry, SL above → inverted
        stop_loss_pct=-0.02,
    )

    res = await BracketOrder(broker).execute(config)

    assert len(broker.placed) == 1, "only the entry should have been placed"
    assert res.raw_payload.get("bracket_unprotected") is True
    assert res.raw_payload.get("unprotected_reason") == "invalid_tp_sl_configuration"
    assert "NO take-profit and NO stop-loss" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_failed_oco_submission_is_flagged_unprotected(capsys):
    """The SL leg is rejected. The entry is already filled — report the position."""
    broker = _Broker(fill_price=100.0, fail_leg=2)   # placed[2] == SL leg
    config = BracketOrderConfig(entry=_entry("market"), take_profit_pct=0.05,
                                stop_loss_pct=0.02)

    res = await BracketOrder(broker).execute(config)   # must NOT raise

    assert res.raw_payload.get("bracket_unprotected") is True
    assert res.raw_payload.get("unprotected_reason") == "oco_submission_failed"
    assert res.raw_payload.get("intended_stop_loss") == pytest.approx(98.0)
    assert res.raw_payload.get("intended_take_profit") == pytest.approx(105.0)
    out = capsys.readouterr().out
    assert "NO stop-loss/take-profit protection" in out


@pytest.mark.asyncio
async def test_successful_bracket_is_not_flagged():
    broker = _Broker(fill_price=100.0)
    config = BracketOrderConfig(entry=_entry("market"), take_profit_pct=0.05,
                                stop_loss_pct=0.02)

    res = await BracketOrder(broker).execute(config)

    assert len(broker.placed) == 3, "entry + TP + SL"
    assert not (res.raw_payload or {}).get("bracket_unprotected")


# ── OCO leg hygiene ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_failed_leg_b_cancels_the_live_leg_a(capsys):
    """Leg A resting with no counterpart is not an OCO — it's an orphan."""
    broker = _Broker(fail_leg=1)     # placed[1] == order_b
    a, b = _entry("market"), _entry("market")

    with pytest.raises(RuntimeError):
        await OCOOrder(broker).execute(a, b)

    assert broker.cancelled == ["ord-1"], "leg A must be cancelled, not left live"
    assert "cancelling the already-live leg A" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_failed_leg_a_cancel_is_escalated(capsys):
    """If the cleanup cancel also fails, the live order must be named loudly."""
    broker = _Broker(fail_leg=1, cancel_raises=True)
    a, b = _entry("market"), _entry("market")

    with pytest.raises(RuntimeError):
        await OCOOrder(broker).execute(a, b)

    assert "is LIVE and unmanaged" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_oco_timeout_marks_the_result():
    """Both legs pulled → the caller's position is bare. Say so on the result."""
    broker = _Broker(leg_status="open")   # neither leg ever fills
    oco = OCOOrder(broker, poll_seconds=0, max_wait_seconds=0)

    res = await oco.execute(_entry("market"), _entry("market"))

    assert res.raw_payload.get("oco_timed_out") is True
    assert res.raw_payload.get("oco_cancel_failed_legs") == []
    assert set(broker.cancelled) == {"ord-1", "ord-2"}


@pytest.mark.asyncio
async def test_oco_timeout_records_which_cancels_failed():
    broker = _Broker(cancel_raises=True, leg_status="open")
    oco = OCOOrder(broker, poll_seconds=0, max_wait_seconds=0)

    res = await oco.execute(_entry("market"), _entry("market"))

    assert res.raw_payload.get("oco_cancel_failed_legs") == ["A", "B"]
