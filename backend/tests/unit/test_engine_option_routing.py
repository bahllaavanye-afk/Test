"""Engine routing for open_option_spread (issue #191).

Proves paper-first is airtight: a real TradeStation order is placed ONLY for a
live TS account with explicit-strike legs; every other case stays alert-only
(returns None). No DB, no network — fakes the session and broker.
"""
from types import SimpleNamespace

import pytest

from app.bots.engine import BotEngine
from app.brokers.base import OrderResult
from app.schemas.bot import OptionLeg


class _FakeResult:
    def __init__(self, obj):
        self._obj = obj

    def scalar_one_or_none(self):
        return self._obj


class _FakeDB:
    def __init__(self, account):
        self._account = account

    async def execute(self, *a, **k):
        return _FakeResult(self._account)


def _bot(account_id="acc1"):
    return SimpleNamespace(id="bot1", account_id=account_id, symbol="SPY")


def _account(mode="live", broker="tradestation", creds=True):
    return SimpleNamespace(
        id="acc1",
        mode=mode,
        broker=broker,
        encrypted_key="ek" if creds else None,
        encrypted_secret="es" if creds else None,
        extra_config={"tradestation_account_id": "TS1"} if creds else {},
    )


_STRIKE_LEGS = [
    OptionLeg(side="sell", option_type="put", strike=440, dte=30, ratio=1),
    OptionLeg(side="buy", option_type="put", strike=430, dte=30, ratio=1),
]
_DELTA_LEGS = [OptionLeg(side="sell", option_type="put", delta=0.25, dte=30, ratio=1)]


@pytest.mark.asyncio
async def test_paper_account_never_routes(monkeypatch):
    eng = BotEngine()
    db = _FakeDB(_account(mode="paper"))
    assert await eng._route_option_spread(_bot(), _STRIKE_LEGS, db) is None


@pytest.mark.asyncio
async def test_live_non_tradestation_never_routes():
    eng = BotEngine()
    db = _FakeDB(_account(mode="live", broker="alpaca"))
    assert await eng._route_option_spread(_bot(), _STRIKE_LEGS, db) is None


@pytest.mark.asyncio
async def test_no_account_returns_none():
    eng = BotEngine()
    db = _FakeDB(None)
    assert await eng._route_option_spread(_bot(account_id=None), _STRIKE_LEGS, db) is None


@pytest.mark.asyncio
async def test_live_ts_delta_legs_not_routed_yet():
    eng = BotEngine()
    db = _FakeDB(_account())
    # delta legs (no explicit strike) require a chain lookup → stays alert-only
    assert await eng._route_option_spread(_bot(), _DELTA_LEGS, db) is None


@pytest.mark.asyncio
async def test_live_ts_missing_creds_returns_none():
    eng = BotEngine()
    db = _FakeDB(_account(creds=False))
    assert await eng._route_option_spread(_bot(), _STRIKE_LEGS, db) is None


@pytest.mark.asyncio
async def test_live_ts_explicit_strikes_routes(monkeypatch):
    eng = BotEngine()
    db = _FakeDB(_account())

    import app.utils.security as sec
    monkeypatch.setattr(sec, "decrypt_secret", lambda v: "decrypted")

    captured = {}

    async def _fake_place(self, legs, quantity=1, order_type="market", limit_price=None, *, opening=True):
        captured["legs"] = legs
        captured["quantity"] = quantity
        return OrderResult(broker_order_id="OID42", status="queued", filled_qty=0.0, avg_fill_price=None)

    from app.brokers.tradestation import TradeStationBroker
    monkeypatch.setattr(TradeStationBroker, "place_option_order", _fake_place)

    oid = await eng._route_option_spread(_bot(), _STRIKE_LEGS, db)
    assert oid == "OID42"
    # legs were built into broker payloads with proper option symbols
    assert len(captured["legs"]) == 2
    assert captured["legs"][0]["symbol"].startswith("SPY ")
    assert captured["legs"][0]["side"] == "sell"
