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

# Constants
BOT_ID = "bot1"
BOT_DEFAULT_ACCOUNT_ID = "acc1"
BOT_SYMBOL = "SPY"

ACCOUNT_ID = "acc1"
ACCOUNT_MODE_LIVE = "live"
ACCOUNT_MODE_PAPER = "paper"
ACCOUNT_BROKER_TRADE_STATION = "tradestation"
ACCOUNT_BROKER_ALPACA = "alpaca"
ACCOUNT_ENCRYPTED_KEY = "ek"
ACCOUNT_ENCRYPTED_SECRET = "es"
ACCOUNT_EXTRA_CONFIG_KEY = "tradestation_account_id"
ACCOUNT_EXTRA_CONFIG_VALUE = "TS1"

STRIKE_PRICE_SELL = 440
STRIKE_PRICE_BUY = 430
DTE_30 = 30
RATIO_1 = 1
DELTA_VALUE = 0.25

SIDE_SELL = "sell"
SIDE_BUY = "buy"
OPTION_TYPE_PUT = "put"

PLACE_ORDER_TYPE = "market"
PLACE_ORDER_QUANTITY = 1

MOCK_ORDER_ID = "OID42"
MOCK_ORDER_STATUS = "queued"


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


def _bot(account_id=BOT_DEFAULT_ACCOUNT_ID):
    return SimpleNamespace(id=BOT_ID, account_id=account_id, symbol=BOT_SYMBOL)


def _account(mode=ACCOUNT_MODE_LIVE, broker=ACCOUNT_BROKER_TRADE_STATION, creds=True):
    return SimpleNamespace(
        id=ACCOUNT_ID,
        mode=mode,
        broker=broker,
        encrypted_key=ACCOUNT_ENCRYPTED_KEY if creds else None,
        encrypted_secret=ACCOUNT_ENCRYPTED_SECRET if creds else None,
        extra_config={ACCOUNT_EXTRA_CONFIG_KEY: ACCOUNT_EXTRA_CONFIG_VALUE} if creds else {},
    )


_STRIKE_LEGS = [
    OptionLeg(side=SIDE_SELL, option_type=OPTION_TYPE_PUT, strike=STRIKE_PRICE_SELL, dte=DTE_30, ratio=RATIO_1),
    OptionLeg(side=SIDE_BUY, option_type=OPTION_TYPE_PUT, strike=STRIKE_PRICE_BUY, dte=DTE_30, ratio=RATIO_1),
]
_DELTA_LEGS = [OptionLeg(side=SIDE_SELL, option_type=OPTION_TYPE_PUT, delta=DELTA_VALUE, dte=DTE_30, ratio=RATIO_1)]


@pytest.mark.asyncio
async def test_paper_account_never_routes(monkeypatch):
    eng = BotEngine()
    db = _FakeDB(_account(mode=ACCOUNT_MODE_PAPER))
    assert await eng._route_option_spread(_bot(), _STRIKE_LEGS, db) is None


@pytest.mark.asyncio
async def test_live_non_tradestation_never_routes():
    eng = BotEngine()
    db = _FakeDB(_account(mode=ACCOUNT_MODE_LIVE, broker=ACCOUNT_BROKER_ALPACA))
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

    async def _fake_place(self, legs, quantity=PLACE_ORDER_QUANTITY, order_type=PLACE_ORDER_TYPE, limit_price=None, *, opening=True):
        captured["legs"] = legs
        captured["quantity"] = quantity
        return OrderResult(broker_order_id=MOCK_ORDER_ID, status=MOCK_ORDER_STATUS, filled_qty=0.0, avg_fill_price=None)

    from app.brokers.tradestation import TradeStationBroker
    monkeypatch.setattr(TradeStationBroker, "place_option_order", _fake_place)

    oid = await eng._route_option_spread(_bot(), _STRIKE_LEGS, db)
    assert oid == MOCK_ORDER_ID
    # legs were built into broker payloads with proper option symbols
    assert len(captured["legs"]) == 2
    assert captured["legs"][0]["symbol"].startswith(f"{BOT_SYMBOL} ")
    assert captured["legs"][0]["side"] == SIDE_SELL