"""Unit tests for TradeStation options response parsing (issue #195).

The request *builders* are tested in test_bot_options.py. These cover the async
methods' response parsing with a mocked httpx client — no live API, no creds.
"""
import httpx
import pytest

from app.brokers import tradestation as ts_mod
from app.brokers.tradestation import TradeStationBroker


class _FakeResp:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)  # type: ignore[arg-type]


class _FakeClient:
    """Stand-in for httpx.AsyncClient as an async context manager."""
    def __init__(self, resp: _FakeResp, captured: dict):
        self._resp = resp
        self._cap = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, params=None, headers=None):
        self._cap.update(method="GET", url=url, params=params, headers=headers)
        return self._resp

    async def post(self, url, json=None, headers=None):
        self._cap.update(method="POST", url=url, json=json, headers=headers)
        return self._resp


def _broker():
    b = TradeStationBroker("cid", "csecret", "ACC123", paper=True)
    return b


def _patch(monkeypatch, broker, resp, captured):
    monkeypatch.setattr(ts_mod.httpx, "AsyncClient", lambda *a, **k: _FakeClient(resp, captured))

    async def _fake_headers():
        return {"Authorization": "Bearer x"}

    monkeypatch.setattr(broker, "_headers", _fake_headers)


@pytest.mark.asyncio
async def test_get_option_chain_returns_options_list(monkeypatch):
    b = _broker()
    cap: dict = {}
    _patch(monkeypatch, b, _FakeResp({"Options": [{"Strike": 440}, {"Strike": 445}]}), cap)
    out = await b.get_option_chain("spy")
    assert [o["Strike"] for o in out] == [440, 445]
    assert cap["method"] == "GET"
    assert "marketdata/options/chains/SPY" in cap["url"]


@pytest.mark.asyncio
async def test_get_option_chain_legs_fallback_and_expiration_format(monkeypatch):
    from datetime import date
    b = _broker()
    cap: dict = {}
    _patch(monkeypatch, b, _FakeResp({"Legs": [{"Strike": 100}]}), cap)
    out = await b.get_option_chain("AAPL", expiration=date(2024, 3, 15))
    assert out == [{"Strike": 100}]               # falls back to Legs when no Options key
    assert cap["params"]["expiration"] == "03-15-2024"  # %m-%d-%Y


@pytest.mark.asyncio
async def test_place_option_order_parses_fields(monkeypatch):
    b = _broker()
    cap: dict = {}
    resp = _FakeResp({"OrderID": "OID9", "Message": "Queued", "FilledQuantity": "2", "AveragePrice": "1.25"})
    _patch(monkeypatch, b, resp, cap)
    legs = [
        {"symbol": "SPY 240119P440", "side": "sell", "ratio": 1},
        {"symbol": "SPY 240119P430", "side": "buy", "ratio": 1},
    ]
    res = await b.place_option_order(legs, quantity=2)
    assert res.broker_order_id == "OID9"
    assert res.status == "queued"
    assert res.filled_qty == 2.0
    assert res.avg_fill_price == 1.25
    # body was built via build_option_order_body
    assert cap["json"]["AccountID"] == "ACC123"
    assert len(cap["json"]["Legs"]) == 2
    assert cap["json"]["Legs"][0]["TradeAction"] == "SELLTOOPEN"


@pytest.mark.asyncio
async def test_place_option_order_handles_missing_average_price(monkeypatch):
    b = _broker()
    cap: dict = {}
    # No AveragePrice / FilledQuantity in response — must not raise TypeError
    _patch(monkeypatch, b, _FakeResp({"OrderID": "OID0", "Message": "Received"}), cap)
    res = await b.place_option_order([{"symbol": "QQQ 240119C400", "side": "buy"}], quantity=1)
    assert res.broker_order_id == "OID0"
    assert res.avg_fill_price is None
    assert res.filled_qty == 0.0
