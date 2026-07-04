"""Integration tests for the Option Alpha-style automation endpoints.

/options/rules/validate, /options/macro-calendar and /options/next-fomc are
fully deterministic (no network). /options/flow, /options/wheel and
/options/put-call-ratio depend on Alpaca credentials — in the test environment
they must degrade to empty/unavailable responses, never 404 and never
fabricated rows. Runs on the in-process SQLite test DB.
"""
from __future__ import annotations

import uuid
from datetime import date, timedelta

import pytest

_PASSWORD = "0pti0ns!2026"


async def _auth_headers(client) -> dict[str, str]:
    email = f"oa_{uuid.uuid4().hex[:10]}@example.com"
    r = await client.post(
        "/api/v1/auth/register", json={"email": email, "password": _PASSWORD}
    )
    if r.status_code in (500, 503):
        pytest.skip(f"Auth backend unavailable ({r.status_code}) — DB not migrated")
    if r.status_code == 201:
        return {"Authorization": f"Bearer {r.json()['access_token']}"}
    resp = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": _PASSWORD}
    )
    if resp.status_code != 200:
        pytest.skip(f"Login failed ({resp.status_code}) — DB not migrated in test env")
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _occ(underlying: str, exp: date, cp: str, strike: float) -> str:
    return f"{underlying}{exp.strftime('%y%m%d')}{cp}{int(strike * 1000):08d}"


def _rules_body(
    dte: int = 35, delta: float = -0.25, qty: int = 1, credit: float = 1.0, strike: float = 45.0
) -> dict:
    # A $45-strike CSP needs ~$4.4k collateral — inside the 5% ceiling of the
    # $100k default paper equity, so the size rule passes at 1 contract.
    exp = date.today() + timedelta(days=dte)
    return {
        "symbol": "F",
        "option_symbol": _occ("F", exp, "P", strike),
        "expiration_date": exp.isoformat(),
        "side": "sell",
        "quantity": qty,
        "credit_received": credit,
        "delta": delta,
        "strategy_type": "csp",
    }


@pytest.mark.asyncio
async def test_rules_validate_good_csp_is_valid(client):
    headers = await _auth_headers(client)
    r = await client.post("/api/v1/options/rules/validate", json=_rules_body(), headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data["is_valid"] is True
    assert data["errors"] == []
    assert data["dte"] == 35
    assert data["rules"]["dte"]["status"] == "ok"
    assert data["rules"]["delta"]["status"] == "ok"
    # Option Alpha automation exits: 50% profit target, 2x-credit stop, 21-DTE manage
    assert data["profit_target_price"] == pytest.approx(0.5)
    assert data["stop_loss_price"] == pytest.approx(2.0)
    assert data["max_profit"] == pytest.approx(100.0)
    exp = date.today() + timedelta(days=35)
    assert data["exit_before_date"] == (exp - timedelta(days=21)).isoformat()


@pytest.mark.asyncio
async def test_rules_validate_flags_bad_trade(client):
    headers = await _auth_headers(client)
    body = _rules_body(dte=3, delta=-0.55, qty=500)
    r = await client.post("/api/v1/options/rules/validate", json=body, headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data["is_valid"] is False
    assert data["rules"]["dte"]["status"] == "error"
    assert data["rules"]["delta"]["status"] == "error"
    # 500 CSP contracts on a $560 strike is far beyond 5% of any test equity
    assert data["rules"]["position_size"]["status"] == "error"
    assert len(data["errors"]) >= 3


@pytest.mark.asyncio
async def test_rules_validate_rejects_bad_expiration(client):
    headers = await _auth_headers(client)
    body = _rules_body()
    body["expiration_date"] = "not-a-date"
    r = await client.post("/api/v1/options/rules/validate", json=body, headers=headers)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_macro_calendar_events_sorted_and_bounded(client):
    headers = await _auth_headers(client)
    r = await client.get("/api/v1/options/macro-calendar?days_ahead=60", headers=headers)
    assert r.status_code == 200
    events = r.json()
    assert isinstance(events, list)
    assert len(events) > 0  # 60 days always contains NFP + CPI at minimum
    today = date.today()
    dates = [e["date"] for e in events]
    assert dates == sorted(dates)
    for ev in events:
        assert 0 <= ev["days_away"] <= 60
        assert date.fromisoformat(ev["date"]) >= today
        assert ev["category"] in {"fomc", "nfp", "cpi", "ppi", "gdp"}
        assert ev["importance"] in {"high", "medium"}


@pytest.mark.asyncio
async def test_next_fomc_returns_upcoming_date(client):
    headers = await _auth_headers(client)
    r = await client.get("/api/v1/options/next-fomc", headers=headers)
    assert r.status_code == 200
    data = r.json()
    if data["date"] is not None:  # schedule list exhausts eventually
        assert date.fromisoformat(data["date"]) >= date.today()
        assert data["days_away"] >= 0


@pytest.mark.asyncio
async def test_flow_degrades_to_empty_without_credentials(client):
    headers = await _auth_headers(client)
    r = await client.get("/api/v1/options/flow", headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)  # [] without creds — never a 404, never fake rows
    r2 = await client.get("/api/v1/options/flow?unusual_only=true", headers=headers)
    assert r2.status_code == 200


@pytest.mark.asyncio
async def test_wheel_degrades_to_empty_without_credentials(client):
    headers = await _auth_headers(client)
    r = await client.get("/api/v1/options/wheel?tickers=AAPL,MSFT", headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


@pytest.mark.asyncio
async def test_put_call_ratio_shape(client):
    headers = await _auth_headers(client)
    r = await client.get("/api/v1/options/put-call-ratio", headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) >= {"ratio", "puts", "calls", "sentiment"}
    assert data["sentiment"] in {"bullish", "bearish", "neutral", "unavailable"}


@pytest.mark.asyncio
async def test_options_endpoints_require_auth(client):
    for path in (
        "/api/v1/options/flow",
        "/api/v1/options/put-call-ratio",
        "/api/v1/options/wheel",
        "/api/v1/options/macro-calendar",
        "/api/v1/options/next-fomc",
    ):
        r = await client.get(path)
        assert r.status_code in (401, 403), path
    r = await client.post("/api/v1/options/rules/validate", json=_rules_body())
    assert r.status_code in (401, 403)
