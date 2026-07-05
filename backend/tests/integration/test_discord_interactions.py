"""Tests for the Discord interactions endpoint.

The Ed25519 signature IS the auth on this route, so the tests generate a real
keypair, monkeypatch the module's public key, and sign requests exactly the
way Discord does (timestamp + raw body). Unsigned/forged requests must 401 —
that's what Discord's own endpoint validation checks first.
"""
from __future__ import annotations

import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

import app.api.v1.discord_interactions as di

_PRIV = Ed25519PrivateKey.generate()
_PUB_HEX = _PRIV.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()

URL = "/api/v1/discord/interactions"


def _signed_headers(body: bytes, timestamp: str = "1700000000") -> dict[str, str]:
    sig = _PRIV.sign(timestamp.encode() + body).hex()
    return {
        "X-Signature-Ed25519": sig,
        "X-Signature-Timestamp": timestamp,
        "Content-Type": "application/json",
    }


@pytest.fixture(autouse=True)
def _use_test_key(monkeypatch):
    monkeypatch.setattr(di, "_PUBLIC_KEY_HEX", _PUB_HEX)


@pytest.mark.asyncio
async def test_ping_pong_with_valid_signature(client):
    body = json.dumps({"type": 1}).encode()
    r = await client.post(URL, content=body, headers=_signed_headers(body))
    assert r.status_code == 200
    assert r.json() == {"type": 1}


@pytest.mark.asyncio
async def test_unsigned_request_is_401(client):
    r = await client.post(URL, json={"type": 1})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_forged_signature_is_401(client):
    body = json.dumps({"type": 1}).encode()
    other = Ed25519PrivateKey.generate()
    headers = {
        "X-Signature-Ed25519": other.sign(b"1700000000" + body).hex(),
        "X-Signature-Timestamp": "1700000000",
        "Content-Type": "application/json",
    }
    r = await client.post(URL, content=body, headers=headers)
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_tampered_body_is_401(client):
    body = json.dumps({"type": 1}).encode()
    headers = _signed_headers(body)
    tampered = json.dumps({"type": 2}).encode()
    r = await client.post(URL, content=tampered, headers=headers)
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_status_command_answers(client):
    body = json.dumps({"type": 2, "data": {"name": "status"}}).encode()
    r = await client.post(URL, content=body, headers=_signed_headers(body))
    assert r.status_code == 200
    data = r.json()
    assert data["type"] == 4
    assert "QuantEdge status" in data["data"]["content"]


@pytest.mark.asyncio
async def test_pnl_command_answers(client):
    body = json.dumps({"type": 2, "data": {"name": "pnl"}}).encode()
    r = await client.post(URL, content=body, headers=_signed_headers(body))
    assert r.status_code == 200
    assert "Equity" in r.json()["data"]["content"]


@pytest.mark.asyncio
async def test_health_command_answers(client):
    body = json.dumps({"type": 2, "data": {"name": "health"}}).encode()
    r = await client.post(URL, content=body, headers=_signed_headers(body))
    assert r.status_code == 200
    assert "database" in r.json()["data"]["content"]


@pytest.mark.asyncio
async def test_run_bot_requires_name(client):
    body = json.dumps({"type": 2, "data": {"name": "run-bot", "options": []}}).encode()
    r = await client.post(URL, content=body, headers=_signed_headers(body))
    assert r.status_code == 200
    assert "Usage" in r.json()["data"]["content"]


@pytest.mark.asyncio
async def test_run_bot_no_match_is_graceful(client):
    body = json.dumps({
        "type": 2,
        "data": {"name": "run-bot", "options": [{"name": "name", "value": "zzz-no-such-bot"}]},
    }).encode()
    r = await client.post(URL, content=body, headers=_signed_headers(body))
    assert r.status_code == 200
    assert "No enabled bot" in r.json()["data"]["content"]


@pytest.mark.asyncio
async def test_unknown_command_is_ephemeral(client):
    body = json.dumps({"type": 2, "data": {"name": "nonsense"}}).encode()
    r = await client.post(URL, content=body, headers=_signed_headers(body))
    assert r.status_code == 200
    data = r.json()
    assert data["data"].get("flags") == 64
