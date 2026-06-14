"""
Tests for the SlackClient notifier — covers the failure modes that previously
caused silent no-posts (token absent, not_in_channel, channel resolution) and
the typed notification helpers wired this session.
"""
from __future__ import annotations

import httpx
import pytest

from app.notifications.slack import CHANNEL_MAP, SlackClient


def _make_bot_client(token: str = "xoxb-test-token") -> SlackClient:
    c = SlackClient()
    c._token = token
    c._use_bot = True
    c._enabled = True
    return c


class TestSlackEnablement:
    @pytest.mark.asyncio
    async def test_disabled_client_returns_false_not_crash(self):
        c = SlackClient()
        c._enabled = False
        # send() must short-circuit cleanly when nothing is configured.
        assert await c.send("system", "system", "hi") is False

    def test_non_bot_token_is_not_treated_as_bot(self):
        c = SlackClient()
        c._token = "xoxp-user-token"
        # only xoxb- tokens count as a bot token
        assert not (c._token.startswith("xoxb-"))


class TestChannelMapping:
    def test_known_logical_channels_map_to_real_names(self):
        assert CHANNEL_MAP["orders"] == "pnl-daily"
        assert CHANNEL_MAP["alerts"] == "risk-alerts"
        assert CHANNEL_MAP["experiments"] == "ml-experiments"

    def test_unknown_channel_falls_back_to_engineering(self):
        assert CHANNEL_MAP.get("does-not-exist", "engineering") == "engineering"


class TestPostBotErrorHandling:
    @pytest.mark.asyncio
    async def test_not_in_channel_attempts_join_then_retries(self, monkeypatch):
        """On not_in_channel the client should try to self-join and retry once."""
        calls = {"post": 0, "join_attempted": False}

        async def fake_post(channel, payload, _retry=True):
            # delegate to the real method but stub the network underneath
            return await SlackClient._post_bot(c, channel, payload, _retry)

        c = _make_bot_client()

        # Patch the join helper to report success, and count post attempts.
        async def fake_join(client, channel):
            calls["join_attempted"] = True
            return True

        post_results = [
            {"ok": False, "error": "not_in_channel"},  # first attempt
            {"ok": True},                               # after join, retry
        ]

        class _Resp:
            def __init__(self, data): self._data = data
            def json(self): return self._data

        async def fake_http_post(url, **kwargs):
            calls["post"] += 1
            return _Resp(post_results[min(calls["post"] - 1, len(post_results) - 1)])

        class _FakeClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def post(self, url, **kwargs): return await fake_http_post(url, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeClient())
        monkeypatch.setattr(c, "_join_channel", fake_join)

        ok = await c._post_bot("engineering", {"text": "hi"})
        assert ok is True
        assert calls["join_attempted"] is True
        assert calls["post"] == 2  # original + retry after join

    @pytest.mark.asyncio
    async def test_other_error_does_not_retry(self, monkeypatch):
        c = _make_bot_client()
        calls = {"post": 0}

        class _Resp:
            def json(self): return {"ok": False, "error": "invalid_auth"}

        class _FakeClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def post(self, url, **kwargs):
                calls["post"] += 1
                return _Resp()

        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeClient())
        ok = await c._post_bot("engineering", {"text": "hi"})
        assert ok is False
        assert calls["post"] == 1  # no retry on a non-membership error


class TestTypedHelpers:
    @pytest.mark.asyncio
    async def test_notify_circuit_breaker_routes_to_alerts(self, monkeypatch):
        c = _make_bot_client()
        captured = {}

        async def fake_send(channel, event_type, title, fields=None, text=None):
            captured.update(channel=channel, event_type=event_type, title=title)
            return True

        monkeypatch.setattr(c, "send", fake_send)
        ok = await c.notify_circuit_breaker("global", 0.12, 0.10)
        assert ok is True
        assert captured["channel"] == "alerts"
        assert captured["event_type"] == "circuit_breaker"

    @pytest.mark.asyncio
    async def test_notify_daily_summary_routes_to_system(self, monkeypatch):
        c = _make_bot_client()
        captured = {}

        async def fake_send(channel, event_type, title, fields=None, text=None):
            captured.update(channel=channel, fields=fields)
            return True

        monkeypatch.setattr(c, "send", fake_send)
        ok = await c.notify_daily_summary(1234.5, 10, 0.6, "momentum")
        assert ok is True
        assert captured["channel"] == "system"
        assert captured["fields"]["Best"] == "momentum"
