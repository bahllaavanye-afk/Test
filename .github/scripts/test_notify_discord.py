"""Guards for Discord bot-API routing in notify.py.

The channel-map load 403'd because bot API calls used a browser User-Agent;
Discord's REST API requires "DiscordBot (url, version)". These tests pin that
the bot path uses the compliant UA (webhooks keep the browser UA) so the
routing can't silently regress back into the #general dump.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def notify(monkeypatch):
    # give the module a bot token so the bot path is active, then (re)import
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/x/y")
    import notify as _notify
    importlib.reload(_notify)
    return _notify


def test_bot_ua_is_discord_compliant(notify):
    assert notify._BOT_UA.startswith("DiscordBot ("), notify._BOT_UA
    assert "Mozilla" not in notify._BOT_UA


def test_bot_req_sends_discordbot_ua_not_browser(notify, monkeypatch):
    """_bot_req must send the DiscordBot UA — a browser UA is what 403'd."""
    captured = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"[]"

    def fake_urlopen(req, timeout=15):
        captured["ua"] = req.get_header("User-agent")
        captured["auth"] = req.get_header("Authorization")
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    notify._bot_req("GET", "/users/@me/guilds")
    assert captured["ua"].startswith("DiscordBot ("), captured["ua"]
    assert captured["auth"] == "Bot test-token"


def test_guild_ids_prefers_pinned_env(notify, monkeypatch):
    monkeypatch.setattr(notify, "_GUILD_ID", "999")
    # with a pinned guild id we must NOT call /users/@me/guilds
    def boom(*a, **k):
        raise AssertionError("should not enumerate guilds when DISCORD_GUILD_ID is set")
    monkeypatch.setattr(notify, "_bot_req", boom)
    assert notify._guild_ids() == ["999"]


def test_channel_routing_falls_back_to_general_prefix_when_unresolved(notify, monkeypatch):
    """When the bot can't resolve a channel, the webhook fallback keeps the
    [#channel] prefix so nothing is silently lost."""
    monkeypatch.setattr(notify, "_load_channel_ids", lambda: {})   # nothing resolves
    sent = {}

    def fake_urlopen(req, timeout=15):
        sent["url"] = req.full_url
        sent["body"] = req.data
        class _R:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b"{}"
        return _R()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    notify.stats["discord_ok"] = 0
    ok = notify.discord_post("alpha-research", "hello", username="Data Team")
    assert ok is True
    assert b"alpha-research" in sent["body"]  # prefix preserved on fallback
