"""
Unit tests for SlackBot.audit_channels — the channel-health audit that finds
what's not working. No network: we stub list_channels and channel_history.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.integrations.slack_bot import SlackBot


def _make_bot() -> SlackBot:
    # Bypass __post_init__ token validation + httpx client construction.
    bot = SlackBot.__new__(SlackBot)
    bot.token = "xoxb-test"
    bot._client = None
    return bot


def _msg(text: str, age_hours: float) -> dict:
    return {"text": text, "ts": str(time.time() - age_hours * 3600)}


class TestAuditChannels:
    def test_missing_channel(self, monkeypatch):
        bot = _make_bot()
        monkeypatch.setattr(bot, "list_channels", lambda include_private=True: [])
        report = bot.audit_channels(names=["does-not-exist"])
        assert report["summary"]["missing"] == 1
        assert report["channels"][0]["status"] == "missing"

    def test_unreadable_when_not_member(self, monkeypatch):
        bot = _make_bot()
        monkeypatch.setattr(bot, "list_channels",
                            lambda include_private=True: [{"name": "risk-alerts", "id": "C1", "is_member": False}])
        report = bot.audit_channels(names=["risk-alerts"])
        assert report["channels"][0]["status"] == "unreadable"
        assert report["summary"]["unreadable"] == 1

    def test_silent_when_no_messages(self, monkeypatch):
        bot = _make_bot()
        monkeypatch.setattr(bot, "list_channels",
                            lambda include_private=True: [{"name": "wins", "id": "C1", "is_member": True}])
        monkeypatch.setattr(bot, "channel_history", lambda channel, limit=50: [])
        report = bot.audit_channels(names=["wins"])
        assert report["channels"][0]["status"] == "silent"

    def test_healthy_recent_activity(self, monkeypatch):
        bot = _make_bot()
        monkeypatch.setattr(bot, "list_channels",
                            lambda include_private=True: [{"name": "engineering", "id": "C1", "is_member": True}])
        monkeypatch.setattr(bot, "channel_history",
                            lambda channel, limit=50: [_msg("shipped the new feature", 1.0)])
        report = bot.audit_channels(names=["engineering"])
        assert report["channels"][0]["status"] == "healthy"

    def test_stale_when_old(self, monkeypatch):
        bot = _make_bot()
        monkeypatch.setattr(bot, "list_channels",
                            lambda include_private=True: [{"name": "deploys", "id": "C1", "is_member": True}])
        monkeypatch.setattr(bot, "channel_history",
                            lambda channel, limit=50: [_msg("old deploy", 100.0)])
        report = bot.audit_channels(names=["deploys"], stale_after_hours=24.0)
        assert report["channels"][0]["status"] == "stale"

    def test_flagged_on_error_keywords(self, monkeypatch):
        bot = _make_bot()
        monkeypatch.setattr(bot, "list_channels",
                            lambda include_private=True: [{"name": "ci-failures", "id": "C1", "is_member": True}])
        monkeypatch.setattr(bot, "channel_history",
                            lambda channel, limit=50: [_msg("build failed with traceback", 1.0)])
        report = bot.audit_channels(names=["ci-failures"])
        assert report["channels"][0]["status"] == "flagged"
        assert report["channels"][0]["flagged_messages"] == 1

    def test_summary_counts_add_up(self, monkeypatch):
        bot = _make_bot()
        chans = [
            {"name": "a", "id": "C1", "is_member": True},
            {"name": "b", "id": "C2", "is_member": False},
        ]
        monkeypatch.setattr(bot, "list_channels", lambda include_private=True: chans)
        monkeypatch.setattr(bot, "channel_history",
                            lambda channel, limit=50: [_msg("all good", 1.0)])
        report = bot.audit_channels(names=["a", "b"])
        s = report["summary"]
        assert s["total"] == 2
        assert s["healthy"] + s["unreadable"] == 2


class TestChannelHistory:
    def test_history_unknown_channel_returns_empty(self, monkeypatch):
        bot = _make_bot()
        monkeypatch.setattr(bot, "find_channel_id", lambda name, channels=None: None)
        assert bot.channel_history("nope") == []
