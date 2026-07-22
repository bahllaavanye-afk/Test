"""notify.read_channel_recent — the CONSUME side of desk posts.

Desk run summaries land durably in Discord; this reads them back (stateless,
via the bot API) so the company brain can ground the morning discussion in real
desk outcomes instead of committing desk state to git. These tests mock the
bot-API layer; no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import notify  # noqa: E402


def _setup(monkeypatch, messages):
    monkeypatch.setattr(notify, "_BOT_TOKEN", "x")
    monkeypatch.setattr(notify, "_load_channel_ids", lambda: {"pnl-daily": "123"})
    monkeypatch.setattr(notify, "_bot_req", lambda m, p, b=None: messages)


def test_reads_recent_messages(monkeypatch):
    _setup(monkeypatch, [
        {"content": "crypto desk — 3 orders", "author": {"username": "DeskBot"}, "timestamp": "t1"},
        {"content": "fx desk — quiet", "author": {"username": "DeskBot"}, "timestamp": "t2"},
    ])
    out = notify.read_channel_recent("#pnl-daily", limit=10)
    assert [m["content"] for m in out] == ["crypto desk — 3 orders", "fx desk — quiet"]
    assert out[0]["author"] == "DeskBot" and out[0]["ts"] == "t1"


def test_strips_bold_username_prefix(monkeypatch):
    _setup(monkeypatch, [{"content": "**QuantEdge Desk** funnel: 49→5 placed", "author": {}}])
    out = notify.read_channel_recent("#pnl-daily")
    assert out[0]["content"] == "funnel: 49→5 placed"
    assert out[0]["author"] == "?"  # missing username → sentinel


def test_empty_messages_are_dropped(monkeypatch):
    _setup(monkeypatch, [{"content": "   "}, {"content": "real post"}])
    out = notify.read_channel_recent("#pnl-daily")
    assert [m["content"] for m in out] == ["real post"]


def test_no_bot_token_returns_empty(monkeypatch):
    monkeypatch.setattr(notify, "_BOT_TOKEN", "")
    assert notify.read_channel_recent("#pnl-daily") == []


def test_unknown_channel_returns_empty(monkeypatch):
    monkeypatch.setattr(notify, "_BOT_TOKEN", "x")
    monkeypatch.setattr(notify, "_load_channel_ids", lambda: {})
    assert notify.read_channel_recent("#nope") == []


def test_fetch_failure_fails_soft(monkeypatch):
    monkeypatch.setattr(notify, "_BOT_TOKEN", "x")
    monkeypatch.setattr(notify, "_load_channel_ids", lambda: {"pnl-daily": "123"})
    def boom(m, p, b=None):
        raise RuntimeError("discord 500")
    monkeypatch.setattr(notify, "_bot_req", boom)
    assert notify.read_channel_recent("#pnl-daily") == []


def test_limit_is_clamped(monkeypatch):
    seen = {}
    monkeypatch.setattr(notify, "_BOT_TOKEN", "x")
    monkeypatch.setattr(notify, "_load_channel_ids", lambda: {"pnl-daily": "123"})
    def capture(m, p, b=None):
        seen["path"] = p
        return []
    monkeypatch.setattr(notify, "_bot_req", capture)
    notify.read_channel_recent("#pnl-daily", limit=500)  # over the 100 cap
    assert "limit=100" in seen["path"]
    notify.read_channel_recent("#pnl-daily", limit=0)    # under the floor
    assert "limit=1" in seen["path"]
