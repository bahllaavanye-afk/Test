"""notify.post_dedup — suppress desk messages identical to a recent channel post.

Stateless dedup (reads channel history via the bot API), so no git-committed
state. These tests mock the bot-API layer; no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import notify  # noqa: E402


def _setup(monkeypatch, recent_bodies):
    monkeypatch.setattr(notify, "_BOT_TOKEN", "x")
    monkeypatch.setattr(notify, "_load_channel_ids", lambda: {"desk-fx-rates": "123"})
    monkeypatch.setattr(notify, "_bot_req",
                        lambda m, p, b=None: [{"content": c} for c in recent_bodies])


def test_duplicate_is_detected(monkeypatch):
    _setup(monkeypatch, ["FX desk — quiet: 10 signals, 0 orders"])
    assert notify._is_recent_duplicate("#desk-fx-rates", "FX desk — quiet: 10 signals, 0 orders")


def test_duplicate_ignores_bold_username_prefix(monkeypatch):
    _setup(monkeypatch, ["**QuantEdge FX Desk** FX desk — quiet: 10 signals, 0 orders"])
    assert notify._is_recent_duplicate("#desk-fx-rates", "FX desk — quiet: 10 signals, 0 orders")


def test_new_message_is_not_duplicate(monkeypatch):
    _setup(monkeypatch, ["FX desk — quiet: 10 signals, 0 orders"])
    assert not notify._is_recent_duplicate(
        "#desk-fx-rates", "FX desk — 2 order(s): EUR_USD BUY@1.08 …")


def test_no_bot_token_fails_open(monkeypatch):
    monkeypatch.setattr(notify, "_BOT_TOKEN", "")
    assert not notify._is_recent_duplicate("#desk-fx-rates", "anything")


def test_post_dedup_skips_duplicate(monkeypatch):
    _setup(monkeypatch, ["same message"])
    posted = {"n": 0}
    monkeypatch.setattr(notify, "post", lambda *a, **k: posted.__setitem__("n", posted["n"] + 1) or True)
    out = notify.post_dedup("#desk-fx-rates", "same message")
    assert out is False and posted["n"] == 0


def test_post_dedup_posts_new(monkeypatch):
    _setup(monkeypatch, ["old message"])
    posted = {"n": 0}
    monkeypatch.setattr(notify, "post", lambda *a, **k: posted.__setitem__("n", posted["n"] + 1) or True)
    out = notify.post_dedup("#desk-fx-rates", "brand new message")
    assert out is True and posted["n"] == 1
