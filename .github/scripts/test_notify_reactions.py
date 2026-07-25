"""Discord message-id return + reactions.

Agent acknowledgements (👀 on receipt, ✅ on success, ❌ on failure) were silent
no-ops: `_post_to_channel_id` discarded the created message's id, and
`chat_call` returned `ts: ""`, so nothing could ever be reacted to. These tests
pin the id round-trip and the reaction call so it cannot regress back to a no-op.
"""
from __future__ import annotations

import importlib.util
import sys
import urllib.parse
from pathlib import Path

import pytest

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

import notify  # noqa: E402


# ── message id round-trip ────────────────────────────────────────────────────

def test_post_to_channel_id_returns_the_created_message_id(monkeypatch):
    seen = {}

    def fake_bot_req(method, path, body=None):
        seen["method"], seen["path"], seen["body"] = method, path, body
        return {"id": "1234567890"}

    monkeypatch.setattr(notify, "_bot_req", fake_bot_req)
    mid = notify._post_to_channel_id_returning("chan1", "hello", "Alex")
    assert mid == "1234567890"
    assert seen["method"] == "POST"
    assert seen["path"] == "/channels/chan1/messages"
    # employee identity rides a bold prefix (bots can't set per-message usernames)
    assert seen["body"]["content"].startswith("**Alex** ")


def test_bool_wrapper_still_works_for_existing_callers(monkeypatch):
    monkeypatch.setattr(notify, "_bot_req", lambda *a, **k: {"id": "42"})
    assert notify._post_to_channel_id("c", "t", "QuantEdge") is True
    monkeypatch.setattr(notify, "_bot_req", lambda *a, **k: {})
    assert notify._post_to_channel_id("c", "t", "QuantEdge") is False


def test_post_returning_id_is_none_without_a_resolvable_channel(monkeypatch):
    """No channel id → the message must STILL be delivered, just not reactable."""
    monkeypatch.setattr(notify, "_load_channel_ids", lambda: {})
    delivered = []
    monkeypatch.setattr(notify, "post", lambda ch, tx, username="QuantEdge": delivered.append(ch) or True)
    assert notify.post_returning_id("engineering", "text") is None
    assert delivered == ["engineering"], "must not silently drop the message"


def test_post_returning_id_empty_text_is_a_noop():
    assert notify.post_returning_id("engineering", "") is None


# ── reactions ────────────────────────────────────────────────────────────────

def test_add_reaction_hits_the_right_endpoint_with_encoded_emoji(monkeypatch):
    seen = {}
    monkeypatch.setattr(notify, "_BOT_TOKEN", "bot-token")
    monkeypatch.setattr(notify, "_load_channel_ids", lambda: {"engineering": "C9"})

    def fake_bot_req(method, path, body=None):
        seen["method"], seen["path"] = method, path
        return {}

    monkeypatch.setattr(notify, "_bot_req", fake_bot_req)
    assert notify.add_reaction("engineering", "777", "white_check_mark") is True
    assert seen["method"] == "PUT"
    assert seen["path"] == f"/channels/C9/messages/777/reactions/{urllib.parse.quote('✅')}/@me"


def test_slack_style_names_map_to_unicode(monkeypatch):
    monkeypatch.setattr(notify, "_BOT_TOKEN", "bot-token")
    monkeypatch.setattr(notify, "_load_channel_ids", lambda: {"eng": "C1"})
    paths = []
    monkeypatch.setattr(notify, "_bot_req", lambda m, p, b=None: paths.append(p) or {})
    for name, char in (("eyes", "👀"), ("x", "❌"), ("hourglass_flowing_sand", "⏳")):
        notify.add_reaction("eng", "1", name)
    assert all(urllib.parse.quote(c) in p for p, c in zip(paths, ["👀", "❌", "⏳"]))


def test_raw_emoji_passes_through(monkeypatch):
    monkeypatch.setattr(notify, "_BOT_TOKEN", "bot-token")
    monkeypatch.setattr(notify, "_load_channel_ids", lambda: {"eng": "C1"})
    seen = {}
    monkeypatch.setattr(notify, "_bot_req", lambda m, p, b=None: seen.update(path=p) or {})
    assert notify.add_reaction("eng", "1", "🔥") is True
    assert urllib.parse.quote("🔥") in seen["path"]


@pytest.mark.parametrize("bot_token,chan_map,msg_id", [
    ("", {"eng": "C1"}, "1"),          # no bot token
    ("tok", {}, "1"),                  # channel not resolvable
    ("tok", {"eng": "C1"}, ""),        # no message id (webhook post)
])
def test_add_reaction_fails_soft(monkeypatch, bot_token, chan_map, msg_id):
    monkeypatch.setattr(notify, "_BOT_TOKEN", bot_token)
    monkeypatch.setattr(notify, "_load_channel_ids", lambda: chan_map)
    monkeypatch.setattr(notify, "_bot_req", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call")))
    assert notify.add_reaction("eng", msg_id, "eyes") is False


def test_add_reaction_never_raises_on_api_error(monkeypatch):
    monkeypatch.setattr(notify, "_BOT_TOKEN", "tok")
    monkeypatch.setattr(notify, "_load_channel_ids", lambda: {"eng": "C1"})
    monkeypatch.setattr(notify, "_bot_req", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("403")))
    assert notify.add_reaction("eng", "1", "eyes") is False
