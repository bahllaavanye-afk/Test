"""chat_post must return the dict shape its callers unpack.

It was annotated `-> dict` but did `return notify.post(...)`, which is a bool.
Every caller doing `.get("ts")` died:

    AttributeError: 'bool' object has no attribute 'get'

That took down the 24/7 Research → Trade pipeline on its very first post —
`research_to_trade.chat()` calls it before any research runs, so the whole
pipeline never executed. `employee_intros.post()` has the same shape.

A signature/arity checker cannot catch this: the call is well-formed, it is the
RETURN shape that lies. Hence a behavioural test.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import llm_common


class _FakeNotify:
    def __init__(self, message_id=None, post_ok=True):
        self.message_id = message_id
        self.post_ok = post_ok
        self.calls = []

    def post_returning_id(self, channel, text, username="QuantEdge"):
        self.calls.append(("post_returning_id", channel))
        return self.message_id

    def post(self, channel, text, username="QuantEdge"):
        self.calls.append(("post", channel))
        return self.post_ok


def _patch(monkeypatch, fake):
    """chat_post imports notify inside the function body."""
    monkeypatch.setitem(sys.modules, "notify", fake)


def test_returns_a_dict_with_the_message_id(monkeypatch):
    fake = _FakeNotify(message_id="1234567890")
    _patch(monkeypatch, fake)

    result = llm_common.chat_post("#desk-research", "hello")

    assert isinstance(result, dict), "callers call .get('ts') on this"
    assert result["ok"] is True
    assert result["ts"] == "1234567890"


def test_webhook_delivery_reports_ok_without_a_ts(monkeypatch):
    """A webhook post lands but cannot report an id — ok, ts=None."""
    fake = _FakeNotify(message_id=None, post_ok=True)
    _patch(monkeypatch, fake)

    result = llm_common.chat_post("#desk-research", "hello")

    assert result == {"ok": True, "ts": None}
    assert ("post", "#desk-research") in fake.calls, "must fall back to plain post"


def test_failed_delivery_is_reported_not_raised(monkeypatch):
    fake = _FakeNotify(message_id=None, post_ok=False)
    _patch(monkeypatch, fake)

    result = llm_common.chat_post("#desk-research", "hello")

    assert result == {"ok": False, "ts": None}


def test_result_supports_the_exact_caller_idiom(monkeypatch):
    """research_to_trade.chat and employee_intros.post both do this."""
    _patch(monkeypatch, _FakeNotify(message_id="42"))

    result = llm_common.chat_post("#desk-research", "x")
    thread_ts = result.get("ts")           # the line that used to AttributeError

    assert thread_ts == "42"


def test_a_bool_return_would_fail_this_idiom():
    """Pin why the old form broke, so the reason stays legible."""
    import pytest

    with pytest.raises(AttributeError):
        True.get("ts")          # type: ignore[attr-defined]
