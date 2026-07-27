"""Error triage must collapse noise into causes, and never spam the tracker.

The fleet posts errors into Discord all day and nothing read them. The failure
mode this guards against is the obvious one: 200 near-identical messages
becoming 200 issues. Signatures normalise ids, timestamps and numbers so that

    insufficient balance (requested: 134.58, available: 6.71)
    insufficient balance (requested: 92.10, available: 4.02)

are ONE cause, not two.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MOD = Path(__file__).resolve().parent / "discord_error_triage.py"
_spec = importlib.util.spec_from_file_location("det_test", _MOD)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)  # type: ignore[union-attr]


# ── signature normalisation ──────────────────────────────────────────────────

def test_same_failure_different_numbers_is_one_signature():
    a = "⚠ alpaca POST /v2/orders → 403: insufficient balance for USD (requested: 134.58, available: 6.71)"
    b = "⚠ alpaca POST /v2/orders → 403: insufficient balance for USD (requested: 92.10, available: 4.02)"
    assert m.signature(a) == m.signature(b)


def test_timestamps_and_uuids_do_not_split_a_signature():
    a = "Bot run failed bot_id=3f2a1c4e-11aa-4bb2-9ccd-0123456789ab at 2026-07-27T09:33:21Z"
    b = "Bot run failed bot_id=aa112233-44bb-55cc-66dd-abcdefabcdef at 2026-07-26T04:02:11Z"
    assert m.signature(a) == m.signature(b)


def test_genuinely_different_failures_stay_separate():
    a = "asset MKR/USD is not active"
    b = "float division by zero"
    assert m.signature(a) != m.signature(b)


def test_signature_is_stable_across_calls():
    text = "🚨 DESK crypto PLACED NOTHING — 9 unfunded, 3 rejected"
    assert m.signature(text) == m.signature(text)


def test_hash_is_short_and_deterministic():
    h1 = m.sig_hash("some signature")
    h2 = m.sig_hash("some signature")
    assert h1 == h2 and len(h1) == 10
    assert m.sig_hash("other") != h1


# ── error detection ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "🚨 Live smoke test FAILED",
    "❌ deploy parity: live=29 repo=61",
    "⚠ place_order failed SHIB/USD buy: float division by zero",
    "Traceback (most recent call last):",
    "TypeError: can't subtract offset-naive and offset-aware datetimes",
])
def test_error_messages_are_detected(text):
    assert m.is_error(text) is True


@pytest.mark.parametrize("text", [
    "",
    "   ",
    "✅ all checks passed",
    "Daily P&L: +$412.33",
    "🩺 delivery self-test 09:12 UTC — if you can read this, Discord routing works.",
])
def test_non_errors_and_noise_are_ignored(text):
    assert m.is_error(text) is False


def test_self_test_noise_is_explicitly_excluded():
    """The delivery self-test posts every smoke run and is not a defect."""
    assert m.is_error("🩺 delivery self-test 05:52 UTC — if you can read this, "
                      "Discord routing works.") is False


# ── ranking ──────────────────────────────────────────────────────────────────

def test_ranking_counts_and_orders_by_frequency():
    msgs = [
        {"channel": "ci-failures", "content": "❌ insufficient balance (requested: 1.00, available: 0.10)"},
        {"channel": "ci-failures", "content": "❌ insufficient balance (requested: 2.00, available: 0.20)"},
        {"channel": "ci-failures", "content": "❌ insufficient balance (requested: 3.00, available: 0.30)"},
        {"channel": "incidents",   "content": "❌ asset MKR/USD is not active"},
    ]
    ranked = m.rank(msgs)
    assert len(ranked) == 2, "three numeric variants must collapse to one cause"
    top_sig, top_n, top_ex = ranked[0]
    assert top_n == 3
    assert "insufficient balance" in top_sig
    assert ranked[1][1] == 1


def test_ranking_keeps_an_example_per_signature():
    msgs = [{"channel": "incidents", "content": "❌ boom 1"},
            {"channel": "incidents", "content": "❌ boom 2"}]
    ranked = m.rank(msgs)
    assert ranked[0][2]["channel"] == "incidents"
    assert "boom" in ranked[0][2]["content"]


def test_empty_input_ranks_to_nothing():
    assert m.rank([]) == []


# ── collection is fail-soft ──────────────────────────────────────────────────

def test_a_broken_channel_does_not_stop_triage(monkeypatch):
    """One unreadable channel must not lose the other channels' signal."""
    import sys, types
    fake = types.ModuleType("notify")

    def _read(channel, limit=100):
        if channel == "incidents":
            raise RuntimeError("channel unreadable")
        return [{"content": "❌ something broke", "author": "bot", "ts": "2026-07-27T09:00:00Z"}]

    fake.read_channel_recent = _read
    monkeypatch.setitem(sys.modules, "notify", fake)

    found = m.collect(limit_per_channel=5)
    channels = {f["channel"] for f in found}
    assert "incidents" not in channels
    assert len(channels) == len(m.ERROR_CHANNELS) - 1, "every other channel still read"


def test_missing_notify_returns_empty_not_crash(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _blocked(name, *a, **kw):
        if name == "notify":
            raise ImportError("no notify here")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    assert m.collect() == []


# ── issue-spam guards ────────────────────────────────────────────────────────

def test_only_recurring_signatures_are_worth_an_issue():
    assert m.MIN_OCCURRENCES >= 2, "a one-off must not open an issue"


def test_issues_per_run_is_capped():
    assert 1 <= m.MAX_ISSUES_PER_RUN <= 10, "never flood the tracker in one run"


def test_issue_title_carries_the_dedupe_hash():
    """existing_issues() re-reads this hash to update instead of duplicating."""
    import re
    sig = "insufficient balance for USD (requested: <num>, available: <num>)"
    h = m.sig_hash(sig)
    title = f"[auto-triage] [{h}] {sig[:90]}"
    assert re.search(r"\[([0-9a-f]{10})\]", title).group(1) == h
