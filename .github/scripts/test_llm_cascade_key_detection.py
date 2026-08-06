"""A provider with three configured keys was excluded from every LLM call.

Brain-health run 31078219842 (2026-08-06 06:42) probed the cascade and reported:

    groq:       {"has_key": false, "ok": false}
    nvidia_nim: {"has_key": true, "ok": true, "ms": 18550}
    "working": ["nvidia_nim"],  "healthy": true

while that same job's environment block showed:

    GROQ_API_KEY:    (empty)
    GROQ_API_KEY_1:  ***
    GROQ_API_KEY_2:  ***
    GROQ_API_KEY_3:  ***

`_provider_keys()` collects `_1.._3` and its docstring says several free keys
behave like a multiple of the per-key quota. `_has_key()` did not, and
`_call_parallel_race` builds the live cascade from `_has_key` — so this was not
a reporting bug. Groq was excluded from actual calls, leaving one provider at
18.5s. The 03:32 run had already failed outright with zero providers.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_common as lc  # noqa: E402

GROQ = {"name": "groq", "key_env": "GROQ_API_KEY"}


def _clear(monkeypatch, *names):
    for n in names:
        monkeypatch.delenv(n, raising=False)


def test_numbered_keys_alone_count_as_having_a_key(monkeypatch):
    """The exact live configuration: base empty, _1.._3 set."""
    _clear(monkeypatch, "GROQ_API_KEY", "GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY_1", "k1")
    monkeypatch.setenv("GROQ_API_KEY_2", "k2")
    monkeypatch.setenv("GROQ_API_KEY_3", "k3")
    assert lc._provider_keys(GROQ) == ["k1", "k2", "k3"]
    assert lc._has_key(GROQ) is True


def test_the_two_answers_cannot_disagree(monkeypatch):
    """`_has_key` and `_provider_keys` answer the same question. Any environment
    where one says yes and the other says no is the bug this file exists for."""
    cases = [
        {},
        {"GROQ_API_KEY": "base"},
        {"GROQ_API_KEY_2": "only-a-numbered-one"},
        {"GROQ_API_KEY": "", "GROQ_API_KEY_1": "k1"},
        {"GROQ_API_KEY": "disabled"},
        {"GROQ_API_KEY": "disabled", "GROQ_API_KEY_3": "k3"},
    ]
    for env in cases:
        _clear(monkeypatch, "GROQ_API_KEY", "GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3")
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        assert lc._has_key(GROQ) == bool(lc._provider_keys(GROQ)), (
            f"disagreement for {env}: _has_key={lc._has_key(GROQ)} "
            f"_provider_keys={lc._provider_keys(GROQ)}")


def test_disabled_is_still_respected(monkeypatch):
    """'disabled' is the documented opt-out and must not be resurrected."""
    _clear(monkeypatch, "GROQ_API_KEY", "GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3")
    monkeypatch.setenv("GROQ_API_KEY", "disabled")
    assert lc._has_key(GROQ) is False


def test_no_keys_at_all_is_false(monkeypatch):
    _clear(monkeypatch, "GROQ_API_KEY", "GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3")
    assert lc._has_key(GROQ) is False


def test_the_live_cascade_is_built_from_the_same_predicate():
    """`_call_parallel_race` selects providers with `_has_key`, which is why this
    was a capability bug and not a reporting one. If that changes, this file's
    rationale needs revisiting."""
    src = (Path(__file__).resolve().parent / "llm_common.py").read_text()
    assert "available = [p for p in _PROVIDERS if _has_key(p)]" in src, (
        "the cascade no longer selects providers via _has_key — re-check whether "
        "key detection still gates real calls")
