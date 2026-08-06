"""`healthy` had exactly two states: alive and dead. Nothing said "nearly dead".

Measured 2026-08-06, run 31090685392 — the canary printed `BRAIN OK` and exited
green with this underneath:

    groq:       ok: true,  ms: 171
    gemini:     ok: false, error: "HTTP Error 429: Too Many Requests"
    nvidia_nim: ok: false, error: "The read operation timed out"

One provider answering, two keyed providers already down, and every signal
available said OK. `healthy = bool(working)` is the right predicate for the
red/green gate — it just cannot express distance from failure.

`single_point_of_failure` is reported ALONGSIDE `healthy`, not folded into it:
whether a degraded cascade should page anyone is an operator decision, and
widening the alarm here would silently make that decision for them.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_common as L  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent


def _status(monkeypatch, working: set[str], keyed: set[str]):
    monkeypatch.setattr(L, "_has_key", lambda p: p["name"] in keyed)
    monkeypatch.setattr(L, "_provider_keys", lambda p: ["k"] if p["name"] in keyed else [])
    monkeypatch.setattr(
        L, "_call_provider",
        lambda p, *a, **k: "OK" if p["name"] in working else (_ for _ in ()).throw(RuntimeError("down")))
    return L.cascade_status(probe=True)


def test_one_working_provider_is_flagged(monkeypatch):
    """The exact live shape: groq up, gemini and nvidia_nim keyed but failing."""
    st = _status(monkeypatch, {"groq"}, {"groq", "gemini", "nvidia_nim"})
    assert st["working"] == ["groq"]
    assert st["healthy"] is True
    assert st["single_point_of_failure"] is True
    assert set(st["keyed"]) == {"groq", "gemini", "nvidia_nim"}


def test_two_working_providers_are_not_flagged(monkeypatch):
    st = _status(monkeypatch, {"groq", "gemini"}, {"groq", "gemini"})
    assert st["healthy"] is True
    assert st["single_point_of_failure"] is False


def test_zero_working_is_unhealthy_and_not_merely_at_risk(monkeypatch):
    """A dead cascade must stay dead, not get relabelled as a lesser warning."""
    st = _status(monkeypatch, set(), {"groq"})
    assert st["healthy"] is False
    assert st["single_point_of_failure"] is False


def test_the_gate_is_unchanged_one_provider_still_exits_zero(monkeypatch, capsys):
    """The whole point of reporting this separately: the red/green verdict and
    the Discord page are untouched. Raising the floor is the operator's call."""
    import brain_health

    monkeypatch.setattr(brain_health.L, "cascade_status",
                        lambda probe=True: {"providers": {}, "working": ["groq"],
                                            "keyed": ["groq", "gemini", "nvidia_nim"],
                                            "healthy": True, "single_point_of_failure": True})
    paged = []
    monkeypatch.setattr(brain_health, "_alert_chat", lambda msg: paged.append(msg))
    rc = brain_health.main()
    out = capsys.readouterr().out
    assert rc == 0, "a degraded-but-alive cascade must not turn the workflow red"
    assert paged == [], "it must not page either — that is the operator's decision"
    assert "BRAIN AT RISK" in out
    assert "1 of 3 keyed provider(s)" in out


def test_a_dead_cascade_still_pages_and_exits_nonzero(monkeypatch):
    import brain_health

    monkeypatch.setattr(brain_health.L, "cascade_status",
                        lambda probe=True: {"providers": {"groq": {"has_key": True}},
                                            "working": [], "keyed": ["groq"],
                                            "healthy": False, "single_point_of_failure": False})
    paged = []
    monkeypatch.setattr(brain_health, "_alert_chat", lambda msg: paged.append(msg))
    assert brain_health.main() == 1
    assert len(paged) == 1 and "DOWN" in paged[0]


def test_the_healthy_path_no_longer_says_only_brain_ok_when_at_risk():
    """String-matching the source, because 'BRAIN OK' printed over a one-provider
    cascade is the exact misleading output this file exists to prevent."""
    src = (SCRIPTS / "brain_health.py").read_text()
    assert "single_point_of_failure" in src, (
        "brain_health no longer consults single_point_of_failure — the healthy "
        "path is back to reporting OK regardless of how close to dead it is")
