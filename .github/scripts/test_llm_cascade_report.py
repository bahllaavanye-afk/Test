"""The cascade report must make a one-deep cascade loud.

`llm_common._record_metric()` already logs every LLM call to
`.github/state/llm_metrics.jsonl`. That path is gitignored and no workflow
reads it, so the metrics are written into the ephemeral runner and discarded
when the job ends — collected every run, visible to nobody.

The live picture matters more than the discarded file: with only GEMINI_API_KEY
populated, one rate-limit takes every agent dark, and nothing said so.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MOD = Path(__file__).resolve().parent / "llm_cascade_report.py"
_spec = importlib.util.spec_from_file_location("lcr_test", _MOD)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)  # type: ignore[union-attr]


def _status(**providers):
    working = [n for n, v in providers.items() if v.get("ok")]
    return {"providers": providers, "working": working, "healthy": bool(working)}


def test_dead_cascade_is_flagged_loudly():
    report, healthy = m.build_report(_status(
        gemini={"has_key": True, "ok": False, "error": "429 rate limited"},
        groq={"has_key": False, "ok": False},
    ))
    assert healthy is False
    assert "DEAD" in report
    assert "silently degraded" in report


def test_one_deep_cascade_is_flagged_even_though_it_works():
    """This is the live state: it 'works' and is one outage from dark."""
    report, healthy = m.build_report(_status(
        gemini={"has_key": True, "ok": True, "ms": 210},
        groq={"has_key": False, "ok": False},
        deepseek={"has_key": False, "ok": False},
    ))
    assert healthy is False, "a single working provider is not a healthy cascade"
    assert "1 deep" in report
    assert "DEAD" not in report


def test_two_working_providers_is_healthy():
    _report, healthy = m.build_report(_status(
        gemini={"has_key": True, "ok": True, "ms": 200},
        groq={"has_key": True, "ok": True, "ms": 150},
    ))
    assert healthy is True


def test_report_lists_every_provider_and_its_key_state():
    report, _ = m.build_report(_status(
        gemini={"has_key": True, "ok": True, "ms": 200},
        cerebras={"has_key": False, "ok": False},
    ))
    assert "gemini" in report and "cerebras" in report
    assert "Keys configured:** 1" in report
    assert "cerebras" in report.split("**No key:**")[1].splitlines()[0]


def test_errors_are_surfaced_but_truncated():
    long_err = "x" * 300
    report, _ = m.build_report(_status(
        gemini={"has_key": True, "ok": False, "error": long_err},
    ))
    assert "x" * 60 in report
    assert "x" * 61 not in report, "error text must be truncated for the table"


def test_empty_status_does_not_crash():
    report, healthy = m.build_report({"providers": {}, "working": []})
    assert healthy is False
    assert "DEAD" in report


def test_healthy_threshold_is_more_than_one():
    assert m.HEALTHY_MIN_PROVIDERS >= 2, (
        "one provider is a single point of failure, not a cascade"
    )


def test_main_never_raises_when_probe_fails(monkeypatch):
    """A reporter must not be able to break the run it reports on."""
    import sys, types
    fake = types.ModuleType("llm_common")

    def _boom(probe=True):
        raise RuntimeError("network down")

    fake.cascade_status = _boom
    monkeypatch.setitem(sys.modules, "llm_common", fake)
    assert m.main() == 0
