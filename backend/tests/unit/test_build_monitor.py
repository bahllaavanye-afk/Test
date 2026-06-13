"""
Unit tests for the hourly build monitor.

Hermetic: build checks and the auto-fix pass are monkeypatched, so no
subprocess, Redis, or disk is touched. We assert the gating logic (lint is soft,
imports/tsc are hard), escalation wiring, and the auto-fix → rebuild flow.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import app.tasks.build_monitor as bm
from app.tasks.build_monitor import (
    BuildMonitor, BuildResult, BuildReport, get_build_monitor,
    _escalation_candidates, HARD_GATES,
)


def _ok(check, component="backend"):
    return BuildResult(component=component, check=check, ok=True)


def _red(check, component="backend"):
    return BuildResult(component=component, check=check, ok=False, detail="boom")


class TestDataclasses:
    def test_build_result_defaults(self):
        r = BuildResult(component="backend", check="ruff", ok=True)
        assert r.skipped is False and r.detail == ""

    def test_build_report_defaults(self):
        rep = BuildReport(timestamp="t", overall_ok=True)
        assert rep.results == [] and rep.issues_escalated == 0


class TestEscalationCandidates:
    def test_shape_and_stable_fingerprint(self):
        red = [_red("imports"), _red("tsc", component="frontend")]
        cands = _escalation_candidates(red)
        assert len(cands) == 2
        assert cands[0]["role"] == "backend"
        assert cands[1]["role"] == "frontend"
        # Stable across calls.
        assert _escalation_candidates(red)[0]["fingerprint"] == cands[0]["fingerprint"]


class TestHardGates:
    def test_only_imports_and_tsc_are_hard(self):
        assert HARD_GATES == {"imports", "tsc"}


class TestRunCycle:
    @pytest.mark.asyncio
    async def test_all_green_no_escalation(self, monkeypatch):
        mon = BuildMonitor()
        monkeypatch.setattr(mon, "_build_all", lambda: [_ok("imports"), _ok("ruff"), _ok("tsc", "frontend")])

        async def _noop(_r):
            return None
        monkeypatch.setattr(mon, "_persist_and_broadcast", _noop)

        report = await mon.run_cycle()
        assert report.overall_ok is True
        assert report.issues_escalated == 0
        assert report.autofixes_applied == []

    @pytest.mark.asyncio
    async def test_lint_red_is_not_a_broken_build(self, monkeypatch):
        """ruff red but imports+tsc green → overall_ok True; autofix attempted."""
        mon = BuildMonitor()
        calls = {"autofix": 0}

        monkeypatch.setattr(mon, "_build_all", lambda: [_ok("imports"), _red("ruff"), _ok("tsc", "frontend")])

        def _fake_autofix():
            calls["autofix"] += 1
            return ["ruff --fix applied to backend/app"]
        monkeypatch.setattr(bm, "apply_autofixes", _fake_autofix)
        monkeypatch.setattr(bm, "_maybe_commit", lambda applied: False)

        async def _noop(_r):
            return None
        monkeypatch.setattr(mon, "_persist_and_broadcast", _noop)

        report = await mon.run_cycle()
        assert report.overall_ok is True       # lint is soft
        assert calls["autofix"] == 1           # but we still tried to clean it
        assert report.issues_escalated == 0

    @pytest.mark.asyncio
    async def test_hard_failure_escalates_when_enabled(self, monkeypatch):
        mon = BuildMonitor()
        # Stays red even after the auto-fix pass.
        monkeypatch.setattr(mon, "_build_all", lambda: [_red("imports"), _ok("ruff")])
        monkeypatch.setattr(bm, "apply_autofixes", lambda: [])

        async def _noop(_r):
            return None
        monkeypatch.setattr(mon, "_persist_and_broadcast", _noop)

        class _FakeEscalator:
            enabled = True
            async def escalate_items(self, cands):
                return {"opened": len(cands), "skipped_existing": 0, "candidates": len(cands), "errors": []}

        import app.tasks.issue_escalation as esc_mod
        monkeypatch.setattr(esc_mod, "get_escalator", lambda: _FakeEscalator())

        report = await mon.run_cycle()
        assert report.overall_ok is False
        assert report.issues_escalated == 1

    @pytest.mark.asyncio
    async def test_hard_failure_fixed_by_autofix(self, monkeypatch):
        """First build red on imports, auto-fix runs, rebuild green → ok, no escalation."""
        mon = BuildMonitor()
        seq = [
            [_red("imports"), _ok("ruff")],   # first build
            [_ok("imports"), _ok("ruff")],    # rebuild after fix
        ]
        monkeypatch.setattr(mon, "_build_all", lambda: seq.pop(0))
        monkeypatch.setattr(bm, "apply_autofixes", lambda: ["deprecated-API rewrite in 1 file(s)"])
        monkeypatch.setattr(bm, "_maybe_commit", lambda applied: False)

        async def _noop(_r):
            return None
        monkeypatch.setattr(mon, "_persist_and_broadcast", _noop)

        report = await mon.run_cycle()
        assert report.overall_ok is True
        assert report.issues_escalated == 0
        assert report.autofixes_applied


class TestSingleton:
    def test_singleton(self):
        assert get_build_monitor() is get_build_monitor()
