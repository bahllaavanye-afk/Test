"""
Unit tests for the red-team static security auditor.

The scan runs against temp files written into a patched source root, so it tests
real regex detection without depending on the live codebase's contents.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import app.tasks.red_team as rt
from app.tasks.red_team import Finding, to_escalations, summarize, scan_codebase


def _make_finding(severity, rule_id="x", file_path="app/x.py", line=1):
    return Finding(severity=severity, rule_id=rule_id, file_path=file_path,
                   line_number=line, description="d", snippet="s")


class TestToEscalations:
    def test_only_high_and_critical_escalated(self):
        findings = [
            _make_finding("low"), _make_finding("medium"),
            _make_finding("high"), _make_finding("critical"),
        ]
        cands = to_escalations(findings)
        assert len(cands) == 2
        prios = sorted(c["priority"] for c in cands)
        assert prios == ["P0", "P1"]

    def test_all_role_security(self):
        cands = to_escalations([_make_finding("critical")])
        assert cands[0]["role"] == "security"

    def test_fingerprint_stable(self):
        f = [_make_finding("high", line=42)]
        assert to_escalations(f)[0]["fingerprint"] == to_escalations(f)[0]["fingerprint"]


class TestSummarize:
    def test_counts_by_severity(self):
        findings = [_make_finding("critical"), _make_finding("high"),
                    _make_finding("high"), _make_finding("low")]
        s = summarize(findings)
        assert s["total"] == 4
        assert s["critical"] == 1
        assert s["high"] == 2
        assert s["low"] == 1


class TestScanCodebase:
    def test_detects_dangerous_patterns(self, monkeypatch, tmp_path):
        # Lay out a fake backend source tree the scanner will walk.
        app_dir = tmp_path / "app" / "danger"
        app_dir.mkdir(parents=True)
        (app_dir / "bad.py").write_text(
            "import subprocess\n"
            "def run(cmd):\n"
            "    subprocess.run(cmd, shell=True)\n"
            "    x = eval('1+1')\n"
            "    return x\n"
        )
        monkeypatch.setattr(rt, "BACKEND_DIR", tmp_path)
        monkeypatch.setattr(rt, "PROJECT_ROOT", tmp_path)

        findings = scan_codebase()
        rule_ids = {f.rule_id for f in findings}
        assert "shell_true" in rule_ids
        assert "eval_use" in rule_ids
        # Locations are real line numbers.
        assert all(f.line_number > 0 for f in findings)

    def test_clean_file_no_findings(self, monkeypatch, tmp_path):
        app_dir = tmp_path / "app" / "clean"
        app_dir.mkdir(parents=True)
        (app_dir / "ok.py").write_text("def add(a, b):\n    return a + b\n")
        monkeypatch.setattr(rt, "BACKEND_DIR", tmp_path)
        monkeypatch.setattr(rt, "PROJECT_ROOT", tmp_path)
        assert scan_codebase() == []

    def test_jwt_no_verify_is_critical(self, monkeypatch, tmp_path):
        app_dir = tmp_path / "app" / "auth"
        app_dir.mkdir(parents=True)
        (app_dir / "a.py").write_text("import jwt\nd = jwt.decode(tok, verify=False)\n")
        monkeypatch.setattr(rt, "BACKEND_DIR", tmp_path)
        monkeypatch.setattr(rt, "PROJECT_ROOT", tmp_path)
        findings = scan_codebase()
        jwt_findings = [f for f in findings if f.rule_id == "jwt_no_verify"]
        assert jwt_findings and jwt_findings[0].severity == "critical"
