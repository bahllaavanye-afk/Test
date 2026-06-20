"""Unit tests for the AI PR reviewer's pure logic (no network)."""
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / ".github" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ai_pr_review as R  # noqa: E402


def test_parse_verdict_variants():
    assert R.parse_verdict("blah\nVERDICT: APPROVE") == "APPROVE"
    assert R.parse_verdict("x\nVERDICT: REQUEST_CHANGES") == "REQUEST_CHANGES"
    assert R.parse_verdict("y\nVERDICT: COMMENT") == "COMMENT"
    assert R.parse_verdict("no verdict line here") == "COMMENT"  # safe default


def test_build_review_prompt_includes_diff_and_title():
    p = R.build_review_prompt("Fix bug", "diff --git a/x b/x\n+code")
    assert "Fix bug" in p and "diff --git" in p
    assert "VERDICT:" in p  # asks the model for a verdict


def test_format_comment_has_marker_and_badge():
    body = R.format_comment("looks good", "APPROVE")
    assert R._MARKER in body
    assert "Approve" in body
    assert "looks good" in body


def test_upsert_creates_when_absent_updates_when_present(monkeypatch):
    calls = {"post": 0, "patch": 0}

    class _Resp:
        ok = True

    class _FakeRequests:
        def post(self, *a, **k):
            calls["post"] += 1
            return _Resp()

        def patch(self, *a, **k):
            calls["patch"] += 1
            return _Resp()

    monkeypatch.setattr(R, "requests", _FakeRequests())

    monkeypatch.setattr(R, "find_existing_comment", lambda repo, num: None)
    assert R.upsert_comment("o/r", "1", "body") is True
    assert calls == {"post": 1, "patch": 0}  # created

    monkeypatch.setattr(R, "find_existing_comment", lambda repo, num: 12345)
    assert R.upsert_comment("o/r", "1", "body") is True
    assert calls == {"post": 1, "patch": 1}  # updated in place


def test_main_skips_without_token(monkeypatch):
    monkeypatch.setattr(R, "GH_TOKEN", "")
    monkeypatch.setattr(R, "PR_NUMBER", "")
    assert R.main() == 0  # advisory: never errors the workflow
