"""Unit tests for the AI PR reviewer's pure logic (no network)."""

import sys
from pathlib import Path

# Ensure the scripts directory is on the import path
SCRIPTS = Path(__file__).resolve().parents[3] / ".github" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ai_pr_review as R  # noqa: E402


def test_parse_verdict_variants():
    """Basic verdict extraction and safe default handling."""
    assert R.parse_verdict("blah\nVERDICT: APPROVE") == "APPROVE"
    assert R.parse_verdict("x\nVERDICT: REQUEST_CHANGES") == "REQUEST_CHANGES"
    assert R.parse_verdict("y\nVERDICT: COMMENT") == "COMMENT"
    # No explicit verdict line – fallback to COMMENT
    assert R.parse_verdict("no verdict line here") == "COMMENT"


def test_parse_verdict_ignores_case_and_spaces():
    """The parser should be case‑insensitive and trim whitespace."""
    assert R.parse_verdict("VERDICT:   approve  ") == "APPROVE"
    assert R.parse_verdict("verdict: request_changes") == "REQUEST_CHANGES"


def test_build_review_prompt_includes_diff_and_title():
    """Prompt must contain title, diff and a request for a verdict."""
    title = "Fix bug"
    diff = "diff --git a/x b/x\n+code"
    prompt = R.build_review_prompt(title, diff)
    assert title in prompt
    assert diff.splitlines()[0] in prompt
    assert "VERDICT:" in prompt  # model is asked for a verdict


def test_build_review_prompt_handles_empty_inputs():
    """Even with empty title or diff the prompt should still be well‑formed."""
    prompt = R.build_review_prompt("", "")
    assert "VERDICT:" in prompt
    # The placeholder for title should still appear (even if empty)
    assert "Title:" in prompt


def test_format_comment_has_marker_and_badge():
    """Formatted comment must contain the internal marker, a badge and the body."""
    body = R.format_comment("looks good", "APPROVE")
    assert R._MARKER in body
    # The badge text is title‑cased for readability
    assert "Approve" in body
    assert "looks good" in body


def test_format_comment_invalid_verdict_falls_back_to_comment():
    """When an unknown verdict is supplied the function should default to COMMENT."""
    body = R.format_comment("needs work", "UNKNOWN")
    assert "Comment" in body
    assert "needs work" in body


def test_upsert_creates_when_absent_updates_when_present(monkeypatch):
    """upsert_comment should POST when no comment exists and PATCH otherwise."""
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

    # No existing comment – POST is used
    monkeypatch.setattr(R, "find_existing_comment", lambda repo, num: None)
    assert R.upsert_comment("owner/repo", "1", "body") is True
    assert calls == {"post": 1, "patch": 0}

    # Existing comment – PATCH is used
    monkeypatch.setattr(R, "find_existing_comment", lambda repo, num: 12345)
    assert R.upsert_comment("owner/repo", "1", "body") is True
    assert calls == {"post": 1, "patch": 1}


def test_upsert_returns_false_on_failed_request(monkeypatch):
    """If the underlying request fails, upsert_comment must return False."""
    class _BadResp:
        ok = False

    class _FakeRequests:
        def post(self, *a, **k):
            return _BadResp()

        def patch(self, *a, **k):
            return _BadResp()

    monkeypatch.setattr(R, "requests", _FakeRequests())
    monkeypatch.setattr(R, "find_existing_comment", lambda repo, num: None)
    assert R.upsert_comment("owner/repo", "1", "body") is False

    monkeypatch.setattr(R, "find_existing_comment", lambda repo, num: 999)
    assert R.upsert_comment("owner/repo", "1", "body") is False


def test_main_skips_without_token(monkeypatch):
    """When GH_TOKEN or PR_NUMBER is missing, main should exit gracefully."""
    monkeypatch.setattr(R, "GH_TOKEN", "")
    monkeypatch.setattr(R, "PR_NUMBER", "")
    assert R.main() == 0  # advisory: never errors the workflow


def test_main_proceeds_with_token_and_pr(monkeypatch):
    """When credentials are present, main should invoke the review flow."""
    # Stub out the heavy functions to avoid network calls
    monkeypatch.setattr(R, "GH_TOKEN", "dummy")
    monkeypatch.setattr(R, "PR_NUMBER", "42")
    called = {"review": False}

    def fake_review():
        called["review"] = True
        return 0

    monkeypatch.setattr(R, "run_review_flow", fake_review)
    assert R.main() == 0
    assert called["review"] is True