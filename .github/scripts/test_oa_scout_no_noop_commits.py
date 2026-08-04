"""The OA scout committed to main 56 times having never found a single bot.

`oa_library.json` has read `{"known": [], ...}` since it was created — every
Options Alpha page redirects to a login and `OA_SESSION_COOKIE` has never been
set, so the scout is a no-op by construction until a human supplies that secret.

It still produced a commit on every run, because it rewrote `last_run` — a
timestamp that moves every run and that **nothing reads**; it is written in
`main()` and defaulted in `_load_state()`, and that is the whole of its lifetime.
Since the file always differed, the workflow's

    git diff --cached --quiet || { git commit ...; git push; }

guard was never satisfied. Fifty-six commits whose entire content was "the clock
moved", stacked onto main, reading like progress.

This is the inverse of the bug this repo usually finds. The usual one is work
that succeeds and produces nothing. This is *nothing* dressed as work — and it is
worse in one way, because a reader scanning `git log` sees a scout that appears
to be actively harvesting.

The fix bumps `last_run` only alongside a real change to `known` or `auth_wall`,
so a no-op run leaves the file byte-identical and the existing guard suppresses
the commit. Nothing is concealed: the workflow already pipes stdout into
`$GITHUB_STEP_SUMMARY`, so each run leaves a record in the Actions UI, and the
auth-wall case posts to Discord asking for the cookie.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent / "oa_scout.py"
_WF = Path(__file__).resolve().parents[1] / "workflows" / "oa-scout.yml"


@pytest.fixture(scope="module")
def src() -> str:
    assert _SRC.exists(), "oa_scout.py is gone"
    return _SRC.read_text()


def test_the_write_is_conditional(src):
    """`STATE_FILE.write_text` must sit inside a branch, not run unconditionally."""
    tree = ast.parse(src)
    writes_at_top_level_of_main = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for stmt in fn.body:  # direct children only — an `if` body is nested
            for node in ast.walk(stmt) if isinstance(stmt, ast.If) else [stmt]:
                pass
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if isinstance(func, ast.Attribute) and func.attr == "write_text":
                    owner = func.value
                    if isinstance(owner, ast.Name) and owner.id == "STATE_FILE":
                        writes_at_top_level_of_main.append(fn.name)
    assert not writes_at_top_level_of_main, (
        f"STATE_FILE.write_text runs unconditionally in {writes_at_top_level_of_main}. "
        f"That rewrites last_run on every run, so the workflow's "
        f"`git diff --cached --quiet` guard never holds and the scout commits to "
        f"main every run — 56 no-op commits before this was caught."
    )


def test_the_change_test_covers_the_meaningful_fields(src):
    """`known` and `auth_wall` are the only fields that carry information."""
    assert 'known != set(state.get("known") or [])' in src, (
        "the scout no longer compares `known` against the persisted value, so a "
        "genuine new find could be dropped instead of committed."
    )
    assert 'now_wall != state.get("auth_wall")' in src, (
        "the auth_wall transition is no longer detected. That flag flipping is "
        "exactly the event worth committing — it means the cookie started or "
        "stopped working."
    )


def test_a_missing_state_file_is_still_written(src):
    """First run, or a deleted file, must recreate it.

    Without this the 'skip when unchanged' logic would refuse to create the file
    at all — the same class of bug as `git diff --quiet` being blind to untracked
    files, which is already recorded in IMPROVEMENTS.
    """
    assert "not STATE_FILE.exists()" in src, (
        "the change test does not special-case a missing state file, so the "
        "scout can never write it the first time."
    )


def test_last_run_is_only_bumped_with_a_real_change(src):
    """The timestamp is the thing that caused the noise; keep it inside the branch."""
    idx_if = src.index("if changed:")
    idx_ts = src.index('"last_run": datetime.now(timezone.utc).isoformat()')
    assert idx_ts > idx_if, (
        "last_run is assigned outside the `if changed:` branch again — that is "
        "precisely what made every run produce a diff."
    )


def test_the_workflow_still_guards_the_commit():
    """The script change only helps if the workflow keeps its guard."""
    wf = _WF.read_text()
    assert "git add .github/state/oa_library.json" in wf, "the state file is no longer staged"
    assert "git diff --cached --quiet" in wf, (
        "the workflow lost its `git diff --cached --quiet` guard, so it will "
        "commit unconditionally regardless of what the script writes."
    )


def test_the_run_output_still_reaches_the_actions_summary():
    """Suppressing the commit is only acceptable because this record remains."""
    wf = _WF.read_text()
    assert 'tee -a "$GITHUB_STEP_SUMMARY"' in wf, (
        "the scout's stdout no longer reaches $GITHUB_STEP_SUMMARY. Without a "
        "commit AND without a summary, a no-op run would leave no evidence it "
        "ran at all — trading one invisibility problem for another."
    )
