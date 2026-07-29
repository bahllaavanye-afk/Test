"""`gh workflow run` without actions:write is a permanent, invisible 403.

The improver dispatches CI on its own PR branch, because a PR opened by
GITHUB_TOKEN gets no `pull_request` workflow run. `workflow_dispatch` is one of
the two events explicitly EXCEPTED from that suppression, so the approach is
sound — the workflow's own comment says as much:

    # Dispatching CI explicitly (allowed for GITHUB_TOKEN) closes the
    # loop: improver → PR → CI → gate merges → CI completion → next improver

But `gh workflow run` posts to `/actions/workflows/{id}/dispatches`, which needs
**actions: write**, and the job granted only `contents: write` and
`pull-requests: write`. So every run since it was written ended:

    could not create workflow dispatch event: HTTP 403: Resource not
    accessible by integration
    ##[error]Process completed with exit code 1

…while `continue-on-error: true` reported the step as **success**. Observed in
runs 30476849972 (17:46) and 30483279439 (19:11) — and in both, the job
conclusion is `success` and the step conclusion is `success`.

That single missing permission is the mechanical cause of the standing problem
that "improver PRs bypass CI, so main can silently break" — restated at the top
of every monitor tick, and blamed on GITHUB_TOKEN event suppression when the
real cause was a permission the workflow never asked for.

`continue-on-error` is kept deliberately: a lost dispatch must not throw away
the improvements. But the failure now emits a `::warning::` annotation, because
a bare `exit 1` under continue-on-error renders as a green step, which is
exactly how this hid for the workflow's whole lifetime.

The sweep below covers the class: any workflow that dispatches another workflow
must grant actions:write.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_WF_DIR = Path(__file__).resolve().parents[1] / "workflows"
_WORKFLOWS = sorted(_WF_DIR.glob("*.yml"))

# Commands that hit an endpoint requiring the `actions` scope for WRITE.
_DISPATCH_PATTERNS = (
    re.compile(r"gh workflow run\b"),
    re.compile(r"/actions/workflows/[^\s]*/dispatches"),
    re.compile(r"gh api[^\n]*\bdispatches\b"),
    re.compile(r"gh run (?:rerun|cancel)\b"),
)


def _dispatches(text: str) -> list[str]:
    return [p.pattern for p in _DISPATCH_PATTERNS if p.search(text)]


def _grants_actions_write(text: str) -> bool:
    """True if any permissions block in the file grants actions: write.

    Deliberately file-scoped rather than job-scoped: a false PASS here would be
    worse than a false FAIL, and job-level parsing without a YAML dependency is
    where this check would get subtly wrong. A file that dispatches from one job
    and grants actions:write in another is still flagged by review, not by this.
    """
    # Trailing comments are normal and must not break the match. The first
    # version anchored with `\s*$` and reported pacemaker.yml as missing the
    # permission — it grants it on line 74 as
    #     actions: write              # required to dispatch CI
    # which would have been published as "the pacemaker has never worked", a
    # false claim about the one workflow holding the fleet's heartbeat together.
    return bool(re.search(r"^\s*actions:\s*write\s*(?:#.*)?$", text, re.M))


def test_at_least_one_workflow_dispatches(  # guards the guard
):
    """If nothing dispatches, this whole file is vacuous and should be deleted."""
    found = [wf.name for wf in _WORKFLOWS if _dispatches(wf.read_text())]
    assert found, (
        "no workflow dispatches another workflow any more — this test is now "
        "vacuously green and should be removed rather than left as decoration"
    )


@pytest.mark.parametrize("wf", _WORKFLOWS, ids=lambda p: p.name)
def test_a_dispatching_workflow_grants_actions_write(wf: Path):
    text = wf.read_text()
    used = _dispatches(text)
    if not used:
        pytest.skip("does not dispatch a workflow")
    assert _grants_actions_write(text), (
        f"{wf.name} dispatches a workflow ({', '.join(used)}) but never grants "
        f"`actions: write`. The API call returns 'HTTP 403: Resource not "
        f"accessible by integration' every single time. If the step also sets "
        f"continue-on-error, that 403 renders as a GREEN step and nothing ever "
        f"surfaces — which is how this went unnoticed for the improver's entire "
        f"lifetime."
    )


def test_the_improver_dispatch_failure_is_visible():
    """continue-on-error must not turn a hard failure into silence."""
    text = (_WF_DIR / "continuous-improvement.yml").read_text()
    assert "gh workflow run test.yml" in text, "the dispatch step moved or was removed"
    assert "::warning" in text, (
        "the dispatch step must emit a ::warning:: annotation on failure. With "
        "continue-on-error: true a non-zero exit is rendered as success, so a "
        "permanent 403 looks identical to a working dispatch."
    )
    assert "continue-on-error: true" in text, (
        "continue-on-error was removed — a failed dispatch would now discard a "
        "run's worth of improvements. Keep it, and keep the warning."
    )


def test_the_dispatch_target_actually_accepts_workflow_dispatch():
    """A dispatch to a workflow without the trigger 404s, not 403s."""
    target = _WF_DIR / "test.yml"
    assert target.exists(), "test.yml is gone — the dispatch target moved"
    assert re.search(r"^\s*workflow_dispatch:", target.read_text(), re.M), (
        "test.yml no longer declares workflow_dispatch, so `gh workflow run` "
        "cannot trigger it regardless of permissions"
    )
