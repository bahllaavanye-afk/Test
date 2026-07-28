"""The security gate has never inspected a single pull request's changes.

Found 2026-07-28 while verifying that a PR was green before merging. Two
independent defects stacked, and each one alone was enough to make the gate
decorative:

1. **It could not RUN.** `security-scan.yml` triggered only on `pull_request`.
   Agent branches get their PR opened by `auto-pr.yml`, so the workflow run's
   actor is `github-actions[bot]` — and a bot-actored PR run lands in
   `action_required`, waiting on a human approval that never arrives. Of the
   last 30 PR runs: 21 `action_required`, and **every one of them bot-actored**
   (all the `improver/*` PRs, plus my own whenever auto-pr won the race):

       20:51 action_required  actor: github-actions[bot]  claude/stoic-...
       20:43 action_required  actor: github-actions[bot]  improver/run-...
       15:57 success          actor: bahllaavanye-afk     claude/stoic-...

   The correlation is exact. Adding a `push` trigger fixes it: a push is
   actored by the pusher, needs no approval, and fires before the PR exists.
   main is excluded on purpose — 27 of its 129 commits in the last 24h touch
   the scanned paths, so it would be pure burn, and the weekly cron already
   covers merged code.

2. **When it DID run, it scanned the wrong tree.** The checkout was pinned
   `ref: main`, unconditionally. So a PR run analysed main, not the code being
   proposed. That is why the five `success` runs on this branch proved nothing
   about this branch — bandit and the red-team probe never saw the diff.

Both are the same failure shape as the `ModuleNotFoundError: sqlalchemy` bug
already recorded in this workflow's own comments: a gate that is permanently
not-red looks exactly like a gate that passed. Nothing distinguishes "approved
and clean" from "never executed".

Scheduled and dispatch runs still pin main deliberately — they have no
meaningful event ref and should scan the latest merged code.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"
_SCAN = _WORKFLOWS / "security-scan.yml"


def _trigger_block(text: str) -> str:
    """The `on:` block, up to the next top-level key."""
    m = re.search(r"^on:\s*$(.*?)^(?=\w)", text, re.MULTILINE | re.DOTALL)
    return m.group(1) if m else ""


@pytest.fixture(scope="module")
def scan_src() -> str:
    assert _SCAN.is_file(), f"missing {_SCAN}"
    return _SCAN.read_text()


# ── defect 1: it could not run ───────────────────────────────────────────────

def test_the_gate_runs_on_push_not_only_on_pull_request(scan_src):
    """`pull_request` alone is unreachable for bot-opened PRs."""
    on = _trigger_block(scan_src)
    assert "push:" in on, (
        "security-scan.yml triggers only on pull_request. Agent PRs are opened "
        "by auto-pr.yml, so their runs are bot-actored and sit in "
        "action_required forever — the gate never executes."
    )


def test_the_push_trigger_excludes_main(scan_src):
    """main takes ~27 qualifying commits a day; the weekly cron covers it."""
    on = _trigger_block(scan_src)
    push = on.split("push:", 1)[1].split("pull_request:", 1)[0]
    assert "branches-ignore" in push and "main" in push, (
        "a push trigger without branches-ignore: [main] would fire on every "
        "state-bot commit that touches backend/ or .github/"
    )


def test_the_push_trigger_keeps_the_paths_filter(scan_src):
    """Without it, docs-only and state-only pushes would scan too."""
    on = _trigger_block(scan_src)
    push = on.split("push:", 1)[1].split("pull_request:", 1)[0]
    assert "paths:" in push and ".github/scripts/**" in push


def test_the_other_triggers_survive(scan_src):
    """The weekly cron and manual dispatch must not be lost in the edit."""
    on = _trigger_block(scan_src)
    assert "schedule:" in on and "cron:" in on
    assert "workflow_dispatch" in on
    assert "pull_request:" in on


# ── defect 2: it scanned the wrong tree ──────────────────────────────────────

def test_the_checkout_is_not_unconditionally_pinned_to_main(scan_src):
    """`ref: main` on a PR run analyses main, never the proposed diff."""
    assert not re.search(r"ref:\s*main\s*[}\n]", scan_src), (
        "checkout is hard-pinned to main, so event-driven runs scan the wrong "
        "tree and cannot gate the change they were triggered by"
    )


def test_event_driven_runs_scan_the_triggering_commit(scan_src):
    assert re.search(r"ref:\s*\$\{\{.*github\.sha.*\}\}", scan_src), (
        "the checkout must resolve to the triggering commit for push/PR runs"
    )


def test_scheduled_runs_still_pin_main(scan_src):
    """No event ref worth scanning — the latest merged code is the subject.

    Also required by test_script_safety.test_scheduled_workflows_checkout_correct_branch.
    """
    m = re.search(r"ref:\s*\$\{\{(.*?)\}\}", scan_src, re.DOTALL)
    assert m, "no templated ref found"
    expr = m.group(1)
    assert "schedule" in expr and "'main'" in expr


def test_exactly_one_ref_key_in_the_checkout_block(scan_src):
    """Two `ref:` keys in one `with:` block silently take the last one."""
    blocks = re.findall(r"with:\s*\n((?:[ \t]+\S.*\n?)+)", scan_src)
    for b in blocks:
        assert len(re.findall(r"^\s+ref:", b, re.MULTILINE)) <= 1, b


# ── the hard gate itself must stay intact ────────────────────────────────────

def test_the_secret_leak_step_is_still_a_hard_gate(scan_src):
    """No `|| true` — this step's failure is the whole point of the workflow."""
    # anchor on the STEP, not the phrase — it also appears in the file header
    step = scan_src.split("- name: Secret-leak guard", 1)[1].split("\n      - name:", 1)[0]
    assert "test_script_safety.py" in step
    assert "--noconftest" in step, "conftest drags in sqlalchemy; see the workflow comment"
    assert "|| true" not in step, "the hard gate must be allowed to fail the run"


def test_the_yaml_still_parses():
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(_SCAN.read_text())
    # `on` is parsed as the boolean True by YAML 1.1 — accept either spelling.
    triggers = doc.get("on", doc.get(True))
    assert triggers, f"no trigger block parsed from {_SCAN}"
    assert set(triggers) >= {"push", "pull_request", "schedule"}
    assert triggers["push"]["branches-ignore"] == ["main"]
