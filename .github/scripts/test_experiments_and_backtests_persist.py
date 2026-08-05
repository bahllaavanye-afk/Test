"""Backtests have run every 15 minutes since inception and never saved a result.

Measured 2026-08-05:

    git log --all -- .github/state/last_backtest.json   ->  0 commits

The workflow reports success every time. Two independent faults, both fixed here.

**1. No `permissions:` block.** `quick-backtest.yml` never declared one, so
GITHUB_TOKEN was read-only and the final push 403'd:

    fatal: unable to access '.../Test/': The requested URL returned error: 403

**2. Only one file was staged.** `quick_backtest_runner.py` writes results into
`agent_memory.json` (line ~218) as well as the summary, but the save step ran
`git add .github/state/last_backtest.json` alone. That left a tracked file
modified, so the push step's rebase refused, on all four retries:

    error: cannot pull with rebase: You have unstaged changes.

Both steps carry `continue-on-error: true`, which is why a run that persisted
nothing still went green.

The ML side had the same shape. `ml_experiment.py` ran a real walk-forward GBC
on real bars (SPY 1399 rows, QQQ 940, NVDA 1399) and its entire output was a
`print`. The workflow's fallback was to append the JSON to a rolling GitHub
issue titled `ml-experiments-log` — **which does not exist**: paging all issues
(500 checked, PRs excluded) finds no such issue, while the step reports success.
So the out-of-sample numbers were produced and dropped every run, and nothing
could answer the one question a recurring experiment exists to answer: is the
edge improving or decaying?

Worth recording alongside: those numbers show the model LOSING to buy-and-hold —
SPY strategy Sharpe 0.56 vs 0.789 buy-hold, QQQ 1.12 vs 1.325. That is a result,
not a failure, and it is exactly the kind of thing that must be kept rather than
printed into a log that rotates away.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

_ROOT = Path(__file__).resolve().parents[2]
_WF = _ROOT / ".github" / "workflows"
_BACKTEST = _WF / "quick-backtest.yml"
_MLEXP = _WF / "ml-experiments.yml"
_MLSRC = _ROOT / ".github" / "scripts" / "ml_experiment.py"


# ── backtest persistence ──────────────────────────────────────────────────────

def test_the_backtest_workflow_can_write_to_the_repo():
    doc = yaml.safe_load(_BACKTEST.read_text())
    perms = doc.get("permissions") or {}
    assert perms.get("contents") == "write", (
        "quick-backtest.yml has no `contents: write`, so GITHUB_TOKEN is "
        "read-only and the push 403s. This ran every 15 minutes and persisted "
        "zero results."
    )


def test_the_save_step_stages_the_whole_state_dir():
    """Staging one file leaves others modified and the rebase aborts."""
    src = _BACKTEST.read_text()
    assert "git add -A .github/state/" in src, (
        "the save step stages a single file again. quick_backtest_runner also "
        "writes agent_memory.json, so the leftover modification makes "
        "`git pull --rebase` fail with 'You have unstaged changes' and nothing "
        "is ever pushed."
    )


def test_the_commit_is_still_guarded():
    """An unconditional commit on a clean tree fails the step."""
    src = _BACKTEST.read_text()
    assert "git diff --cached --quiet ||" in src, (
        "the empty-diff guard was dropped; a no-change run would error"
    )


def test_the_backtest_commit_does_not_retrigger_ci():
    """Every 15 minutes is far too often to be dispatching CI."""
    src = _BACKTEST.read_text()
    i = src.index("backtest: update results")
    assert "[skip ci]" in src[i:i + 160], (
        "the backtest commit no longer carries [skip ci]; at 96 commits/day "
        "this would drive the whole workflow_run fleet."
    )


# ── ML experiment persistence ─────────────────────────────────────────────────

def test_the_experiment_writes_a_repo_local_history():
    src = _MLSRC.read_text()
    assert "ml_experiments.json" in src, (
        "ml_experiment.py no longer persists to .github/state/. Its results "
        "existed only as stdout and a GitHub issue that does not exist."
    )
    assert "history.append(payload)" in src, "results are no longer appended"


def test_the_history_is_bounded():
    """agent_memory.json reached 47% of the git repo by not doing this."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mlx", _MLSRC)
    mod = importlib.util.module_from_spec(spec)
    # Executing the module would run the experiment; read the constant instead.
    src = _MLSRC.read_text()
    assert "_HISTORY_KEEP" in src, "no rolling cap on the experiment history"
    ns: dict = {}
    for line in src.splitlines():
        if line.startswith("_HISTORY_KEEP"):
            exec(line, ns)  # noqa: S102
    keep = ns.get("_HISTORY_KEEP")
    assert isinstance(keep, int) and 0 < keep <= 500, (
        f"_HISTORY_KEEP={keep!r} is missing or unbounded"
    )
    assert f"history[-_HISTORY_KEEP:]" in src, "the cap is defined but not applied"


def test_persistence_cannot_fail_the_experiment():
    """Bookkeeping must never lose a completed walk-forward run."""
    src = _MLSRC.read_text()
    i = src.index("ml_experiments.json")
    block = src[max(0, i - 600):i + 900]
    assert "except Exception" in block, (
        "the persistence block is unguarded — a corrupt or unwritable history "
        "file would discard the run that just completed"
    )


def test_the_print_is_kept():
    """The workflow tees stdout into /tmp/ml_results.json for its artifact."""
    src = _MLSRC.read_text()
    assert "print(json.dumps(payload, indent=2))" in src, (
        "stdout output was removed; the workflow's `| tee /tmp/ml_results.json` "
        "and its uploaded artifact both depend on it"
    )


def test_the_experiment_workflow_can_persist_what_it_produces():
    """`contents: read` cannot commit the new state file."""
    doc = yaml.safe_load(_MLEXP.read_text())
    perms = doc.get("permissions") or {}
    assert perms.get("contents") == "write", (
        "ml-experiments.yml still has contents: read, so the history file it "
        "now writes can never be committed — the results stay ephemeral."
    )
