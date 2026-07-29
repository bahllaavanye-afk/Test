"""The P&L attribution file was recomputed and thrown away on every run.

`fill_tracker.py` fetches filled orders, attributes them back to strategies via
the `client_order_id` encoding, and writes cumulative stats to

    backend/performance_log/strategy_performance.json

`fill-tracking.yml` ran it on schedule and reported success — but never
committed the output. Actions runners are ephemeral, so the file existed for
the length of the job and vanished. It has never been in the repository.

THREE consumers read that exact path, and all three were therefore inert:

  strategy_trimmer.py     load_perf() -> {} -> no strategy ever evaluated,
                          so .github/state/strategy_trims.json is never written
  strategy_auto_tuner.py  prints "not found — no data to tune from" and stops
  desk_order_placer.py    _trimmed_strategies() reads the trims file that the
                          trimmer never produces

So the file-based retirement path was dead end-to-end, at the source rather
than at the consumer. (The desk's OTHER pruning mechanism — attribution weights
from /api/v1/leaderboard/live, which sets weight 0.0 and skips the order with a
`✂ pruned by attribution` line — is live and unaffected. That distinction
matters: losing strategies WERE being stopped; the redundant file-based trimmer
was the part that never worked.)

Same shape as the dead `run_desk()` in test_no_dead_desk_path.py: machinery
that runs on schedule, exits zero, and produces nothing that outlives the job.
A workflow that writes a file it does not commit is indistinguishable from one
that does — until something tries to read it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_WF = Path(__file__).resolve().parents[1] / "workflows" / "fill-tracking.yml"
_SCRIPTS = Path(__file__).resolve().parent
_PERF_PATH = "backend/performance_log/strategy_performance.json"


@pytest.fixture(scope="module")
def wf() -> str:
    assert _WF.is_file(), f"missing {_WF}"
    return _WF.read_text()


def test_the_workflow_commits_the_file_it_writes(wf):
    assert _PERF_PATH in wf, (
        "fill-tracking.yml never references the file its script writes, so the "
        "attribution data cannot outlive the runner"
    )
    assert "git commit" in wf and "git push" in wf, (
        "the tracker's output is written to an ephemeral runner and discarded"
    )


def test_it_has_write_permission(wf):
    """A commit step without contents: write fails at push time, not at parse."""
    m = re.search(r"^permissions:\s*$(.*?)^\S", wf, re.MULTILINE | re.DOTALL)
    assert m, "no permissions block — the default may be read-only"
    assert "contents: write" in m.group(1)


def test_the_commit_is_conditional(wf):
    """An unconditional commit fails the job on a no-op run."""
    assert "git diff --quiet" in wf, "commit must be skipped when nothing changed"


def test_the_commit_does_not_retrigger_ci(wf):
    """This workflow triggers on CI completion — an unmarked commit would loop."""
    body = wf.split("Persist strategy performance", 1)[1]
    assert "[skip ci]" in body, (
        "fill-tracking runs on workflow_run:[CI]; a commit without [skip ci] "
        "risks a self-sustaining loop"
    )


def test_the_push_retries(wf):
    """Concurrent state-bot pushes to main are routine in this repo."""
    body = wf.split("Persist strategy performance", 1)[1]
    assert "for i in" in body and "sleep" in body, "push should back off and retry"


# ── the consumers must keep agreeing on the path ─────────────────────────────

@pytest.mark.parametrize("script", ["strategy_trimmer.py", "strategy_auto_tuner.py", "fill_tracker.py"])
def test_producer_and_consumers_use_the_same_path(script):
    """A silent path divergence here would restore the exact original bug."""
    src = (_SCRIPTS / script).read_text()
    assert '"performance_log"' in src or "performance_log" in src, script
    assert "strategy_performance.json" in src, script


def test_the_trimmer_still_fails_soft_on_a_missing_file():
    """Until the first commit lands, the file is absent — that must not crash."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "trimmer_under_test", _SCRIPTS / "strategy_trimmer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert mod.load_perf() == {} or isinstance(mod.load_perf(), dict)
    assert isinstance(mod.load_trims(), dict)
