"""The roll call announced "0 total runs · 0% success rate" for 61 real runs.

`.github/state/agent_status.json` reported, across eighteen agents whose
workflows were visibly succeeding:

    "total_runs": 0, "success_rate_pct": 0

That is not evidence the agents were idle. `improvement_stats` has **two
writers with incompatible key spaces and incompatible schemas**, sharing one
dict:

    continuous_improver.record_success()   key: improvement_type ("cleanup")
                                           schema: successes / failures / test_pass
    SharedContext.record_success()         key: agent_name ("signal_runner")
                                           schema: runs / successes / last_summary

The reporter read `runs` — the SECOND schema — indexed by agent name — the
SECOND key space. `SharedContext.record_success()` has zero call sites outside
its own docstring example, so that dimension has never been written by
anything. The live file holds only the first writer's entries, none of which
carries a `runs` key, so `v.get("runs", 0)` returned 0 for all of them and
`total_runs` summed to zero. The `if total_runs else 0` guard then zeroed the
percentage too, so both numbers agreed and both were wrong.

The measured truth in the live file at the time of the fix: 61 recorded
attempts across 8 improvement types.

Two things this deliberately does NOT do:

* It does not invent a per-agent run count. Nothing writes that dimension, so
  the roll call now omits the "(N runs)" suffix rather than printing "(0 runs)"
  next to every name, which read as "this agent has never run".
* It does not publish a 100% success rate. The only live writer calls
  `record_success()` and never increments `failures`, so the ratio is
  definitionally 100 — quoting it as a *measured* rate would be a fabricated
  metric. The header says "no failures recorded" until a failure is, and the
  status file ships `failures_recorded` so a consumer can tell a real 100%
  from an untracked one.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "agent_status_checker.py"
sys.path.insert(0, str(_HERE))


def _load_runs_fn():
    """Import `_runs` without executing the module (it hits the network/LLM)."""
    tree = ast.parse(_SRC.read_text())
    fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_runs"),
        None,
    )
    assert fn is not None, "_runs() is gone — the count is being read raw again"
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<_runs>", "exec"), ns)
    return ns["_runs"]


@pytest.fixture(scope="module")
def runs():
    return _load_runs_fn()


def test_the_real_shape_counts_instead_of_returning_zero(runs):
    """The exact entry shape the live file holds — this is the bug."""
    assert runs({"successes": 10, "failures": 0, "test_pass": 10}) == 10
    assert runs({"successes": 3, "failures": 2, "test_pass": 3}) == 5


def test_an_explicit_runs_key_still_wins(runs):
    """If the per-agent dimension ever starts being written, honour it."""
    assert runs({"runs": 7, "successes": 7}) == 7
    assert runs({"runs": 0, "successes": 99}) == 0


@pytest.mark.parametrize("bad", [None, [], "nope", {"successes": "x"}, {}])
def test_garbage_does_not_raise(runs, bad):
    assert runs(bad) == 0


def test_the_live_memory_file_now_totals_more_than_zero(runs):
    """Against real state, not a fixture — the fix must move the actual number.

    A unit test on a hand-made dict would have passed against the broken code
    just as happily if I had written the dict in the broken shape. This reads
    what production actually wrote.
    """
    mem_file = _HERE.parent / "state" / "agent_memory.json"
    assert mem_file.exists(), (
        f"{mem_file} is missing. This assertion is deliberate: the first draft "
        f"pointed at the wrong directory and pytest.skip()'d green, which is "
        f"the same silent-absence failure this whole file is about."
    )
    stats = json.loads(mem_file.read_text()).get("improvement_stats", {})
    if not stats:
        pytest.skip("no improvement_stats recorded yet")

    old_total = sum(v.get("runs", 0) for v in stats.values() if isinstance(v, dict))
    new_total = sum(runs(v) for v in stats.values())

    assert old_total == 0, (
        "the old read now finds data — re-check whether this bug still exists"
    )
    assert new_total > 0, (
        f"recorded improvements exist ({stats}) but the count is still 0"
    )


def test_no_success_rate_is_published_without_a_denominator():
    """A rate that can only ever be 100% must not be presented as measured."""
    src = _SRC.read_text()
    assert "no failures recorded" in src, (
        "the header quotes a success rate unconditionally again; with failures "
        "never incremented, that number is always 100 and means nothing"
    )
    assert "failures_recorded" in src, (
        "the status file must ship the failure counter so a consumer can tell a "
        "real 100% from an untracked one"
    )


def test_the_per_agent_suffix_is_omitted_when_there_is_no_data():
    """'(0 runs)' beside every agent read as 'this agent has never run'."""
    src = _SRC.read_text()
    assert "({s['runs']} runs)\\n" not in src, (
        "the roll-call line prints '(0 runs)' unconditionally again"
    )
    assert "if s[\"runs\"] else \"\"" in src, (
        "the run-count suffix must be conditional on there being a count"
    )
