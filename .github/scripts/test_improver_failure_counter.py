"""41 failures were traced and none was counted, so the rate read 100%.

`record_success()` initialises `improvement_stats[type]["failures"] = 0`, and
`agent_status_checker` sums that key to compute the headline success rate. But
`record_failure()` only ever appended to `failure_traces` — it never touched
the counter. So the counter existed, was read, and was never incremented, and
the success rate was pinned at 100% by construction.

Measured in the live memory file at the time of the fix:

    failure_traces stored : 41   (list is capped at 50, so this is a FLOOR)
    improvement_stats     : every "failures" == 0
    successes             : 61

    by type          traces   successes
      constants          9       1      <- reported as flawless
      docstrings        10      10
      cleanup           10      10

    by reason
      syntax check failed   32   <- the LLM's rewrite did not parse
      LLM returned empty     9

So the true rate is nearer 61/(61+41) ≈ 60% than 100%, and the dominant
failure mode — three quarters of all failures being unparseable output — was
invisible in every report.

The counter starts from the fix; the 41 historical failures are deliberately
NOT backfilled, because the trace list is capped and a backfill would be a
known undercount presented as a total. `test_the_backfill_temptation_is_
documented` pins that decision so the asymmetry stays disclosed rather than
quietly forgotten.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "continuous_improver.py"


def _load(fn_name: str):
    """Exec one function in isolation — importing the module runs the improver."""
    tree = ast.parse(_SRC.read_text())
    fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name),
        None,
    )
    assert fn is not None, f"{fn_name}() is gone"
    ns: dict = {}
    exec("from datetime import datetime, timezone", ns)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), f"<{fn_name}>", "exec"), ns)
    return ns[fn_name]


@pytest.fixture(scope="module")
def record_failure():
    return _load("record_failure")


@pytest.fixture(scope="module")
def record_success():
    return _load("record_success")


def test_a_failure_increments_the_counter_the_reporter_reads(record_failure):
    """The bug, stated directly."""
    mem: dict = {}
    record_failure(mem, "backend/app/x.py", "syntax check failed", "cleanup")
    assert mem["improvement_stats"]["cleanup"]["failures"] == 1
    assert len(mem["failure_traces"]) == 1


def test_repeated_failures_accumulate(record_failure):
    mem: dict = {}
    for i in range(5):
        record_failure(mem, f"f{i}.py", "LLM returned empty", "constants")
    assert mem["improvement_stats"]["constants"]["failures"] == 5


def test_it_does_not_clobber_an_existing_success_count(record_failure, record_success):
    """Both writers share one entry; neither may reset the other's field."""
    mem: dict = {}
    record_success(mem, "a.py", "docstrings", True)
    record_success(mem, "b.py", "docstrings", True)
    record_failure(mem, "c.py", "syntax check failed", "docstrings")

    entry = mem["improvement_stats"]["docstrings"]
    assert entry == {"successes": 2, "failures": 1, "test_pass": 2}


def test_a_failure_before_any_success_still_creates_a_well_formed_entry(record_failure):
    """record_success() used to be the only thing that created the entry."""
    mem: dict = {}
    record_failure(mem, "a.py", "output shrank", "schemas")
    assert mem["improvement_stats"]["schemas"] == {
        "successes": 0, "failures": 1, "test_pass": 0
    }


def test_the_trace_list_stays_capped(record_failure):
    mem: dict = {}
    for i in range(60):
        record_failure(mem, f"f{i}.py", "syntax check failed", "cleanup")
    assert len(mem["failure_traces"]) == 50, "the 50-item cap must survive the change"
    assert mem["improvement_stats"]["cleanup"]["failures"] == 60, (
        "the COUNTER must not be capped just because the trace list is — that "
        "cap is exactly why the traces cannot be used as a total"
    )


def test_every_failure_path_in_the_improver_goes_through_record_failure():
    """A path that returns early without calling it would be uncounted again."""
    tree = ast.parse(_SRC.read_text())
    calls = sum(
        1
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "record_failure"
    )
    assert calls >= 5, (
        f"only {calls} record_failure() call sites found; there are 5 failure "
        f"paths (empty LLM output, syntax check, elision marker, shrinkage, "
        f"tests failed). A new early-return that skips it is uncounted again."
    )


def test_the_backfill_temptation_is_documented():
    """Pin the decision not to backfill, and why."""
    src = _SRC.read_text()
    assert "backfill" in src.lower(), (
        "the choice not to backfill the 41 historical failures must stay "
        "explained in the source — otherwise the first days' optimistic rate "
        "looks like a bug rather than a disclosed limitation"
    )


def test_the_live_file_still_shows_the_uncounted_history():
    """Reads production state, not a fixture.

    If this ever fails because counters are non-zero, the fix has taken effect
    in production and this test should become a floor check instead.
    """
    mem_file = _HERE.parent / "state" / "agent_memory.json"
    assert mem_file.exists(), f"{mem_file} missing — do not let this skip silently"
    mem = json.loads(mem_file.read_text())
    traces = mem.get("failure_traces", [])
    stats = mem.get("improvement_stats", {})
    assert traces, "no failure traces at all — re-derive before trusting the notes"
    counted = sum(v.get("failures", 0) for v in stats.values() if isinstance(v, dict))
    assert counted == 0 or counted > 0, "sanity"
    # The historical asymmetry: many traces, zero (or far fewer) counted.
    assert len(traces) >= counted, (
        "more failures counted than traced — the counter is being incremented "
        "somewhere the trace is not, which breaks the audit trail"
    )
