"""A file too large to send was logged as "LLM returned empty". It was never sent.

`improve_file()` returns `None` for two unrelated reasons:

    the file exceeds MAX_FILE_CHARS   -> deliberate policy skip, no LLM call
    the model returned nothing        -> a real failed attempt

The caller could not tell them apart, so both landed on:

    record_failure(mem, target, "LLM returned empty", improvement_type)

Measured in run 30476849972 (2026-07-29 17:46), the last full pre-fix run —
10 attempts, and every one of its 5 failures was this:

    · backend/app/backtest/cpcv.py is 13806 chars (> 8000) — skipped…
      ✗ LLM returned nothing for backend/app/backtest/cpcv.py
    · backend/app/comparison/report_builder.py is 9242 chars (> 8000) — skipped…
      ✗ LLM returned nothing for backend/app/comparison/report_builder.py
    · backend/tests/unit/test_all_employees.py is 21032 chars (> 8000) — skipped…
    · backend/app/api/v1/discord_interactions.py is 8609 chars (> 8000) — skipped…

Half that run's attempt budget, all charged to a model that never saw the
input. The counter fixed in #1235 had just started working, and this was
already corrupting it — the first thing it would have measured is a failure
rate dominated by non-failures.

Two things follow, and both are asserted here:

1. An oversized file must be a SKIP, not a failure. The caller checks the size
   itself and `continue`s without recording anything.
2. The guard inside `improve_file()` stays. It is load-bearing — the comment
   there records that a whole-file rewrite of an 8000+ char file is how PR #420
   deleted 6 of 7 methods from `brokers/alpaca.py` — so it is kept as defence in
   depth rather than moved.

Not addressed here, and worth its own look: burning 5 of 10 attempts on files
the guard will always reject means `pick_target_file()` is choosing candidates
it cannot act on. Filtering by size at selection time would roughly double the
useful work per run. Logged in IMPROVEMENTS.md rather than fixed blind.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent / "continuous_improver.py"


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(_SRC.read_text())


def _main(tree: ast.Module) -> ast.FunctionDef:
    fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"),
        None,
    )
    assert fn is not None, "main() not found"
    return fn


def test_the_size_limit_is_a_named_constant(tree):
    """Two call sites must agree; a second literal 8000 would silently drift."""
    assigns = [
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "MAX_FILE_CHARS" for t in n.targets)
    ]
    assert assigns, "MAX_FILE_CHARS is gone — the two size checks can now disagree"
    assert assigns[0].value.value == 8000

    literals = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and n.value == 8000
    ]
    assert len(literals) == 1, (
        f"{len(literals)} bare 8000 literals — the limit must live in "
        f"MAX_FILE_CHARS only, or the caller and improve_file() will drift apart"
    )


def test_the_caller_checks_size_before_calling_improve_file(tree):
    """Order matters: checking after the call cannot distinguish the two Nones."""
    main = _main(tree)
    size_check = call = None
    for node in ast.walk(main):
        if (
            size_check is None
            and isinstance(node, ast.Compare)
            and any(getattr(c, "id", None) == "MAX_FILE_CHARS" for c in node.comparators)
        ):
            size_check = node.lineno
        if (
            call is None
            and isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "improve_file"
        ):
            call = node.lineno
    assert size_check is not None, (
        "main() no longer checks MAX_FILE_CHARS — oversized files fall back "
        "through improve_file() and are recorded as LLM failures again"
    )
    assert call is not None, "main() no longer calls improve_file()"
    assert size_check < call, (
        f"the size check (line {size_check}) must precede the improve_file() "
        f"call (line {call}); after it, the two None cases are indistinguishable"
    )


def test_an_oversized_file_records_no_failure(tree):
    """The skip branch must not call record_failure()."""
    main = _main(tree)
    for node in ast.walk(main):
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and any(getattr(c, "id", None) == "MAX_FILE_CHARS" for c in node.test.comparators)
        ):
            continue
        calls = {
            getattr(n.func, "id", None)
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
        }
        assert "record_failure" not in calls, (
            "the oversized-file branch records a failure. The model was never "
            "called; charging it to the failure rate makes the metric wrong."
        )
        assert any(isinstance(n, ast.Continue) for n in ast.walk(node)), (
            "the oversized-file branch must `continue` to the next candidate"
        )
        return
    pytest.fail("no MAX_FILE_CHARS guard found in main()")


def test_the_guard_inside_improve_file_is_still_there(tree):
    """Defence in depth — this one is load-bearing (the PR #420 lesson)."""
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "improve_file"
    )
    guards = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Compare)
        and any(getattr(c, "id", None) == "MAX_FILE_CHARS" for c in n.comparators)
    ]
    assert guards, (
        "improve_file() no longer guards its own input size. The caller's check "
        "is not a substitute — any other call site would bypass it, and a "
        "whole-file rewrite above this size is how PR #420 destroyed alpaca.py."
    )


def test_the_run_summary_reports_skips_separately(tree):
    """A skip that is invisible is indistinguishable from work not attempted."""
    main = _main(tree)
    names = {
        t.id
        for n in ast.walk(main)
        if isinstance(n, (ast.Assign, ast.AugAssign))
        for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
        if isinstance(t, ast.Name)
    }
    assert "skipped_too_large" in names, (
        "the skip count is not tracked, so a run that burns its whole budget on "
        "oversized files looks identical to one that simply found little to do"
    )
    assert "skipped_too_large" in _SRC.read_text().split("✓ Committed {improved_count}")[-1], (
        "the skip count must be printed in the run summary, not just counted"
    )
