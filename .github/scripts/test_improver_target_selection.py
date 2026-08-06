"""Half the improver's attempt budget went to files it can never improve.

`improve_file()` refuses anything over `MAX_FILE_CHARS`, and `main()` caps the
loop at 10 attempts for 5 wanted improvements. But `pick_target_file()` did not
know about the limit, so it kept handing back files guaranteed to be rejected.
Measured in run 30476849972 (2026-07-29 17:46) — 10 attempts, 5 committed, and
every one of the 5 failures was this:

    · backend/app/backtest/cpcv.py is 13806 chars (> 8000) — skipped…
    · backend/app/comparison/report_builder.py is 9242 chars (> 8000) — skipped…
    · backend/tests/unit/test_all_employees.py is 21032 chars (> 8000) — skipped…
    · backend/app/api/v1/discord_interactions.py is 8609 chars (> 8000) — skipped…

Filtering at selection **changes nothing about which files can be improved** —
those were already rejected 100% of the time. It only stops spending a scarce
attempt on a certain rejection. That is why this is safe to do without changing
policy, and it is the distinction the tests below pin.

Two properties matter and are easy to get wrong:

1. The check must apply to the FALLBACK list too. `pick_target_file()` retries
   with a repo-wide glob when the hour's pattern yields nothing; filtering only
   the first list would leak oversized files back in through the second.
2. It must not reintroduce a bare literal. The limit is read in three places
   now (selector, caller, guard) and they must agree — a second `8000` is how
   they drift.

Deliberately NOT changed here: `backend/tests/unit/*.py` stays in
CANDIDATE_PATTERNS. The improver writing false invariants into a shared test
(#1246: "Consecutive non-zero signals must alternate sign", false for any
trend-follower) is a real problem, but `test_cases` is a *configured*
improvement type — "Add 2-3 new unit test cases for edge cases not currently
tested" — so removing tests from selection would disable a designed function.
That trade belongs to a human, and is logged in IMPROVEMENTS.md.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "continuous_improver.py"
_REPO = _HERE.parents[1]


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(_SRC.read_text())


def _fn(tree: ast.Module, name: str) -> ast.FunctionDef:
    fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name),
        None,
    )
    assert fn is not None, f"{name}() not found"
    return fn


def _load_too_large():
    """Exec the predicate alone — importing the module runs the improver."""
    fn = _fn(ast.parse(_SRC.read_text()), "_too_large")
    ns: dict = {"os": os, "MAX_FILE_CHARS": 8000}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<t>", "exec"), ns)
    return ns["_too_large"]


def test_an_oversized_file_is_rejected(tmp_path):
    too_large = _load_too_large()
    big = tmp_path / "big.py"
    big.write_text("x = 1\n" * 3000)          # ~18 KB
    assert too_large(str(big)) is True


def test_a_normal_file_is_accepted(tmp_path):
    too_large = _load_too_large()
    ok = tmp_path / "ok.py"
    ok.write_text("def f():\n    return 1\n")
    assert too_large(str(ok)) is False


def test_a_missing_file_is_treated_as_unusable(tmp_path):
    """getsize() raises on a vanished file; the picker must not crash."""
    too_large = _load_too_large()
    assert too_large(str(tmp_path / "gone.py")) is True


def test_the_real_offenders_from_the_measured_run_are_now_excluded():
    """Against the actual repo, not a fixture — these are the files that wasted
    the budget in run 30476849972."""
    too_large = _load_too_large()
    offenders = [
        "backend/app/backtest/cpcv.py",
        "backend/app/comparison/report_builder.py",
        "backend/tests/unit/test_all_employees.py",
        "backend/app/api/v1/discord_interactions.py",
    ]
    present = [p for p in offenders if (_REPO / p).exists()]
    assert present, "none of the measured offenders still exist — re-derive the list"
    for path in present:
        assert too_large(str(_REPO / path)), (
            f"{path} was rejected by the guard in the measured run but the "
            f"selector would still pick it"
        )


def test_every_candidate_list_applies_the_filter(tree):
    """The fallback glob is a second path back to the same wasted attempts.

    This used to count `_too_large` calls and require two, one per candidate
    list. The selector now routes every list through a single `_usable()`
    helper, so counting would demand duplication that no longer exists. The
    property that actually matters is unchanged and is now checked directly:
    NO glob result may reach the caller without passing through the filter.
    That is strictly stronger — it fails for a third candidate list too, which
    the old count would have waved through."""
    picker = _fn(tree, "pick_target_file")
    unfiltered = []
    for node in ast.walk(picker):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "glob"):
            continue
        # Every glob() must sit directly inside a _usable(...) call.
        wrapped = any(
            isinstance(outer, ast.Call)
            and getattr(outer.func, "id", None) == "_usable"
            and any(a is node for a in outer.args)
            for outer in ast.walk(picker)
        )
        if not wrapped:
            unfiltered.append(ast.unparse(node))
    assert not unfiltered, (
        f"glob call(s) not wrapped in _usable(): {unfiltered}. Every candidate "
        f"list — the type's patterns and the repo-wide fallback — must be "
        f"filtered, or oversized and protected files leak back in."
    )
    # And the filter itself must still apply both checks.
    helper = _fn(tree, "_usable")
    names = {getattr(n.func, "id", None) for n in ast.walk(helper)
             if isinstance(n, ast.Call)}
    assert {"_is_protected", "_too_large"} <= names, (
        f"_usable() no longer applies both filters (calls: {names})")


def test_the_limit_is_still_a_single_constant():
    """Three readers now (selector, caller, guard) — they must not drift."""
    tree = ast.parse(_SRC.read_text())
    literals = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and n.value == 8000
    ]
    assert len(literals) == 1, (
        f"{len(literals)} bare 8000 literals. The limit is read by "
        f"pick_target_file(), main() and improve_file(); a second literal is how "
        f"they start disagreeing."
    )


def test_selection_still_finds_something_in_this_repo():
    """A filter that starves the picker would be worse than the waste.

    Runs the real selector over the real tree for every hour slot. If some slot
    yields nothing the fallback must cover it, so `None` is only acceptable when
    the whole repo has no eligible file — which would itself be a finding.
    """
    src = _SRC.read_text()
    tree = ast.parse(src)
    ns: dict = {"os": os, "glob": __import__("glob"), "random": __import__("random")}
    for name in ("MAX_FILE_CHARS", "CANDIDATE_PATTERNS", "PROTECTED_PREFIXES"):
        node = next(
            n for n in tree.body
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", None) == name for t in n.targets)
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<c>", "exec"), ns)
    for name in ("_is_protected", "_too_large", "_usable", "pick_target_file"):
        exec(
            compile(ast.Module(body=[_fn(tree, name)], type_ignores=[]), "<f>", "exec"),
            ns,
        )

    cwd = os.getcwd()
    os.chdir(_REPO)                       # patterns are repo-relative
    try:
        empty: list[int] = []
        for hour in range(24):
            if ns["pick_target_file"](hour, set()) is None:
                empty.append(hour)
    finally:
        os.chdir(cwd)
    assert not empty, (
        f"pick_target_file() returned None for hour slot(s) {empty} — the size "
        f"filter has starved selection and those runs would improve nothing"
    )
