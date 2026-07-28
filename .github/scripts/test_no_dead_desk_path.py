"""Three live-looking guards were wired into a function nothing called.

`run_desk()` was a complete SECOND implementation of the desk loop — fetch
bars, analyze, size, place — superseded by the staged pipeline in `main()` and
then left in the file with **zero call sites**. Nobody noticed, because it
reads exactly like the code that runs. Three guards lived only inside it:

    _filter_tradable_crypto()   the delisted-pair filter
    _apply_denylist()           the "broker refuses it" denylist
    _trimmed_strategies()       retired strategies must not trade

None of them had ever executed in production.

The cost was not just the missing behaviour — it was a **wrong conclusion**.
Across three ticks I instrumented the tradable filter, watched live runs, and
reasoned: "the filter runs unconditionally for every desk and prints whatever
it drops; it printed nothing; therefore Alpaca lists MKR/USD as active while
its order engine refuses it." The premise was false. The filter never ran, so
the absence of a `skipping` line said nothing at all about Alpaca's metadata,
and a fix shipped into `run_desk` was inert while passing all of its tests.

That is the specific danger of a dead parallel implementation: a change made
there looks correct, tests green, and changes nothing. The strategy trimmer is
the starkest case — the whole performance-pruning loop was decorative, so a
strategy retired for losing money kept trading.

This test fails if any of those helpers loses its production call site again.
It deliberately checks the CALL, not the definition: an unused function that
still exists is exactly the failure being guarded against.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).parent / "desk_order_placer.py"


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(_SRC.read_text())


def _called_names(node: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def _function(tree: ast.Module, name: str):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    return None


@pytest.mark.parametrize("helper", [
    "_filter_tradable_crypto",
    "_apply_denylist",
    "_denylisted_assets",
    "_trimmed_strategies",
])
def test_the_guard_is_actually_called_from_main(tree, helper):
    """Defined-but-uncalled is the bug. Assert the call, not the def."""
    main = _function(tree, "main")
    assert main is not None, "main() not found — the pipeline entry point moved"
    assert helper in _called_names(main), (
        f"{helper}() is never called from main(). It was wired into run_desk(), "
        f"which nothing called, so it silently did nothing in production."
    )


def test_run_desk_is_gone(tree):
    """The dead parallel implementation must not come back."""
    assert _function(tree, "run_desk") is None, (
        "run_desk() is back. It duplicates the staged pipeline; a fix applied "
        "to it looks correct, passes tests, and changes nothing."
    )


def test_every_top_level_helper_is_reachable(tree):
    """Catch the NEXT one of these before it costs another session.

    Any private helper that nothing calls is either dead or a decoy. Genuine
    entry points and the deliberately-unwired are listed explicitly, so adding
    to the exemption list is a conscious act rather than an oversight.
    """
    _ENTRY_POINTS = {"main"}
    defined, referenced = {}, set()
    for n in tree.body:                       # top level only
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined[n.name] = n.lineno
    for n in ast.walk(tree):
        # ANY load counts, not just a Call: `asyncio.to_thread(_alpaca_get_sync,
        # ...)` passes the function by reference and is a perfectly live use.
        # Counting calls only here would flag every threaded helper.
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            referenced.add(n.id)
        elif isinstance(n, ast.Attribute):
            referenced.add(n.attr)

    # A function referenced ONLY inside its own body is still dead (recursion
    # does not make it reachable), so discount self-references.
    orphans = {}
    for name, line in defined.items():
        if name in _ENTRY_POINTS:
            continue
        own = _function(tree, name)
        self_refs = {
            id(n) for n in ast.walk(own)
            if isinstance(n, ast.Name) and n.id == name
        } if own else set()
        external = any(
            isinstance(n, ast.Name) and n.id == name and id(n) not in self_refs
            for n in ast.walk(tree)
        )
        if not external:
            orphans[name] = line
    assert not orphans, (
        "These functions are defined but never called anywhere in the file — "
        "dead code that reads like live code:\n"
        + "\n".join(f"  {n} (line {l})" for n, l in sorted(orphans.items(), key=lambda kv: kv[1]))
    )


def test_the_universe_is_trimmed_before_bars_are_fetched(tree):
    """Order matters: filtering after the fetch would still spend the request.

    The point of moving this into the pipeline was to stop paying for symbols
    that cannot trade — bar requests, analyze() calls, and a scarce top-K slot.
    """
    src = _SRC.read_text()
    filt = src.index("_filter_tradable_crypto(list(")
    fetch = src.index("_get_bars_batch(all_symbols)")
    assert filt < fetch, "the universe must be trimmed BEFORE the bars batch request"


def test_the_trimmer_runs_before_strategies_are_loaded(tree):
    src = _SRC.read_text()
    trim = src.index("_trimmed = _trimmed_strategies()")
    load = src.index("s = _load_strategy(sname)")
    assert trim < load, "retired strategies must be excluded before they are instantiated"
