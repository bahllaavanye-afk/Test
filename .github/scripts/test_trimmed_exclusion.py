"""A retired strategy must not be instantiated. Nothing tested that it isn't.

The trims pipeline is now correct end to end — fill_tracker writes attribution,
the trimmer reads the payload (not the envelope), the desk expands truncated
keys. The last link is the desk actually EXCLUDING a retired strategy, and that
was the one step covered only by presence:

    test_no_dead_desk_path  asserts `_trimmed_strategies()` is CALLED, and
                            called before `_load_strategy`

Neither assertion would notice if the `continue` were deleted. The exclusion
lived inline inside main()'s signal-generation stage, so nothing could reach it.

That gap is not theoretical in this pipeline. Twice a helper carried thorough
tests while its caller was silently broken:

  run_desk()                held this exact check and was never called at all
  strategy_trimmer.run()    iterated the envelope while evaluate_trim() — which
                            had seven passing tests — was always correct

Both times the tested unit was fine and the untested caller did nothing. So the
selection is now `_desk_strategies(names, trimmed)`, and these tests drive it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pandas")

_MOD = Path(__file__).parent / "desk_order_placer.py"
_spec = importlib.util.spec_from_file_location("dop_excl_test", _MOD)
dop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dop)  # type: ignore[union-attr]


class _Fake:
    def __init__(self, name): self.name = name


@pytest.fixture
def loader(monkeypatch):
    """Record which strategies were actually instantiated."""
    loaded: list[str] = []

    def fake_load(name):
        loaded.append(name)
        return _Fake(name)

    monkeypatch.setattr(dop, "_load_strategy", fake_load)
    return loaded


# ── the regression ───────────────────────────────────────────────────────────

def test_a_retired_strategy_is_never_instantiated(loader):
    """The whole point: a trimmed name must not even be loaded."""
    out = dop._desk_strategies(["momentum", "avellaneda_stoikov_mm", "mean_reversion"],
                               {"avellaneda_stoikov_mm"})
    assert [s.name for s in out] == ["momentum", "mean_reversion"]
    assert "avellaneda_stoikov_mm" not in loader, (
        "the retired strategy was instantiated — exclusion happens after load, "
        "or not at all"
    )


def test_the_live_case(loader):
    """`avellaneda_stoikov_mm` is what the first real trim resolves to."""
    out = dop._desk_strategies(["avellaneda_stoikov_mm"], {"avellaneda_stoikov_mm"})
    assert out == [] and loader == []


# ── it must not over-exclude ─────────────────────────────────────────────────

def test_an_empty_trim_set_excludes_nothing(loader):
    names = ["momentum", "mean_reversion", "supertrend"]
    out = dop._desk_strategies(names, set())
    assert [s.name for s in out] == names


def test_a_trim_naming_nothing_on_this_desk_is_a_no_op(loader):
    out = dop._desk_strategies(["momentum"], {"some_other_strategy"})
    assert [s.name for s in out] == ["momentum"]


def test_matching_is_exact_not_prefix(loader):
    """`supertrend` trimmed must NOT take out `supertrend_rsi_tv`.

    Prefix confusion between those two is a real hazard in this codebase — see
    the truncated-key expansion, which deliberately refuses to expand
    `supertrend` for exactly this reason.
    """
    out = dop._desk_strategies(["supertrend", "supertrend_rsi_tv"], {"supertrend"})
    assert [s.name for s in out] == ["supertrend_rsi_tv"]


# ── shape and safety ─────────────────────────────────────────────────────────

def test_an_unloadable_strategy_is_skipped_not_fatal(monkeypatch):
    """_load_strategy returns None for anything not in the registry."""
    monkeypatch.setattr(dop, "_load_strategy", lambda n: None if n == "ghost" else _Fake(n))
    out = dop._desk_strategies(["ghost", "momentum"], set())
    assert [s.name for s in out] == ["momentum"]


def test_trimming_every_strategy_yields_an_empty_desk(loader):
    """Deliberate: an all-retired desk trades nothing. Unlike the symbol
    denylist, there is no keep-at-least-one fallback — trading a strategy the
    system judged as bleeding would be worse than idling the desk."""
    out = dop._desk_strategies(["a", "b"], {"a", "b"})
    assert out == [] and loader == []


def test_order_is_preserved(loader):
    out = dop._desk_strategies(["c", "a", "b"], {"a"})
    assert [s.name for s in out] == ["c", "b"]
