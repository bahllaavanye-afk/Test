"""A trim keyed by a truncated name retires nothing.

The attribution pipeline finally produced its artifact on 2026-07-29 10:19,
after a five-change chase. Its first content:

    strategies: {"avellaneda": {trades: 10, win_rate: 0.6,
                                total_return_pct: -7.9157}}

`evaluate_trim` on that is `(True, "cumulative return -7.9% <= -5.0% over 10
trades")` — so the trimmer will retire it on its next run. But the key is
`avellaneda`, TRUNCATED: those ten fills were placed before the full-name
client_order_id fix, when orders were tagged `qe-{name[:10]}-...`.

The desk checks `if sname in _trimmed` using the FULL registry name,
`avellaneda_stoikov_mm`. Truncated key vs full name never matches, so the trim
would be written and retire nothing — a phantom entry, and the strategy keeps
bleeding.

New fills now carry full names, so this self-resolves as the old data ages out
of the 7-day window. This closes the tail instead of waiting: existing
attribution becomes usable immediately.

The expansion rule mirrors
backend/app/tasks/desk_trade_sync.parse_strategy_from_coid — expand only when
EXACTLY ONE registry entry shares the prefix — because the ambiguity is real:

    commodity_  -> commodity_momentum, commodity_reversion, commodity_trend
    supertrend  -> supertrend (a real strategy!), supertrend_rsi_tv

`supertrend` is the dangerous case: it is simultaneously a valid registry name
and the 10-char prefix of another strategy. Expanding it would retire the wrong
one, so a key that is itself in the registry is returned untouched.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pandas")

_MOD = Path(__file__).parent / "desk_order_placer.py"
_spec = importlib.util.spec_from_file_location("dop_trim_test", _MOD)
dop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_spec and dop)  # type: ignore[arg-type]

_REG = {"avellaneda_stoikov_mm", "vol_of_vol_timing", "supertrend",
        "supertrend_rsi_tv", "commodity_momentum", "commodity_reversion",
        "commodity_trend", "mean_reversion"}


# ── the live case ────────────────────────────────────────────────────────────

def test_the_live_truncated_key_expands():
    """The exact key the artifact landed with on 2026-07-29."""
    assert dop._expand_truncated("avellaneda", _REG) == "avellaneda_stoikov_mm"


def test_vol_of_vol_expands():
    assert dop._expand_truncated("vol_of_vol", _REG) == "vol_of_vol_timing"


# ── ambiguity must NOT be guessed ────────────────────────────────────────────

def test_a_name_that_is_itself_a_strategy_is_left_alone():
    """`supertrend` is a real strategy AND a prefix of supertrend_rsi_tv.

    Expanding it would retire the wrong strategy — the one that was never
    judged.
    """
    assert dop._expand_truncated("supertrend", _REG) == "supertrend"


def test_a_prefix_matching_several_strategies_is_left_alone():
    assert dop._expand_truncated("commodity_", _REG) == "commodity_"


def test_an_unknown_key_is_left_alone():
    assert dop._expand_truncated("no_such_thing", _REG) == "no_such_thing"


@pytest.mark.parametrize("junk", ["", None])
def test_empty_input_is_safe(junk):
    assert dop._expand_truncated(junk, _REG) == junk


def test_a_full_name_passes_through_unchanged():
    assert dop._expand_truncated("mean_reversion", _REG) == "mean_reversion"


# ── the loader keeps BOTH forms ──────────────────────────────────────────────

def test_the_trims_loader_keeps_both_forms(tmp_path, monkeypatch):
    """A trims file may hold either form; matching must work for both."""
    import json
    scripts = tmp_path / "scripts"; scripts.mkdir()
    state = tmp_path / "state"; state.mkdir()
    (state / "strategy_trims.json").write_text(json.dumps({
        "avellaneda": {"trimmed_at": "x", "reason": "y"},
    }))
    monkeypatch.setattr(dop, "__file__", str(scripts / "desk_order_placer.py"))

    import sys, types
    fake = types.ModuleType("app.strategies")
    fake.STRATEGY_REGISTRY = {n: object for n in _REG}
    monkeypatch.setitem(sys.modules, "app.strategies", fake)

    out = dop._trimmed_strategies()
    assert "avellaneda_stoikov_mm" in out, "expanded name missing — trim is a phantom"
    assert "avellaneda" in out, "original key dropped — a full-name trims file would break"


def test_a_missing_trims_file_is_empty(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"; scripts.mkdir()
    monkeypatch.setattr(dop, "__file__", str(scripts / "desk_order_placer.py"))
    assert dop._trimmed_strategies() == set()
