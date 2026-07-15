"""Desk configuration guard: every DESKS entry must be tradeable as written.

A desk with a typo'd strategy name silently trades with FEWER strategies than
configured (the loader skips unknown names) — no crash, no alert, just missing
coverage. Same for a strategy missing from the regime map: it falls back to
_DEFAULT_REGIMES and trades in regimes it was never validated for. These pin
the config so adding a desk (Commodities was added 2026-07-15) can't ship
half-wired.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")

_MOD = Path(__file__).parent / "desk_order_placer.py"
_BACKEND = Path(__file__).parent.parent.parent / "backend"

os.environ.setdefault("SECRET_KEY", "test-secret-key-32-bytes-hex-xxxxxx")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_desk_cfg.db")


def _load():
    spec = importlib.util.spec_from_file_location("dop_cfg_test", _MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


dop = _load()


def _registry():
    sys.path.insert(0, str(_BACKEND))
    from app.strategies import STRATEGY_REGISTRY
    return STRATEGY_REGISTRY


def test_desk_names_unique():
    names = [d.name for d in dop.DESKS]
    assert len(names) == len(set(names)), f"duplicate desk names: {names}"


def test_every_desk_strategy_exists_in_registry():
    registry = _registry()
    missing = {
        f"{d.name}/{s}"
        for d in dop.DESKS
        for s in d.strategy_names
        if registry.get(s) is None
    }
    assert not missing, f"desks reference unknown/unloadable strategies: {sorted(missing)}"


def test_every_desk_has_symbols_and_sane_limits():
    for d in dop.DESKS:
        assert d.symbols, f"{d.name}: empty symbol list"
        assert len(d.symbols) == len(set(d.symbols)), f"{d.name}: duplicate symbols"
        assert d.notional_usd > 0, f"{d.name}: non-positive notional"
        assert 0.0 < d.confidence_min <= 1.0, f"{d.name}: bad confidence_min"
        assert d.slack_channel.startswith("#"), f"{d.name}: bad channel {d.slack_channel!r}"


def test_desk_strategies_have_explicit_regime_mapping():
    """Every desk-wired strategy should be consciously placed in the regime map —
    an unmapped one trades on _DEFAULT_REGIMES without anyone deciding that."""
    unmapped = {
        s
        for d in dop.DESKS
        for s in d.strategy_names
        if s not in dop._STRATEGY_REGIME_MAP
    }
    assert not unmapped, (
        f"desk strategies missing from _STRATEGY_REGIME_MAP (add an explicit "
        f"regime entry): {sorted(unmapped)}"
    )


def test_crypto_desk_is_always_open_and_others_arent_mislabeled():
    for d in dop.DESKS:
        is_crypto = any("/" in s for s in d.symbols)
        if is_crypto:
            assert d.always_open, f"{d.name}: crypto desk must be always_open"
        else:
            assert not d.always_open, (
                f"{d.name}: equity-hours desk marked always_open — it would "
                f"place orders into a closed exchange all night"
            )
