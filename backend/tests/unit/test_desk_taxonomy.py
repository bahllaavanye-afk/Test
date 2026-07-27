"""Desk taxonomy tests: ensure every registered strategy resolves to exactly one real desk.

These tests guard the unified cross‑desk view — a new strategy can't silently fall off a
desk (it would show up as 'Unknown' and fail here).
"""
from typing import Dict, Type

from app.strategies import (
    STRATEGY_REGISTRY,
    desk_of,
    list_desks,
    strategies_by_desk,
)


def _live() -> Dict[str, Type]:
    """Return a mapping of live strategy names to their classes, excluding
    placeholders (where the registry entry is ``None``).
    """
    return {name: cls for name, cls in STRATEGY_REGISTRY.items() if cls is not None}


def test_every_strategy_resolves_to_a_real_desk():
    """All live strategies must map to a known desk."""
    orphans = [name for name in _live() if desk_of(name) == "Unknown"]
    assert not orphans, f"strategies with no desk: {orphans}"


def test_desks_partition_the_registry():
    """Each live strategy should appear in exactly one desk grouping."""
    grouped = strategies_by_desk()
    flat = [strategy_name for members in grouped.values() for strategy_name in members]
    assert sorted(flat) == sorted(_live()), "registry coverage mismatch"
    assert len(flat) == len(set(flat)), "duplicate strategy assignment detected"


def test_core_desks_present():
    """Core desks must always be present in the desk list."""
    desks = set(list_desks())
    # Equities + Crypto always exist; the finer desks are derived by convention/lists.
    assert {"Equities", "Crypto"} <= desks
    assert {"Options", "TradingView Indicators", "Macro", "Rates"} <= desks


def test_explicit_desk_attribute_wins(monkeypatch):
    """If a strategy defines an explicit ``desk`` attribute, it should override the
    default mapping.
    """
    name = next(iter(_live()))
    cls = STRATEGY_REGISTRY[name]
    monkeypatch.setattr(cls, "desk", "Macro", raising=False)
    try:
        assert desk_of(name) == "Macro"
    finally:
        monkeypatch.delattr(cls, "desk", raising=False)