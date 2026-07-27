"""Desk taxonomy: every registered strategy must resolve to exactly one real desk.

This guards the unified cross-desk view — a new strategy can't silently fall off a
desk (it would show up as 'Unknown' and fail here).
"""
from app.strategies import (
    STRATEGY_REGISTRY,
    desk_of,
    list_desks,
    strategies_by_desk,
)


def _live():
    return {n: c for n, c in STRATEGY_REGISTRY.items() if c is not None}


def test_every_strategy_resolves_to_a_real_desk():
    orphans = [n for n in _live() if desk_of(n) == "Unknown"]
    assert not orphans, f"strategies with no desk: {orphans}"


def test_desks_partition_the_registry():
    grouped = strategies_by_desk()
    flat = [n for members in grouped.values() for n in members]
    assert sorted(flat) == sorted(_live())          # covers every live strategy
    assert len(flat) == len(set(flat))              # each strategy in exactly one desk


def test_core_desks_present():
    desks = set(list_desks())
    # Equities + Crypto always exist; the finer desks are derived by convention/lists.
    assert {"Equities", "Crypto"} <= desks
    assert {"Options", "TradingView Indicators", "Macro", "Rates"} <= desks


def test_explicit_desk_attribute_wins(monkeypatch):
    name = next(iter(_live()))
    cls = STRATEGY_REGISTRY[name]
    monkeypatch.setattr(cls, "desk", "Macro", raising=False)
    try:
        assert desk_of(name) == "Macro"
    finally:
        monkeypatch.delattr(cls, "desk", raising=False)


def test_desk_of_unknown_strategy_returns_unknown():
    """A strategy name that is not in the registry should resolve to 'Unknown'."""
    assert desk_of("non_existent_strategy") == "Unknown"


def test_empty_desk_attribute_falls_back_to_unknown(monkeypatch):
    """If a strategy's explicit desk attribute is empty, it should be treated as unknown."""
    name = next(iter(_live()))
    cls = STRATEGY_REGISTRY[name]
    monkeypatch.setattr(cls, "desk", "", raising=False)
    try:
        assert desk_of(name) == "Unknown"
    finally:
        monkeypatch.delattr(cls, "desk", raising=False)


def test_strategies_by_desk_with_no_strategies(monkeypatch):
    """When the strategy registry is empty, strategies_by_desk should return an empty mapping."""
    original_registry = STRATEGY_REGISTRY.copy()
    try:
        monkeypatch.setattr('app.strategies.STRATEGY_REGISTRY', {}, raising=False)
        result = strategies_by_desk()
        assert result == {}
    finally:
        monkeypatch.setattr('app.strategies.STRATEGY_REGISTRY', original_registry, raising=False)