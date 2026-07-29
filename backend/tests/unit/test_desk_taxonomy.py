"""Desk taxonomy: every registered strategy must resolve to exactly one real desk.

This guards the unified cross-desk view — a new strategy can't silently fall off a
desk (it would show up as 'Unknown' and fail here).
"""
import pytest
from app.strategies import (
    STRATEGY_REGISTRY,
    desk_of,
    list_desks,
    strategies_by_desk,
)


def _live():
    """Return a dict of live strategies, safely handling None or malformed registry."""
    if not isinstance(STRATEGY_REGISTRY, dict):
        return {}
    return {n: c for n, c in STRATEGY_REGISTRY.items() if c is not None}


def test_every_strategy_resolves_to_a_real_desk():
    live_strategies = _live()
    if not live_strategies:
        pytest.skip("No live strategies to test.")
    orphans = [n for n in live_strategies if desk_of(n) in (None, "Unknown")]
    assert not orphans, f"strategies with no desk: {orphans}"


def test_desks_partition_the_registry():
    grouped = strategies_by_desk()
    if not isinstance(grouped, dict) or not grouped:
        pytest.skip("No desk grouping available.")
    flat = [n for members in grouped.values() for n in members]
    live_strategies = _live()
    # covers every live strategy
    assert sorted(flat) == sorted(live_strategies)
    # each strategy in exactly one desk
    assert len(flat) == len(set(flat))


def test_core_desks_present():
    desks = set(list_desks() or [])
    # Equities + Crypto always exist; the finer desks are derived by convention/lists.
    required_core = {"Equities", "Crypto"}
    required_extended = {"Options", "TradingView Indicators", "Macro", "Rates"}
    assert required_core <= desks, f"Missing core desks: {required_core - desks}"
    assert required_extended <= desks, f"Missing extended desks: {required_extended - desks}"


def test_explicit_desk_attribute_wins(monkeypatch):
    live_strategies = _live()
    if not live_strategies:
        pytest.skip("No live strategies to test explicit desk attribute.")
    name = next(iter(live_strategies))
    cls = STRATEGY_REGISTRY[name]
    monkeypatch.setattr(cls, "desk", "Macro", raising=False)
    try:
        assert desk_of(name) == "Macro"
    finally:
        monkeypatch.delattr(cls, "desk", raising=False)