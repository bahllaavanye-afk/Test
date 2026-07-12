"""Desk taxonomy: every registered strategy must resolve to exactly one real desk.

This guards the unified cross-desk view — a new strategy can't silently fall off a
desk (it would show up as 'Unknown' and fail here).
"""
from typing import Dict

from pydantic import BaseModel, Field, validator

from app.strategies import (
    STRATEGY_REGISTRY,
    desk_of,
    list_desks,
    strategies_by_desk,
)


class StrategyDeskInfo(BaseModel):
    """Schema representing a strategy and its assigned desk.

    Attributes
    ----------
    name: str
        The unique identifier of the strategy as registered in the registry.
    desk: str
        The desk name the strategy is assigned to. Must not be ``"Unknown"``.
    """

    name: str = Field(..., description="Unique strategy name from the registry", example="mean_rev_20_1.5")
    desk: str = Field(..., description="Desk to which the strategy belongs", example="Equities")

    @validator("desk")
    def desk_must_be_known(cls, v: str) -> str:
        """Validate that the desk is not the placeholder ``Unknown``."""
        if v == "Unknown":
            raise ValueError("Desk cannot be 'Unknown'")
        return v

    class Config:
        schema_extra = {
            "example": {"name": "mean_rev_20_1.5", "desk": "Equities"}
        }


def _live() -> Dict[str, type]:
    """Return a mapping of live strategy names to their classes, filtering out None entries."""
    return {n: c for n, c in STRATEGY_REGISTRY.items() if c is not None}


def test_every_strategy_resolves_to_a_real_desk():
    """All live strategies must resolve to a known desk."""
    orphans = [n for n in _live() if desk_of(n) == "Unknown"]
    assert not orphans, f"strategies with no desk: {orphans}"


def test_desks_partition_the_registry():
    """Each strategy should belong to exactly one desk and all desks cover the registry."""
    grouped = strategies_by_desk()
    flat = [n for members in grouped.values() for n in members]
    assert sorted(flat) == sorted(_live())          # covers every live strategy
    assert len(flat) == len(set(flat))              # each strategy in exactly one desk


def test_core_desks_present():
    """Core desks must always be present in the desk list."""
    desks = set(list_desks())
    # Equities + Crypto always exist; the finer desks are derived by convention/lists.
    assert {"Equities", "Crypto"} <= desks
    assert {"Options", "TradingView Indicators", "Macro", "Rates"} <= desks


def test_explicit_desk_attribute_wins(monkeypatch):
    """Explicit desk attribute on a strategy class should override derived desk."""
    name = next(iter(_live()))
    cls = STRATEGY_REGISTRY[name]
    monkeypatch.setattr(cls, "desk", "Macro", raising=False)
    try:
        assert desk_of(name) == "Macro"
    finally:
        monkeypatch.delattr(cls, "desk", raising=False)