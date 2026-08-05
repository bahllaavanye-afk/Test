from typing import Iterable, List

from app.models.bot import MARKET_TYPES


def _validate_market_types(market_types: Iterable) -> None:
    """
    Validate the MARKET_TYPES constant.

    Ensures that:
    - market_types is an iterable of strings
    - there are no duplicate entries
    - the collection is non‑empty

    Raises:
        ValueError: If any validation rule is violated.
    """
    if not isinstance(market_types, (list, tuple, set)):
        raise ValueError(
            f"MARKET_TYPES must be a list, tuple, or set, got {type(market_types).__name__}"
        )

    if not market_types:
        raise ValueError("MARKET_TYPES cannot be empty")

    if not all(isinstance(item, str) for item in market_types):
        raise ValueError("All entries in MARKET_TYPES must be strings")

    if len(set(market_types)) != len(list(market_types)):
        raise ValueError("MARKET_TYPES contains duplicate entries")


def test_market_types_include_all_desks():
    _validate_market_types(MARKET_TYPES)

    originals = ["equity", "crypto", "polymarket"]
    new_desks = {"options", "macro", "rates"}

    assert MARKET_TYPES[:3] == originals
    assert new_desks.issubset(set(MARKET_TYPES))
    assert len(MARKET_TYPES) == len(set(MARKET_TYPES))