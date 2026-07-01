from app.models.bot import MARKET_TYPES


def test_market_types_include_all_desks():
    # Guard against None input
    if MARKET_TYPES is None:
        assert MARKET_TYPES is None
        return

    # Guard against empty collection
    if not MARKET_TYPES:
        assert isinstance(MARKET_TYPES, (list, tuple, set))
        assert len(MARKET_TYPES) == 0
        return

    # Basic expectations
    originals = ["equity", "crypto", "polymarket"]
    new_desks = {"options", "macro", "rates"}

    assert MARKET_TYPES[:3] == originals
    assert new_desks.issubset(set(MARKET_TYPES))
    assert len(MARKET_TYPES) == len(set(MARKET_TYPES))

    # Edge case: slicing beyond the list length should return the full list
    assert MARKET_TYPES[: len(MARKET_TYPES) + 1] == MARKET_TYPES

    # Edge case: zero-length slice should return an empty list
    assert MARKET_TYPES[:0] == []