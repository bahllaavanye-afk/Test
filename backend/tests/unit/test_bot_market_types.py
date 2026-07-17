from app.models.bot import MARKET_TYPES


def test_market_types_include_all_desks():
    """Validate that MARKET_TYPES contains the expected original market types,
    includes the newly added desks, preserves order for the original entries,
    and contains no duplicates.
    """
    # Expected original market types in order
    expected_originals = ["equity", "crypto", "polymarket"]
    # Newly introduced market desks
    new_desks = {"options", "macro", "rates"}

    # Verify the first three entries match the original list exactly
    assert MARKET_TYPES[:3] == expected_originals, (
        f"First three market types should be {expected_originals}, got {MARKET_TYPES[:3]}"
    )

    # Ensure all new desks are present somewhere in the full list
    missing = new_desks - set(MARKET_TYPES)
    assert not missing, f"Missing expected market desks: {missing}"

    # Check for duplicate entries
    assert len(MARKET_TYPES) == len(set(MARKET_TYPES)), (
        "MARKET_TYPES contains duplicate entries"
    )

    # Verify total count matches the sum of originals and new desks
    expected_total = len(expected_originals) + len(new_desks)
    assert len(MARKET_TYPES) == expected_total, (
        f"MARKET_TYPES length should be {expected_total}, got {len(MARKET_TYPES)}"
    )