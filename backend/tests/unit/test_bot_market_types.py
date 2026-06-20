"""The expanded bot market-type set (desk consolidation stage 2)."""
from app.models.bot import MARKET_TYPES


def test_market_types_include_all_desks():
    assert MARKET_TYPES[:3] == ["equity", "crypto", "polymarket"]  # originals preserved
    assert {"options", "macro", "rates"} <= set(MARKET_TYPES)      # new desks added
    assert len(MARKET_TYPES) == len(set(MARKET_TYPES))             # no dupes
