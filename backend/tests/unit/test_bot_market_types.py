from app.models.bot import MARKET_TYPES

def test_market_types_include_all_desks():
    originals = ["equity", "crypto", "polymarket"]
    new_desks = {"options", "macro", "rates"}
    assert MARKET_TYPES[:3] == originals
    assert new_desks.issubset(set(MARKET_TYPES))
    assert len(MARKET_TYPES) == len(set(MARKET_TYPES))