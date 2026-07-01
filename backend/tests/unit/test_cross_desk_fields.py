"""Cross-desk tracking fields on Position/Order (desk consolidation stage 3)."""
from app.models.order import Order
from app.models.position import Position

CROSS_DESK_FIELDS = {
    "asset_class",
    "underlying_symbol",
    "expiry",
    "strike",
    "option_right",
    "contract_multiplier",
}
DEFAULT_ASSET_CLASS = "equity"


def test_position_has_cross_desk_fields():
    assert CROSS_DESK_FIELDS <= set(Position.__table__.columns.keys())


def test_order_has_cross_desk_fields():
    assert CROSS_DESK_FIELDS <= set(Order.__table__.columns.keys())


def test_asset_class_defaults_to_equity():
    for model in (Position, Order):
        col = model.__table__.columns["asset_class"]
        assert col.default.arg == DEFAULT_ASSET_CLASS
        assert col.nullable is False