"""Cross-desk tracking fields on Position/Order (desk consolidation stage 3)."""
from app.models.order import Order
from app.models.position import Position

_FIELDS = {"asset_class", "underlying_symbol", "expiry", "strike",
           "option_right", "contract_multiplier"}


def test_position_has_cross_desk_fields():
    assert _FIELDS <= set(Position.__table__.columns.keys())


def test_order_has_cross_desk_fields():
    assert _FIELDS <= set(Order.__table__.columns.keys())


def test_asset_class_defaults_to_equity():
    for model in (Position, Order):
        col = model.__table__.columns["asset_class"]
        assert col.default.arg == "equity"
        assert col.nullable is False
