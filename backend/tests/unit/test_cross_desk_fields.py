"""Cross-desk tracking fields on Position/Order (desk consolidation stage 3)."""
from app.models.order import Order
from app.models.position import Position

_FIELDS = {"asset_class", "underlying_symbol", "expiry", "strike",
           "option_right", "contract_multiplier"}


def _validate_model_has_columns(model):
    """Validate that a SQLAlchemy model exposes a columns collection.

    Args:
        model: The SQLAlchemy model class to validate.

    Raises:
        ValueError: If the model does not have a __table__ attribute or the
            table lacks a columns mapping.
    """
    if not hasattr(model, "__table__"):
        raise ValueError(f"Model {model.__name__} is missing a '__table__' attribute.")
    if not hasattr(model.__table__, "columns"):
        raise ValueError(
            f"Model {model.__name__} '__table__' does not contain a 'columns' attribute."
        )


def test_position_has_cross_desk_fields():
    _validate_model_has_columns(Position)
    assert _FIELDS <= set(Position.__table__.columns.keys())


def test_order_has_cross_desk_fields():
    _validate_model_has_columns(Order)
    assert _FIELDS <= set(Order.__table__.columns.keys())


def test_asset_class_defaults_to_equity():
    for model in (Position, Order):
        _validate_model_has_columns(model)
        col = model.__table__.columns["asset_class"]
        assert col.default.arg == "equity"
        assert col.nullable is False