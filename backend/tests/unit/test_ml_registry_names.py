"""Guard against ML registry name-mismatch regressions.

`app.ml.models.__init__` imports each model under a canonical public name and falls
back to `None` on failure. Two models (`TransformerPredictor`, `iTransformerPredictor`)
were silently `None` for every environment — even with torch — because the classes are
actually named `TFTModel` / `iTransformer`. These modules import without torch (their
base falls back to `object`), so the registry must always expose them.
"""
import pytest
import app.ml.models as M


@pytest.mark.parametrize(
    "public_name,expected_class_name",
    [
        ("TransformerPredictor", "TFTModel"),
        ("iTransformerPredictor", "iTransformer"),
    ],
)
def test_registry_name_resolves(public_name: str, expected_class_name: str):
    """Validate that the public registry name resolves to the correct class.

    The test asserts:
    1. The attribute exists on the module.
    2. The attribute is not ``None`` (i.e., import succeeded).
    3. The underlying class name matches the expected implementation class.
    """
    # Ensure the attribute is present on the module.
    assert hasattr(M, public_name), (
        f"{public_name} attribute missing from app.ml.models module"
    )

    model = getattr(M, public_name)

    # The registry should never expose ``None`` for a valid model.
    assert model is not None, (
        f"{public_name} is None — registry name mismatch regressed "
        f"(class is {expected_class_name}; __init__ must import a matching name)"
    )

    # Confirm that the underlying class name matches the expected implementation.
    assert getattr(model, "__name__", None) == expected_class_name, (
        f"{public_name} resolves to class {getattr(model, '__name__', None)} "
        f"instead of expected {expected_class_name}"
    )