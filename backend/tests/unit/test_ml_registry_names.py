"""Guard against ML registry name‑mismatch regressions.

`app.ml.models.__init__` imports each model under a canonical public name and
falls back to ``None`` on failure. Two models (``TransformerPredictor`` and
``iTransformerPredictor``) were silently ``None`` for every environment — even
with torch — because the classes are actually named ``TFTModel`` /
``iTransformer``. These modules import without torch (their base falls back to
``object``), so the registry must always expose them.
"""

from __future__ import annotations

import pytest
import app.ml.models as M


@pytest.mark.parametrize(
    "public_name,expected_class_name,description",
    [
        (
            "TransformerPredictor",
            "TFTModel",
            "TransformerPredictor is None — registry name mismatch regressed "
            "(class is TFTModel; __init__ must import a matching name)",
        ),
        (
            "iTransformerPredictor",
            "iTransformer",
            "iTransformerPredictor is None — registry name mismatch regressed "
            "(class is iTransformer)",
        ),
    ],
)
def test_ml_registry_name_resolves(
    public_name: str, expected_class_name: str, description: str
) -> None:
    """Validate that the public registry entry resolves to the correct class.

    Args:
        public_name: The attribute name exported by ``app.ml.models``.
        expected_class_name: The actual class name the attribute should reference.
        description: Custom error message for clarity on failure.
    """
    model = getattr(M, public_name, None)
    assert model is not None, description
    assert model.__name__ == expected_class_name, (
        f"{public_name} resolved to {model.__name__}, expected {expected_class_name}"
    )