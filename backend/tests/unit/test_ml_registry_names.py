"""Guard against ML registry name-mismatch regressions.

`app.ml.models.__init__` imports each model under a canonical public name and falls
back to `None` on failure. Two models (`TransformerPredictor`, `iTransformerPredictor`)
were silently `None` for every environment — even with torch — because the classes are
actually named `TFTModel` / `iTransformer`. These modules import without torch (their
base falls back to `object`), so the registry must always expose them.
"""
import app.ml.models as M


def _resolve_model(name: str):
    """Resolve a model from the registry safely.

    Handles:
    * Empty or None name inputs.
    * Missing attributes on the module.
    * Attributes that resolve to ``None``.
    """
    if not name:
        raise AssertionError("Model name is empty or None")
    model = getattr(M, name, None)
    assert model is not None, (
        f"{name} is None — registry name mismatch regressed "
        f"(expected a concrete class, got None)"
    )
    return model


def test_transformer_registry_name_resolves():
    model = _resolve_model("TransformerPredictor")
    assert model.__name__ == "TFTModel", (
        "TransformerPredictor resolved to wrong class "
        f"(got {model.__name__}, expected 'TFTModel')"
    )


def test_itransformer_registry_name_resolves():
    model = _resolve_model("iTransformerPredictor")
    assert model.__name__ == "iTransformer", (
        "iTransformerPredictor resolved to wrong class "
        f"(got {model.__name__}, expected 'iTransformer')"
    )


def test_registry_expected_models_resolve():
    """Validate that the expected registry entries are present.

    This protects against accidental removal of entries and also guards against
    off‑by‑one errors when the list of expected models is modified.
    """
    expected_models = ["TransformerPredictor", "iTransformerPredictor"]
    assert expected_models, "Expected models list is empty"
    for name in expected_models:
        _resolve_model(name)