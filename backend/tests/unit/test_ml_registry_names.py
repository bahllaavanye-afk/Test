"""Guard against ML registry name-mismatch regressions.

`app.ml.models.__init__` imports each model under a canonical public name and falls
back to `None` on failure. Two models (`TransformerPredictor`, `iTransformerPredictor`)
were silently `None` for every environment — even with torch — because the classes are
actually named `TFTModel` / `iTransformer`. These modules import without torch (their
base falls back to `object`), so the registry must always expose them.
"""
import app.ml.models as M


def test_transformer_registry_name_resolves():
    assert M.TransformerPredictor is not None, (
        "TransformerPredictor is None — registry name mismatch regressed "
        "(class is TFTModel; __init__ must import a matching name)"
    )
    assert M.TransformerPredictor.__name__ == "TFTModel"


def test_itransformer_registry_name_resolves():
    assert M.iTransformerPredictor is not None, (
        "iTransformerPredictor is None — registry name mismatch regressed "
        "(class is iTransformer)"
    )
    assert M.iTransformerPredictor.__name__ == "iTransformer"
