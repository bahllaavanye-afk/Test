"""ml_signal bot condition — ML predictions as OA-style decision recipes.

The user ask: "use ml signal and other indicators as well" in Option-Alpha
format. A bot condition of type ``ml_signal`` passes only when the trained
model predicts the configured direction with confidence ≥ min_confidence.
Pins the fail-safe contract: no trained model / no inference → False, never
an exception (bots must not fire on a missing model).
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("pandas")

from app.bots.engine import evaluate_condition  # noqa: E402
from app.schemas.bot import ConditionConfig  # noqa: E402


def _df(n: int = 50) -> pd.DataFrame:
    return pd.DataFrame({"close": [100.0 + i * 0.1 for i in range(n)]})


def _cond(**kw) -> ConditionConfig:
    return ConditionConfig(type="ml_signal", **kw)


def test_passes_when_model_confirms_direction():
    ml = {"prediction": "up", "confidence": 0.80}
    assert evaluate_condition(_cond(direction="up", min_confidence=0.65), _df(), 105.0, ml_result=ml)


def test_fails_when_confidence_below_threshold():
    ml = {"prediction": "up", "confidence": 0.60}
    assert not evaluate_condition(_cond(direction="up", min_confidence=0.65), _df(), 105.0, ml_result=ml)


def test_fails_on_wrong_direction():
    ml = {"prediction": "down", "confidence": 0.90}
    assert not evaluate_condition(_cond(direction="up"), _df(), 105.0, ml_result=ml)


def test_down_direction_supported():
    ml = {"prediction": "down", "confidence": 0.71}
    assert evaluate_condition(_cond(direction="down", min_confidence=0.7), _df(), 105.0, ml_result=ml)


def test_defaults_direction_up_and_065_confidence():
    assert evaluate_condition(_cond(), _df(), 105.0, ml_result={"prediction": "up", "confidence": 0.66})
    assert not evaluate_condition(_cond(), _df(), 105.0, ml_result={"prediction": "up", "confidence": 0.64})


def test_no_model_is_false_not_error():
    # No trained model / inference failed → ml_result None → condition simply False.
    assert not evaluate_condition(_cond(direction="up"), _df(), 105.0, ml_result=None)


def test_malformed_confidence_is_false_not_error():
    assert not evaluate_condition(_cond(), _df(), 105.0, ml_result={"prediction": "up", "confidence": "bad"})


def test_schema_accepts_ml_signal_type():
    c = ConditionConfig(type="ml_signal", direction="up", min_confidence=0.7)
    assert c.type == "ml_signal" and c.min_confidence == 0.7
