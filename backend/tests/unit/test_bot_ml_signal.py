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

# Constants
DEFAULT_DF_ROWS: int = 50
DEFAULT_CLOSE_START: float = 100.0
DEFAULT_CLOSE_INCREMENT: float = 0.1
DEFAULT_PRICE: float = 105.0

CONDITION_TYPE_ML_SIGNAL: str = "ml_signal"
DIRECTION_UP: str = "up"
DIRECTION_DOWN: str = "down"

MIN_CONFIDENCE_DEFAULT: float = 0.65
CONFIDENCE_THRESHOLD: float = 0.7
CONFIDENCE_ABOVE_THRESHOLD: float = 0.71
CONFIDENCE_HIGH: float = 0.80
CONFIDENCE_LOW: float = 0.60
CONFIDENCE_VERY_HIGH: float = 0.90
CONFIDENCE_ABOVE_DEFAULT: float = 0.66
CONFIDENCE_BELOW_DEFAULT: float = 0.64


def _df(n: int = DEFAULT_DF_ROWS) -> pd.DataFrame:
    return pd.DataFrame(
        {"close": [DEFAULT_CLOSE_START + i * DEFAULT_CLOSE_INCREMENT for i in range(n)]}
    )


def _cond(**kw) -> ConditionConfig:
    return ConditionConfig(type=CONDITION_TYPE_ML_SIGNAL, **kw)


def test_passes_when_model_confirms_direction():
    ml = {"prediction": DIRECTION_UP, "confidence": CONFIDENCE_HIGH}
    assert evaluate_condition(
        _cond(direction=DIRECTION_UP, min_confidence=MIN_CONFIDENCE_DEFAULT),
        _df(),
        DEFAULT_PRICE,
        ml_result=ml,
    )


def test_fails_when_confidence_below_threshold():
    ml = {"prediction": DIRECTION_UP, "confidence": CONFIDENCE_LOW}
    assert not evaluate_condition(
        _cond(direction=DIRECTION_UP, min_confidence=MIN_CONFIDENCE_DEFAULT),
        _df(),
        DEFAULT_PRICE,
        ml_result=ml,
    )


def test_fails_on_wrong_direction():
    ml = {"prediction": DIRECTION_DOWN, "confidence": CONFIDENCE_VERY_HIGH}
    assert not evaluate_condition(_cond(direction=DIRECTION_UP), _df(), DEFAULT_PRICE, ml_result=ml)


def test_down_direction_supported():
    ml = {"prediction": DIRECTION_DOWN, "confidence": CONFIDENCE_ABOVE_THRESHOLD}
    assert evaluate_condition(
        _cond(direction=DIRECTION_DOWN, min_confidence=CONFIDENCE_THRESHOLD),
        _df(),
        DEFAULT_PRICE,
        ml_result=ml,
    )


def test_defaults_direction_up_and_065_confidence():
    assert evaluate_condition(
        _cond(),
        _df(),
        DEFAULT_PRICE,
        ml_result={"prediction": DIRECTION_UP, "confidence": CONFIDENCE_ABOVE_DEFAULT},
    )
    assert not evaluate_condition(
        _cond(),
        _df(),
        DEFAULT_PRICE,
        ml_result={"prediction": DIRECTION_UP, "confidence": CONFIDENCE_BELOW_DEFAULT},
    )


def test_no_model_is_false_not_error():
    # No trained model / inference failed → ml_result None → condition simply False.
    assert not evaluate_condition(_cond(direction=DIRECTION_UP), _df(), DEFAULT_PRICE, ml_result=None)


def test_malformed_confidence_is_false_not_error():
    assert not evaluate_condition(
        _cond(),
        _df(),
        DEFAULT_PRICE,
        ml_result={"prediction": DIRECTION_UP, "confidence": "bad"},
    )


def test_schema_accepts_ml_signal_type():
    c = ConditionConfig(type=CONDITION_TYPE_ML_SIGNAL, direction=DIRECTION_UP, min_confidence=CONFIDENCE_THRESHOLD)
    assert c.type == CONDITION_TYPE_ML_SIGNAL and c.min_confidence == CONFIDENCE_THRESHOLD