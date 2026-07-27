"""A single NaN anywhere in a response is a hard 500.

Starlette's JSONResponse renders with `allow_nan=False`, so one NaN or
Infinity raises `ValueError: Out of range float values are not JSON compliant`
and the endpoint 500s — with nothing in the response saying which field did it.

/analytics/tearsheet hit this live. It returned 404 while there were no trades
(so the smoke test passed), and 500'd the moment trades existed again. The
tearsheet is especially exposed: pandas `.std()` on a single trading day is NaN
(ddof=1), and the yfinance SPY benchmark can hand back NaN for a thin series —
which is why the failure was intermittent.
"""
from __future__ import annotations

import json
import math

import pytest

from app.api.v1.analytics import _json_safe


def _render(payload):
    """Exactly what Starlette's JSONResponse.render does."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False,
                      indent=None, separators=(",", ":"))


def test_starlette_rejects_nan_this_is_the_500():
    with pytest.raises(ValueError, match="Out of range float"):
        _render({"sharpe": float("nan")})


def test_nan_becomes_none():
    assert _json_safe(float("nan")) is None


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_all_non_finite_values_are_neutralised(bad):
    assert _json_safe(bad) is None
    _render(_json_safe({"x": bad}))          # must not raise


def test_finite_values_pass_through_unchanged():
    payload = {"sharpe": 0.62, "n": 9, "name": "x", "flag": True, "none": None}
    assert _json_safe(payload) == payload


def test_nested_structures_are_cleaned():
    payload = {
        "metrics": {"sharpe": float("nan"), "sortino": 1.5},
        "curve": [{"equity": 100.0}, {"equity": float("inf")}],
        "tuple_field": (1.0, float("nan")),
    }
    out = _json_safe(payload)
    assert out["metrics"]["sharpe"] is None
    assert out["metrics"]["sortino"] == 1.5
    assert out["curve"][1]["equity"] is None
    assert out["tuple_field"] == [1.0, None]
    _render(out)                              # the whole payload now serialises


def test_a_realistic_tearsheet_payload_survives():
    """Single trading day: pandas std() is NaN, benchmark may be NaN too."""
    import pandas as pd

    one_day = pd.Series([0.0003])
    assert math.isnan(one_day.std()), "ddof=1 on one sample is NaN — the trap"

    payload = {
        "sharpe": 0.0,
        "sortino": 0.0,
        "benchmark_sharpe_spy": float("nan"),      # thin yfinance series
        "benchmark_return_spy": float("nan"),
        "equity_curve": [{"date": "2026-07-27", "equity": 100003.0}],
        "monthly_returns": [{"month": "Jul 2026", "ret": float("nan")}],
    }
    with pytest.raises(ValueError):
        _render(payload)                       # what the endpoint did

    safe = _json_safe(payload)
    _render(safe)                              # what it does now
    assert safe["benchmark_sharpe_spy"] is None
    assert safe["monthly_returns"][0]["ret"] is None
    assert safe["sharpe"] == 0.0
