"""The scanners and their response schema disagreed about both fields.

Found by writing the first-ever test for `/api/v1/scanners` — the module had
0% coverage while being mounted and live. The polymarket desk returned:

    500  2 validation errors for ScanResultOut
         score: Input should be less than or equal to 1  [input_value=75.0]
         side:  Value error, side must be one of {'sell', 'none', ...}

It is not one desk. ALL THREE producers emit `min(score, 100)` — a 0-100 scale
— against a schema declaring `ge=0.0, le=1.0`, and they emit
`long` / `short` / `long_yes` / `long_no` against a validator allowing only
`{buy, sell, neutral, none}`. Every non-empty scan result 500'd, on every desk.

Equity and crypto only *looked* fine in the endpoint test because they returned
nothing in that environment, so the serialisation path never ran with rows.
That is exactly why this file exercises the normaliser directly rather than
relying on whatever data happens to exist.

Invisible from outside, too: an anonymous probe of these routes gets 401.
"""
from __future__ import annotations

import pytest

from app.api.v1.scanners import ScanResultOut, _normalise_scan_item


class _ScanResult:
    """Stand-in for tasks.stock_scanners.ScanResult (attribute access)."""

    def __init__(self, symbol, desk, score, signals, side, data=None):
        self.symbol, self.desk, self.score = symbol, desk, score
        self.signals, self.side, self.data = signals, side, data or {}


# ── the exact production failures ────────────────────────────────────────────

def test_the_polymarket_result_that_returned_500_now_validates():
    item = _ScanResult("will-x-happen", "polymarket", 75.0, ["binary_arb_3%"], "long_yes")
    out = ScanResultOut(**_normalise_scan_item(item))
    assert out.score == pytest.approx(0.75)
    assert out.side == "buy"


@pytest.mark.parametrize("side,expected", [
    ("long", "buy"),          # equity + crypto directional
    ("short", "sell"),        # equity + crypto directional
    ("long_yes", "buy"),      # polymarket
    ("long_no", "sell"),      # polymarket
    ("neutral", "neutral"),
    ("buy", "buy"),           # already-canonical values pass through
    ("sell", "sell"),
])
def test_every_side_the_scanners_actually_emit_is_accepted(side, expected):
    item = _ScanResult("X", "equity", 50.0, ["s"], side)
    assert ScanResultOut(**_normalise_scan_item(item)).side == expected


def test_an_unknown_side_degrades_to_none_rather_than_500ing():
    """A new scanner inventing a side must not take the endpoint down."""
    item = _ScanResult("X", "equity", 50.0, ["s"], "moon")
    assert ScanResultOut(**_normalise_scan_item(item)).side == "none"


# ── score scaling ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    (100.0, 1.0),      # producers cap at 100
    (75.0, 0.75),
    (10.0, 0.10),      # the scanners' own minimum threshold
    (0.0, 0.0),
])
def test_the_0_to_100_producer_scale_maps_onto_the_documented_0_to_1(raw, expected):
    item = _ScanResult("X", "equity", raw, ["s"], "long")
    assert ScanResultOut(**_normalise_scan_item(item)).score == pytest.approx(expected)


@pytest.mark.parametrize("already_normalised", [0.0, 0.5, 1.0])
def test_values_already_on_the_0_to_1_scale_are_left_alone(already_normalised):
    """Cached rows written after this fix must round-trip unchanged."""
    item = _ScanResult("X", "equity", already_normalised, ["s"], "long")
    assert ScanResultOut(**_normalise_scan_item(item)).score == pytest.approx(already_normalised)


def test_a_score_beyond_the_cap_is_clamped_not_rejected():
    item = _ScanResult("X", "equity", 250.0, ["s"], "long")
    assert ScanResultOut(**_normalise_scan_item(item)).score == 1.0


# ── the cached path uses dicts, not objects ──────────────────────────────────

def test_cached_rows_are_normalised_too():
    """Redis holds the producer's raw form; that path 500'd identically."""
    cached = {"symbol": "BTC/USD", "desk": "crypto", "score": 65.0,
              "signals": ["rsi_28"], "side": "long", "data": {"rsi": 28}}
    out = ScanResultOut(**_normalise_scan_item(cached))
    assert out.score == pytest.approx(0.65)
    assert out.side == "buy"
    assert out.data == {"rsi": 28}


# ── it must never be the thing that breaks the endpoint ──────────────────────

@pytest.mark.parametrize("bad_score", [None, "junk", float("nan"), float("inf")])
def test_a_malformed_score_does_not_raise(bad_score):
    item = _ScanResult("X", "equity", bad_score, ["s"], "long")
    result = _normalise_scan_item(item)
    assert 0.0 <= result["score"] <= 1.0


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf")])
def test_a_malformed_score_fails_SAFE_not_to_maximum_confidence(bad_score):
    """`min(1.0, nan)` returns 1.0 in Python — every NaN comparison is False.

    Clamping naively would turn a malformed score into MAXIMUM confidence on a
    ranking signal, which is the worst direction to round in. It must go to 0.
    """
    item = _ScanResult("X", "equity", bad_score, ["s"], "long")
    assert _normalise_scan_item(item)["score"] == 0.0


def test_missing_fields_degrade_instead_of_raising():
    out = _normalise_scan_item({})
    assert out["symbol"] == "" and out["side"] == "none" and out["score"] == 0.0
    assert out["signals"] == [] and out["data"] == {}
