"""A NaN confidence produced the LARGEST possible position, not none.

Confidence has three consumers that all trust a [0, 1] range — Kelly position
sizing, the desk confidence gate, and the cross-strategy conflict resolution
added 2026-07-28 — and none of them were protected. `Signal.confidence` is
annotated "0.0 to 1.0" with nothing enforcing it.

The old read, `getattr(signal, "confidence", 1.0) or 1.0`, failed OPEN three
separate ways:

    NaN        every comparison against NaN is False, so `if conf < threshold:
               skip` did NOT skip, and the signal was approved and sized.
               Compounding it, the clamp idiom several strategies use —
               yield_curve_momentum's `min(0.90, 0.60 + abs(z) * 0.10)` —
               returns 0.90 when z is NaN, so one bad bar became MAXIMUM
               conviction rather than none.
    0.0        `or 1.0` treats a legitimate zero-conviction signal as falsy
               and promotes it to 1.0, the largest size available.
    >1 / junk  passed straight through into sizing.

Fixed in the direction the scanner normaliser was fixed on the same day: a
malformed number means "no conviction", never "total conviction".

SCOPE. The natural home is `Signal.__post_init__`, which would also cover the
backend bot path. `backend/app/strategies/CLAUDE.md` states base.py must never
be modified, so this covers the DESK path only — which is the one that places
the live paper orders. The backend bot path remains unprotected and needs a
decision on relaxing that rule.
"""
from __future__ import annotations

import math

import pytest

from desk_order_placer import _sane_confidence

DESK_CONFIDENCE_MIN = 0.60


# ── the dangerous case ───────────────────────────────────────────────────────

def test_nan_fails_SAFE_at_zero_not_open_at_one():
    """THE BUG: a NaN must not become maximum conviction."""
    assert _sane_confidence(float("nan")) == 0.0


def test_a_nan_signal_is_now_gated_out():
    """The gate is `conf < threshold`, which raw NaN silently passed."""
    assert _sane_confidence(float("nan")) < DESK_CONFIDENCE_MIN


def test_the_raw_idiom_that_produced_the_nan_is_still_dangerous():
    """Documents WHY sanitising at the read matters — the strategies' own
    clamp cannot defend itself. This is yield_curve_momentum's expression."""
    nan = float("nan")
    assert min(0.90, nan) == 0.90        # a bad bar -> maximum confidence
    assert not (nan < DESK_CONFIDENCE_MIN)  # and the gate would not catch it


def test_the_old_expression_promoted_zero_to_one():
    """`or 1.0` turned zero conviction into full conviction. Pinned as the
    regression: the new path must return 0.0, not 1.0."""
    old = (0.0 or 1.0)
    assert old == 1.0                     # what it used to do
    assert _sane_confidence(0.0) == 0.0   # what it does now


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
def test_infinities_also_fail_safe(bad):
    assert _sane_confidence(bad) == 0.0


# ── ordinary clamping ────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    (1.5, 1.0), (2.0, 1.0), (-0.5, 0.0), (-10.0, 0.0),
])
def test_out_of_range_values_are_clamped(raw, expected):
    assert _sane_confidence(raw) == expected


@pytest.mark.parametrize("good", [0.16, 0.5, 0.72, 0.9, 1.0])
def test_valid_confidences_pass_through_unchanged(good):
    assert _sane_confidence(good) == pytest.approx(good)


@pytest.mark.parametrize("junk", [None, "high", object(), [], {}])
def test_non_numeric_confidence_fails_safe_rather_than_raising(junk):
    """A strategy returning junk must not crash the whole desk run."""
    assert _sane_confidence(junk) == 0.0


def test_a_missing_attribute_is_zero_not_one():
    """`getattr(signal, "confidence", None)` -> None -> 0.0.

    The old default was 1.0, so a Signal without the attribute would have been
    treated as maximum conviction.
    """
    assert _sane_confidence(None) == 0.0


# ── the invariant the consumers rely on ──────────────────────────────────────

def test_the_result_is_always_a_finite_number_in_range():
    for raw in [float("nan"), float("inf"), float("-inf"), -3, 0, 0.5, 1, 7, None, "x"]:
        c = _sane_confidence(raw)
        assert isinstance(c, float) and math.isfinite(c) and 0.0 <= c <= 1.0


def test_the_ensemble_combination_stays_finite():
    """`1 - prod(1-ci)` over sanitised inputs cannot produce NaN.

    A NaN reaching the conflict resolver would compare False against its
    threshold and silently stand every symbol aside.
    """
    confs = [_sane_confidence(x) for x in (float("nan"), 0.9, 2.0, -1.0)]
    p = 1.0
    for c in confs:
        p *= (1.0 - c)
    combined = 1.0 - p
    assert math.isfinite(combined) and 0.0 <= combined <= 1.0
