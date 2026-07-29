"""CompositeExit failure semantics.

A swallowed error in an exit check is the money-path failure that matters most:
if the stop-loss strategy raises, should_exit() returns (False, None) — which the
caller cannot tell apart from "checked, nothing to do" — and the position keeps
running with no stop. These tests pin the behaviour: keep evaluating the other
strategies, but escalate loudly when NOTHING evaluated.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, validator
from app.execution.position_exit import CompositeExit


class Position(BaseModel):
    """Schema representing a trading position payload.

    Attributes
    ----------
    symbol: str
        Ticker symbol of the asset (e.g., ``\"AAPL\"``).
    qty: int
        Quantity of the asset. Must be a non‑negative integer.
    """
    symbol: str = Field(..., description="Ticker symbol of the asset", example="AAPL")
    qty: int = Field(..., description="Number of shares/contracts", example=1, ge=0)

    @validator("symbol")
    def symbol_must_be_nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError("symbol must be a non‑empty string")
        return v


class _Ok:
    def __init__(self, triggered: bool = False, reason: str = "tp"):
        self._t, self._r = triggered, reason

    def should_exit(self, position, current_price, context):
        return self._t, self._r


class _Boom:
    def should_exit(self, position, current_price, context):
        raise RuntimeError("bad position payload")


POS = {"symbol": "AAPL", "qty": 1}


def test_first_trigger_wins():
    c = CompositeExit([_Ok(False), _Ok(True, "stop_loss"), _Ok(True, "take_profit")])
    assert c.should_exit(POS, 100.0, {}) == (True, "stop_loss")


def test_one_broken_strategy_does_not_disable_the_others():
    """A raising rule must not prevent a healthy rule from firing."""
    c = CompositeExit([_Boom(), _Ok(True, "stop_loss")])
    assert c.should_exit(POS, 100.0, {}) == (True, "stop_loss")


def test_no_trigger_is_a_clean_false():
    c = CompositeExit([_Ok(False), _Ok(False)])
    assert c.should_exit(POS, 100.0, {}) == (False, None)


# NOTE: the app logs through structlog, which writes straight to stdout and does
# not propagate to stdlib logging — so caplog sees nothing here and capsys is the
# correct probe. (Asserting via caplog silently passes/fails for the wrong reason.)

def test_all_strategies_failing_is_logged_as_an_error_not_a_warning(capsys):
    """Every rule raised => the position has NO exit protection at all."""
    c = CompositeExit([_Boom(), _Boom()])
    assert c.should_exit(POS, 100.0, {}) == (False, None)
    out = capsys.readouterr().out
    assert "UNPROTECTED" in out, "all-failed must escalate loudly"
    assert "error" in out, "escalation must be at error level, not warning"
    assert "AAPL" in out, "the alert must name the position"


def test_partial_failure_is_not_escalated(capsys):
    """One rule evaluated fine, so the position is still protected — warn only."""
    c = CompositeExit([_Boom(), _Ok(False)])
    assert c.should_exit(POS, 100.0, {}) == (False, None)
    out = capsys.readouterr().out
    assert "UNPROTECTED" not in out, "partial failure must not cry wolf"
    assert "Exit strategy check failed" in out, "the broken rule is still reported"


def test_empty_strategy_list_does_not_escalate(capsys):
    assert CompositeExit([]).should_exit(POS, 100.0, {}) == (False, None)
    assert "UNPROTECTED" not in capsys.readouterr().out