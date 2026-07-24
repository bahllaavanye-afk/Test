"""Unit tests for the reward gate's pure decision logic (no network)."""
import sys
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, validator

SCRIPTS = Path(__file__).resolve().parents[3] / ".github" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import reward_gate as G


class CIRun(BaseModel):
    """Schema representing a CI run entry."""

    status: Literal["in_progress", "completed"] = Field(
        ...,
        description="Current status of the CI run.",
        examples=["completed", "in_progress"],
    )
    conclusion: Optional[Literal["success", "failure", "skipped", "neutral"]] = Field(
        None,
        description="Final conclusion of the CI run when status is 'completed'.",
        examples=["success", "failure"],
    )
    name: Optional[str] = Field(
        None,
        description="Human‑readable identifier of the CI job.",
        examples=["test", "Reward Gate"],
    )

    @validator("conclusion", always=True)
    def check_conclusion_for_completed(cls, v, values):
        """Validate that a conclusion is present when status is completed."""
        if values.get("status") == "completed" and v is None:
            raise ValueError("conclusion must be set for completed runs")
        return v

    @validator("name")
    def name_must_not_be_empty(cls, v):
        """Ensure that a provided name is not an empty string."""
        if v is not None and not v.strip():
            raise ValueError("name cannot be empty")
        return v

    class Config:
        schema_extra = {
            "example": {
                "status": "completed",
                "conclusion": "success",
                "name": "test",
            }
        }


class CommitStatus(BaseModel):
    """Schema representing a commit status entry."""

    state: Literal["success", "failure", "pending", "error"] = Field(
        ...,
        description="Overall state of the commit status.",
        examples=["failure"],
    )
    description: Optional[str] = Field(
        None,
        description="Optional human‑readable description of the state.",
        examples=["Unit tests failed"],
    )
    context: Optional[str] = Field(
        None,
        description="Context string used by the status API.",
        examples=["ci/build"],
    )

    class Config:
        schema_extra = {
            "example": {
                "state": "failure",
                "description": "Unit tests failed",
                "context": "ci/build",
            }
        }


def test_ci_conclusion_pending_failure_success():
    assert G.ci_conclusion([{"status": "in_progress"}], []) == "pending"
    assert G.ci_conclusion([{"status": "completed", "conclusion": "failure"}], []) == "failure"
    assert G.ci_conclusion([{"status": "completed", "conclusion": "success"}], []) == "success"
    # neutral/skipped count as ok
    assert G.ci_conclusion([{"status": "completed", "conclusion": "skipped"}], []) == "success"
    # commit statuses are honoured too
    assert G.ci_conclusion([], [{"state": "failure"}]) == "failure"


def test_ci_ignores_the_reward_gate_own_check():
    runs = [
        {"status": "completed", "conclusion": "success", "name": "test"},
        {"status": "in_progress", "name": "Reward Gate"},  # must be ignored
    ]
    assert G.ci_conclusion(runs, []) == "success"


def test_parse_judge_fail_closed():
    assert G.parse_judge("reasoning\nREWARD: PASS") is True
    assert G.parse_judge("reasoning\nREWARD: FAIL") is False
    assert G.parse_judge("no verdict at all") is False          # fail-closed
    assert G.parse_judge("[LLM unavailable — all tiers failed]") is False  # fail-closed


def test_decide_merges_only_on_full_reward():
    assert G.decide("success", True, True, True)[0] is True
    assert G.decide("success", False, True, True)[0] is False   # judge fail
    assert G.decide("failure", True, True, True)[0] is False    # CI red
    assert G.decide("success", True, False, True)[0] is False   # coverage regressed
    assert G.decide("pending", True, True, True)[0] is False    # CI running
    assert G.decide("success", True, True, False)[0] is False   # not labelled