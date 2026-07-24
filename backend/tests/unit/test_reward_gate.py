"""Unit tests for the reward gate's pure decision logic (no network)."""
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / ".github" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import reward_gate as G


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


def test_ci_conclusion_mixed_and_empty_cases():
    # Pending status should dominate over completed successes
    runs = [
        {"status": "completed", "conclusion": "success"},
        {"status": "in_progress"},
    ]
    assert G.ci_conclusion(runs, []) == "pending"

    # Mixed commit states: any failure should cause overall failure
    commits = [{"state": "success"}, {"state": "failure"}]
    assert G.ci_conclusion([], commits) == "failure"

    # No runs and no commit statuses – treat as success (no failures)
    assert G.ci_conclusion([], []) == "success"


def test_parse_judge_fail_closed():
    assert G.parse_judge("reasoning\nREWARD: PASS") is True
    assert G.parse_judge("reasoning\nREWARD: FAIL") is False
    assert G.parse_judge("no verdict at all") is False          # fail-closed
    assert G.parse_judge("[LLM unavailable — all tiers failed]") is False  # fail-closed


def test_parse_judge_whitespace_and_case():
    # Extra spaces before the verdict are tolerated
    assert G.parse_judge("REWARD:   PASS") is True
    # Case sensitivity – only uppercase 'REWARD' is recognized
    assert G.parse_judge("reward: PASS") is False
    # Verdict not at the very end but still present should be accepted
    assert G.parse_judge("REWARD: PASS\nadditional info") is True


def test_decide_merges_only_on_full_reward():
    assert G.decide("success", True, True, True)[0] is True
    assert G.decide("success", False, True, True)[0] is False   # judge fail
    assert G.decide("failure", True, True, True)[0] is False    # CI red
    assert G.decide("success", True, False, True)[0] is False   # coverage regressed
    assert G.decide("pending", True, True, True)[0] is False    # CI running
    assert G.decide("success", True, True, False)[0] is False   # not labelled


def test_decide_unknown_ci_state():
    # Any CI state other than 'success' should result in no merge
    assert G.decide("unknown", True, True, True)[0] is False