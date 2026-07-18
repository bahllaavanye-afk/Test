"""Unit tests for the reward gate's pure decision logic (no network)."""
import sys
import logging
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / ".github" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import reward_gate as G

_logger = logging.getLogger(__name__)

def _call_with_logging(func, *args, **kwargs):
    """
    Invoke ``func`` with the supplied arguments, logging any exception that occurs.
    Specific exception information is captured in a structured log entry before
    re‑raising the exception so that test failures remain visible.
    """
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover
        _logger.error(
            "Exception in %s",
            func.__name__,
            exc_info=True,
            extra={
                "function": func.__name__,
                "args": args,
                "kwargs": kwargs,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )
        raise


def test_ci_conclusion_pending_failure_success():
    assert _call_with_logging(G.ci_conclusion, [{"status": "in_progress"}], []) == "pending"
    assert _call_with_logging(G.ci_conclusion, [{"status": "completed", "conclusion": "failure"}], []) == "failure"
    assert _call_with_logging(G.ci_conclusion, [{"status": "completed", "conclusion": "success"}], []) == "success"
    # neutral/skipped count as ok
    assert _call_with_logging(G.ci_conclusion, [{"status": "completed", "conclusion": "skipped"}], []) == "success"
    # commit statuses are honoured too
    assert _call_with_logging(G.ci_conclusion, [], [{"state": "failure"}]) == "failure"


def test_ci_ignores_the_reward_gate_own_check():
    runs = [
        {"status": "completed", "conclusion": "success", "name": "test"},
        {"status": "in_progress", "name": "Reward Gate"},  # must be ignored
    ]
    assert _call_with_logging(G.ci_conclusion, runs, []) == "success"


def test_parse_judge_fail_closed():
    assert _call_with_logging(G.parse_judge, "reasoning\nREWARD: PASS") is True
    assert _call_with_logging(G.parse_judge, "reasoning\nREWARD: FAIL") is False
    assert _call_with_logging(G.parse_judge, "no verdict at all") is False          # fail-closed
    assert _call_with_logging(G.parse_judge, "[LLM unavailable — all tiers failed]") is False  # fail-closed


def test_decide_merges_only_on_full_reward():
    assert _call_with_logging(G.decide, "success", True, True, True)[0] is True
    assert _call_with_logging(G.decide, "success", False, True, True)[0] is False   # judge fail
    assert _call_with_logging(G.decide, "failure", True, True, True)[0] is False    # CI red
    assert _call_with_logging(G.decide, "success", True, False, True)[0] is False   # coverage regressed
    assert _call_with_logging(G.decide, "pending", True, True, True)[0] is False    # CI running
    assert _call_with_logging(G.decide, "success", True, True, False)[0] is False   # not labelled