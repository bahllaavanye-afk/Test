"""Promotion criteria thresholds for each stage."""
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any


@dataclass(frozen=True)
class StageCriteria:
    min_days: int
    min_sharpe: float
    min_win_rate: float
    max_drawdown: float  # negative number, e.g. -0.08
    min_trades: int
    require_p_value: bool  # whether ML statistical significance is required


CRITERIA = {
    "paper_to_shadow": StageCriteria(
        min_days=14,
        min_sharpe=0.5,
        min_win_rate=0.45,
        max_drawdown=-0.15,
        min_trades=20,
        require_p_value=False,
    ),
    "shadow_to_staging": StageCriteria(
        min_days=30,
        min_sharpe=0.8,
        min_win_rate=0.50,
        max_drawdown=-0.12,
        min_trades=40,
        require_p_value=False,
    ),
    "staging_to_live": StageCriteria(
        min_days=60,
        min_sharpe=1.2,
        min_win_rate=0.52,
        max_drawdown=-0.10,
        min_trades=60,
        require_p_value=True,
    ),
}

TRANSITION_MAP = {
    "paper": "paper_to_shadow",
    "shadow": "shadow_to_staging",
    "staging": "staging_to_live",
}


def _check_min_days(metrics: Dict[str, Any], criteria: StageCriteria) -> str | None:
    days = metrics.get("days_in_stage", 0)
    if days < criteria.min_days:
        return f"Too few days: {days} < {criteria.min_days}"
    return None


def _check_sharpe(metrics: Dict[str, Any], criteria: StageCriteria) -> str | None:
    sharpe = metrics.get("sharpe", 0.0)
    if sharpe < criteria.min_sharpe:
        return f"Sharpe too low: {sharpe:.2f} < {criteria.min_sharpe}"
    return None


def _check_win_rate(metrics: Dict[str, Any], criteria: StageCriteria) -> str | None:
    win_rate = metrics.get("win_rate", 0.0)
    if win_rate < criteria.min_win_rate:
        return (
            f"Win rate too low: {win_rate:.2%} < {criteria.min_win_rate:.2%}"
        )
    return None


def _check_drawdown(metrics: Dict[str, Any], criteria: StageCriteria) -> str | None:
    max_dd = metrics.get("max_drawdown", -1.0)
    if max_dd < criteria.max_drawdown:
        return (
            f"Max drawdown too large: {max_dd:.2%} < {criteria.max_drawdown:.2%}"
        )
    return None


def _check_trades(metrics: Dict[str, Any], criteria: StageCriteria) -> str | None:
    num_trades = metrics.get("num_trades", 0)
    if num_trades < criteria.min_trades:
        return f"Too few trades: {num_trades} < {criteria.min_trades}"
    return None


def _check_p_value(metrics: Dict[str, Any], criteria: StageCriteria) -> str | None:
    if not criteria.require_p_value:
        return None
    p_value = metrics.get("p_value")
    if p_value is None or p_value >= 0.05:
        return f"ML significance not established: p_value={p_value} (need < 0.05)"
    return None


def check_criteria(metrics: dict, transition: str) -> Tuple[bool, List[str]]:
    """Validate a set of performance metrics against promotion criteria.

    Returns:
        (passed, failures): ``passed`` is True when all criteria are satisfied.
        ``failures`` contains human‑readable messages for each unmet criterion.
    """
    criteria = CRITERIA.get(transition)
    if not criteria:
        return False, [f"Unknown transition: {transition}"]

    checks = [
        _check_min_days,
        _check_sharpe,
        _check_win_rate,
        _check_drawdown,
        _check_trades,
        _check_p_value,
    ]

    failures: List[str] = []
    for check in checks:
        result = check(metrics, criteria)
        if result:
            failures.append(result)

    return len(failures) == 0, failures