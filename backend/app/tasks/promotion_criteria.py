"""Promotion criteria thresholds for each stage."""
from dataclasses import dataclass


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


def check_criteria(metrics: dict, transition: str) -> tuple[bool, list[str]]:
    """Returns (passed, list_of_failures)"""
    c = CRITERIA.get(transition)
    if not c:
        return False, [f"Unknown transition: {transition}"]

    failures = []
    days = metrics.get("days_in_stage", 0)
    if days < c.min_days:
        failures.append(f"Too few days: {days} < {c.min_days}")

    sharpe = metrics.get("sharpe", 0.0)
    if sharpe < c.min_sharpe:
        failures.append(f"Sharpe too low: {sharpe:.2f} < {c.min_sharpe}")

    win_rate = metrics.get("win_rate", 0.0)
    if win_rate < c.min_win_rate:
        failures.append(f"Win rate too low: {win_rate:.2%} < {c.min_win_rate:.2%}")

    max_dd = metrics.get("max_drawdown", -1.0)
    if max_dd < c.max_drawdown:
        failures.append(f"Max drawdown too large: {max_dd:.2%} < {c.max_drawdown:.2%}")

    num_trades = metrics.get("num_trades", 0)
    if num_trades < c.min_trades:
        failures.append(f"Too few trades: {num_trades} < {c.min_trades}")

    if c.require_p_value:
        p_value = metrics.get("p_value")
        if p_value is None or p_value >= 0.05:
            failures.append(
                f"ML significance not established: p_value={p_value} (need < 0.05)"
            )

    return len(failures) == 0, failures
