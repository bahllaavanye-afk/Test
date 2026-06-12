"""
Validation gate: enforces that experiments pass OOS walk-forward validation
before they can be marked 'done'. Prevents fake in-sample-only results.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass
class ValidationReport:
    passed: bool
    oos_sharpe: float                    # average OOS Sharpe across walk-forward windows
    oos_drawdown: float                  # average OOS max drawdown
    n_windows: int                       # number of walk-forward windows evaluated
    window_results: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    MIN_SHARPE = 0.3                     # minimum acceptable OOS Sharpe
    MIN_WINDOWS = 2                      # need at least 2 OOS windows
    MAX_DRAWDOWN = -0.40                 # reject if avg drawdown worse than -40%


def validate_experiment(
    signals_fn,                          # callable(train_prices, test_prices) -> pd.Series
    prices: pd.Series,
    *,
    min_sharpe: float = ValidationReport.MIN_SHARPE,
    min_windows: int = ValidationReport.MIN_WINDOWS,
    max_drawdown: float = ValidationReport.MAX_DRAWDOWN,
) -> ValidationReport:
    """
    Run walk-forward validation and return a ValidationReport.
    signals_fn receives (train_prices, test_prices) and returns signal Series for test period.
    """
    from app.backtest.walk_forward import walk_forward

    wf_result = walk_forward(signals_fn, prices)
    windows = wf_result.windows

    valid_windows = [w for w in windows if "sharpe" in w and "error" not in w]
    failures = []
    warnings = []

    if len(valid_windows) < min_windows:
        failures.append(
            f"Only {len(valid_windows)} valid OOS windows; need at least {min_windows}"
        )

    avg_sharpe = wf_result.avg_sharpe
    avg_drawdown = wf_result.avg_drawdown

    if avg_sharpe < min_sharpe:
        failures.append(
            f"OOS Sharpe {avg_sharpe:.3f} below minimum {min_sharpe:.3f}"
        )
    elif avg_sharpe < 0.5:
        warnings.append(f"OOS Sharpe {avg_sharpe:.3f} is borderline")

    if avg_drawdown < max_drawdown:
        failures.append(
            f"OOS avg drawdown {avg_drawdown:.1%} exceeds limit {max_drawdown:.1%}"
        )

    # Consistency check: require at least half of windows to be profitable
    profitable = [w for w in valid_windows if w.get("total_return", 0) > 0]
    if valid_windows and len(profitable) / len(valid_windows) < 0.4:
        failures.append(
            f"Only {len(profitable)}/{len(valid_windows)} OOS windows are profitable"
        )

    return ValidationReport(
        passed=len(failures) == 0,
        oos_sharpe=avg_sharpe,
        oos_drawdown=avg_drawdown,
        n_windows=len(valid_windows),
        window_results=valid_windows,
        failures=failures,
        warnings=warnings,
    )


def summarize_for_results(report: ValidationReport) -> dict:
    """Convert a ValidationReport to a dict suitable for saving in experiments/results/."""
    return {
        "validation": {
            "passed": report.passed,
            "oos_sharpe": report.oos_sharpe,
            "oos_drawdown": report.oos_drawdown,
            "n_windows": report.n_windows,
            "failures": report.failures,
            "warnings": report.warnings,
        }
    }
