from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional

import pandas as pd

from .walk_forward import walk_forward, WalkForwardResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Container for validation results."""

    # Threshold constants – kept as class attributes for easy reference
    MIN_SHARPE: float = 0.3
    MIN_WINDOWS: int = 2
    MAX_DRAWDOWN: float = -0.40

    passed: bool = False
    oos_sharpe: float = 0.0
    oos_drawdown: float = 0.0
    n_windows: int = 0
    window_results: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float, falling back to default on TypeError/ValueError."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def validate_experiment(
    signal_fn: Optional[Callable[[pd.Series, pd.Series], pd.Series]],
    prices: Optional[pd.Series],
) -> ValidationReport:
    """
    Run walk‑forward validation on a signal generation function.

    Edge‑case handling:
    * ``signal_fn`` or ``prices`` being ``None`` yields a failing report.
    * Empty ``prices`` series is treated as missing data.
    * Empty ``WalkForwardResult.windows`` is handled gracefully.
    * Off‑by‑one window counts are guarded by the MIN_WINDOWS check.
    """
    report = ValidationReport()

    # ------------------------------------------------------------------
    # Basic input validation
    # ------------------------------------------------------------------
    if signal_fn is None:
        report.failures.append("Signal function is None")
        logger.error("validate_experiment called with signal_fn=None")
        return report

    if prices is None:
        report.failures.append("Price series is None")
        logger.error("validate_experiment called with prices=None")
        return report

    if not isinstance(prices, pd.Series):
        report.failures.append("Price data is not a pandas Series")
        logger.error("validate_experiment received non‑Series price data: %s", type(prices))
        return report

    if prices.empty:
        report.failures.append("Price series is empty")
        logger.error("validate_experiment received an empty price series")
        return report

    # ------------------------------------------------------------------
    # Run walk‑forward; any exception is captured and turned into a failure.
    # ------------------------------------------------------------------
    try:
        wf_result: WalkForwardResult = walk_forward(signal_fn, prices)
    except Exception as exc:  # pragma: no cover
        report.failures.append(f"Walk‑forward execution error: {exc}")
        logger.exception("walk_forward raised an exception")
        return report

    # Defensive handling for a possibly malformed result
    windows = wf_result.windows if isinstance(wf_result.windows, list) else []
    report.n_windows = len(windows)

    # Guard against empty windows – this is a common edge case when the
    # price series is too short for the default window parameters.
    if report.n_windows == 0:
        report.failures.append("Walk‑forward produced no windows")
        logger.warning("WalkForwardResult contains no windows")
        return report

    # Store per‑window results (deep copy not required – data are immutable)
    report.window_results = windows

    # Compute OOS Sharpe and drawdown safely
    report.oos_sharpe = _safe_float(getattr(wf_result, "avg_sharpe", None))
    report.oos_drawdown = _safe_float(getattr(wf_result, "avg_drawdown", None))

    # ------------------------------------------------------------------
    # Validation logic – each check adds a failure message if not met.
    # ------------------------------------------------------------------
    if report.n_windows < ValidationReport.MIN_WINDOWS:
        report.failures.append(
            f"Insufficient windows: {report.n_windows} (minimum required {ValidationReport.MIN_WINDOWS})"
        )
        logger.info("Validation failed: insufficient windows")

    if report.oos_sharpe < ValidationReport.MIN_SHARPE:
        report.failures.append(
            f"OOS Sharpe {report.oos_sharpe:.3f} below minimum {ValidationReport.MIN_SHARPE}"
        )
        logger.info("Validation failed: Sharpe below threshold")

    if report.oos_drawdown < ValidationReport.MAX_DRAWDOWN:
        report.failures.append(
            f"OOS Drawdown {report.oos_drawdown:.3f} exceeds maximum {ValidationReport.MAX_DRAWDOWN}"
        )
        logger.info("Validation failed: Drawdown exceeds threshold")

    # Determine overall pass/fail status
    report.passed = len(report.failures) == 0

    # ------------------------------------------------------------------
    # Optional warnings – e.g., borderline Sharpe values.
    # ------------------------------------------------------------------
    if 0.0 < report.oos_sharpe < ValidationReport.MIN_SHARPE:
        report.warnings.append(f"OOS Sharpe {report.oos_sharpe:.2f} is borderline")

    return report


def summarize_for_results(report: ValidationReport) -> Dict[str, Any]:
    """
    Convert a ValidationReport into a nested dictionary suitable for
    downstream JSON serialisation.

    The function is defensive: missing fields are substituted with sensible
    defaults so that downstream consumers never encounter ``None`` where a
    numeric value is expected.
    """
    return {
        "validation": {
            "passed": bool(report.passed),
            "oos_sharpe": _safe_float(report.oos_sharpe),
            "oos_drawdown": _safe_float(report.oos_drawdown),
            "n_windows": int(report.n_windows),
            "failures": list(report.failures),
            "warnings": list(report.warnings),
        }
    }