"""
Investor-facing comparison report builder.

Generates structured JSON reports comparing manual vs ML-enhanced strategies
against SPY, QQQ, BRK-B, All Weather benchmarks.

Used by: GET /api/v1/comparison/report
"""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape as _h
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator, root_validator
from app.comparison.engine import ComparisonResult
from app.comparison.benchmarks import get_benchmark_stats


class StrategyMetrics(BaseModel):
    """Performance metrics for a single strategy or benchmark."""

    name: str = Field(
        ...,
        description="Human readable name of the strategy or benchmark.",
        example="Manual Strategy",
    )
    sharpe: float = Field(
        ...,
        description="Sharpe ratio (annualized).",
        example=1.23,
        ge=-1000,
        le=1000,
    )
    sortino: float = Field(
        ...,
        description="Sortino ratio (annualized).",
        example=1.45,
        ge=-1000,
        le=1000,
    )
    annual_return_pct: float = Field(
        ...,
        description="Annual return expressed as a percentage.",
        example=12.5,
        ge=-1000,
        le=1000,
    )
    max_drawdown_pct: float = Field(
        ...,
        description="Maximum drawdown expressed as a percentage.",
        example=-15.3,
        le=0,
        ge=-1000,
    )
    win_rate: float = Field(
        ...,
        description="Proportion of winning trades (0‑1).",
        example=0.58,
        ge=0.0,
        le=1.0,
    )
    total_trades: int = Field(
        ...,
        description="Total number of trades executed.",
        example=124,
        ge=0,
    )
    avg_hold_days: float = Field(
        ...,
        description="Average holding period in days.",
        example=3.7,
        ge=0.0,
    )
    calmar: float = Field(
        ...,
        description="Calmar ratio (annual return / max drawdown).",
        example=0.82,
        ge=-1000,
        le=1000,
    )

    @validator("max_drawdown_pct")
    def ensure_negative_drawdown(cls, v: float) -> float:
        """Drawdown should be a non‑positive percentage."""
        if v > 0:
            raise ValueError("max_drawdown_pct must be zero or negative")
        return v


class ComparisonReport(BaseModel):
    """Full comparison report returned by the API."""

    strategy_name: str = Field(
        ...,
        description="Name of the evaluated strategy.",
        example="Momentum Strategy",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol the strategy was applied to.",
        example="AAPL",
    )
    interval: str = Field(
        ...,
        description="Data granularity (e.g., '1d', '5m').",
        example="1d",
    )
    period: str = Field(
        ...,
        description='Date range of the back‑test in the form "YYYY-MM-DD to YYYY-MM-DD".',
        example="2021-01-01 to 2024-12-31",
    )
    manual: StrategyMetrics = Field(
        ...,
        description="Metrics for the manually‑executed version of the strategy.",
    )
    ml_enhanced: StrategyMetrics = Field(
        ...,
        description="Metrics for the ML‑enhanced version of the strategy.",
    )
    benchmarks: Dict[str, StrategyMetrics] = Field(
        ...,
        description="Mapping of benchmark name to its performance metrics.",
        example={"SPY": {"name": "S&P 500", "sharpe": 0.8, "sortino": 0.9, "annual_return_pct": 10.2,
                      "max_drawdown_pct": -12.5, "win_rate": 0.55, "total_trades": 0,
                      "avg_hold_days": 0.0, "calmar": 0.82}},
    )
    ml_improvement_pct: float = Field(
        ...,
        description="Percentage improvement of ML Sharpe over manual Sharpe.",
        example=34.5,
    )
    is_statistically_significant: bool = Field(
        ...,
        description="Whether the Sharpe difference passes the chosen significance test.",
        example=True,
    )
    t_statistic: float = Field(
        ...,
        description="t‑statistic from the significance test.",
        example=2.45,
    )
    p_value: float = Field(
        ...,
        description="p‑value from the significance test.",
        example=0.0143,
        ge=0.0,
        le=1.0,
    )
    winner: str = Field(
        ...,
        description="Identifier of the winning entity (manual, ml, or benchmark name).",
        example="ml",
    )
    equity_curves: Dict[str, List[float]] = Field(
        ...,
        description="Normalized equity curve values for each entity, indexed by name.",
        example={"Manual Strategy": [100, 102, 101.5], "ML-Enhanced Strategy": [100, 105, 107]},
    )
    generated_at: str = Field(
        ...,
        description="ISO‑8601 timestamp when the report was generated.",
        example="2024-08-15T12:34:56Z",
    )

    @validator("period")
    def validate_period(cls, v: str) -> str:
        """Ensure the period string follows the expected pattern."""
        import re

        pattern = r"^\d{4}-\d{2}-\d{2}\s+to\s+\d{4}-\d{2}-\d{2}$"
        if not re.match(pattern, v.strip()):
            raise ValueError(
                "period must be in the format 'YYYY-MM-DD to YYYY-MM-DD'"
            )
        return v

    @validator("generated_at")
    def validate_generated_at(cls, v: str) -> str:
        """Ensure generated_at is a valid UTC ISO‑8601 timestamp."""
        try:
            datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as exc:
            raise ValueError(
                "generated_at must be in UTC ISO‑8601 format like '2024-08-15T12:34:56Z'"
            ) from exc
        return v

    @root_validator
    def check_improvement_consistency(cls, values):
        """Cross‑field validation for improvement percentage."""
        manual = values.get("manual")
        ml = values.get("ml_enhanced")
        imp = values.get("ml_improvement_pct")
        if manual and ml and imp is not None:
            expected = (
                round((ml.sharpe - manual.sharpe) / abs(manual.sharpe) * 100, 2)
                if manual.sharpe != 0
                else round((ml.sharpe - manual.sharpe) * 100, 2)
            )
            if abs(expected - imp) > 0.01:
                raise ValueError(
                    f"ml_improvement_pct ({imp}) does not match computed improvement ({expected})"
                )
        return values


class ReportBuilder:
    """Builds investor‑facing ComparisonReport objects from ComparisonEngine results."""

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def build(self, comparison_result: ComparisonResult) -> ComparisonReport:
        """Build investor report from ComparisonEngine result."""
        cr = comparison_result

        manual_metrics = self._backtest_to_strategy_metrics(
            "Manual Strategy", cr.manual
        )
        ml_metrics = self._backtest_to_strategy_metrics(
            "ML-Enhanced Strategy", cr.ml_enhanced
        )

        # Build benchmark StrategyMetrics from static stats using a dict comprehension
        benchmark_metrics: Dict[str, StrategyMetrics] = {
            key: StrategyMetrics(
                name=stats.get("name", key),
                sharpe=float(stats.get("sharpe", 0.0)),
                sortino=float(stats.get("sharpe", 0.0)) * 1.15,
                annual_return_pct=round(float(stats.get("annual_return", 0.0)) * 100, 2),
                max_drawdown_pct=round(float(stats.get("max_dd", 0.0)) * 100, 2),
                win_rate=0.0,
                total_trades=0,
                avg_hold_days=0.0,
                calmar=round(
                    float(stats.get("annual_return", 0.0))
                    / max(abs(float(stats.get("max_dd", 1.0))), 1e-9),
                    4,
                ),
            )
            for key, stats in cr.benchmark_stats.items()
        }

        # Sharpe improvement expressed as a percentage of manual Sharpe
        manual_sharpe = manual_metrics.sharpe
        ml_improvement_pct = (
            round((ml_metrics.sharpe - manual_sharpe) / abs(manual_sharpe) * 100, 2)
            if manual_sharpe != 0
            else round((ml_metrics.sharpe - manual_sharpe) * 100, 2)
        )

        # Determine winner, also consider benchmarks
        winner = self._determine_winner(
            ml_metrics, manual_metrics, benchmark_metrics, cr.winner
        )

        # Build normalized equity curves (start = 100) with memoization
        equity_curves = self._extract_equity_curves(cr)

        period_str = f"{cr.start_date} to {cr.end_date}"

        return ComparisonReport(
            strategy_name=cr.strategy_name,
            symbol=cr.symbol,
            interval=cr.interval,
            period=period_str,
            manual=manual_metrics,
            ml_enhanced=ml_metrics,
            benchmarks=benchmark_metrics,
            ml_improvement_pct=ml_improvement_pct,
            is_statistically_significant=cr.is_significant,
            t_statistic=cr.t_statistic,
            p_value=cr.p_value,
            winner=winner,
            equity_curves=equity_curves,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def to_dict(self, report: ComparisonReport) -> dict:
        """Serialize for API response / JSON storage."""
        return report.dict()

    def executive_summary(self, report: ComparisonReport) -> str:
        """Plain English summary: 'ML Momentum outperforms manual by 34% Sharpe...'"""
        direction = "outperforms" if report.ml_improvement_pct > 0 else "underperforms"
        abs_improvement = abs(report.ml_improvement_pct)

        sig_phrase = (
            "statistically significant (p={:.4f})".format(report.p_value)
            if report.is_statistically_significant
            else "not statistically significant (p={:.4f})".format(report.p_value)
        )

        best_benchmark = self._best_benchmark(report)

        lines = [
            f"ML {report.strategy_name} {direction} the manual strategy by "
            f"{abs_improvement:.1f}% on a risk-adjusted Sharpe basis "
            f"({report.ml_enhanced.sharpe:.2f} vs {report.manual.sharpe:.2f}), "
            f"and the result is {sig_phrase}.",
            "",
            f"ML-Enhanced:  annual return {report.ml_enhanced.annual_return_pct:.1f}%, "
            f"max drawdown {report.ml_enhanced.max_drawdown_pct:.1f}%, "
            f"win rate {report.ml_enhanced.win_rate * 100:.1f}%, "
            f"Calmar {report.ml_enhanced.calmar:.2f}.",
            "",
            f"Manual:       annual return {report.manual.annual_return_pct:.1f}%, "
            f"max drawdown {report.manual.max_drawdown_pct:.1f}%, "
            f"win rate {report.manual.win_rate * 100:.1f}%, "
            f"Calmar {report.manual.calmar:.2f}.",
        ]

        if best_benchmark:
            bm = report.benchmarks[best_benchmark]
            lines.append(
                f"\nBest benchmark: {bm.name} — Sharpe {bm.sharpe:.2f}, "
                f"annual return {bm.annual_return_pct:.1f}%."
            )

        lines.append(f"\nOverall winner: {report.winner}.")
        return "\n".join(lines)

    def to_html(self, report: ComparisonReport) -> str:
        """Generate simple HTML report for email/PDF conversion.

        Bloomberg dark theme — dark background, green for positive, red for negative.
        """
        summary_html = self.executive_summary(report).replace("\n", "<br>")
        metrics_table = self._metrics_table_rows(report)
        equity_section = self._equity_curve_section(report)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>QuantEdge Comparison Report — {_h(report.strategy_name)} / {_h(report.symbol)}</title>
  <style>
    body {{ background:#0d1117; color:#c9d1d9; font-family:Arial,Helvetica,sans-serif; padding:24px; }}
    h1 {{ color:#58a6ff; }}
    .summary {{ background:#161b22; padding:12px; margin-bottom:20px; border-left:4px solid #58a6ff; }}
    table {{ width:100%; border-collapse:collapse; font-size:0.9rem; }}
    th {{ background:#161b22; color:#58a6ff; padding:8px; text-align:left; }}
    td {{ padding:8px; border-bottom:1px solid #21262d; }}
    .pos {{ color:#3fb950; }}
    .neg {{ color:#f85149; }}
  </style>
</head>
<body>
  <h1>QuantEdge Comparison Report</h1>
  <div class="summary">{summary_html}</div>
  {metrics_table}
  {equity_section}
</body>
</html>"""
        return html

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _backtest_to_strategy_metrics(self, name: str, backtest) -> StrategyMetrics:
        """Convert a back‑test result object to a StrategyMetrics instance."""
        #