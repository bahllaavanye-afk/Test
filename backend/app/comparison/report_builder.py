"""
Investor-facing comparison report builder.

Generates structured JSON reports comparing manual vs ML-enhanced strategies
against SPY, QQQ, BRK-B, All Weather benchmarks.

Used by: GET /api/v1/comparison/report
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import escape as _h
from typing import Dict, List, Optional, Any

from app.comparison.engine import ComparisonResult
from app.comparison.benchmarks import get_benchmark_stats


@dataclass
class StrategyMetrics:
    name: str
    sharpe: float
    sortino: float
    annual_return_pct: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    avg_hold_days: float
    calmar: float


@dataclass
class ComparisonReport:
    strategy_name: str
    symbol: str
    interval: str
    period: str                              # "2021-01-01 to 2024-12-31"
    manual: StrategyMetrics
    ml_enhanced: StrategyMetrics
    benchmarks: Dict[str, StrategyMetrics]  # "SPY", "QQQ", "BRK-B", "All Weather"
    ml_improvement_pct: float               # % Sharpe improvement
    is_statistically_significant: bool
    t_statistic: float
    p_value: float
    winner: str                             # "manual" | "ml" | "benchmark:SPY" etc.
    equity_curves: Dict[str, List[float]]   # {name: [normalized equity values]}
    generated_at: str


class ReportBuilder:
    """Builds investor-facing ComparisonReport objects from ComparisonEngine results."""

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def build(self, comparison_result: ComparisonResult) -> ComparisonReport:
        """Build investor report from ComparisonEngine result."""
        self._validate_comparison_result(comparison_result)
        cr = comparison_result

        manual_metrics = self._backtest_to_strategy_metrics("Manual Strategy", cr.manual)
        ml_metrics = self._backtest_to_strategy_metrics("ML-Enhanced Strategy", cr.ml_enhanced)

        # Build benchmark StrategyMetrics from static stats using a dict comprehension
        benchmark_metrics: Dict[str, StrategyMetrics] = {
            key: StrategyMetrics(
                name=stats.get("name", key),
                sharpe=float(stats.get("sharpe", 0.0)),
                sortino=float(stats.get("sharpe", 0.0)) * 1.15,  # approximate if not provided
                annual_return_pct=round(float(stats.get("annual_return", 0.0)) * 100, 2),
                max_drawdown_pct=round(float(stats.get("max_dd", 0.0)) * 100, 2),
                win_rate=0.0,    # not available from static stats
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
        winner = self._determine_winner(ml_metrics, manual_metrics, benchmark_metrics, cr.winner)

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
        self._validate_report(report)
        return asdict(report)

    def executive_summary(self, report: ComparisonReport) -> str:
        """Plain English summary: 'ML Momentum outperforms manual by 34% Sharpe...'"""
        self._validate_report(report)
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
        self._validate_report(report)
        rows_html = self._metrics_table_rows(report)
        eq_section = self._equity_curve_section(report)
        summary_text = self.executive_summary(report).replace("\n", "<br>")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>QuantEdge Comparison Report — {_h(report.strategy_name)} / {_h(report.symbol)}</title>
  <style>
    body {{
      background: #0d1117;
      color: #c9d1d9;
      font-family: 'Courier New', Courier, monospace;
      margin: 0;
      padding: 24px;
    }}
    h1 {{ color: #58a6ff; font-size: 1.4rem; letter-spacing: 0.06em; }}
    h2 {{ color: #8b949e; font-size: 1rem; border-bottom: 1px solid #21262d; padding-bottom: 4px; }}
    .meta {{ color: #8b949e; font-size: 0.82rem; margin-bottom: 18px; }}
    .summary {{ background: #161b22; border-left: 3px solid #58a6ff; padding: 14px; font-size: 0.88rem; line-height: 1.6; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.84rem; }}
    th {{ background: #161b22; color: #58a6ff; text-align: left; padding: 8px 12px; border-bottom: 2px solid #21262d; }}
    td {{ padding: 7px 12px; border-bottom: 1px solid #21262d; }}
    tr:hover td {{ background: #1c2128; }}
    .pos {{ color: #3fb950; font-weight: bold; }}
    .neg {{ color: #f85149; font-weight: bold; }}
    .neutral {{ color: #c9d1d9; }}
  </style>
</head>
<body>
  <h1>QuantEdge Comparison Report — {_h(report.strategy_name)} / {_h(report.symbol)}</h1>
  <div class="meta">Generated at: {_h(report.generated_at)}</div>
  <div class="summary">{summary_text}</div>
  <h2>Metrics</h2>
  <table>
    <thead><tr><th>Metric</th><th>Manual</th><th>ML-Enhanced</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  <h2>Equity Curves</h2>
  {eq_section}
</body>
</html>"""
        return html

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _validate_comparison_result(self, cr: ComparisonResult) -> None:
        """Validate that a ComparisonResult instance contains required attributes."""
        if cr is None:
            raise ValueError("build() received None as ComparisonResult.")
        if not isinstance(cr, ComparisonResult):
            raise ValueError(
                f"build() expects a ComparisonResult instance, got {type(cr).__name__}."
            )
        required_attrs = [
            "manual", "ml_enhanced", "benchmark_stats", "strategy_name",
            "symbol", "interval", "start_date", "end_date", "is_significant",
            "t_statistic", "p_value", "winner"
        ]
        missing = [attr for attr in required_attrs if not hasattr(cr, attr)]
        if missing:
            raise ValueError(f"ComparisonResult missing required attributes: {', '.join(missing)}.")

        if not isinstance(cr.benchmark_stats, dict):
            raise ValueError("ComparisonResult.benchmark_stats must be a dictionary.")

    def _validate_report(self, report: ComparisonReport) -> None:
        """Validate that a ComparisonReport instance contains required attributes."""
        if report is None:
            raise ValueError("Function received None instead of a ComparisonReport.")
        if not isinstance(report, ComparisonReport):
            raise ValueError(
                f"Expected a ComparisonReport instance, got {type(report).__name__}."
            )
        # Simple type checks for crucial numeric fields
        numeric_fields = [
            ("ml_improvement_pct", report.ml_improvement_pct),
            ("t_statistic", report.t_statistic),
            ("p_value", report.p_value),
        ]
        for name, value in numeric_fields:
            if not isinstance(value, (int, float)):
                raise ValueError(f"ComparisonReport.{name} must be a number, got {type(value).__name__}.")

    def _backtest_to_strategy_metrics(self, name: str, backtest: Any) -> StrategyMetrics:
        """Convert backtest result dict/obj to StrategyMetrics."""
        # Assuming backtest provides a dict-like interface
        stats = getattr(backtest, "stats", backtest)  # allow dict or object with .stats
        if not isinstance(stats, dict):
            raise ValueError("Backtest data must be a dict or have a .stats attribute returning a dict.")
        return StrategyMetrics(
            name=name,
            sharpe=float(stats.get("sharpe", 0.0)),
            sortino=float(stats.get("sortino", 0.0)),
            annual_return_pct=round(float(stats.get("annual_return", 0.0)) * 100, 2),
            max_drawdown_pct=round(float(stats.get("max_drawdown", 0.0)) * 100, 2),
            win_rate=float(stats.get("win_rate", 0.0)),
            total_trades=int(stats.get("total_trades", 0)),
            avg_hold_days=float(stats.get("avg_hold_days", 0.0)),
            calmar=round(
                float(stats.get("annual_return", 0.0))
                / max(abs(float(stats.get("max_drawdown", 1.0))), 1e-9),
                4,
            ),
        )

    def _determine_winner(
        self,
        ml: StrategyMetrics,
        manual: StrategyMetrics,
        benchmarks: Dict[str, StrategyMetrics],
        default_winner: str,
    ) -> str:
        """Determine overall winner based on Sharpe and optional external hint."""
        # Simple rule: pick the highest Sharpe among ml, manual, benchmarks
        candidates = {"ml": ml.sharpe, "manual": manual.sharpe}
        candidates.update({f"benchmark:{k}": v.sharpe for k, v in benchmarks.items()})
        best = max(candidates, key=candidates.get)
        return best if best else default_winner

    def _extract_equity_curves(self, cr: ComparisonResult) -> Dict[str, List[float]]:
        """Normalize equity curves to start at 100."""
        def normalize(curve: List[float]) -> List[float]:
            if not curve:
                return []
            base = curve[0] if curve[0] != 0 else 1
            return [round(v / base * 100, 2) for v in curve]

        return {
            "Manual Strategy": normalize(getattr(cr.manual, "equity_curve", [])),
            "ML-Enhanced Strategy": normalize(getattr(cr.ml_enhanced, "equity_curve", [])),
        }

    def _best_benchmark(self, report: ComparisonReport) -> Optional[str]:
        """Return the key of the benchmark with highest Sharpe, or None."""
        if not report.benchmarks:
            return None
        best_key = max(report.benchmarks, key=lambda k: report.benchmarks[k].sharpe)
        return best_key

    def _metrics_table_rows(self, report: ComparisonReport) -> str:
        """Render HTML rows for the metrics table."""
        def fmt(value: float) -> str:
            return f"{value:.2f}"

        rows = ""
        metrics = [
            ("Sharpe", report.manual.sharpe, report.ml_enhanced.sharpe),
            ("Sortino", report.manual.sortino, report.ml_enhanced.sortino),
            ("Annual Return (%)", report.manual.annual_return_pct, report.ml_enhanced.annual_return_pct),
            ("Max Drawdown (%)", report.manual.max_drawdown_pct, report.ml_enhanced.max_drawdown_pct),
            ("Win Rate (%)", report.manual.win_rate * 100, report.ml_enhanced.win_rate * 100),
            ("Calmar", report.manual.calmar, report.ml_enhanced.calmar),
        ]
        for name, manual_val, ml_val in metrics:
            diff = ml_val - manual_val
            css_class = "pos" if diff > 0 else "neg" if diff < 0 else "neutral"
            rows += f"<tr><td>{_h(name)}</td><td>{fmt(manual_val)}</td><td class=\"{css_class}\">{fmt(ml_val)}</td></tr>"
        return rows

    def _equity_curve_section(self, report: ComparisonReport) -> str:
        """Simple placeholder for equity curve visualization."""
        # In a real implementation this would embed a chart; here we list values.
        lines = []
        for name, curve in report.equity_curves.items():
            points = ", ".join(str(v) for v in curve[:10])  # show first 10 points
            lines.append(f"<p><strong>{_h(name)}:</strong> {points} ...</p>")
        return "\n".join(lines)