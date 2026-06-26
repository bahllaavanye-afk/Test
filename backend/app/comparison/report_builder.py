"""
Investor-facing comparison report builder.

Generates structured JSON reports comparing manual vs ML-enhanced strategies
against SPY, QQQ, BRK-B, All Weather benchmarks.

Used by: GET /api/v1/comparison/report
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape as _h

from app.comparison.engine import ComparisonResult


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
    benchmarks: dict[str, StrategyMetrics]  # "SPY", "QQQ", "BRK-B", "All Weather"
    ml_improvement_pct: float               # % Sharpe improvement
    is_statistically_significant: bool
    t_statistic: float
    p_value: float
    winner: str                             # "manual" | "ml" | "benchmark:SPY" etc.
    equity_curves: dict[str, list[float]]  # {name: [normalized equity values]}
    generated_at: str


class ReportBuilder:
    """Builds investor-facing ComparisonReport objects from ComparisonEngine results."""

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def build(self, comparison_result: ComparisonResult) -> ComparisonReport:
        """Build investor report from ComparisonEngine result."""
        cr = comparison_result

        manual_metrics = self._backtest_to_strategy_metrics("Manual Strategy", cr.manual)
        ml_metrics = self._backtest_to_strategy_metrics("ML-Enhanced Strategy", cr.ml_enhanced)

        # Build benchmark StrategyMetrics from static stats
        benchmark_metrics: dict[str, StrategyMetrics] = {}
        for key, stats in cr.benchmark_stats.items():
            display_name = stats.get("name", key)
            benchmark_metrics[key] = StrategyMetrics(
                name=display_name,
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

        # Sharpe improvement expressed as a percentage of manual Sharpe
        manual_sharpe = manual_metrics.sharpe
        if manual_sharpe != 0:
            ml_improvement_pct = round(
                (ml_metrics.sharpe - manual_sharpe) / abs(manual_sharpe) * 100, 2
            )
        else:
            ml_improvement_pct = round((ml_metrics.sharpe - manual_sharpe) * 100, 2)

        # Determine winner, also consider benchmarks
        winner = self._determine_winner(ml_metrics, manual_metrics, benchmark_metrics, cr.winner)

        # Build normalized equity curves (start = 100)
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
        # asdict recursively converts dataclasses to plain dicts
        return asdict(report)

    # ------------------------------------------------------------------ #
    # Summary generation helpers                                          #
    # ------------------------------------------------------------------ #

    def executive_summary(self, report: ComparisonReport) -> str:
        """Plain English summary: 'ML Momentum outperforms manual by 34% Sharpe...'"""
        lines = [
            self._summary_header(report),
            "",
            self._summary_ml_details(report),
            "",
            self._summary_manual_details(report),
        ]

        best_benchmark = self._best_benchmark(report)
        if best_benchmark:
            lines.append(self._summary_best_benchmark_line(report, best_benchmark))

        lines.append(self._summary_winner_line(report))
        return "\n".join(lines)

    def _summary_header(self, report: ComparisonReport) -> str:
        direction = "outperforms" if report.ml_improvement_pct > 0 else "underperforms"
        abs_improvement = abs(report.ml_improvement_pct)
        sig_phrase = (
            f"statistically significant (p={report.p_value:.4f})"
            if report.is_statistically_significant
            else f"not statistically significant (p={report.p_value:.4f})"
        )
        return (
            f"ML {report.strategy_name} {direction} the manual strategy by "
            f"{abs_improvement:.1f}% on a risk-adjusted Sharpe basis "
            f"({report.ml_enhanced.sharpe:.2f} vs {report.manual.sharpe:.2f}), "
            f"and the result is {sig_phrase}."
        )

    def _summary_ml_details(self, report: ComparisonReport) -> str:
        m = report.ml_enhanced
        return (
            f"ML-Enhanced:  annual return {m.annual_return_pct:.1f}%, "
            f"max drawdown {m.max_drawdown_pct:.1f}%, "
            f"win rate {m.win_rate * 100:.1f}%, "
            f"Calmar {m.calmar:.2f}."
        )

    def _summary_manual_details(self, report: ComparisonReport) -> str:
        m = report.manual
        return (
            f"Manual:       annual return {m.annual_return_pct:.1f}%, "
            f"max drawdown {m.max_drawdown_pct:.1f}%, "
            f"win rate {m.win_rate * 100:.1f}%, "
            f"Calmar {m.calmar:.2f}."
        )

    def _summary_best_benchmark_line(self, report: ComparisonReport, best_key: str) -> str:
        bm = report.benchmarks[best_key]
        return (
            f"\nBest benchmark: {bm.name} — Sharpe {bm.sharpe:.2f}, "
            f"annual return {bm.annual_return_pct:.1f}%."
        )

    def _summary_winner_line(self, report: ComparisonReport) -> str:
        return f"\nOverall winner: {report.winner}."

    # ------------------------------------------------------------------ #
    # HTML generation helpers                                             #
    # ------------------------------------------------------------------ #

    def to_html(self, report: ComparisonReport) -> str:
        """Generate simple HTML report for email/PDF conversion.

        Bloomberg dark theme — dark background, green for positive, red for negative.
        """
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
    .winner-badge {{
      display: inline-block;
      background: #3fb950;
      color: #0d1117;
      padding: 3px 7px;
      border-radius: 4px;
      font-weight: bold;
    }}
  </style>
</head>
<body>
  <h1>QuantEdge Comparison Report</h1>
  <div class="meta">
    <strong>Strategy:</strong> {_h(report.strategy_name)}<br>
    <strong>Symbol:</strong> {_h(report.symbol)}<br>
    <strong>Interval:</strong> {_h(report.interval)}<br>
    <strong>Period:</strong> {_h(report.period)}<br>
    <strong>Generated:</strong> {_h(report.generated_at)}
  </div>
  <div class="summary">{summary_text}</div>
  <h2>Performance Metrics</h2>
  <table>
    <thead>
      <tr>
        <th>Component</th><th>Sharpe</th><th>Sortino</th><th>Annual Return %</th><th>Max DD %</th><th>Win Rate %</th><th>Calmar</th>
      </tr>
    </thead>
    <tbody>
{rows_html}
    </tbody>
  </table>
  <h2>Equity Curves</h2>
  {eq_section}
</body>
</html>"""
        return html

    # ------------------------------------------------------------------ #
    # Placeholder methods – implementations exist elsewhere or are stub #
    # ------------------------------------------------------------------ #

    def _backtest_to_strategy_metrics(self, name: str, backtest) -> StrategyMetrics:
        """Convert a backtest result object into a StrategyMetrics dataclass."""
        # Placeholder implementation – real logic resides elsewhere.
        raise NotImplementedError

    def _determine_winner(
        self,
        ml: StrategyMetrics,
        manual: StrategyMetrics,
        benchmarks: dict[str, StrategyMetrics],
        engine_winner: str,
    ) -> str:
        """Determine the overall winner based on Sharpe and other criteria."""
        # Placeholder implementation – real logic resides elsewhere.
        raise NotImplementedError

    def _extract_equity_curves(self, comparison_result: ComparisonResult) -> dict[str, list[float]]:
        """Extract normalized equity curves for each component."""
        # Placeholder implementation – real logic resides elsewhere.
        raise NotImplementedError

    def _best_benchmark(self, report: ComparisonReport) -> str | None:
        """Return the key of the benchmark with the highest Sharpe, or None if empty."""
        # Placeholder implementation – real logic resides elsewhere.
        raise NotImplementedError

    def _metrics_table_rows(self, report: ComparisonReport) -> str:
        """Render HTML rows for the metrics table."""
        # Placeholder implementation – real logic resides elsewhere.
        raise NotImplementedError

    def _equity_curve_section(self, report: ComparisonReport) -> str:
        """Render HTML for the equity curve chart(s)."""
        # Placeholder implementation – real logic resides elsewhere.
        raise NotImplementedError