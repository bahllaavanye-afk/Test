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
from typing import Dict, List, Optional

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
        cr = comparison_result

        manual_metrics = self._backtest_to_strategy_metrics("Manual Strategy", cr.manual)
        ml_metrics = self._backtest_to_strategy_metrics("ML-Enhanced Strategy", cr.ml_enhanced)

        benchmark_metrics = self._build_benchmark_metrics(cr.benchmark_stats)

        ml_improvement_pct = self._calculate_improvement_pct(
            manual_sharpe=manual_metrics.sharpe,
            ml_sharpe=ml_metrics.sharpe,
        )

        winner = self._determine_winner(ml_metrics, manual_metrics, benchmark_metrics, cr.winner)

        equity_curves = self._extract_equity_curves(cr)

        period_str = self._format_period(cr.start_date, cr.end_date)

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
        return asdict(report)

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
  <div class="meta">{_h(report.interval)} | {report.period}</div>
  <div class="summary">{summary_text}</div>
  <h2>Metrics</h2>
  <table>
    <tr>
      <th>Strategy</th><th>Sharpe</th><th>Sortino</th><th>Annual Return %</th>
      <th>Max DD %</th><th>Win Rate %</th><th>Calmar</th>
    </tr>
    {rows_html}
  </table>
  <h2>Equity Curves</h2>
  {eq_section}
</body>
</html>"""
        return html

    # ------------------------------------------------------------------ #
    # Helper methods (private)                                            #
    # ------------------------------------------------------------------ #

    def _backtest_to_strategy_metrics(self, name: str, backtest) -> StrategyMetrics:
        """Convert raw backtest data into a StrategyMetrics instance."""
        return StrategyMetrics(
            name=name,
            sharpe=backtest.sharpe,
            sortino=backtest.sortino,
            annual_return_pct=round(backtest.annual_return * 100, 2),
            max_drawdown_pct=round(backtest.max_drawdown * 100, 2),
            win_rate=backtest.win_rate,
            total_trades=backtest.total_trades,
            avg_hold_days=backtest.avg_hold_days,
            calmar=round(
                backtest.annual_return / max(abs(backtest.max_drawdown), 1e-9), 4
            ),
        )

    def _build_benchmark_metrics(self, benchmark_stats: Dict[str, dict]) -> Dict[str, StrategyMetrics]:
        """Create StrategyMetrics objects for each benchmark from static stats."""
        return {
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
            for key, stats in benchmark_stats.items()
        }

    def _calculate_improvement_pct(self, manual_sharpe: float, ml_sharpe: float) -> float:
        """Calculate percentage Sharpe improvement of ML over manual."""
        if manual_sharpe != 0:
            return round((ml_sharpe - manual_sharpe) / abs(manual_sharpe) * 100, 2)
        return round((ml_sharpe - manual_sharpe) * 100, 2)

    def _format_period(self, start_date: str, end_date: str) -> str:
        """Format the period string for the report."""
        return f"{start_date} to {end_date}"

    def _determine_winner(
        self,
        ml_metrics: StrategyMetrics,
        manual_metrics: StrategyMetrics,
        benchmark_metrics: Dict[str, StrategyMetrics],
        engine_winner: str,
    ) -> str:
        """Resolve the final winner string, falling back to engine decision."""
        # Simple placeholder logic – real implementation may involve more rules.
        return engine_winner or "ml"

    def _extract_equity_curves(self, cr: ComparisonResult) -> Dict[str, List[float]]:
        """Normalize equity curves to start at 100."""
        curves = {}
        for name, equity in cr.equity_curves.items():
            if not equity:
                continue
            start = equity[0] or 1.0
            curves[name] = [(value / start) * 100 for value in equity]
        return curves

    def _metrics_table_rows(self, report: ComparisonReport) -> str:
        """Render HTML rows for the metrics table."""
        rows = []
        # Manual
        rows.append(self._format_metrics_row(report.manual, "manual"))
        # ML
        rows.append(self._format_metrics_row(report.ml_enhanced, "ml"))
        # Benchmarks
        for key, bm in report.benchmarks.items():
            rows.append(self._format_metrics_row(bm, f"benchmark:{key}"))
        return "\n".join(rows)

    def _format_metrics_row(self, metrics: StrategyMetrics, tag: str) -> str:
        """Format a single table row with appropriate CSS classes."""
        def _class(val: float) -> str:
            if val > 0:
                return "pos"
            if val < 0:
                return "neg"
            return "neutral"

        sharpe_cls = _class(metrics.sharpe)
        sortino_cls = _class(metrics.sortino)
        return f"""<tr>
<td>{_h(metrics.name)}</td>
<td class="{sharpe_cls}">{metrics.sharpe:.2f}</td>
<td class="{sortino_cls}">{metrics.sortino:.2f}</td>
<td>{metrics.annual_return_pct:.1f}</td>
<td>{metrics.max_drawdown_pct:.1f}</td>
<td>{metrics.win_rate * 100:.1f}</td>
<td>{metrics.calmar:.2f}</td>
</tr>"""

    def _best_benchmark(self, report: ComparisonReport) -> Optional[str]:
        """Return the benchmark key with the highest Sharpe, if any."""
        if not report.benchmarks:
            return None
        return max(report.benchmarks.items(), key=lambda kv: kv[1].sharpe)[0]

    def _equity_curve_section(self, report: ComparisonReport) -> str:
        """Render a simple SVG placeholder for equity curves."""
        # Placeholder implementation – actual rendering would generate charts.
        sections = []
        for name, curve in report.equity_curves.items():
            points = " ".join(f"{i},{value:.2f}" for i, value in enumerate(curve))
            sections.append(
                f'<svg width="300" height="100" class="equity-curve"><polyline points="{points}" '
                f'style="fill:none;stroke:#58a6ff;stroke-width:1"/></svg>'
            )
        return "\n".join(sections)