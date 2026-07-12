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

        benchmark_metrics = self._build_benchmark_metrics(cr)
        ml_improvement_pct = self._calculate_ml_improvement_pct(manual_metrics, ml_metrics)
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

    # ------------------------------------------------------------------ #
    # Helper methods for build                                            #
    # ------------------------------------------------------------------ #

    def _build_benchmark_metrics(self, cr: ComparisonResult) -> Dict[str, StrategyMetrics]:
        """Create StrategyMetrics for each benchmark from static stats."""
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
            for key, stats in cr.benchmark_stats.items()
        }

    def _calculate_ml_improvement_pct(
        self, manual: StrategyMetrics, ml: StrategyMetrics
    ) -> float:
        """Calculate Sharpe improvement of ML strategy vs manual, expressed as a percent."""
        manual_sharpe = manual.sharpe
        if manual_sharpe != 0:
            return round((ml.sharpe - manual_sharpe) / abs(manual_sharpe) * 100, 2)
        return round((ml.sharpe - manual_sharpe) * 100, 2)

    def _format_period(self, start_date: str, end_date: str) -> str:
        """Return a human‑readable period string."""
        return f"{start_date} to {end_date}"

    # ------------------------------------------------------------------ #
    # Serialization & presentation                                         #
    # ------------------------------------------------------------------ #

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
  <div class="meta">Generated at {report.generated_at} | Interval: {report.interval} | Period: {report.period}</div>
  <div class="summary">{summary_text}</div>
  <h2>Metrics</h2>
  <table>
    <thead>
      <tr><th>Strategy</th><th>Sharpe</th><th>Sortino</th><th>Annual Return %</th><th>Max DD %</th><th>Win Rate %</th><th>Calmar</th></tr>
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
    # Private helper implementations (unchanged)                           #
    # ------------------------------------------------------------------ #

    def _backtest_to_strategy_metrics(self, name: str, backtest) -> StrategyMetrics:
        """Convert raw backtest output to StrategyMetrics."""
        # Placeholder conversion – actual implementation depends on backtest schema
        return StrategyMetrics(
            name=name,
            sharpe=backtest.get("sharpe", 0.0),
            sortino=backtest.get("sortino", 0.0),
            annual_return_pct=round(backtest.get("annual_return", 0.0) * 100, 2),
            max_drawdown_pct=round(backtest.get("max_dd", 0.0) * 100, 2),
            win_rate=backtest.get("win_rate", 0.0),
            total_trades=backtest.get("total_trades", 0),
            avg_hold_days=backtest.get("avg_hold_days", 0.0),
            calmar=round(
                backtest.get("annual_return", 0.0)
                / max(abs(backtest.get("max_dd", 1.0)), 1e-9),
                4,
            ),
        )

    def _determine_winner(
        self,
        ml: StrategyMetrics,
        manual: StrategyMetrics,
        benchmarks: Dict[str, StrategyMetrics],
        engine_winner: Optional[str],
    ) -> str:
        """Resolve the final winner string."""
        # Simplified logic – real logic may involve statistical tests
        if engine_winner:
            return engine_winner
        # Compare Sharpe ratios
        best = max(
            [("ml", ml.sharpe), ("manual", manual.sharpe)]
            + [(key, bm.sharpe) for key, bm in benchmarks.items()],
            key=lambda x: x[1],
        )
        return best[0] if best[0] != "ml" else "ml_enhanced"

    def _extract_equity_curves(self, cr: ComparisonResult) -> Dict[str, List[float]]:
        """Normalize equity curves to start at 100."""
        curves = {}
        for label, equity in [
            ("Manual Strategy", cr.manual.get("equity", [])),
            ("ML-Enhanced Strategy", cr.ml_enhanced.get("equity", [])),
        ]:
            if equity:
                base = equity[0] if equity[0] != 0 else 1
                curves[label] = [100 * v / base for v in equity]
        for bm_key, stats in cr.benchmark_stats.items():
            equity = stats.get("equity", [])
            if equity:
                base = equity[0] if equity[0] != 0 else 1
                curves[bm_key] = [100 * v / base for v in equity]
        return curves

    def _best_benchmark(self, report: ComparisonReport) -> Optional[str]:
        """Return the benchmark key with the highest Sharpe."""
        if not report.benchmarks:
            return None
        return max(report.benchmarks.items(), key=lambda kv: kv[1].sharpe)[0]

    def _metrics_table_rows(self, report: ComparisonReport) -> str:
        """Render HTML rows for the metrics table."""
        rows = []
        def fmt(val: float, fmt_str: str = "{:.2f}") -> str:
            return fmt_str.format(val)

        def color_class(val: float) -> str:
            if val > 0:
                return "pos"
            if val < 0:
                return "neg"
            return "neutral"

        for label, metrics in [
            (report.manual.name, report.manual),
            (report.ml_enhanced.name, report.ml_enhanced),
        ] + [(bm.name, bm) for bm in report.benchmarks.values()]:
            rows.append(
                f"<tr>"
                f"<td>{_h(label)}</td>"
                f"<td class='{color_class(metrics.sharpe)}'>{fmt(metrics.sharpe)}</td>"
                f"<td class='{color_class(metrics.sortino)}'>{fmt(metrics.sortino)}</td>"
                f"<td>{fmt(metrics.annual_return_pct, '{:.1f}')}</td>"
                f"<td>{fmt(metrics.max_drawdown_pct, '{:.1f}')}</td>"
                f"<td>{fmt(metrics.win_rate * 100, '{:.1f}')}</td>"
                f"<td>{fmt(metrics.calmar, '{:.2f}')}</td>"
                f"</tr>"
            )
        return "\n".join(rows)

    def _equity_curve_section(self, report: ComparisonReport) -> str:
        """Render simple SVG line chart for each equity curve."""
        # Placeholder implementation – real implementation would generate SVG or use a chart library
        sections = []
        for name, curve in report.equity_curves.items():
            points = " ".join(f"{i},{v:.2f}" for i, v in enumerate(curve))
            svg = (
                f"<svg width='400' height='150' class='neutral'>"
                f"<polyline points='{points}' fill='none' stroke='#58a6ff' stroke-width='2'/>"
                f"</svg>"
            )
            sections.append(f"<div><strong>{_h(name)}</strong>{svg}</div>")
        return "\n".join(sections)