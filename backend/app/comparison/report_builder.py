"""Investor-facing comparison report builder.

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

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
MANUAL_STRATEGY_LABEL = "Manual Strategy"
ML_ENHANCED_STRATEGY_LABEL = "ML-Enhanced Strategy"

DEFAULT_SORTINO_MULTIPLIER = 1.15
DEFAULT_MAX_DD_FALLBACK = 1.0
MIN_DENOMINATOR = 1e-9

SHARPE_IMPROVEMENT_FORMAT = "{:.2f}"
P_VALUE_FORMAT = "{:.4f}"
STAT_SIGNIFICANT_TEMPLATE = "statistically significant (p={})"
STAT_NOT_SIGNIFICANT_TEMPLATE = "not statistically significant (p={})"

BEST_BENCHMARK_PREFIX = "Best benchmark: "
OVERALL_WINNER_PREFIX = "Overall winner: "

HTML_TITLE_PREFIX = "QuantEdge Comparison Report — "
HTML_H1_PREFIX = "QuantEdge Comparison Report — "

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


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

        manual_metrics = self._backtest_to_strategy_metrics(MANUAL_STRATEGY_LABEL, cr.manual)
        ml_metrics = self._backtest_to_strategy_metrics(ML_ENHANCED_STRATEGY_LABEL, cr.ml_enhanced)

        # Build benchmark StrategyMetrics from static stats using a dict comprehension
        benchmark_metrics: Dict[str, StrategyMetrics] = {
            key: StrategyMetrics(
                name=stats.get("name", key),
                sharpe=float(stats.get("sharpe", 0.0)),
                sortino=float(stats.get("sharpe", 0.0)) * DEFAULT_SORTINO_MULTIPLIER,
                annual_return_pct=round(float(stats.get("annual_return", 0.0)) * 100, 2),
                max_drawdown_pct=round(float(stats.get("max_dd", 0.0)) * 100, 2),
                win_rate=0.0,    # not available from static stats
                total_trades=0,
                avg_hold_days=0.0,
                calmar=round(
                    float(stats.get("annual_return", 0.0))
                    / max(abs(float(stats.get("max_dd", DEFAULT_MAX_DD_FALLBACK))), MIN_DENOMINATOR),
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
        return asdict(report)

    def executive_summary(self, report: ComparisonReport) -> str:
        """Plain English summary: 'ML Momentum outperforms manual by 34% Sharpe...'"""
        direction = "outperforms" if report.ml_improvement_pct > 0 else "underperforms"
        abs_improvement = abs(report.ml_improvement_pct)

        if report.is_statistically_significant:
            sig_phrase = STAT_SIGNIFICANT_TEMPLATE.format(P_VALUE_FORMAT.format(report.p_value))
        else:
            sig_phrase = STAT_NOT_SIGNIFICANT_TEMPLATE.format(P_VALUE_FORMAT.format(report.p_value))

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
                f"\n{BEST_BENCHMARK_PREFIX}{bm.name} — Sharpe {bm.sharpe:.2f}, "
                f"annual return {bm.annual_return_pct:.1f}%."
            )

        lines.append(f"\n{OVERALL_WINNER_PREFIX}{report.winner}.")
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
  <title>{HTML_TITLE_PREFIX}{_h(report.strategy_name)} / {_h(report.symbol)}</title>
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
  <h1>{HTML_H1_PREFIX}{_h(report.strategy_name)} / {_h(report.symbol)}</h1>
  <div class="summary">{summary_text}</div>
  <section class="metrics">
    <h2>Metrics</h2>
    <table>
      <thead>
        <tr><th>Metric</th><th>Manual</th><th>ML-Enhanced</th></tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </section>
  <section class="equity">
    <h2>Equity Curves</h2>
    {eq_section}
  </section>
</body>
</html>"""
        return html

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _backtest_to_strategy_metrics(self, name: str, backtest) -> StrategyMetrics:
        # Placeholder implementation – assumes backtest object has needed attributes
        return StrategyMetrics(
            name=name,
            sharpe=getattr(backtest, "sharpe", 0.0),
            sortino=getattr(backtest, "sortino", 0.0),
            annual_return_pct=getattr(backtest, "annual_return_pct", 0.0),
            max_drawdown_pct=getattr(backtest, "max_drawdown_pct", 0.0),
            win_rate=getattr(backtest, "win_rate", 0.0),
            total_trades=getattr(backtest, "total_trades", 0),
            avg_hold_days=getattr(backtest, "avg_hold_days", 0.0),
            calmar=getattr(backtest, "calmar", 0.0),
        )

    def _determine_winner(
        self,
        ml_metrics: StrategyMetrics,
        manual_metrics: StrategyMetrics,
        benchmarks: Dict[str, StrategyMetrics],
        engine_winner: Optional[str],
    ) -> str:
        # Simplified placeholder logic
        if engine_winner:
            return engine_winner
        if ml_metrics.sharpe > manual_metrics.sharpe:
            return "ml"
        return "manual"

    def _extract_equity_curves(self, cr: ComparisonResult) -> Dict[str, List[float]]:
        # Placeholder – normalize equity curves to start at 100
        curves = {}
        for name, equity in cr.equity_curves.items():
            if not equity:
                curves[name] = []
                continue
            start = equity[0] if equity[0] != 0 else 1
            curves[name] = [e / start * 100 for e in equity]
        return curves

    def _best_benchmark(self, report: ComparisonReport) -> Optional[str]:
        # Return benchmark key with highest Sharpe
        if not report.benchmarks:
            return None
        best = max(report.benchmarks.items(), key=lambda kv: kv[1].sharpe)
        return best[0]

    def _metrics_table_rows(self, report: ComparisonReport) -> str:
        # Simplified placeholder rendering
        rows = []
        for metric in ["sharpe", "sortino", "annual_return_pct", "max_drawdown_pct", "win_rate", "calmar"]:
            manual_val = getattr(report.manual, metric)
            ml_val = getattr(report.ml_enhanced, metric)
            rows.append(
                f"<tr><td>{metric.replace('_', ' ').title()}</td>"
                f"<td>{manual_val:.2f}</td>"
                f"<td>{ml_val:.2f}</td></tr>"
            )
        return "\n".join(rows)

    def _equity_curve_section(self, report: ComparisonReport) -> str:
        # Simplified placeholder rendering
        sections = []