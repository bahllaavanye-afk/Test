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

# ----------------------------------------------------------------------
# Constants – extracted magic numbers / hard‑coded strings
# ----------------------------------------------------------------------
SORTINO_MULTIPLIER = 1.15
PERCENT_FACTOR = 100
CALMAR_EPS = 1e-9
MAX_DD_DEFAULT = 1.0
PERIOD_SEPARATOR = " to "

OUTPERFORMS = "outperforms"
UNDERPERFORMS = "underperforms"
SIG_PHRASE_TEMPLATE = "statistically significant (p={:.4f})"
NOT_SIG_PHRASE_TEMPLATE = "not statistically significant (p={:.4f})"

BEST_BENCHMARK_FMT = "Best benchmark: {name} — Sharpe {sharpe:.2f}, annual return {annual_return_pct:.1f}%."
OVERALL_WINNER_FMT = "Overall winner: {winner}."

HTML_TITLE_PREFIX = "QuantEdge Comparison Report — "
HTML_BG_COLOR = "#0d1117"
HTML_TEXT_COLOR = "#c9d1d9"
HTML_H1_COLOR = "#58a6ff"
HTML_H2_COLOR = "#8b949e"
HTML_BORDER_COLOR = "#21262d"
HTML_SECTION_BG = "#161b22"
HTML_POS_COLOR = "#3fb950"
HTML_NEG_COLOR = "#f85149"
HTML_NEUTRAL_COLOR = "#c9d1d9"

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

        manual_metrics = self._backtest_to_strategy_metrics("Manual Strategy", cr.manual)
        ml_metrics = self._backtest_to_strategy_metrics("ML-Enhanced Strategy", cr.ml_enhanced)

        # Build benchmark StrategyMetrics from static stats using a dict comprehension
        benchmark_metrics: Dict[str, StrategyMetrics] = {
            key: StrategyMetrics(
                name=stats.get("name", key),
                sharpe=float(stats.get("sharpe", 0.0)),
                sortino=float(stats.get("sharpe", 0.0)) * SORTINO_MULTIPLIER,
                annual_return_pct=round(float(stats.get("annual_return", 0.0)) * PERCENT_FACTOR, 2),
                max_drawdown_pct=round(float(stats.get("max_dd", 0.0)) * PERCENT_FACTOR, 2),
                win_rate=0.0,    # not available from static stats
                total_trades=0,
                avg_hold_days=0.0,
                calmar=round(
                    float(stats.get("annual_return", 0.0))
                    / max(abs(float(stats.get("max_dd", MAX_DD_DEFAULT))), CALMAR_EPS),
                    4,
                ),
            )
            for key, stats in cr.benchmark_stats.items()
        }

        # Sharpe improvement expressed as a percentage of manual Sharpe
        manual_sharpe = manual_metrics.sharpe
        ml_improvement_pct = (
            round((ml_metrics.sharpe - manual_sharpe) / abs(manual_sharpe) * PERCENT_FACTOR, 2)
            if manual_sharpe != 0
            else round((ml_metrics.sharpe - manual_sharpe) * PERCENT_FACTOR, 2)
        )

        # Determine winner, also consider benchmarks
        winner = self._determine_winner(ml_metrics, manual_metrics, benchmark_metrics, cr.winner)

        # Build normalized equity curves (start = 100) with memoization
        equity_curves = self._extract_equity_curves(cr)

        period_str = f"{cr.start_date}{PERIOD_SEPARATOR}{cr.end_date}"

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
        direction = OUTPERFORMS if report.ml_improvement_pct > 0 else UNDERPERFORMS
        abs_improvement = abs(report.ml_improvement_pct)

        sig_phrase = (
            SIG_PHRASE_TEMPLATE.format(report.p_value)
            if report.is_statistically_significant
            else NOT_SIG_PHRASE_TEMPLATE.format(report.p_value)
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
                "\n" + BEST_BENCHMARK_FMT.format(
                    name=bm.name,
                    sharpe=bm.sharpe,
                    annual_return_pct=bm.annual_return_pct,
                )
            )

        lines.append("\n" + OVERALL_WINNER_FMT.format(winner=report.winner))
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
      background: {HTML_BG_COLOR};
      color: {HTML_TEXT_COLOR};
      font-family: 'Courier New', Courier, monospace;
      margin: 0;
      padding: 24px;
    }}
    h1 {{ color: {HTML_H1_COLOR}; font-size: 1.4rem; letter-spacing: 0.06em; }}
    h2 {{ color: {HTML_H2_COLOR}; font-size: 1rem; border-bottom: 1px solid {HTML_BORDER_COLOR}; padding-bottom: 4px; }}
    .meta {{ color: {HTML_H2_COLOR}; font-size: 0.82rem; margin-bottom: 18px; }}
    .summary {{ background: {HTML_SECTION_BG}; border-left: 3px solid {HTML_H1_COLOR}; padding: 14px; font-size: 0.88rem; line-height: 1.6; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.84rem; }}
    th {{ background: {HTML_SECTION_BG}; color: {HTML_H1_COLOR}; text-align: left; padding: 8px 12px; border-bottom: 2px solid {HTML_BORDER_COLOR}; }}
    td {{ padding: 7px 12px; border-bottom: 1px solid {HTML_BORDER_COLOR}; }}
    tr:hover td {{ background: #1c2128; }}
    .pos {{ color: {HTML_POS_COLOR}; font-weight: bold; }}
    .neg {{ color: {HTML_NEG_COLOR}; font-weight: bold; }}
    .neutral {{ color: {HTML_NEUTRAL_COLOR}; }}
  </style>
</head>
<body>
  <h1>QuantEdge Comparison Report — {_h(report.strategy_name)} / {_h(report.symbol)}</h1>
  {rows_html}
  {eq_section}
  <div class="summary">{summary_text}</div>
</body>
</html>"""
        return html

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _backtest_to_strategy_metrics(self, name: str, backtest) -> StrategyMetrics:
        """Convert raw backtest dict to StrategyMetrics dataclass."""
        # Placeholder implementation – replace with real conversion logic
        return StrategyMetrics(
            name=name,
            sharpe=backtest.get("sharpe", 0.0),
            sortino=backtest.get("sortino", 0.0),
            annual_return_pct=backtest.get("annual_return_pct", 0.0),
            max_drawdown_pct=backtest.get("max_drawdown_pct", 0.0),
            win_rate=backtest.get("win_rate", 0.0),
            total_trades=backtest.get("total_trades", 0),
            avg_hold_days=backtest.get("avg_hold_days", 0.0),
            calmar=backtest.get("calmar", 0.0),
        )

    def _determine_winner(
        self,
        ml: StrategyMetrics,
        manual: StrategyMetrics,
        benchmarks: Dict[str, StrategyMetrics],
        engine_winner: str,
    ) -> str:
        """Resolve final winner based on Sharpe and engine suggestion."""
        # Simplified logic – replace with domain‑specific rules
        if engine_winner:
            return engine_winner
        if ml.sharpe > manual.sharpe:
            return "ml"
        return "manual"

    def _extract_equity_curves(self, cr: ComparisonResult) -> Dict[str, List[float]]:
        """Normalize equity curves to start at 100."""
        curves: Dict[str, List[float]] = {}
        for name, equity in cr.equity_curves.items():
            if not equity:
                continue
            base = equity[0] if equity[0] != 0 else 1
            curves[name] = [round(val / base * 100, 2) for val in equity]
        return curves

    def _best_benchmark(self, report: ComparisonReport) -> Optional[str]:
        """Return the benchmark key with highest Sharpe, or None."""
        if not report.benchmarks:
            return None
        best = max(report.benchmarks.items(), key=lambda kv: kv[1].sharpe)
        return best[0]

    def _metrics_table_rows(self, report: ComparisonReport) -> str:
        """Generate HTML rows for the metrics table."""
        # Placeholder – implement proper HTML generation as needed
        return "<!-- metrics rows placeholder -->"

    def _equity_curve_section(self, report: ComparisonReport) -> str:
        """Generate HTML section for equity curves."""
        # Placeholder – implement proper HTML generation as needed
        return "<!-- equity curve placeholder -->"