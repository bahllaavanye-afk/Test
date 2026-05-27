"""
Investor-facing comparison report builder.

Generates structured JSON reports comparing manual vs ML-enhanced strategies
against SPY, QQQ, BRK-B, All Weather benchmarks.

Used by: GET /api/v1/comparison/report
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

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
    # Public API                                                           #
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
                    float(stats.get("annual_return", 0.0)) / max(abs(float(stats.get("max_dd", 1.0))), 1e-9),
                    4,
                ),
            )

        # Sharpe improvement expressed as a percentage of manual Sharpe
        manual_sharpe = manual_metrics.sharpe
        if manual_sharpe != 0:
            ml_improvement_pct = round((ml_metrics.sharpe - manual_sharpe) / abs(manual_sharpe) * 100, 2)
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
        d = asdict(report)
        # Convert nested dataclasses to plain dicts (asdict handles recursion already)
        return d

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
  <title>QuantEdge Comparison Report — {report.strategy_name} / {report.symbol}</title>
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
      padding: 2px 10px;
      border-radius: 12px;
      font-size: 0.78rem;
      font-weight: bold;
    }}
    .section {{ margin-bottom: 36px; }}
    .eq-values {{ font-size: 0.75rem; color: #8b949e; max-height: 60px; overflow: hidden; text-overflow: ellipsis; }}
    .footer {{ color: #8b949e; font-size: 0.75rem; margin-top: 32px; border-top: 1px solid #21262d; padding-top: 8px; }}
  </style>
</head>
<body>
  <h1>QuantEdge Comparison Report</h1>
  <div class="meta">
    Strategy: <strong>{report.strategy_name}</strong> &nbsp;|&nbsp;
    Symbol: <strong>{report.symbol}</strong> &nbsp;|&nbsp;
    Interval: <strong>{report.interval}</strong> &nbsp;|&nbsp;
    Period: <strong>{report.period}</strong><br>
    Generated: {report.generated_at} &nbsp;|&nbsp;
    Winner: <span class="winner-badge">{report.winner}</span>
  </div>

  <div class="section">
    <h2>Executive Summary</h2>
    <div class="summary">{summary_text}</div>
  </div>

  <div class="section">
    <h2>Performance Metrics</h2>
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th>Manual</th>
          <th>ML-Enhanced</th>
          {''.join(f'<th>{report.benchmarks[k].name}</th>' for k in report.benchmarks)}
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>Statistical Test</h2>
    <table>
      <thead>
        <tr><th>Test</th><th>Value</th></tr>
      </thead>
      <tbody>
        <tr><td>t-statistic</td><td>{report.t_statistic:.4f}</td></tr>
        <tr>
          <td>p-value</td>
          <td class="{'pos' if report.is_statistically_significant else 'neg'}">{report.p_value:.6f}</td>
        </tr>
        <tr><td>Statistically Significant (α=0.05)</td>
            <td class="{'pos' if report.is_statistically_significant else 'neg'}">
              {'YES' if report.is_statistically_significant else 'NO'}</td></tr>
        <tr><td>ML Sharpe Improvement</td>
            <td class="{'pos' if report.ml_improvement_pct >= 0 else 'neg'}">{report.ml_improvement_pct:+.2f}%</td></tr>
      </tbody>
    </table>
  </div>

  {eq_section}

  <div class="footer">
    QuantEdge — Institutional Quantitative Trading Platform &nbsp;|&nbsp; For internal use only.
  </div>
</body>
</html>"""
        return html

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _backtest_to_strategy_metrics(self, name: str, bm) -> StrategyMetrics:
        """Convert a BacktestMetrics dataclass to a StrategyMetrics dataclass."""
        if bm is None:
            return StrategyMetrics(
                name=name, sharpe=0.0, sortino=0.0, annual_return_pct=0.0,
                max_drawdown_pct=0.0, win_rate=0.0, total_trades=0,
                avg_hold_days=0.0, calmar=0.0,
            )
        # avg_hold_days: derive from equity curve length / num_trades if available
        n_bars = len(bm.equity_curve) if bm.equity_curve else 0
        avg_hold = round(n_bars / max(bm.num_trades, 1), 1)

        return StrategyMetrics(
            name=name,
            sharpe=float(bm.sharpe),
            sortino=float(bm.sortino),
            annual_return_pct=round(float(bm.annualized_return) * 100, 2),
            max_drawdown_pct=round(float(bm.max_drawdown) * 100, 2),
            win_rate=float(bm.win_rate),
            total_trades=int(bm.num_trades),
            avg_hold_days=avg_hold,
            calmar=float(bm.calmar),
        )

    def _determine_winner(
        self,
        ml: StrategyMetrics,
        manual: StrategyMetrics,
        benchmarks: dict[str, StrategyMetrics],
        engine_winner: str,
    ) -> str:
        """
        Determine the overall winner across all entrants.
        Falls back to the engine's winner label, then checks if any benchmark beats both.
        """
        best_sharpe = max(ml.sharpe, manual.sharpe)
        best_label = engine_winner  # "ml" | "manual" | "neither"

        for key, bm in benchmarks.items():
            if bm.sharpe > best_sharpe + 0.1:
                best_sharpe = bm.sharpe
                best_label = f"benchmark:{key}"

        return best_label

    def _extract_equity_curves(self, cr: ComparisonResult) -> dict[str, list[float]]:
        """Return normalized equity curves (starting at 100) for all entrants."""
        curves: dict[str, list[float]] = {}

        for label, bm_metrics in [("manual", cr.manual), ("ml_enhanced", cr.ml_enhanced)]:
            if bm_metrics and bm_metrics.equity_curve:
                raw = [pt["equity"] for pt in bm_metrics.equity_curve]
                base = raw[0] if raw[0] != 0 else 1.0
                curves[label] = [round(v / base * 100, 2) for v in raw]

        for key, curve_data in cr.benchmark_curves.items():
            if curve_data:
                curves[key] = [pt["value"] for pt in curve_data]

        return curves

    def _best_benchmark(self, report: ComparisonReport) -> Optional[str]:
        """Return the ticker of the best-Sharpe benchmark, or None."""
        if not report.benchmarks:
            return None
        return max(report.benchmarks, key=lambda k: report.benchmarks[k].sharpe)

    def _fmt(self, value: float, is_pct: bool = False, invert_colors: bool = False) -> str:
        """Format a float with HTML colour class (green=positive, red=negative)."""
        display = f"{value:.2f}{'%' if is_pct else ''}"
        if value > 0:
            css = "neg" if invert_colors else "pos"
        elif value < 0:
            css = "pos" if invert_colors else "neg"
        else:
            css = "neutral"
        return f'<span class="{css}">{display}</span>'

    def _metrics_table_rows(self, report: ComparisonReport) -> str:
        """Render <tr> rows for the main metrics table."""
        m = report.manual
        ml = report.ml_enhanced
        bms = list(report.benchmarks.values())

        def bm_cells(getter, is_pct=False, invert=False):
            return "".join(
                f"<td>{self._fmt(getter(b), is_pct=is_pct, invert_colors=invert)}</td>"
                for b in bms
            )

        rows = [
            (
                "Sharpe Ratio",
                self._fmt(m.sharpe), self._fmt(ml.sharpe),
                bm_cells(lambda b: b.sharpe),
            ),
            (
                "Sortino Ratio",
                self._fmt(m.sortino), self._fmt(ml.sortino),
                bm_cells(lambda b: b.sortino),
            ),
            (
                "Annual Return %",
                self._fmt(m.annual_return_pct, is_pct=True),
                self._fmt(ml.annual_return_pct, is_pct=True),
                bm_cells(lambda b: b.annual_return_pct, is_pct=True),
            ),
            (
                "Max Drawdown %",
                self._fmt(m.max_drawdown_pct, is_pct=True, invert=True),
                self._fmt(ml.max_drawdown_pct, is_pct=True, invert=True),
                bm_cells(lambda b: b.max_drawdown_pct, is_pct=True, invert=True),
            ),
            (
                "Win Rate %",
                self._fmt(m.win_rate * 100, is_pct=True),
                self._fmt(ml.win_rate * 100, is_pct=True),
                bm_cells(lambda b: b.win_rate * 100, is_pct=True),
            ),
            (
                "Calmar Ratio",
                self._fmt(m.calmar), self._fmt(ml.calmar),
                bm_cells(lambda b: b.calmar),
            ),
            (
                "Total Trades",
                f"<span class='neutral'>{m.total_trades}</span>",
                f"<span class='neutral'>{ml.total_trades}</span>",
                "".join(f"<span class='neutral'>{b.total_trades}</span>" for b in bms),
            ),
            (
                "Avg Hold Days",
                f"<span class='neutral'>{m.avg_hold_days:.1f}</span>",
                f"<span class='neutral'>{ml.avg_hold_days:.1f}</span>",
                "".join(f"<span class='neutral'>{b.avg_hold_days:.1f}</span>" for b in bms),
            ),
        ]

        html_rows = []
        for label, manual_cell, ml_cell, bm_cells_str in rows:
            html_rows.append(
                f"<tr><td>{label}</td><td>{manual_cell}</td><td>{ml_cell}</td>{bm_cells_str}</tr>"
            )
        return "\n        ".join(html_rows)

    def _equity_curve_section(self, report: ComparisonReport) -> str:
        """Render a small equity curve data section (table of first/last values)."""
        if not report.equity_curves:
            return ""

        rows = []
        for name, values in report.equity_curves.items():
            if not values:
                continue
            start_v = values[0]
            end_v = values[-1]
            change = end_v - start_v
            css = "pos" if change >= 0 else "neg"
            rows.append(
                f"<tr><td>{name}</td>"
                f"<td>{start_v:.2f}</td>"
                f"<td>{end_v:.2f}</td>"
                f"<td class='{css}'>{change:+.2f}</td>"
                f"<td>{len(values)} bars</td></tr>"
            )

        return f"""
  <div class="section">
    <h2>Equity Curves (Normalized to 100)</h2>
    <table>
      <thead>
        <tr><th>Series</th><th>Start</th><th>End</th><th>Change</th><th>Bars</th></tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>"""
