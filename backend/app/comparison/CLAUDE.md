# Quantitative Analyst — Comparison & Benchmarking Guide

## Your Role
You prove that ML-enhanced strategies outperform their manual counterparts — and benchmark everything against SPY, BRK.B, QQQ, and Ray Dalio's All Weather portfolio. Your output is the investor pitch.

## Owned Files (safe to modify)
```
backend/app/comparison/
  engine.py          # Runs manual vs ML side-by-side on same data
  benchmarks.py      # Downloads SPY, QQQ, BRK.B, GLD, TLT for All Weather
  report_builder.py  # Builds investor-facing JSON/HTML report

backend/app/api/v1/comparison.py   # REST endpoint: POST /comparison/run
```

## Do NOT Modify
- Strategy files — comparison engine calls `backtest_signals()` without modifying strategies
- `backend/app/risk/` — risk engine is bypassed during backtest (applied in live only)

## Comparison Engine Flow
```python
ComparisonResult = await engine.run_comparison(
    strategy_name="momentum",
    symbol="SPY",
    interval="1d",
    start_date=date(2021, 1, 1),
    end_date=date(2024, 1, 1),
)

# Result contains:
{
  "manual":   {"sharpe": 0.82, "return": 0.31, "max_dd": -0.18, "equity_curve": [...]},
  "ml":       {"sharpe": 1.24, "return": 0.49, "max_dd": -0.12, "equity_curve": [...]},
  "spy":      {"sharpe": 0.47, "return": 0.14, "max_dd": -0.34},
  "all_weather": {"sharpe": 0.67, "return": 0.11, "max_dd": -0.20},
  "ml_improvement_pct": 51.2,
  "t_statistic": 2.31,
  "p_value": 0.021,
  "is_significant": true,
  "winner": "ml"
}
```

## Statistical Significance Rules
- Use Welch's t-test on rolling 12-month excess returns (not raw returns)
- Threshold: p < 0.05 → label as statistically significant
- Bootstrap (1000 resample) as secondary confirmation
- If significant but Sharpe improvement < 10%: mark as "marginal"
- Report both tests; never claim significance from only one

## Benchmark Definitions
```python
# benchmarks.py — fetched via Alpaca historical bars (not yfinance)
BENCHMARKS = {
    "SPY":   "SPDR S&P 500 ETF",
    "QQQ":   "Invesco NASDAQ-100 ETF",
    "BRK-B": "Berkshire Hathaway B",
    "GLD":   "SPDR Gold Shares",
}

ALL_WEATHER_WEIGHTS = {
    "TLT": 0.40,  # Long-term bonds
    "IEF": 0.15,  # Intermediate bonds
    "VTI": 0.30,  # Stocks
    "GLD": 0.075, # Gold
    "DJP": 0.075, # Commodities (use PDBC if DJP not available on Alpaca)
}
```

## Investor Report Format (for pitch decks)
The `report_builder.py` produces a JSON that the frontend renders as:
1. **Hero KPIs**: OOS Sharpe, Annual Return, Max Drawdown vs all benchmarks
2. **Equity curve chart**: all strategies + benchmarks on same axis (log scale)
3. **Monthly returns heatmap**: colour-coded calendar grid
4. **Statistical significance badge**: green "Significant at 95% confidence" or grey
5. **ML improvement table**: per-metric delta (Sharpe, Win Rate, Drawdown)

## Adding a New Benchmark
1. Add ticker to `BENCHMARKS` dict in `benchmarks.py`
2. Add a `_fetch_<name>()` method that calls Alpaca `get_bars()`
3. Update `report_builder.py` to include it in the output JSON

## Running a Comparison
```bash
./scripts/compare.sh momentum SPY
# → outputs report to experiments/results/comparison_<id>.json
# → accessible via GET /api/v1/comparison/<id>
```
