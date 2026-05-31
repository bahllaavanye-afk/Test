# Backtesting Engineer — Module Guide

## Your Role
You ensure every strategy's backtest is rigorous, free of look-ahead bias, and walk-forward validated. Your job is to catch overfitting before capital is at risk.

## Owned Files (safe to modify)
```
backend/app/backtest/
  engine.py           # Vectorised engine: open fills, volume-adaptive slippage
  metrics.py          # Full performance metrics suite
  walk_forward.py     # Rolling train/test windows
  monte_carlo.py      # Bootstrap simulation for robustness
  stress_test.py      # Historical crisis overlay (GFC, COVID, rate-hikes, etc.)
backend/experiments/
  debug/
    debug_feature_leak.py   # Detect look-ahead bias
    debug_overfitting.py    # Train vs val loss plots
    debug_slippage.py       # Realized vs expected fill prices
    debug_signal_quality.py # IC/IR analysis
  configs/*.yaml            # Experiment definitions
  run_experiment.py         # CLI entry point
```

## Do NOT Modify
- Strategy source files (`app/strategies/**/*.py`)
- The risk engine — backtest ignores risk limits by design (they are applied in live)
- DB migration files

## Walk-Forward Protocol (MANDATORY for all strategies)

```
Total data: 100%
├── In-sample:      70%  (parameter fitting)
├── Validation:     15%  (threshold selection)
└── Out-of-sample:  15%  (report this number only)

Window type: anchored (expanding) for factors, rolling for mean-reversion
Step size:   1 month
Min windows: 12 (at least 1 year of OOS)
```

A strategy is only approved for paper trading if OOS Sharpe ≥ 0.7 across all 12+ windows.

## Detecting Look-Ahead Bias (critical checklist)
Every indicator must use `.shift(1)` before use in a signal:
```python
# WRONG — uses today's RSI to trade today
signal = rsi > 70

# RIGHT — uses yesterday's RSI to decide today's trade
signal = rsi.shift(1) > 70
```

Run `python experiments/debug/debug_feature_leak.py --strategy <name>` to verify.

## Standard Metrics Suite (`engine.py`)
| Metric                | Target          | Notes |
|-----------------------|-----------------|-------|
| Sharpe Ratio          | > 1.0 OOS       | rf=5% annualised |
| Sortino Ratio         | > 1.5           | downside dev only |
| Calmar Ratio          | > 0.5           | ann_return / max_dd |
| Omega Ratio           | > 1.5           | gains/losses above rf |
| Ulcer Index           | < 5             | RMS drawdown — penalises prolonged underwater |
| Max Drawdown          | < 20%           | |
| Win Rate              | > 50%           | |
| Profit Factor         | > 1.5           | |
| Expectancy            | > 0.3%/trade    | avg_win×win_rate − avg_loss×loss_rate |
| Annualised Return     | > 15%           | |
| Benchmark Alpha       | > 3% vs SPY     | |

## Engine Execution Realism
The engine models realistic fill assumptions:

```python
# Open-price fills (default)
run_backtest(signals, prices, opens=opens, fill_at_open=True)

# Volume-adaptive slippage (Kyle sqrt model)
# slippage = base * sqrt(trade_size / daily_volume_usd)
run_backtest(signals, prices, volume=volume)
```

This means:
- **Flat % slippage is the floor** — slippage scales up for large trades
- **Fills happen at the open**, not the signal-bar close, so no bar-close lookahead

## Stress Testing (`stress_test.py`)
Overlay any strategy on 7 built-in crisis windows:

```python
from app.backtest.stress_test import run_stress_tests, stress_summary

results = run_stress_tests(signals, prices, opens=opens, volume=volume)
summary = stress_summary(results)
# → {"gfc": {"total_return_pct": -12.4, "max_drawdown_pct": -18.1, ...}, ...}
```

Available scenarios: GFC 2008, Euro Crisis 2011, China Flash 2015,
Vol Spike Feb-2018, COVID Crash 2020, Rate Hikes 2022, SVB Crisis 2023.

REST access: `GET /api/v1/backtests/scenarios`

## Running a Backtest
```bash
# Single strategy
./scripts/launch.sh backtest momentum SPY 1d 2021-01-01 2024-01-01

# Walk-forward
python -m backend.app.backtest.walk_forward --strategy momentum --symbol SPY \
  --start 2018-01-01 --end 2024-01-01 --windows 24

# Monte Carlo robustness
python -m backend.app.backtest.monte_carlo --strategy momentum --simulations 1000
```

## Adding a New Backtest Configuration
1. Create `experiments/configs/<strategy>_<symbol>_<interval>.yaml`
2. Fill in all required fields (see existing configs for schema)
3. Run: `python experiments/run_experiment.py --config <filename>.yaml`
4. Results auto-saved to `experiments/results/<run_id>.json`

## Common Bugs to Watch For
- **Survivorship bias**: only use data from tickers that existed at the *start* of the period
- **Transaction costs**: always apply 5bps commission + estimated slippage per trade
- **Overnight gap risk**: for intraday strategies, close positions at 15:55 ET
- **Look-ahead bias**: use `.shift(1)` on EVERY feature before generating a signal
- **Overfitting**: if OOS Sharpe < 0.5 * IS Sharpe, the strategy is overfit — reject it
