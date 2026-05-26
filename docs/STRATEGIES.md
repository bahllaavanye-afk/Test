# Trading Strategies Reference

Every strategy has a manual version (pure indicators) and an ML-enhanced version. The ML version uses the manual signal logic plus an ML classifier filter that must agree.

## Adding a New Strategy
See [`backend/app/strategies/CLAUDE.md`](../backend/app/strategies/CLAUDE.md).

## Strategy Catalog

### Momentum (`momentum.py`)
- **Logic**: Jegadeesh-Titman (1993) 12-1 month total return ranking. Long top-decile, monthly rebalance.
- **Entry**: 12-month return > rolling threshold
- **Exit**: monthly rebalance
- **Risk Bucket**: directional
- **Best on**: SPY, QQQ, large-cap equities
- **Avoid on**: short-time-horizon (<3mo) or mean-reverting periods

### Mean Reversion (`mean_reversion.py`)
- **Logic**: Bollinger Band (20, 2) + RSI(14) confirmation
- **Entry**: Close < lower BB AND RSI < 30 (oversold bounce)
- **Exit**: Close crosses BB middle
- **Risk Bucket**: directional
- **Best on**: range-bound stocks, hours/days holding

### RSI + MACD (`rsi_macd.py`)
- **Logic**: Classic dual-indicator combo. RSI for overbought/oversold, MACD for trend confirmation.
- **Entry Long**: RSI < 30 AND MACD line crosses above signal
- **Entry Short**: RSI > 70 AND MACD line crosses below signal
- **Risk Bucket**: directional

### Breakout (`breakout.py`)
- **Logic**: Close > 252-day high AND volume > 1.5x 20-day average AND ATR clearance
- **Risk Bucket**: directional
- **Best on**: trending stocks during earnings season

### Supertrend (`supertrend.py`)
- **Logic**: ATR-based trend follower. Flips long/short when price crosses ATR multiplier band.
- **Risk Bucket**: directional
- **Sharpe Target**: 0.6-0.9

### Low Volatility (`low_volatility.py`)
- **Logic**: Baker et al. (2011) — long bottom 30th percentile rolling 252-day vol, with EMA50 uptrend filter.
- **Risk Bucket**: directional
- **Best on**: equities, monthly rebalance

### Pairs Trading (`pairs_trading.py`)
- **Logic**: Engle-Granger cointegration test. Long the underperformer, short the outperformer, exit when spread reverts.
- **Entry**: spread z-score > 2.0
- **Exit**: spread z-score < 0.5
- **Stop**: spread z-score > 4.0 (cointegration broken)
- **Risk Bucket**: arbitrage
- **Sharpe Target**: 1.5-2.5

### Triangular Arbitrage (`triangular_arb.py`)
- **Logic**: BTC → ETH → USDT → BTC profitability after Binance fees (0.075% × 3 = 0.225%)
- **Entry**: profit > 0.15% net of fees
- **Risk Bucket**: arbitrage
- **Sharpe Target**: >2.0
- **Execution**: must be near-simultaneous on all 3 pairs

### Polymarket Binary Arb (`poly_binary_arb.py`)
- **Logic**: When YES + NO < $0.97, buying both is risk-free (gas costs included)
- **Risk Bucket**: arbitrage (true arb)
- **Capital cap**: 5% per opportunity

## ML-Enhanced Strategies

### `ml_momentum` (`ml_momentum.py`)
- Manual momentum signal + LSTM probability > 0.6 to confirm
- Expected: +20-40% Sharpe vs manual
- Model: 60-bar BiLSTM with attention

### `ml_mean_reversion` (`ml_mean_reversion.py`)
- Manual mean reversion signal + XGBoost probability > 0.65
- Expected: -30% false signals
- Model: XGBoost with 100-trial Optuna HPO

### `ml_breakout` (`ml_breakout.py`)
- Manual breakout signal + ensemble (LSTM + XGBoost + Lorentzian) probability > 0.65
- Expected: +25-35% win rate

### `lorentzian_knn` (`lorentzian_knn.py`)
- Pure Lorentzian Classification (jdehorty's TV indicator port)
- Lorentzian distance more robust to outliers than Euclidean
- Features: RSI(14), CCI(20), ADX(20), EMA fast/slow deltas
- k=8 nearest neighbors, lookback=2000 bars

### `ensemble` (`ensemble.py`)
- Weighted average of LSTM + XGBoost + Lorentzian + TFT
- Weights optimized via Optuna on validation set
- Confidence = abs(prob - 0.5) * 2; only trade if confidence > 0.65

## Backtesting Parameters

Default backtest settings (`backend/app/backtest/engine.py`):
- **Commission**: 0.10% per trade (Alpaca = 0% but conservative default)
- **Slippage**: 5 bps assumed (LimitFirst typically achieves better)
- **Risk-free rate**: 5% annualized (daily = 5%/252)
- **Initial equity**: $100,000

## Walk-Forward Validation (Mandatory)

All strategies must pass walk-forward before live activation:
- Train window: 2 years
- Test window: 6 months
- Roll forward by 6 months each iteration
- Average Sharpe across windows must exceed 1.0

```bash
./scripts/launch.sh backtest momentum SPY 1d 2018-01-01 2024-01-01
```

## Statistical Significance

`StrategyComparisonEngine` runs Welch's t-test on daily returns between manual and ML versions. Requires p < 0.05 to declare an "ML wins" verdict.
