# QuantEdge Strategy Engineer Guide

## Welcome

This guide is for engineers joining the QuantEdge strategy team. Your job is to find, implement, validate, and operate quantitative trading strategies that generate genuine, durable alpha. The bar is high: every strategy must be grounded in peer-reviewed research, survive realistic transaction cost assumptions, and pass a walk-forward out-of-sample test before going live. Read this guide completely before writing a single line of strategy code.

---

## 1. Philosophy: Institutional vs. Retail Strategies

The most important filter you can apply to any strategy idea is: *"Could a well-read retail trader implement this from a blog post?"* If yes, the edge is probably gone. Institutional-grade strategies share four properties retail strategies almost never have:

**Academic backing.** Every strategy must cite a peer-reviewed paper documenting the anomaly's source. This is not a formality — it forces you to understand *why* the edge exists (and therefore *when* it will stop working). The paper must show a documented Sharpe ratio of at least 0.5 net of transaction costs, or gross Sharpe of 1.0 with explicit cost estimates. Acceptable journals: Journal of Finance, Journal of Financial Economics, Review of Financial Studies, Journal of Portfolio Management, Management Science. Blog posts, Seeking Alpha articles, and YouTube tutorials do not qualify.

**Out-of-sample validation mandate.** Any paper can be overfit. Our requirement is a minimum 2-year out-of-sample (OOS) test on data the strategy developer never touched during model selection. "OOS" means the developer made no decisions — no parameter choices, no universe filters — based on that period. If your strategy was published in 2020, your OOS period starts in 2022 at the earliest.

**Risk-adjusted metrics first.** We care about Sharpe ratio, not raw return. A strategy returning 40% per year with 80% volatility is useless. Minimum bar: Sharpe ≥ 0.5 on the OOS period *before* considering ML enhancement. If you can only achieve Sharpe 0.3, go back and find a better strategy. Calmar ratio (annual return / max drawdown) must exceed 0.5 as well.

**Alpha vs. beta distinction.** You must quantify how much of a strategy's return comes from genuine alpha (skill-based, market-neutral) versus beta (passive market exposure). Run a regression of strategy returns on SPY, QQQ, TLT, VIX, and sector ETFs. Anything with R² > 0.4 is probably just systematic beta dressed as alpha. We want alpha: the intercept (annualized) must be statistically significant (t-stat > 2.0) at the strategy level.

---

## 2. Signal Quality Framework

Before building a full strategy, evaluate the quality of the underlying signal with these metrics:

### Information Coefficient (IC)

IC is the Spearman rank correlation between your signal at time T and forward returns at time T+N. It measures raw predictive power.

```
IC_t = rank_corr(signal_t, return_{t+N})
Mean IC = average IC over all periods
IC t-stat = Mean IC × sqrt(T) / std(IC)
```

- IC > 0.05 is worth pursuing (signals with IC > 0.10 are rare and valuable)
- IC t-stat > 2.0 indicates statistical significance
- Mean IC < 0.02 is noise — abandon the signal

### Information Ratio (IR)

IR combines signal quality with breadth (number of independent bets):

```
IR = IC × sqrt(Breadth)
```

where Breadth = number of independent signals generated per year. A strategy trading 50 stocks daily has Breadth ≈ 50 × 252 = 12,600; a monthly rebalanced 20-stock portfolio has Breadth ≈ 240.

The Fundamental Law of Active Management (Grinold 1989) says Sharpe ≈ IR. You need IR > 0.5 as a minimum viable signal.

### Decay Analysis

Every signal has a half-life — the horizon at which IC drops to 50% of peak. Measure IC at horizons: 1 bar, 5 bars, 20 bars, 60 bars, 252 bars. Plot the decay curve.

- Microstructure signals: half-life < 5 minutes
- Short-term behavioral (momentum, PEAD): half-life 1-90 days
- Fundamental factors (accruals, value): half-life 6-18 months

Your execution infrastructure must match the signal decay. A signal with a 5-minute half-life cannot survive a 30-minute execution window.

### Turnover Cost Analysis

Gross alpha means nothing after costs. Compute:

```
Net alpha = Gross alpha − Turnover × Round-trip cost
```

For US equities, use round-trip cost = 0.05% (Alpaca zero-commission + spread). For a daily-rebalanced strategy with 20% daily turnover:

```
Cost drag = 0.20 × 0.05% × 252 = 2.52% per year
```

The signal must generate > 2× expected cost drag in gross alpha to be viable. Strategies with very high IC but > 100% monthly turnover rarely survive cost analysis.

---

## 3. Walk-Forward Validation Protocol

Overfitting is the strategy team's primary enemy. The walk-forward protocol is mandatory and non-negotiable:

### Data Splits

```
Full data: 2010 – present (minimum 10 years of daily data)
├── Train:    2010 – 2018 (70%) — fit parameters, select universe, tune thresholds
├── Validate: 2018 – 2021 (15%) — hyperparameter selection ONLY (Sharpe ≥ 0.4)
└── Test:     2022 – present (15%) — single evaluation, NEVER touched during development
```

The test set is sacred. You look at it **once**, after all decisions are frozen. If you look at test-set results and adjust parameters, you have contaminated the test set and must wait for fresh OOS data to accumulate before re-evaluating.

### Re-validation Requirement

After any modification to a live strategy — signal logic, universe, parameters, risk limits — the full walk-forward must be re-run. The strategy is placed back on 2-week paper trading before live reactivation. No exceptions. Even a "minor tweak" can induce overfitting.

### Walk-Forward Windows (for adaptive strategies)

For strategies with rolling re-estimation (e.g., Kalman filter pairs), use expanding or rolling walk-forward windows:

```
Window 1: train 2010-2014, test 2015
Window 2: train 2010-2015, test 2016
...
Window N: train 2010-2021, test 2022
```

Report the distribution of Sharpe ratios across windows. If any window shows Sharpe < -0.5, investigate before proceeding.

---

## 4. Alpha Taxonomy

Understand which category your strategy belongs to. Each category has different implementation requirements, decay speed, and capacity limits.

### Microstructure Alpha
*Decay: minutes to hours | Capacity: low (<$1M per strategy)*

Examples: order flow imbalance, bid-ask bounce, inventory effects, short-term price impact reversal. These signals require 1-minute or tick data and near-zero latency execution. They are difficult to scale because alpha decays as position size grows. OFI strategies fall here.

### Short-Term Behavioral Alpha
*Decay: days to months | Capacity: medium ($10M–$100M)*

Examples: post-earnings drift (PEAD), short-term momentum (1-12 months), earnings surprise reactions. These signals derive from investor under-reaction to public information. They require event data (earnings dates, analyst revisions) and daily rebalancing. Capacity is limited by market impact.

### Long-Term Fundamental Alpha
*Decay: months to years | Capacity: high (>$100M)*

Examples: accruals (Sloan), value (P/B, P/E), profitability (ROE, gross margin), investment factor. These signals derive from fundamental valuation or quality metrics. They are slow, high-capacity, and relatively immune to crowding because they require patience that most institutions cannot sustain.

### Cross-Asset / Macro Alpha
*Decay: weeks to months | Capacity: very high*

Examples: carry (currency, bond, equity), momentum across asset classes, crisis alpha (managed futures). These strategies exploit structural risk premia — investors pay a persistent premium for insurance or liquidity that patient capital can harvest. Carry and cross-asset momentum fall here.

---

## 5. Failure Modes (What Kills Most Strategies)

### Overfitting: Too Many Parameters

The most common cause of strategy failure. A strategy with 10 free parameters fitted on 252 data points effectively has no predictive power. Follow these rules:
- Parameters ≤ 5 for daily strategies with < 5 years of training data
- Every parameter must have economic intuition — no grid-searched magic numbers
- Use information criteria (AIC, BIC) to penalize model complexity
- If your in-sample Sharpe is 2.0 but OOS Sharpe is 0.2, you are overfit

### Transaction Costs: The Silent Killer

Most academic papers use frictionless assumptions. In live trading, a strategy generating 8% gross alpha with 150% annual turnover earns only 8% − (1.50 × 0.05% × 252) = −10.9% net. Always compute net alpha first. Reject any strategy where gross alpha < 2× cost drag.

### Correlation: Portfolio-Level Thinking

A strategy may have Sharpe 0.8 in isolation but add zero value if it is 90% correlated with existing strategies. The marginal contribution to portfolio Sharpe is:

```
ΔSharpe ≈ IC_new_with_portfolio × sqrt(1 - ρ²)
```

A new strategy must have correlation < 0.60 with the existing strategy portfolio, or it must have Sharpe > 1.5 to justify the correlated exposure.

### Regime Change: Factor Crowding and Structural Breaks

A strategy that worked in 2020-2022 (zero interest rates, retail trading surge, meme stocks) may be entirely explained by that regime. Check:
- Does the strategy's IC decay after 2022 (rate hike cycle)?
- Has the signal become crowded? (Sharpe declining over rolling 1-year windows)
- Is there a macro regime explanation for when the strategy works vs. fails?

Acceptable strategies should work across multiple regimes (pre-GFC, post-GFC, COVID, post-COVID normalization). If it only works in one regime, it is a beta bet, not alpha.

---

## 6. How to Add a Strategy: Step-by-Step

### Step 1: Find the Academic Foundation

Search Google Scholar, SSRN, or AQR's white papers for the anomaly you want to implement. Requirements:
- Published in peer-reviewed journal (or NBER/SSRN working paper by known academics)
- Documents Sharpe ≥ 0.5 or gross return > 10% with transaction cost analysis
- Has been replicated by independent researchers (search for replication papers)
- Is not based solely on US data from a single short period

Write a 1-paragraph "theory of the trade" explaining *why* the anomaly exists and *why* it has not been fully arbitraged away.

### Step 2: Implement in Python

Create the file in `backend/app/strategies/manual/my_strategy.py`. Follow these rules:

```python
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

class MyStrategy(AbstractStrategy):
    name = "my_strategy"
    display_name = "My Strategy"
    market_type = "equity"       # equity | crypto | polymarket
    strategy_type = "manual"
    risk_bucket = "directional"  # directional | arbitrage
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.60

    def __init__(self, params: dict | None = None):
        super().__init__(params)   # MANDATORY — always call super()

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        # ...compute signal...
        return Signal(
            symbol=symbol,
            side="buy",                          # 'buy' | 'sell'
            confidence=0.75,                     # 0.0 to 1.0
            strategy_name=self.name,             # REQUIRED
            strategy_type=self.strategy_type,    # REQUIRED
            risk_bucket=self.risk_bucket,        # REQUIRED
            metadata={"reason": "..."},
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        # ALL indicators must use .shift(1) to prevent lookahead bias
        signal = compute_signal(df).shift(1).fillna(False)
        return BacktestSignals(entries=signal, exits=~signal)
```

Key rules:
- `analyze()` is async — use `httpx.AsyncClient` for all HTTP calls
- `backtest_signals()` is synchronous and pure — no I/O, no side effects
- Every signal indicator in `backtest_signals()` must be `.shift(1)` before use
- `Signal` objects must include `strategy_name`, `strategy_type`, `risk_bucket`
- `BacktestSignals` must have both `entries` and `exits` as boolean pd.Series

### Step 3: Backtest on 10+ Years of Data

```bash
./scripts/backtest.sh my_strategy SPY 1d 2012-01-01 2024-01-01
```

Review the output for:
- Sharpe ratio (must be ≥ 0.5)
- Max drawdown (must be < 30%)
- Calmar ratio (must be ≥ 0.5)
- Number of trades (must be ≥ 30 for statistical significance)
- Win rate (context-dependent; must exceed cost hurdle)

If these metrics pass on the full period, proceed to walk-forward analysis.

### Step 4: Out-of-Sample Validation

Run exclusively on the 2022-2024 holdout period (never touched before):

```bash
./scripts/backtest.sh my_strategy SPY 1d 2022-01-01 2024-01-01
```

Minimum bar: **Sharpe ≥ 0.3 on OOS period**. Below this, the in-sample results are likely overfit and the strategy should not proceed to paper trading.

### Step 5: ML Enhancement Check

If OOS Sharpe ≥ 0.3, check whether ML adds value:

```bash
./scripts/compare.sh my_strategy SPY
```

The ML version should show ≥ 10% relative Sharpe improvement. If the improvement is < 10%, the manual strategy is preferred (simpler, more interpretable, less prone to ML overfitting).

ML enhancements that work well:
- Feature selection (remove noisy signal variants)
- Regime detection (suppress signal in unfavorable regimes)
- Position sizing (Kelly fraction optimization)
- Signal combination (ensemble of related signals)

ML enhancements to avoid:
- End-to-end price prediction (too many parameters, unstable)
- Deep learning on OHLCV alone (insufficient alpha, overfit)
- Reinforcement learning without strong priors (sample inefficient)

### Step 6: Register and Submit

Register in `backend/app/strategies/__init__.py`:

```python
from app.strategies.manual.my_strategy import MyStrategy
STRATEGY_REGISTRY["my_strategy"] = MyStrategy
```

Submit a pull request with:
- Strategy implementation file (`manual/my_strategy.py`)
- Full backtest results in `experiments/results/my_strategy_backtest.json`
- OOS results in `experiments/results/my_strategy_oos.json`
- A 1-paragraph description in the PR body covering: academic basis, signal mechanics, Sharpe (in-sample and OOS), risk considerations

The PR will be reviewed by the head of quantitative research. Expect questions about parameter sensitivity, regime behavior, and capacity limits.

---

## 7. Golden Rules

1. **Never look at the test set twice.** Once you see those results, you are anchored. The test set is sacred.
2. **Costs kill.** Every strategy review starts with a transaction cost analysis. If you haven't done it, the review will end immediately.
3. **Simple beats complex, all else equal.** A 3-parameter strategy that survives 10 years OOS beats a 20-parameter neural network any day.
4. **Breadth beats IC.** A signal with IC = 0.04 applied to 500 stocks daily beats a signal with IC = 0.12 applied to 10 stocks monthly. Increase breadth first.
5. **Understand what you own.** If you cannot explain in two sentences why your signal predicts future returns, you do not understand it well enough to trade it.

---

*For questions, contact the quantitative research team or open a discussion thread in the `#strategy-development` Slack channel.*
