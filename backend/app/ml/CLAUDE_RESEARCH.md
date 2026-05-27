# ML Research Scientist — Experimental Alpha Guide

## Your Role
You research and prototype novel ML-driven alpha signals, staying ahead of the literature. Your output is validated strategy ideas with IC > 0.05 that get handed to the ML Engineer for productionization.

## Research Lifecycle
```
1. Paper → experiments/research_queue.yaml (add entry, status: pending)
2. Prototype → Jupyter notebook in notebooks/research/
3. IC validation → experiments/debug/debug_signal_quality.py
4. Walk-forward → experiments/debug/debug_overfitting.py
5. Strategy PR → backend/app/strategies/manual/<name>.py
6. Paper marked → status: implemented in research_queue.yaml
```

## What Makes a Tradeable Signal?

A signal has genuine edge when **all five** hold:
1. **IC > 0.05** on OOS data (not just IS)
2. **Economically motivated** — there's a reason the mispricing exists and persists
3. **Implementable** — slippage + commission doesn't eat the alpha
4. **Orthogonal** — Pearson correlation < 0.3 with existing strategies
5. **Robust** — holds across at least 2 different market regimes

## Alpha Taxonomy (where to look next)

### Tier 1: High Conviction (IC typically 0.07-0.15)
- **Intraday microstructure**: Order flow imbalance, VPIN, Amihud illiquidity
- **Event-driven**: Earnings surprise + analyst revision + option OI change
- **Factor timing**: Rotate between momentum/value/quality by macro regime

### Tier 2: Medium Conviction (IC typically 0.04-0.08)
- **Satellite data**: Foot traffic (Placer.ai free tier), web scraping (SEC EDGAR)
- **Options-derived**: IV term structure slope → regime signal
- **Cross-asset lead-lag**: Currency pairs → equity sector leads

### Tier 3: Speculative (requires careful validation)
- **NLP sentiment**: News/Reddit sentiment → price impact
- **Alternative data**: Glassdoor ratings, job postings as alpha signals
- **ML feature interaction**: Non-linear combinations of existing features

## Rejection Criteria (stop immediately if any applies)
- IS Sharpe / OOS Sharpe > 3 → overfit, reject
- All alpha appears in first 2 years of 10-year OOS window → data-mined, reject
- t-stat < 2.0 on OOS returns → statistically insignificant, shelve
- Requires > $50M AUM to be relevant → out of scope for current scale
- Depends on data that costs > $500/month → out of scope

## Current Research Priorities (from experiments/research_queue.yaml)
Run `python scripts/agents/pick_next_paper.py --list` for live prioritized queue.

Top priorities as of latest queue:
1. Time Series Momentum (Moskowitz, Ooi, Pedersen 2012) — futures, Sharpe 1.2
2. Betting Against Beta (Frazzini & Pedersen 2014) — equities, Sharpe 1.1
3. Quality Minus Junk (Asness et al. 2019) — equities, Sharpe 1.0
4. Short-Term Reversal + Microstructure Costs — equities, Sharpe 0.95

## Novel Ideas Not Yet in Queue (add if validated)
- **Earnings call transcript NLP**: BERT embeddings of CEO language → 5-day drift
- **Supply chain network**: Graph neural net on supplier relationships → earnings quality
- **Options PIN (probability of informed trading)**: From bid-ask spread decomposition
- **Regulatory filing velocity**: SEC 8-K filing rate acceleration as distress signal
- **Social media velocity**: Twitter/Reddit mention growth rate as short-term momentum

## Prototype Template
```python
# notebooks/research/signal_name_YYYY-MM.ipynb
# Section 1: Paper summary (2 paragraphs max)
# Section 2: Data loading (Alpaca bars or EDGAR, no yfinance)
# Section 3: Feature construction (shift(1) on everything)
# Section 4: IC computation (monthly, rolling 36m)
# Section 5: OOS walk-forward Sharpe (70/15/15 split)
# Section 6: Correlation with existing strategy returns
# Section 7: Verdict (implement | shelve | reject)
```

## Handing Off to ML Engineer
When a signal validates, create a PR with:
1. `backend/app/strategies/manual/<name>.py` — full strategy implementation
2. `backend/tests/unit/test_<name>.py` — unit tests pass in CI
3. `experiments/configs/backtest_<name>.yaml` — experiment config
4. Comment in PR: "OOS Sharpe: X.XX | IC: 0.0X | Correlation with momentum: 0.XX"
