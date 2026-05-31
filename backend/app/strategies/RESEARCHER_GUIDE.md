# QuantEdge Researcher Guide

A field manual for AI agents (QuantEdge AI, GPT, and successors) and human quants who are tasked with continuously discovering, vetting, and implementing new trading strategies for the QuantEdge platform. Read this guide in full before writing a single line of strategy code. Re-read sections 2 and 3 before every PR.

The mandate is simple: find real, durable, capacity-adequate edge — and reject everything else, no matter how clever it looks in-sample.

---

## 1. Sources for Non-Obvious Alpha

The best alpha is almost never in mainstream finance media. It hides in academic working papers, practitioner research notes, conference proceedings, and the occasional Twitter thread from a respected practitioner. Below is the canonical research surface you should crawl on a rotating cadence.

**Academic primary sources**
- SSRN Financial Economics Network — https://papers.ssrn.com/sol3/JELJOUR_Results.cfm?form_name=journalBrowse&journal_id=203
- Journal of Finance — https://onlinelibrary.wiley.com/journal/15406261
- Journal of Financial Economics — https://www.sciencedirect.com/journal/journal-of-financial-economics
- Review of Financial Studies — https://academic.oup.com/rfs
- arXiv q-fin section (newest preprints, often months ahead of journals) — https://arxiv.org/list/q-fin/recent

**Practitioner research libraries (peer-reviewed quality, written by people who actually run money)**
- Robeco Quantitative Investing — https://www.robeco.com/en-int/insights/category/quantitative-investing
- AQR Capital Research Library — https://www.aqr.com/Insights/Research
- Two Sigma Insights — https://www.twosigma.com/research/
- Man AHL Academic — https://www.man.com/insights
- Quantpedia free strategy database — https://quantpedia.com/strategies/

**Signal channels (lower signal-to-noise, but occasionally first to spot regime shifts)**
- Cliff Asness (AQR) on Twitter/X — https://twitter.com/CliffordAsness
- Adam Butler / ReSolve Asset Management — https://investresolve.com/blog/
- Corey Hoffstein (Newfound Research) — https://blog.thinknewfound.com/
- Ernie Chan blog — https://epchan.blogspot.com/

Rotate at least one new source per week. If two independent sources converge on the same finding within a 90-day window, prioritise it.

---

## 2. How to Filter for REAL Edge

Most "alpha" in published research is fake. It is either data-mined, already arbitraged away, or only profitable on paper because the author ignored transaction costs. Apply this filter aggressively — if a paper fails even one check, drop it.

1. **Out-of-sample horizon ≥ 10 years.** The paper must document the strategy working on data the authors did not have when they designed it. In-sample-only studies are worthless.
2. **Net Sharpe ratio ≥ 0.7 after costs.** Use 35bps round-trip for liquid US equities, 15bps for futures, 5bps for spot FX, 50bps for crypto majors. If the paper omits costs, recompute or discard.
3. **Author independence.** Reject anything authored or co-authored by someone selling a paid product, fund, or newsletter built on the same idea. Academic authors with tenure are the gold standard.
4. **Publication maturity ≥ 3 years.** Fresh papers haven't faced an out-of-sample test. A strategy that still works 3+ years after publication is far more likely to be real edge versus a data-mining artifact.
5. **Citation robustness.** A finding cited by 20+ independent papers across multiple universities is robust. A finding cited only by the original author's follow-ups is suspect.

Bonus filters: does the strategy make economic sense (risk premium, behavioral bias, structural friction)? Does it work in multiple markets/countries? Does it have an obvious capacity ceiling? If you cannot articulate WHY the edge exists in one sentence, do not implement it.

---

## 3. Implementation Checklist

Every new strategy PR must check every box. No exceptions. Reviewers should reject any PR with an unchecked item.

- [ ] **Academic citation in docstring.** Format: `Author(s), Year. "Title." Journal, vol(issue), pages. URL.`
- [ ] **Exact mathematical formula in comments.** A reader should be able to reproduce the signal without reading the paper.
- [ ] **No look-ahead bias.** Every rolling/EMA/zscore indicator must be followed by `.shift(1)` before being used as a signal. Confirm via unit test.
- [ ] **Transaction cost reality check.** Gross alpha must exceed `2 × annual_turnover × 35bps`. If turnover is 4x/year, gross alpha must exceed 280bps.
- [ ] **Correlation check.** `corr(new_strategy_returns, existing_portfolio_returns) < 0.6` over a 3-year sample. Anything tighter is a duplicate exposure, not a new strategy.
- [ ] **Walk-forward Sharpe ≥ 0.5 on 2022-2024 out-of-sample.** This is the harshest recent regime (zero-rate unwind, three rate-shock years, two crypto crashes). Survival here matters.
- [ ] **Maximum drawdown < 25%** in the full backtest window. If it breaches 25%, add a vol-target overlay or risk-parity sizing before resubmitting.
- [ ] Unit tests in `backend/tests/strategies/test_<strategy>.py` covering: signal generation on a known synthetic series, no-NaN output, no look-ahead.
- [ ] CLAUDE.md updated in the strategy's parent module if the strategy introduces a new dependency or data source.

---

## 4. Common Failure Modes

These are the recurring ways "great" strategies turn out to be garbage in production. Internalize them.

- **Survivorship bias.** Backtesting on today's S&P 500 constituents ignores companies that were delisted, went bankrupt, or were acquired. The classic example: a "buy low P/B stocks" strategy looks great on current-constituent data but loses badly on point-in-time data because the lowest-P/B stocks are often the ones that went to zero. Always use point-in-time universe data (CRSP, Compustat with PIT flag, or Norgate).
- **Look-ahead bias.** Using `df['close'].rolling(20).mean()` as today's signal uses today's close to make today's trade. Fix: `df['close'].rolling(20).mean().shift(1)`. This single bug has destroyed more "great" backtests than any other.
- **Overfitting.** A strategy with 8 free parameters tuned on 5 years of data is curve-fitting noise. Rule of thumb: at least 100 trades per free parameter, and any parameter should be optimal across a wide plateau (±20% perturbation should not collapse Sharpe).
- **Regime brittleness.** A strategy that only works in low-vol bull markets will blow up. Always backtest through 2008-09 (GFC), March 2020 (COVID crash), 2022 (rate shock). If Sharpe goes negative in any of these, it's not robust.
- **Liquidity ignorance.** A strategy that assumes you can trade $10M of a stock whose average daily volume is $5M is fantasy. Cap position size at 5% of 20-day median ADV per name. For crypto, cap at 1% of 24h volume on the venue.
- **Tail risk hidden in average.** Mean return of +20% looks great until you notice skewness of -2.5 and that 80% of the return came from being short vol in 2017. Always report skew, kurtosis, worst-day, and Sortino alongside Sharpe.
- **Cost amnesia.** Slippage, financing, borrow fees, exchange fees, FX conversion, taxes. Equity short-sellers pay borrow that can exceed 10% APR on hard-to-borrow names. Crypto perp funding can flip from +30% to -30% APR in days.

---

## 5. Categories of Edge (Taxonomy)

Knowing where a strategy fits tells you its decay horizon, capacity ceiling, and what infrastructure it needs.

| Category | Decay | Capacity | Examples |
|---|---|---|---|
| Microstructure | Hours | <$10M | Order flow imbalance, queue position, latency arb |
| Cross-sectional | 1-12 mo | $1B+ | Momentum, value, low-vol, quality, profitability |
| Time-series | Days-weeks | $100M | Trend-following, mean reversion, breakout |
| Volatility | Mins-days | $100M | VIX term structure, dispersion, vol-of-vol, VRP |
| Behavioral | 1-6 mo | $500M | PEAD, accruals, attention anomaly, lottery stocks |
| Calendar | 1-30 d | $50M | Turn-of-month, FOMC drift, end-of-quarter pension rebalance |

QuantEdge currently overweights cross-sectional and time-series (the 70/30 risk budget). We are explicitly looking for more volatility and behavioral edges to diversify.

---

## 6. The 5-Step Research Process

Follow this sequence on every new strategy. Skipping steps is the fastest way to ship a losing strategy.

1. **Read three papers on the topic.** Read the original formulation, one critical follow-up, and one recent extension. Understand the economic mechanism, not just the formula.
2. **Implement in `backend/app/strategies/manual/<strategy>.py`.** Inherit from `BaseStrategy`. Single file, no clever multi-module abstractions on a first pass.
3. **Backtest 2010-2020 in-sample.** Use `./scripts/launch.sh backtest <strategy> <ticker> <bar> 2010-01-01 2020-01-01`. Tune parameters on this window only.
4. **Walk-forward validate on 2020-2024.** Use the same script with the post-2020 window. Sharpe must stay above 0.5 with NO further parameter changes. If you re-tune on the OOS window, the OOS test is contaminated and you've defeated the purpose.
5. **Open PR** with academic citation, performance table (Sharpe, Sortino, MaxDD, Calmar, skew, kurtosis, turnover, hit rate), and correlation matrix vs existing portfolio.

---

## 7. Strategy Graveyard

These are alphas that have been demonstrated to be decayed, crowded, or unimplementable at scale. Do NOT re-implement them as standalone strategies. They may still appear as features inside a larger ensemble — that's fine — but not as their own strategy.

- **Pure momentum (1-12 month lookback) on US large caps.** Crowded since 2003, alpha effectively zero by 2010 after costs. Still works in EM and small caps.
- **Calendar anomalies (sell-in-May, January effect).** Statistical significance evaporated post-2000 once they were widely publicised.
- **Post-earnings-announcement-drift (PEAD) WITHOUT a SUE or analyst-revision filter.** Plain PEAD was arbitraged by 2015. The conditional version (high SUE + positive guidance) still has a pulse.
- **High-frequency mean reversion on liquid ETFs (SPY, QQQ).** Killed by HFT market makers by 2012. Sub-second microstructure on illiquid names still works but is outside our latency budget.
- **Naive variance risk premium (short VIX futures).** Worked 2004-2017, blew up February 2018 (XIV / SVXY). Needs a vol-of-vol filter and tail hedge to be viable post-2018.
- **Low-beta / BAB on US equities at the index level.** Discovered in 2014, alpha mostly gone by 2020 as factor ETFs absorbed it.

---

## 8. Reference: Cutting-Edge Papers to Read (2022-2024)

Prioritize these in the research queue. URLs verified at time of writing — if a URL 404s, search SSRN/arXiv by title.

1. **"Trading Volume Alpha"** — Lou, Polk, Skouras (2024). SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3637061
2. **"Man vs. Machine Learning: The Term Structure of Earnings Expectations and Conditional Biases"** — van Binsbergen, Han, Lopez-Lira (2023). Review of Financial Studies. https://academic.oup.com/rfs/article/36/6/2361/6957931
3. **"Cross-Sectional Factor Dynamics and Momentum Returns"** — Ehsani, Linnainmaa (2022). Journal of Financial Economics. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3300521
4. **"Is There a Replication Crisis in Finance?"** — Jensen, Kelly, Pedersen (2023). Journal of Finance. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3774514
5. **"Carry"** — Koijen, Moskowitz, Pedersen, Vrugt (updated 2023). Journal of Financial Economics. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2298565
6. **"Empirical Asset Pricing via Machine Learning"** — Gu, Kelly, Xiu (2020, still the most-cited ML-finance paper). Review of Financial Studies. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3159577
7. **"The Volatility Risk Premium Revisited"** — Andersen, Bondarenko, Todorov (2022). arXiv. https://arxiv.org/abs/2207.05933
8. **"Intraday Momentum and the FOMC Cycle"** — Lucca, Moench (updated 2023). SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2210331
9. **"Common Risk Factors in Cryptocurrency"** — Liu, Tsyvinski, Wu (2022). Journal of Finance. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3169130
10. **"Order Flow and Prices"** — Brogaard, Hendershott, Riordan (2023). SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1361347

When you finish a paper, log it in `experiments/research_log.jsonl` with verdict (implement / shelve / reject) and one-paragraph reasoning. Future agents will read your notes.

---

## Final Word

You are not paid (or, in your case, optimised) to find clever ideas. You are paid to find ideas that make money out-of-sample, after costs, at our capacity, without blowing up. Most of your work will be rejecting strategies. That is the job. Be ruthless on the filter; be generous on the search.
