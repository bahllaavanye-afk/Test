# Competitor Analytics Depth Report — 2025/2026

_Research date: June 2026. Sources: Bloomberg Professional, QuantConnect docs, IBKR guides, TradeStation help, Two Sigma Venn, The TRADE Algo Survey 2025, arxiv papers._

---

## Executive Summary

- **Institutional TCA has converged on five core metrics** — Implementation Shortfall (arrival-price), VWAP/TWAP deviation, fill rate, dispersion across venues, and market impact — but only Bloomberg BTCA and dedicated TCA vendors (Tradeweb, S&P) expose all five natively; retail-facing platforms expose one or two at most.
- **PnL attribution depth is the sharpest dividing line**: Bloomberg PORT and Two Sigma Venn provide factor-level + sector-level + benchmark decomposition (alpha/beta split); QuantConnect, TradeStation, and Alpaca stop at portfolio-level Sharpe/drawdown with no live factor attribution.
- **Self-healing trading systems are moving from academic prototype to production**: Google DeepMind's AlphaEvolve (June 2025) and the MadEvolve framework (May 2025) demonstrate LLM-driven code mutation loops that measurably improve strategy Sharpe ratios across generations — a capability no commercial retail platform yet offers.

---

## Platform-by-Platform TCA / Analytics Comparison

| Platform | Execution Analytics (TCA) | PnL Attribution Depth | Benchmark Options | Institutional-Grade Gaps |
|---|---|---|---|---|
| **QuantConnect LEAN** | Slippage models (Constant, VolumeShare, MarketImpact), backtest vs live reconciliation overlay, fill simulation (no live market impact modeling) | Portfolio-level: Sharpe, Sortino, Alpha, Beta, Tracking Error, Info Ratio, rolling 6/12m stats | Single custom benchmark; no factor lens | No NBBO price improvement, no IS calculation, no partial-fill attribution, no real-time factor attribution |
| **Interactive Brokers PortfolioAnalyst** | Real-time unrealized/realized P&L via API; post-trade execution report with effective spread; price improvement data per order | Up to 35 metrics; sector allocation vs. benchmark (Brinson attribution: allocation + selection effects); 300+ benchmarks + custom-weighted; Value at Risk | 300+ standard + custom benchmark blends; 165 added 2023 | No IS benchmark, no VWAP/TWAP deviation tracking, no market impact model; factor attribution absent (no Fama-French or similar) |
| **Bloomberg BTCA + PORT** | 100+ benchmarks: IS (arrival price), VWAP, TWAP, PWP, reversions; tags orders by rule/wheel/routing type; compares IS + fill rate + dispersion like-for-like; 30% execution performance improvement documented for RBLD users (2024 study, 23M orders) | PORT: full factor decomposition, sector/region/currency attribution, alpha/beta split, active share; AIM integrates pre-trade cost estimates | Any Bloomberg index; custom blends | Requires terminal subscription (~$24K/yr); not accessible without AIM/PORT license |
| **Two Sigma Venn** | No execution/TCA layer — portfolio-level only | 28-factor Two Sigma Factor Lens (4 groups: Core Macro, Secondary Macro, Macro Style, Equity Style); Factor Contributions to Risk + Return; scenario analysis; private asset cash-flow modeling (2024) | Multi-asset; custom blends | No execution analytics at all; designed for portfolio construction/attribution, not order execution; institutional pricing |
| **Alpaca Broker API** | Portfolio history endpoint (equity + P&L timeseries); no native TCA; community-requested performance attribution API not yet shipped (as of 2025 forum posts) | Raw P&L only; no attribution layer in native API; third-party (TradesViz) adds MFE/MAE, R/R ratios, running P&L | None natively | No IS, no VWAP deviation, no sector attribution, no benchmark comparison in native dashboard |
| **TradeStation** | Q1 2026: avg price improvement $3.04/equity order, $18.34/options order; post-trade slippage vs VWAP/TWAP/arrival; fill rate tracking; venue-level performance; entry/exit efficiency metrics | Strategy Performance Report: gross P&L, K-Ratio, Return Retracement Ratio (RRR), trade efficiency (entry/exit), t-test significance on strategy returns | Single symbol/portfolio benchmark only | No cross-strategy factor attribution, no IS benchmark, no real-time risk decomposition |
| **Saxo Bank (institutional)** | 71,000+ instruments; best-execution documentation; real-time risk management; venue-level execution reporting; no public TCA feature list | Portfolio concentration reports; some sector-level exposure | Index benchmarks available | Detailed TCA features not publicly documented; institutional clients receive customized reporting |

---

## Key Gaps vs Institutional (What Most Retail Platforms Miss)

**1. Implementation Shortfall (IS) calculation is absent**
No retail-facing platform (QuantConnect, Alpaca, TradeStation standard tier) calculates IS — the difference between the decision price at order arrival and the final weighted-average fill. BTCA, Tradeweb TCA, and S&P TCA all do this as a baseline. IS is the primary benchmark used by 53% of institutional desks (The TRADE Algo Survey 2025).

**2. Real-time factor attribution is missing**
Two Sigma Venn offers factor attribution post-hoc, but no retail quant platform provides live factor decomposition of P&L (e.g., how much of today's loss was momentum factor vs. sector rotation vs. idiosyncratic). Bloomberg PORT does this continuously. Fama-French 5-factor or Two Sigma's 28-factor lens is completely absent from QuantConnect, Alpaca, and TradeStation.

**3. Market impact modeling is not applied to live orders**
QuantConnect's own docs note: "QuantConnect doesn't currently model market impact." VolumeShareSlippageModel exists in backtests but is not fed by real-time order book data. Institutional desks use pre-trade market impact estimates (Almgren-Chriss or proprietary) before routing.

**4. NBBO price improvement tracking per order**
TradeStation reports aggregate price improvement figures quarterly but does not expose per-order NBBO improvement in its strategy analytics API. Alpaca exposes none. This metric is required for MiFID II best-execution and SEC Rule 606 reporting.

**5. No cross-strategy, time-of-day, or partial-fill attribution**
None of the retail platforms segment P&L by time-of-day bucket (open auction vs. intraday vs. close), which is standard in institutional post-trade analysis. Partial-fill attribution (opportunity cost of unfilled quantity) is also absent across all reviewed retail platforms.

---

## Recommended Priority Metrics for QuantEdge

Ranked by differentiation impact for an institutional-grade retail quant platform:

1. **Per-strategy IS calculator** — compare each order's arrival price to final fill; expose as a daily metric per strategy. This alone would be unique among retail platforms.
2. **Live factor attribution dashboard** — map P&L in real-time against 5–10 standard factors (momentum, value, size, volatility, sector beta). Use the Fama-French 5-factor model as a baseline; upgrade to a proprietary lens later.
3. **Time-of-day P&L decomposition** — segment fills and returns by market session (pre-market, open 30m, mid-day, close 30m, after-hours). Supports both strategy debugging and execution timing optimization.
4. **Partial-fill opportunity cost** — for each unfilled or partially-filled order, calculate the theoretical cost of the missed quantity using the subsequent price move over a configurable horizon (e.g., 5m, 1h).
5. **VWAP/TWAP deviation heatmap per strategy** — show each strategy's execution quality over time as a z-score vs. daily VWAP; flag systematic degradation automatically.

---

## Self-Healing Systems — State of the Art

### Production / Near-Production

**AlphaEvolve (Google DeepMind, June 2025)**
A Gemini-powered coding agent that iteratively mutates algorithm code, scores each mutation with an evaluator, and selects improvements via evolutionary search. Applied to Bitcoin trading, it evolved a naive strategy from a -2.06 Sharpe to +3.99 Sharpe across generations. Open-source implementation: OpenEvolve (GitHub: `algorithmicsuperintelligence/openevolve`). Mechanism: LLM proposes code diffs → evaluator runs backtests → fitness score fed back → population evolves.

**MadEvolve (arxiv 2605.23007, May 2025)**
Open-source framework combining LLM mutation with evolutionary search, applied to Bitcoin algo trading. Jointly evolves feature pipeline + execution strategy. Ensemble of 5 LLMs achieves 3–64% improvement rates depending on task. Published at `madevolve.org`. Key finding: separate mutation of signal generation vs. execution components outperforms joint mutation on complex strategies.

**Self-Improving Coding Agent (arxiv 2504.15228, May 2025)**
Demonstrated that an LLM coding agent can autonomously edit its own tooling and achieve 17–53% gains on SWE-Bench Verified. Not trading-specific but directly applicable to strategy code self-repair.

### Practical Implementations

A documented self-healing trading bot (dev.to, 2025) tracks lessons from live failures — e.g., an iron condor partial fill creating a naked short put, triggering automatic migration to Alpaca MLeg atomic order type. The system logs failure signatures and patches its own order logic.

### Key Takeaway for QuantEdge

No commercial platform currently offers automated strategy self-repair triggered by execution analytics observations. The combination of (a) IS/fill-rate degradation detection + (b) LLM-guided strategy code mutation + (c) walk-forward validation before redeployment is an open, differentiating capability. The academic infrastructure (AlphaEvolve, MadEvolve) is ready for productization.

---

## Sources

- [Bloomberg EET 2025: Technology Algo and TCA Trends](https://www.bloomberg.com/professional/insights/artificial-intelligence/eet-2025-technology-algo-and-tca-trends/)
- [Bloomberg: How Automation, TCA and Broker Wheels Work Together](https://www.bloomberg.com/professional/insights/trading/how-automation-tca-and-broker-wheels-work-together-in-modern-equity-ems/)
- [Bloomberg Study: Automated Equity Trading Results in Improved Performance (March 2024)](https://www.bloomberg.com/company/press/bloomberg-study-finds-automated-equity-trading-results-in-improved-performance/)
- [Interactive Brokers PortfolioAnalyst Features](https://www.interactivebrokers.com/en/portfolioanalyst/features.php)
- [IBKR: Attribution vs. Benchmark Guide](https://www.ibkrguides.com/portfolioanalyst/performanceandstatistics/pa-attribution-vs-benchmark.htm)
- [IBKR Adds 165 Benchmarks to Performance Attribution](https://fxnewsgroup.com/forex-news/platforms/interactive-brokers-adds-165-benchmarks-to-performance-attribution-report-in-portfolioanalyst/)
- [QuantConnect LEAN Slippage Models](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/slippage/supported-models)
- [QuantConnect Backtest Results](https://www.quantconnect.com/docs/v2/cloud-platform/backtesting/results)
- [QuantConnect Reconciliation](https://www.quantconnect.com/docs/v2/cloud-platform/live-trading/reconciliation)
- [Two Sigma Venn: Understanding Factor Analysis Output](https://help.venn.twosigma.com/en/articles/7136452-understanding-factor-analysis-output)
- [Two Sigma Venn: Factor Lens FAQ](https://help.venn.twosigma.com/en/articles/1392786-two-sigma-factor-lens-faq)
- [TradeStation Strategy Performance Report](https://help.tradestation.com/10_00/eng/tradestationhelp/subsystems/spr_topics/spr/spr_performance_summary_tab.htm)
- [TradeStation Institutional Order Execution](https://www.tradestation.com/insights/institutional-order-execution-at-tradestation/)
- [Alpaca: Performance Attribution API Request (forum)](https://forum.alpaca.markets/t/performance-attribution-api/12413)
- [The TRADE: Algorithmic Trading Survey HF 2025](https://www.thetradenews.com/wp-content/uploads/2025/06/Algo-Survey-HF-2025.pdf)
- [A-Team: Top 12 TCA Solutions in 2024](https://a-teaminsight.com/blog/the-top-12-transaction-cost-analysis-tca-solutions-in-2024/)
- [MadEvolve: Evolutionary Optimization of Trading Systems with LLMs (arxiv 2605.23007)](https://arxiv.org/abs/2605.23007)
- [AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery (arxiv 2506.13131)](https://arxiv.org/abs/2506.13131)
- [Auto-Improve Bitcoin Algo Trading Strategies with LLMs](https://trilogyai.substack.com/p/auto-improve-bitcoin-algo-trading)
- [A Self-Improving Coding Agent (arxiv 2504.15228)](https://arxiv.org/abs/2504.15228)
- [Saxo Bank: Capital Market Solutions for Institutional Clients](https://www.home.saxo/institutional-and-partners)
