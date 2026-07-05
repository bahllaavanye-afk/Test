# Per-Desk SOTA Research — 2026-07-05

Desk-by-desk survey of current (mid-2026) research, open-source projects, and
publicly known practices of top firms — mapped to what each QuantEdge desk
runs today and what it should build next. Companion to
`STRATEGY_SOTA_2026.md` (whole-company audit + scaling plan).

Method note: "firm secrets" below means *publicly documented operating
principles* of Jane Street / Citadel Securities / HRT / Jump — the real
proprietary edge (signals, infrastructure latencies) is not published, and any
source claiming otherwise is selling something.

---

## 1. Equities desk

**SOTA now**
- ML-predicted *factor timing* is mainstream: gradient boosting predicting
  which factors (value/momentum/quality) outperform next period
  ([Cakici et al., JPM 2025](https://alphaarchitect.com/predict-factor-returns/);
  [Morningstar coverage](https://global.morningstar.com/en-ca/markets/machine-learning-can-forecast-which-stock-factors-will-outperform)).
- **Grammar-guided alpha search** ([arXiv 2601.22119](https://arxiv.org/pdf/2601.22119)):
  formal-grammar constrained search over factor formulas — the rigorous
  version of LLM alpha mining; interpretable, overfitting-resistant.
- Hybrid regime-adaptive systems (TA + ML + sentiment) beat single-paradigm
  models ([ComSIA 2026](https://arxiv.org/html/2601.19504v1)).
- VAEs for latent factor spaces and GANs for synthetic factors to dodge
  factor crowding ([AI factor-investing survey](https://link.springer.com/chapter/10.1007/978-981-95-0887-7_12)).

**We have:** 12 desk strategies (momentum, cross-sectional, VWAP reversion,
residual momentum, idio-vol anomaly…) + 21 equity bots.
**Build next:** (1) factor-timing overlay — our weekly GBC experiment already
predicts direction; extend it to predict *which desk strategy* to overweight;
(2) LLM alpha miner constrained to a factor grammar, walk-forward gated.

## 2. Crypto desk

**SOTA now**
- Basis/funding-rate arb has compressed from 30–50% (2021) to **5–15%
  annualized** in normal regimes ([Quantt 2026](https://www.quantt.co.uk/resources/crypto-quant-strategies-2026)) —
  but event-window studies still show outsized bursts, up to +115.9%/6mo with
  1.9% max loss on CEX/DEX funding arb ([ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S2096720925000818)).
- Funding rate is now modeled as an *algorithmic feedback rule* inducing a
  mean-reverting basis — i.e., the basis itself is the tradable signal
  ([SSRN 6185958](https://papers.ssrn.com/sol3/Delivery.cfm/6185958.pdf?abstractid=6185958&mirid=1)).
- Information-driven bars + triple-barrier labeling + DL beats time bars
  net of costs ([Financial Innovation 2025](https://link.springer.com/article/10.1186/s40854-025-00866-w)).

**We have:** funding_rate_arb, basis_carry, btc_eth_stat_arb, mvrv_zscore,
on-chain netflow, whale momentum — good coverage; desk finally trades 24/7
after today's clock fix.
**Build next:** (1) make funding-rate arb the desk's core allocation (highest
evidence-to-effort ratio); (2) CUSUM/volume bars for the crypto data loader —
cheap, proven uplift; (3) DEX-side funding legs remain out of scope (gas +
custody complexity beyond free tier).

## 3. Options desk

**SOTA now**
- **0DTE is the market**: ~59% of all SPX volume in 2025–26
  ([AInvest](https://www.ainvest.com/news/0dte-volatility-drives-market-strategies-2026-2602/)).
  0DTE carries a statistically significant variance risk premium, and the
  *jump-risk* premium is ~2x the diffusion+vol premia combined
  ([SSRN 5223127](https://papers.ssrn.com/sol3/Delivery.cfm/5223127.pdf?abstractid=5223127&mirid=1);
  annotated replication: [vilkovgr/0dte-strategies](https://github.com/vilkovgr/0dte-strategies)).
- Classic VRP selling still earns 0.5–1.5%/day on ATM short-dated premium but
  with -800% tail episodes — position sizing IS the strategy
  ([Quantpedia VRP](https://quantpedia.com/strategies/volatility-risk-premium-effect)).
- Institutional 0DTE playbook: open-auction strangle selling with continuous
  delta hedging ([Resonanz](https://resonanzcapital.com/insights/same-day-options-same-day-alpha-institutional-lessons-from-0-dtes-boom)).

**We have:** vix_mean_reversion, gamma_exposure, skew_arb, vrp_systematic,
dispersion, term-structure + 4 options bots + (new) rules-validate/wheel/flow
endpoints with true IV rank accruing.
**Build next:** (1) defined-risk VRP only (spreads/condors, never naked —
matches the -800% tail evidence); (2) a 21–45 DTE premium-selling bot sized by
the 5%-equity rule we just shipped; 0DTE needs intraday hedging cadence our
free-tier cron (15-min) cannot honestly support — **defer 0DTE**, document why.

## 4. Polymarket desk

**SOTA now**
- Academia documented **$40M+ arbitrage extracted from Polymarket**
  (Apr 2024–Apr 2025, 86M bets, IMDEA) — but 78% of low-volume opportunities
  fail on execution ([overview](https://www.trevorlasn.com/blog/how-prediction-market-polymarket-kalshi-arbitrage-works)).
- Working open-source cross-venue bots exist today:
  [ImMike/polymarket-arbitrage](https://github.com/ImMike/polymarket-arbitrage) (10k+ markets watched),
  [realfishsam/prediction-market-arbitrage-bot](https://github.com/realfishsam/prediction-market-arbitrage-bot).
- Economics: YES+NO < $1 intra-venue, cross-venue Polymarket↔Kalshi spreads;
  maker orders are free on Polymarket; spreads under ~5% die to fees; capital
  lockup makes *rotation* (exit when spread closes) beat hold-to-resolution;
  execution windows shrank from minutes (2024) to ~30s (2026)
  ([guide](https://launchpoly.com/blog/polymarket-kalshi-arbitrage-guide)).
- New instrument class: perpetuals ON prediction markets
  ([arXiv 2605.10400](https://arxiv.org/pdf/2605.10400)).

**We have:** poly_binary_arb, poly_calibration_arb, poly_late_resolution,
sentiment momentum — plus Kalshi public reads already wired (#203).
**Build next:** (1) the **Polymarket↔Kalshi market-matching table** (the
hard part per every practitioner) — an LLM text-similarity job is a perfect
free-cascade task, verified by resolution-source comparison; (2) rotation
exits instead of hold-to-resolution in poly_binary_arb.

## 5. Macro/FX desk

**SOTA now**
- Trend following: positive risk-adjusted return 2004–2025 with ~zero SPX
  correlation; the frontier is *market-mix optimization* — which markets to
  trend, not how ([Man Group](https://www.man.com/insights/trend-following-optimal-market-mix)).
- Trend premia across assets are substantially **redundant** — diversification
  across trend variants hides overlap; fewer, cleaner sleeves win
  ([arXiv 2510.23150](https://arxiv.org/pdf/2510.23150)).
- Macro-informed trend (macro data + price) beats price-only trend
  ([Macrosynergy](https://macrosynergy.com/research/equity-trend-following-with-market-and-macro-data/)).

**We have:** cross_asset_carry, sector_rotation, time_series_momentum,
fx_trend, fx_reversion, dollar_carry, bond_equity_rotation on ETF proxies.
**Build next:** (1) prune trend redundancy — our tsmom/sector_rotation/
fx_trend likely load on one premium; correlation-cluster them in the
leaderboard before scaling any; (2) macro-filter: gate trend entries on the
FOMC/CPI calendar we just shipped (`/options/macro-calendar`).

## 6. StatArb desk

**SOTA now**
- Comprehensive 2025 survey of ML/DL/RL pairs trading
  ([U. Warsaw WP 22/2025](https://www.wne.uw.edu.pl/application/files/5617/5819/7786/WNE_WP485.pdf)):
  ML/DL variants beat distance/cointegration classics on nonlinear spreads,
  but transaction-cost honesty separates papers that replicate from those
  that don't.
- Best practice stack: cointegration screen → Kalman dynamic hedge ratio →
  regime filter → cost-aware execution
  ([stat-arb models deep-dive](https://coincryptorank.com/blog/stat-arb-models-deep-dive)).
- CE-PPO (cluster-embedding RL) and news-aware direct RL are research-grade
  ([MDPI](https://www.mdpi.com/2073-8994/18/1/112), [arXiv 2510.19173](https://arxiv.org/pdf/2510.19173)) — skip until the P&L loop exists.

**We have:** pairs_trading, kalman_pairs, pca_stat_arb, triangular_arb,
stablecoin_depeg_arb — architecturally aligned with SOTA already.
**Build next:** universe expansion (ETF pairs beyond the 5 index proxies) +
per-pair cost model; the survey's conclusion is that costs, not models, kill
stat arb.

## 7. What top firms actually do (publicly known)

- **Market making at scale is the durable business**: Citadel Securities
  handles ~25% of US equity volume ($12.2B 2025 revenue); Jane Street
  ($39.6B 2025) earns spread across enormous instrument universes rather than
  betting direction ([comparison](https://www.quantt.co.uk/resources/citadel-jane-street-two-sigma-comparison),
  [tier list](https://www.quantvps.com/blog/top-quant-trading-firms)).
- **Engineering = trading**: HRT/Jane Street blur the trader/developer line;
  Jane Street standardizes on one language (OCaml) and collaborative review —
  the "secret" is verification culture, not a magic signal.
- **Infrastructure as edge**: Jump/HRT invest in FPGAs and microwave links —
  a latency game we deliberately do NOT play; our edge must come from
  slower-horizon signals where free infrastructure competes.
- Lesson for QuantEdge: our Avellaneda-Stoikov MM strategy is the only
  market-making sleeve we have; on free-tier latency it can only work on slow
  venues — **prediction markets are the one venue where our latency is
  competitive** (30s windows vs microseconds).

## 8. Agentic AI trading — the latest

- **Production adoption is real**: Arcesium "Intelligence" (May 2026) and
  Broadridge run agentic AI in production capital-markets ops; Man Group's
  CTO reports alpha-generating strategies emerging from agentic workflows
  ([Founderland](https://www.founderland.ai/articles/the-race-to-build-fully-autonomous-ai-hedge-funds-mq6jgzt7)).
- **Standard architecture converged** to exactly what we run: planner agents →
  execution agents → post-trade risk agents with feedback
  ([Digiqt](https://digiqt.com/blog/ai-agents-in-hedge-funds/),
  [WunderTrading](https://wundertrading.com/journal/en/agentic-trading)).
- **Factor crowding from LLMs**: 95% of funds prompt the same frontier models
  on the same public data — differentiation comes from private data
  (our own paper P&L history) and verification loops, not the LLM itself.
- **Regulatory direction**: EU AI Act phase 2 + SEC guidance require
  *traceable decision chains* and a named accountable human — our audit-log +
  reward-gate design is already the right shape.
- Frameworks: [TradingAgents v0.3](https://github.com/TauricResearch/TradingAgents)
  (role debate), FinAgent (reflection memory), AI-Hedge-Fund (45k stars,
  analyst-agent ensemble) — all validate the pattern; none publish durable
  live alpha. The moat is the feedback loop, which is why closing our P&L
  loop (STRATEGY_SOTA_2026 §3.1) outranks adopting any framework.

---

## Cross-desk priority queue (evidence-weighted)

1. **P&L feedback loop** (all desks) — prerequisite; nothing scales blind.
2. **Crypto funding-rate arb as core sleeve** — best documented risk/return.
3. **Polymarket↔Kalshi matcher + rotation exits** — the venue where our
   latency is actually competitive; LLM does the matching, code verifies.
4. **Options: defined-risk VRP bot (21–45 DTE)** — premium evidence is
   strong; 0DTE deferred (needs intraday hedging we can't honestly run).
5. **Equities: factor-timing overlay** on the weekly GBC experiment.
6. **Macro: trend-sleeve deduplication** before any scale-up.
7. **StatArb: cost model + universe expansion** — costs kill stat arb, not
   models.
