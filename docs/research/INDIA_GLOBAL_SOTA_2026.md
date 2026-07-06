# India + Global SOTA — Markets, Algos, Infra, AI (2026)

_Research doc. Companion to `GLOBAL_QUANT_FIRMS_2026.md` (US/EU/Asia firms) and
`OPTIONS_ALPHA_DEEP_2026.md`. Focus here: (1) Indian markets — what its quant
ecosystem does uniquely well and what transfers to QuantEdge; (2) the "better
everything" upgrade map: tech, AI, models, multi-agent, infra._

## 1. India — why it matters to a quant platform
India (NSE) is the **world's largest derivatives market by contract volume** —
NSE has ranked #1 globally in exchange-traded derivatives contracts for years,
driven by index options (NIFTY/BANKNIFTY weeklies). Its ecosystem specialties:

### What Indian quant firms/prop shops are known for
| Area | Detail | Transfers to QuantEdge? |
|---|---|---|
| **Weekly index-options selling** | NIFTY/BANKNIFTY weekly expiries dominate; straddle/strangle selling with strict intraday stop discipline (the "expiry-day theta" trade) | ✅ maps to SPX/SPY 0DTE-weekly income (backlog P2) |
| **Expiry-day gamma effects** | Documented intraday pinning/unwinding patterns on expiry days | ✅ `gamma_exposure` strategy already models dealer gamma; add expiry-day gate |
| **Retail-flow alpha** | Massive retail options flow → systematic edge for flow-aware makers (why SEBI found ~90% of retail F&O traders lose) | ⚠️ ethical/regulatory note; the transferable piece is put-call-ratio + flow imbalance signals (already have PCR endpoint) |
| **STT/cost-aware execution** | High transaction taxes force strict cost modeling into every backtest | ✅ add per-trade fee/slippage model to backtests (fees column already on Trade) |
| **HFT + colocation at NSE** | Tight latency games (co-lo racks at BKC) | ❌ not our tier — we're minutes-cadence on free infra |
| **Momentum in mid/small caps** | Strong documented momentum premia in Indian equities vs US | ✅ validates cross_sectional_momentum; consider ADR/ETF universe (INDA, EPI, SMIN) |

### Trading Indian markets from QuantEdge (honest constraints)
- **Direct NSE access requires an Indian broker** (Zerodha Kite Connect ₹2k/mo,
  Upstox, Angel One SmartAPI — all have REST APIs) + Indian KYC. Not free-tier.
- **Zero-manual-step path today:** trade India **via US-listed ETFs on Alpaca**:
  `INDA` (MSCI India), `EPI` (earnings-weighted), `SMIN` (small-cap momentum
  premium), `INDY` (Nifty 50). Action: add an **India sleeve** to the Macro/FX
  desk symbols — no new broker, real paper fills, P&L-loop ranked.
- **GIFT Nifty** (SGX→GIFT City) trades nearly 21h/day — useful as an overnight
  risk signal for the India sleeve (public quotes).

### Indian-market strategy candidates (ETF-implementable now)
1. **India momentum sleeve** — cross-sectional momentum over INDA/EPI/SMIN/INDY vs EEM.
2. **India-US overnight gap** — India closes before US opens; INDA's US-hours drift
   vs NSE close is a documented lead-lag (time-zone arbitrage class).
3. **Monsoon/fiscal seasonality** — budget-day and monsoon-progress seasonal tilts (low capacity, but real).
4. **INR carry proxy** — USDINR carry via rate differentials (ETF proxy weak; note only).

## 2. "Better everything" — the upgrade map (with status)
| Ask | Now shipped | Next (concrete) |
|---|---|---|
| **Automation** | key-relay (secrets→Render), bot lifecycle, channel/command self-setup, smoke gate | Alembic auto-migration check in CI; auto-rollback on smoke failure |
| **AI/models** | LLM cascade + Haiku backstop; weekly GBC walk-forward | Regime-conditional model selection; LightGBM + purged K-fold CV; drift monitor on live features |
| **Multi-agent** | 19 employee workflows + shared brain + reward gate | Team-lead orchestrator triaging `agent-fix-needed` → assigns → reviews (design exists) |
| **Algos/trades** | 112-strategy registry; income wired w/ iv_rank; perf-weighted sizing (self-scaling) | Vol-target position sizing; portfolio-level optimizer (HRP across desks); India sleeve |
| **Infra** | autoDeploy on, keep-alive ×2, smoke test, security scan | Durable Postgres (user), UptimeRobot (user), Langfuse tracing, pgvector memory |
| **Research** | 6 research docs + ML experiments + model audit | Alpha-miner: LLM proposes → backtest gate verifies → auto-PR (reward-gated) |

## 3. Advanced-strategy inventory (answer to "I hope we're beyond wheel")
The registry holds **112 strategies**. Wheel is the *retail-facing floor*, not the
ceiling — it was added for Options-Alpha template parity. The professional tier
already present includes: `dispersion_trading` (index-vs-components correlation
trade — a real vol-desk staple), `skew_arb`, `vrp_systematic` (variance risk
premium), `vol_term_structure`, `gamma_exposure` (dealer-positioning), `pca_stat_arb`
(eigenportfolio mean-reversion), `kalman_pairs` (adaptive hedge ratios),
`cross_sectional_momentum`, `residual_momentum` (factor-neutral), `pead_sue`
(post-earnings drift on standardized surprise), `idio_vol_anomaly`,
`avellaneda_stoikov_mm` (inventory-aware market making), `triple_barrier_momentum`
(López de Prado labeling), `a3c_lstm` (RL), plus TFT/LSTM/XGB/LightGBM ML models.
The gap is not sophistication — it's **evidence**: none have live track records
until fills flow. The P&L loop + self-scaling now ranks them by realized results.
