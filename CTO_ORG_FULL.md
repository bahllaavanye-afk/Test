# QuantEdge — Full Operating Org (92 people, 24×7 coverage)

> Org doctrine: every role is an **autonomous agent** following a CLAUDE.md role file.
> Agents work 24×7 (no human sleep cycle). Performance reviewed **biweekly**.
> All roles below are real engineering specifications, not figureheads.

---

## C-Suite (5)

| Title | Owner | Mandate | Bi-weekly KPI |
|-------|-------|---------|---------------|
| **CEO** | Anika Sharma | Vision, fundraise, board, investor relations | Active investor pipeline ≥ 10, Series A close by D90 |
| **CTO** | QuantEdge AI | Engineering org, architecture, ship cadence | ≥ 50 commits/day across org, zero P0 unresolved > 24h |
| **CFO** | Rohan Kapoor | Runway, vendor costs, accounting, audit | Cash > 12 months runway, gross margin > 70%, infra cost per trade < $0.001 |
| **CRO** (Chief Risk Officer) | Marina Volkov | Firm-wide risk, regulatory compliance, kill switches | Zero risk breaches, daily VaR/CVaR delivered ≤ 09:00 UTC |
| **General Counsel** | David Chen | Legal, KYC, FINRA, GDPR, contracts | All trading licenses current, zero compliance findings |

---

## VPs (8)

| Title | Owner | Reports to | Mandate |
|-------|-------|-----------|---------|
| VP Research | Marcus Polk (ex-Renaissance) | CTO | New alphas — 5 promoted/quarter |
| VP Engineering | Ada Pang (ex-DeepMind) | CTO | Platform reliability + velocity |
| VP Product | Sarah Kim | CTO + CEO | Roadmap, OKRs, customer voice |
| VP DevOps & SRE | Liu Wei (ex-Stripe) | CTO | 99.95% uptime, mean-time-to-deploy < 5min |
| VP Security | Naoko Tanaka (ex-Cloudflare) | CRO + CTO | Zero P0 sec incidents, full pen-test passing |
| VP Quant Research | Dmitri Sokolov (ex-DE Shaw) | VP Research | Cross-asset signal discovery |
| VP Machine Learning | Hiroshi Yamada (ex-Jane Street ML) | VP Engineering | Production ML, model leaderboard |
| VP Frontend | Priya Iyer (ex-Bloomberg) | VP Engineering | Bloomberg-grade dashboard |

---

## Directors (16)

| Title | Owner | Reports to | Squad size |
|-------|-------|-----------|------------|
| Dir Alpha Research | Aleksandr Petrov | VP Quant Research | 8 quants |
| Dir Microstructure | Yuki Nakamura | VP Quant Research | 4 microstructure ICs |
| Dir Fundamental Research | Marcus Polk (dual-hat) | VP Research | 4 fundamentals ICs |
| Dir Alternative Data | Lin Zhang | VP Research | 3 alt-data engineers |
| Dir ML Modeling | Hiroshi Yamada (dual-hat) | VP ML | 6 ML engineers |
| Dir ML Infrastructure | Felix Andersen | VP ML | 4 ML infra ICs |
| Dir Backend Platform | Aaron Bell | VP Engineering | 5 backend engineers |
| Dir Frontend | Priya Iyer (dual-hat) | VP Frontend | 4 frontend engineers |
| Dir Data Engineering | Jiwoo Park | VP Engineering | 4 data ICs |
| Dir Execution | Ying Chen | VP Engineering | 4 execution ICs |
| Dir Risk Engineering | Sven Larsen | CRO | 4 risk ICs |
| Dir Security Engineering | Naoko Tanaka (dual-hat) | VP Security | 3 sec ICs |
| Dir DevOps | Liu Wei (dual-hat) | VP DevOps | 4 SREs |
| Dir QA & Test Automation | Maria Garcia | VP Engineering | 3 QA engineers |
| Dir Compliance Engineering | David Chen (dual-hat) | General Counsel | 2 compliance ICs |
| Dir Finance Engineering | Rohan Kapoor (dual-hat) | CFO | 2 finance/accounting ICs |

---

## Product Managers (6)

| Title | Owner | Owned roadmap |
|-------|-------|---------------|
| PM Alpha Strategies | Jordan Williams | Strategy backlog, paper-to-live promotion, capacity |
| PM Trading Infrastructure | Vikram Sundar | Brokers, execution, market data |
| PM ML Platform | Mei Chen | Experiment tracking, model registry, training infra |
| PM User Experience | Lin Tsai | Dashboard, onboarding, investor reports |
| PM Risk & Compliance | Andrea Rossi | Risk dashboard, audit trail, regulatory reporting |
| PM Growth | Daniel Okafor | Landing page, investor demo, conversion analytics |

---

## Individual Contributors (57)

### Alpha Research (8 quants under Dir Alpha Research)
- IC1 Equities Factor Research: **Sarah Tanaka** — momentum/value/quality/low-vol/QMJ
- IC2 Crypto Quant: **Carlos Rivera** — TSMOM, funding-arb, on-chain
- IC3 Options Quant: **Emma Schmidt** — VRP, skew, gamma exposure
- IC4 FX/Macro Quant: **Raj Patel** — carry, term structure, regime
- IC5 Stat Arb: **Yuki Tanaka** — pairs, PCA, residuals
- IC6 Event-Driven: **Olivia Foster** — earnings, M&A, PEAD
- IC7 Polymarket Quant: **Tom Müller** — prediction-market microstructure
- IC8 Newcomer rotating: 6-month onboarding

### Microstructure (4 under Dir Microstructure)
- IC1 Order Book Modeling: **Hannah Choi** — OFI, queue dynamics, OBI
- IC2 Liquidity Modeling: **Devon Brooks** — Kyle's lambda, impact models
- IC3 Auction Dynamics: **Linh Pham** — open/close, MOC, halts
- IC4 Tick Data Specialist: **Andre Silva** — high-frequency ETL

### ML Engineers (6 under Dir ML Modeling)
- IC1 Sequence Models (LSTM, Transformer): **Ravi Kumar**
- IC2 Tree Models (XGBoost, LightGBM): **Sofia Reyes**
- IC3 KNN / Distance Models (Lorentzian, kNN): **Mark Brown**
- IC4 RL (PPO, A3C-LSTM): **Aisha Khan**
- IC5 Ensembles + Stacking: **Henrik Nordström**
- IC6 Feature Engineering: **Camille Dubois**

### ML Infrastructure (4 under Dir ML Infra)
- IC1 Training Pipeline: **Wei Cheng** — Kaggle/Colab/Lightning workflows
- IC2 Inference Service: **Diego Hernandez** — < 50ms p99
- IC3 Model Registry: **Tara Brennan** — versioning, A/B
- IC4 Experiment Tracking: **Khalid Ahmed** — MLflow, configs

### Backend Platform (5 under Dir Backend)
- IC1 API Lead: **Jonas Weber** — FastAPI routes, schemas
- IC2 DB & ORM: **Mei Lin** — SQLAlchemy, migrations
- IC3 WebSocket: **Ethan Wright** — real-time fan-out
- IC4 Auth & Sessions: **Fatima Said** — JWT, RBAC, audit logs
- IC5 Background Tasks: **Sebastian Marquez** — APScheduler, supervisor

### Frontend (4 under Dir Frontend)
- IC1 Dashboard & Charts: **Yuki Sato** — TradingView widgets, LWCharts
- IC2 Order Entry & Trading UX: **Liam O'Connor** — order forms, execution
- IC3 Analytics & Reports: **Ananya Verma** — investor reports, P&L
- IC4 Design System: **Noah Park** — shadcn/Tailwind primitives

### Data Engineering (4 under Dir Data)
- IC1 Real-Time Feed: **Pedro Mendes** — Alpaca/Binance WS
- IC2 Historical Pipeline: **Greta Roth** — OHLCV warehouse
- IC3 Redis & Cache: **Jamal Carter** — TTL/pipeline/eviction
- IC4 Alt-Data Pipeline: **Anya Volkov** — news, sentiment, on-chain

### Execution (4 under Dir Execution)
- IC1 TWAP/VWAP: **Faraz Sheikh** — slicing algorithms
- IC2 Limit-First: **Kenta Yoshida** — fallback timing, queue position
- IC3 Smart Routing: **Isabella Costa** — broker selection
- IC4 Slippage Analyst: **Theo Lambert** — attribution, reporting

### Risk Engineering (4 under Dir Risk)
- IC1 Position Sizing: **Aditi Mehta** — Kelly, vol targeting
- IC2 Drawdown & Circuit Breakers: **Lucas Berg** — per-bucket caps
- IC3 Correlation Limits: **Hina Tanaka** — cluster detection
- IC4 VaR/CVaR Reporting: **Omar Hassan** — daily reports

### Security (3 under Dir Security)
- IC1 AppSec: **Zoe Martinez** — SAST, dependency scan
- IC2 SecOps: **Hassan Ali** — secret rotation, incident response
- IC3 Penetration Testing: **Robin Sanders** — quarterly pen-test

### DevOps / SRE (4 under Dir DevOps)
- IC1 CI/CD Pipeline: **Maya Joshi** — GitHub Actions, deploy speed
- IC2 Observability: **Akira Suzuki** — logs, traces, alerts
- IC3 Incident Response: **Felix Brown** — on-call rotation
- IC4 Cost Optimization: **Lara Vasquez** — Render/Vercel/Upstash spend

### QA / Test (3 under Dir QA)
- IC1 Unit Coverage: **Sanjay Reddy** — > 80% coverage target
- IC2 Integration & E2E: **Erin Walker** — Playwright + API contract
- IC3 Backtest QA: **Hayden Liu** — walk-forward validation gate

### Compliance Engineering (2 under Dir Compliance)
- IC1 KYC & AML: **Ravi Nair** — onboarding flows
- IC2 Trade Reporting: **Sofia Garcia** — FINRA / SEC integration

### Finance Engineering (2 under Dir Finance Eng)
- IC1 Accounting Integration: **Tom Walker** — Stripe, ledgers, AR/AP
- IC2 Cost Telemetry: **Yumi Tanaka** — per-trade cost attribution

---

## 24×7 Coverage Plan

Three shifts × 8 hours, but every agent runs continuously. Shifts only determine **on-call escalation routing**.

| Shift (UTC) | Window | On-call lead | Backup |
|-------------|--------|--------------|--------|
| Asia | 00:00–08:00 | Hiroshi Yamada (VP ML) | Yuki Nakamura |
| Europe | 08:00–16:00 | Marcus Polk (VP Research) | Sven Larsen (Dir Risk Eng) |
| Americas | 16:00–24:00 | Ada Pang (VP Eng) | Liu Wei (VP DevOps) |
| **Critical security** | 24×7 | Naoko Tanaka (VP Sec) | Hassan Ali |
| **Critical risk** | 24×7 | Marina Volkov (CRO) | Sven Larsen |

---

## Biweekly Performance Review (every 14 days)

### Tracked per IC
| Metric | Weight | Source |
|--------|--------|--------|
| Commits shipped | 15% | GitHub API |
| PRs reviewed (≥ 1 substantive comment) | 15% | GitHub API |
| Tests added / coverage delta | 10% | pytest-cov reports |
| Strategies / models in production | 20% | Strategy registry + ML registry |
| P&L attributed YTD | 20% | trades DB → strategy attribution |
| Incidents caused (negative) | 10% | Incident DB |
| Training hours completed | 10% | Notion training-attendance |

### Promotion ladder
- L1 Junior → L2 IC: composite score > 60 for 3 consecutive cycles
- L2 IC → L3 Senior: composite > 75 + owns a strategy in production
- L3 → L4 Staff: composite > 85 + mentors ≥ 2 L1/L2
- L4 → L5 Director: composite > 90 + owns a squad outcome
- L5 → L6 VP: composite > 95 + drives org-level metric

### Bottom 5% process
- Coaching plan for 1 cycle
- PIP for 1 cycle if no improvement
- Rotation to different squad or exit at end of PIP

### Top 5% process
- Spot bonus equivalent to 10% of comp
- Public recognition in CTO weekly notes
- First-pick of new strategy or project ownership

---

## Strategy Discovery Pipeline (continuous)

Every weekday produces ≥ 1 new candidate strategy. The pipeline:

1. **Source** — VP Research curates a 50-paper backlog refreshed monthly from:
   - SSRN top-downloads in quant categories
   - arXiv q-fin
   - Journal of Financial Economics, Review of Financial Studies, JoF
   - AQR / Two Sigma / Renaissance white papers
   - On-chain analytics (Glassnode, Dune dashboards)
2. **Triage** — Dir Alpha + PM Alpha assign expected Sharpe and feasibility scores
3. **Implementation** — IC owns paper → ships strategy in 1-3 days
4. **Backtest** — walk-forward only; reject if OOS Sharpe < 1.0
5. **Paper trial** — 14-day paper run, daily P&L attribution
6. **Promotion** — pass all 8 criteria → goes live with $1k tranche
7. **Scale-up** — every promoted strategy doubles capital every 14 days if Sharpe > 1.5

Target: **70+ strategies in production by Day 60**, **30+ paper-promoted to live by Day 60**.

---

## Daily / Weekly / Monthly Cadence

| Cadence | Event | Time UTC | Owner | Output |
|---------|-------|----------|-------|--------|
| Daily 13:00 | All-hands standup (15min) | 13:00 | CTO | Notion daily standup page |
| Daily 13:30 | Squad standups (10min × 16 squads, parallel) | 13:30 | Directors | Squad standup pages |
| Daily 14:00 | Open coding window (3hr) | 14:00 | All ICs | Commits, PRs |
| Daily 17:00 | Alpha review (5 new strategies presented) | 17:00 | VP Research | Strategy review doc |
| Daily 18:00 | Code review window | 18:00 | All ICs | PR approvals |
| Daily 19:00 | Deploy window | 19:00 | VP DevOps | Deploy log |
| Daily 20:30 | Risk EOD report | 20:30 | CRO | VaR/CVaR/exposure report |
| Daily 21:00 | P&L attribution by strategy | 21:00 | CTO + PM Alpha | P&L Notion DB row |
| Weekly Mon | C-suite strategy sync (1hr) | 14:00 Mon | CEO | Weekly OKR update |
| Weekly Wed | All-hands engineering town hall (45min) | 15:00 Wed | CTO | Town hall recording + Q&A doc |
| Weekly Fri | Show & Tell — every IC demos | 16:00 Fri | Squad leads | 5-min demos |
| Bi-weekly | Performance review cycle close | Sun 23:00 | Managers | Updated skills DB |
| Monthly | Board meeting | First Tue 14:00 | CEO + CFO + CTO | Board deck |
| Monthly | Penetration test | Last Wed 22:00 | VP Security | Pen-test report |
| Quarterly | Strategy comp portfolio rebalance | Quarter end | CRO + PM Alpha | Allocation update |
| Quarterly | Calibration committee — promotions | Quarter end | CTO + all VPs | Promotion list |

---

## CFO Cost Tracking (daily updated)

| Category | Daily target | YTD budget | Owner |
|----------|--------------|------------|-------|
| Render API (free tier → paid at scale) | $0–$25/day | $9k | Lara Vasquez |
| Vercel (free tier sufficient) | $0 | $0 | Lara Vasquez |
| Supabase (free → Pro at scale) | $0–$1/day | $360 | Lara Vasquez |
| Upstash Redis (10k req/day free) | $0–$3/day | $1k | Lara Vasquez |
| Kaggle GPU (free 30hrs/week) | $0 | $0 | Wei Cheng |
| Alpaca paper trading | $0 | $0 | (CFO) |
| Slack notifications | $0 (free tier) | $0 | (CFO) |
| Notion workspace | $0–$10/mo | $120 | (CFO) |
| Domain + SSL | $1/mo | $12 | Lara Vasquez |
| **Total burn** | **< $50/day** | **< $18k YTD** | CFO |

If burn rate > $50/day, CFO triggers cost-review meeting next morning.

---

## Tools per role

| Tool | Used by | Configured by |
|------|---------|---------------|
| GitHub | All engineers | VP DevOps |
| Notion | Everyone (standups, P&L, skills, papers) | CTO |
| Slack | Everyone (realtime discussion) | CTO |
| Google Docs | C-suite + Directors (long-form docs, OKRs, postmortems) | CEO assistant |
| Render | DevOps (deploys) | Liu Wei |
| Vercel | DevOps (frontend deploys) | Liu Wei |
| Supabase | Backend, Data (DB) | Jiwoo Park |
| Upstash | Backend, Data (cache) | Jiwoo Park |
| Alpaca dashboard | Execution, Risk (broker positions) | Ying Chen |
| Kaggle / Colab | ML Engineers (free GPU training) | Wei Cheng |
| MLflow | ML Engineers (experiment tracking) | Khalid Ahmed |
| Sentry (planned) | All (error tracking) | Akira Suzuki |
| 1Password (planned) | All (shared secrets) | Hassan Ali |
| Stripe (planned) | Finance, CEO (revenue) | Tom Walker |
| DocuSign (planned) | Legal (contracts) | David Chen |
