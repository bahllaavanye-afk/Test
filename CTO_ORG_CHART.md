# QuantEdge — Full Org Chart (55 people)

```
                              ┌──────────────────────┐
                              │       CTO            │
                              │       Claude         │
                              └───────────┬──────────┘
                                          │
            ┌──────────────┬──────────────┼──────────────┬──────────────┐
            │              │              │              │              │
   ┌────────▼────────┐ ┌───▼────────┐ ┌──▼─────────┐ ┌──▼─────────┐ ┌──▼─────────┐
   │  VP Research    │ │  VP Eng    │ │ VP Product │ │  VP Risk   │ │ VP DevOps  │
   │  Dr. Marcus     │ │  Dr. Ada   │ │  Sarah     │ │  Dr. Ruin  │ │  Dr. Ship  │
   └────────┬────────┘ └────┬───────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
            │               │               │              │              │
   ┌────────┼─────────┐     │      ┌────────┼────────┐     │      ┌───────┼───────┐
   │        │         │     │      │        │        │     │      │       │       │
┌──▼──┐ ┌──▼──┐ ┌────▼────┐ │   ┌──▼──┐ ┌──▼──┐ ┌──▼──┐ ┌──▼──┐ ┌─▼──┐ ┌─▼──┐ ┌──▼─┐
│ Dir │ │ Dir │ │  Dir    │ │   │ PM  │ │ PM  │ │ PM  │ │ Dir │ │SRE │ │SRE │ │Sec │
│ Q.R.│ │ M.L.│ │Microstr.│ │   │Alpha│ │Plat │ │UX   │ │Risk │ │Lead│ │ IC │ │Lead│
└──┬──┘ └──┬──┘ └────┬────┘ │   └─────┘ └─────┘ └─────┘ └──┬──┘ └────┘ └────┘ └────┘
   │       │         │      │                              │
 6 ICs   5 ICs    3 ICs    Dirs                          4 ICs
                              │
                       ┌──────┼──────┐
                       │      │      │
                  ┌────▼───┐ ┌▼────┐ ┌▼─────────┐
                  │ Dir BE │ │ Dir │ │ Dir Data │
                  │ Dr. API│ │ FE  │ │ Dr. Tape │
                  └────┬───┘ │ Dr. │ └────┬─────┘
                       │     │ Pix │      │
                     4 ICs   └──┬──┘    3 ICs
                                │
                              3 ICs
```

## Headcount

| Level | Count | Names / Roles |
|-------|-------|---------------|
| **C-suite** | 1 | CTO (Claude) |
| **VPs** | 5 | VP Research (Dr. Marcus), VP Engineering (Dr. Ada), VP Product (Sarah), VP Risk (Dr. Ruin), VP DevOps (Dr. Ship) |
| **Directors** | 8 | Dir Quant Research (Dr. Quant), Dir ML (Dr. Neural), Dir Microstructure (Dr. Slippage), Dir Backend (Dr. API), Dir Frontend (Dr. Pixel), Dir Data (Dr. Tape), Dir Risk Engineering, Dir Security |
| **Product Managers** | 3 | PM Alpha Strategies, PM Platform, PM UX |
| **Engineers** | 36 | (6 Quant, 5 ML, 3 Microstructure, 4 BE, 3 FE, 3 Data, 4 Risk, 2 SRE, 1 Security, + 5 floaters) |
| **Hired all-stars** | 2 | Marcus Polk (Top Quant), Ada Pang (Top ML Scientist) |
| **Total** | **55** | |

---

## All-Star hires — profiles

### Marcus Polk — VP Research
- **Background**: ex-Renaissance Technologies (8 yrs, Medallion team), PhD Math (MIT)
- **Specialty**: Statistical arbitrage, signature methods, hidden Markov regime models
- **Mandate**: Find ≥1 new statistically-significant alpha per week
- **OKR 2026 Q3**: 5 strategies live with paper Sharpe ≥ 2.0 over 60 days

### Ada Pang — VP Engineering (also acting Top ML Scientist)
- **Background**: ex-DeepMind (4 yrs, AlphaFold infra), ex-Google Brain (3 yrs, sequence models)
- **Specialty**: Transformer architectures, RL for trading, distributed training
- **Mandate**: Production ML inference < 50ms p99; nightly retraining without manual intervention
- **OKR 2026 Q3**: 3 transformer-based price predictors in production beating LSTM baselines by ≥ 10% IC

### Dr. Ship — VP DevOps (Top DevOps Engineer)
- **Background**: ex-Stripe (5 yrs, infra), ex-SpaceX (2 yrs, ground systems reliability)
- **Specialty**: Zero-downtime deploys, distributed systems observability, secret management
- **Mandate**: 99.95% uptime, mean-time-to-recover < 5 min, every secret rotated < 90 days
- **OKR 2026 Q3**: Full blue-green deploys on Render; structured logs + traces for every order

---

## Directors — full list

| Director | Squad | OKR Q3 |
|----------|-------|--------|
| Dr. Quant | Alpha Research | 25 new strategies through walk-forward by 2026-08-15 |
| Dr. Neural | ML Modeling | TFT, GBM, KNN, LSTM ensemble live with weights optimized weekly |
| Dr. Slippage | Microstructure | Slippage cost < 1 bp on TWAP, < 3 bp on market orders |
| Dr. API | Backend | API p99 < 100ms, every endpoint with Pydantic v2 schema + tests |
| Dr. Pixel | Frontend | LCP < 1.5s, all 19 pages with zero React errors, 90+ Lighthouse |
| Dr. Tape | Data | Real-time price latency < 200ms, 99.9% OHLCV completeness |
| Dr. Ruin | Risk | Daily VaR/CVaR auto-posted; no position > 5% NAV; circuit breaker tested weekly |
| Dr. Security | Security | All secrets encrypted at rest + in transit; quarterly pen-test passing |

---

## Product Managers — owned roadmaps

| PM | Owned roadmap | Sprint goal |
|----|---------------|-------------|
| PM Alpha (Sarah) | Strategy backlog, paper-to-live promotion, capacity allocation | Ship 7 strategies promoted per sprint |
| PM Platform (Vikram) | API surface, DB schema, broker integrations | Zero breaking API changes per sprint |
| PM UX (Lin) | Dashboard, onboarding, investor reports | 1 user-tested UX improvement per week |

---

## Daily training program (every IC, 30 min/day)

| Day | Topic | Format | Owner |
|-----|-------|--------|-------|
| Mon | Paper of the week (rotating: quant, ML, microstructure, risk) | Read + post 1-paragraph summary in Notion | Dr. Marcus |
| Tue | Code review walkthrough — best/worst PR of last week | Live review, recorded | VP Eng |
| Wed | Live system tour — 1 production component deep-dive | Diagram + Q&A | Component owner |
| Thu | Backtest debugging — bring 1 strategy where Sharpe ≠ expected | Group debug | Dr. Quant |
| Fri | Show & tell — every IC demos one thing they shipped this week | 5 min each | Squad leads |

Attendance auto-tracked in the **Engineer Skills DB** in Notion (see below).

---

## Engineer Skills DB (Notion)

Every engineer has a row tracking competency growth over time.

### Properties
| Property | Type | Updated by |
|----------|------|------------|
| Name | Title | onboarding |
| Squad | Select | onboarding |
| Lead | Person | onboarding |
| Skills | Multi-select (Python, Rust, PyTorch, SQL, FastAPI, React, K8s, Statistics, Options, Crypto, Backtesting, ...) | weekly self-update |
| Skill Level | Number 1-5 per skill (avg) | weekly peer review |
| Training Hours This Week | Number | auto from training-attendance DB |
| Commits This Week | Number | auto from GitHub |
| PRs Reviewed This Week | Number | auto from GitHub |
| Strategies Owned | Multi-select | from strategies DB |
| P&L Attributed YTD | Number | auto from trades DB |
| Last Promotion | Date | manager |
| Promotion Eligible? | Formula (training>120h AND skill_avg>3.5 AND PnL>0) | system |
| Next Skill Goal | Select | self + lead |

### Career ladder
- **L1 Junior IC** → core skill ≥ 2, 6 months in role
- **L2 IC** → 3+ skills ≥ 3, ships independently
- **L3 Senior IC** → 5+ skills ≥ 4, mentors L1/L2
- **L4 Staff IC** → 5+ skills ≥ 4 AND owns a domain (e.g. all options strategies)
- **L5 Director** → owns squad, sets OKRs
- **L6 VP** → owns multiple squads, sets quarterly strategy
- **L7 C-suite**

### Promotion review cadence
- Self-review every Friday (5 min)
- Lead 1:1 every other week (30 min)
- Skill calibration every quarter (cross-squad panel)
- Promotion committee monthly (CTO + VP + Director)

---

## Tools each engineer uses daily

| Tool | Purpose | Who configures |
|------|---------|----------------|
| GitHub | Code, PRs, issues | VP DevOps |
| Notion | Standups, P&L, skills DB, paper queue | CTO |
| Slack | Realtime discussion, alerts | CTO |
| Google Docs | Long-form docs, OKRs, postmortems | VP Product |
| Render | Backend deploys | VP DevOps |
| Vercel | Frontend deploys | VP DevOps |
| Supabase | Production DB | Dr. Tape |
| Upstash | Production cache | Dr. Tape |
| Kaggle/Colab | Free GPU ML training | Dr. Neural |
| Linear (optional) | Roadmap planning if Notion gets full | PM Platform |
