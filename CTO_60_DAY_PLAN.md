# QuantEdge — CTO 60-Day Operating Plan
**Owner:** QuantEdge AI (CTO) · **Start:** 2026-05-28 · **End:** 2026-07-27
**North-star metric:** Daily P&L positive on at least 5 of 7 strategies, paper account growing ≥ 0.3% / day, ≥ 1 commit/team/day shipped.

---

## Org Structure — 8 squads, 1 lead each

| # | Squad | Lead | Headcount | Mission | Daily output |
|---|-------|------|-----------|---------|--------------|
| 1 | **Alpha Research** | Dr. Quant | 6 | Find new edges from papers, news, microstructure | 1 backtested strategy/day |
| 2 | **ML Modeling** | Dr. Neural | 5 | Train, tune, deploy ML predictors and ensembles | 1 model retrain + 1 HPO sweep/day |
| 3 | **Risk & Portfolio** | Dr. Ruin | 4 | Position sizing, drawdown caps, correlation limits, stress tests | Daily VaR/CVaR report |
| 4 | **Execution & Microstructure** | Dr. Slippage | 4 | TWAP/VWAP/limit-first, fill quality, smart routing | Daily slippage attribution |
| 5 | **Broker & Market Data** | Dr. Tape | 3 | Alpaca/Binance/Polymarket integrations, OHLCV pipelines | Live data SLA ≥ 99.5% |
| 6 | **Backend Platform** | Dr. API | 4 | FastAPI, DB, Redis, WebSocket, observability | Zero P0 incidents/day |
| 7 | **Frontend** | Dr. Pixel | 3 | React dashboard, charts, real-time UX | 1 UX improvement shipped/day |
| 8 | **DevOps & Release** | Dr. Ship | 2 | CI/CD, deploys, secrets, monitoring | Mean-time-to-deploy < 5 min |

Total: **31 engineers** (28 ICs + 8 leads, including CTO).

Each "engineer" is an autonomous agent following its CLAUDE.md role file. Squad leads also coordinate cross-squad reviews.

---

## Daily cadence (every weekday)

| Time (UTC) | Event | Owner | Artifact |
|------------|-------|-------|----------|
| 13:00 | Standup — each squad lead posts yesterday-shipped / today-planned / blockers | All leads | Notion daily-standup-YYYY-MM-DD page |
| 13:30 | CTO triage of standup blockers + priority overrides | CTO | Issue comments on blocked items |
| 14:00 | Squad deep work (3 hours uninterrupted) | All ICs | Commits, PRs |
| 17:00 | Strategy review — Alpha Research presents 1 new candidate, Risk vets | Alpha + Risk | Notion strategy-review-YYYY-MM-DD |
| 18:00 | Code review window — every open PR gets ≥1 review before merge | All ICs | PR review comments |
| 19:00 | Deploy window — anything merged before 18:30 ships to paper | DevOps | Deploy log |
| 21:00 | EOD P&L report + tomorrow-priorities — auto-posted | CTO | Notion daily-pnl-YYYY-MM-DD |

Weekends: research-only. No deploys. No new features. ML training runs allowed on GPU.

---

## 60-day milestones

### Sprint 1 (Days 1-7): Production foundation
- [x] All 47 strategies in `STRATEGY_REGISTRY`
- [x] 283 tests passing
- [x] Notion sync workflow ready
- [ ] Backend deployed to Render
- [ ] Frontend deployed to Vercel
- [ ] Supabase migrated, first 100 trades recorded
- [ ] Real-time dashboard shows P&L from live paper trades

### Sprint 2 (Days 8-14): Paper trading at scale
- 14-day paper run across all 47 strategies
- Daily P&L attribution by strategy posted to Notion
- Drop bottom-decile strategies (Sharpe < 0.7 OOS); enable top decile with 3× capital
- Target portfolio Sharpe ≥ 1.0

### Sprint 3 (Days 15-21): ML alpha unlock
- Train LSTM, XGBoost, TFT, Lorentzian KNN on Kaggle GPU
- Deploy artifacts; 7 ML-enhanced strategies generating signals
- Beat manual versions by ≥ +15% Sharpe on the comparison engine

### Sprint 4 (Days 22-28): Execution quality
- TWAP/VWAP slippage < 2 bps median
- Limit-first fill rate ≥ 80%
- Order book imbalance signals integrated into smart router

### Sprint 5 (Days 29-35): Risk hardening
- Monte Carlo flash-crash stress tests (–10% in 5 min) passed by all strategies
- Per-bucket drawdown breakers verified in paper
- Correlation cluster cap forces diversification when ρ > 0.7

### Sprint 6 (Days 36-42): Multi-asset expansion
- 20 new crypto symbols on Alpaca live feed
- 10 new options strategies on liquid SPY/QQQ options
- 5 Polymarket prediction-market arbs running

### Sprint 7 (Days 43-49): Live trading prep
- Audit: every order goes through risk_manager.check_order
- Audit: every secret is encrypted, never logged
- Compliance: KYC, FINRA Reg-T margin rules wired in
- Paper-to-live promotion flow with mandatory 14-day Sharpe ≥ 1.5

### Sprint 8 (Days 50-56): First live trade
- Single $1,000 live trade per strategy that passed promotion
- Daily live P&L vs. paper P&L tracking error ≤ 5%
- Slippage in live vs. backtest ≤ 3× expected

### Sprint 9 (Days 57-60): Scale-up
- Increase live capital to $25k across diversified strategy mix
- Daily investor-facing PDF report generation
- Public landing page goes live

---

## What gets tracked daily (Notion DBs)

| DB name | Properties | Updated by |
|---------|------------|------------|
| **Engineering Tasks** | Title, Status, Priority, Role, GitHub Issue, Sprint | All squads |
| **Daily Standups** | Date, Squad, Shipped, Planned, Blockers | Each lead, 13:00 UTC |
| **Strategy Reviews** | Date, Strategy, Sharpe, Maxdd, IC, Decision | Alpha lead, 17:00 UTC |
| **Daily P&L** | Date, Strategy, P&L, Win Rate, Slippage, Notes | CTO, 21:00 UTC |
| **Incidents** | Date, Severity, Component, Description, Resolution | DevOps, on-call |
| **ML Experiments** | Run ID, Model, Symbol, Val Sharpe, Test Sharpe, Status | ML lead |
| **Research Queue** | Paper, Expected Sharpe, Status, Assignee | Alpha lead |

---

## Promotion criteria (paper → live)

A strategy is eligible for live trading only when ALL hold:

1. Backtest: walk-forward Sharpe ≥ 1.0 over ≥ 5 years of data
2. Paper: 14 consecutive trading days with Sharpe ≥ 1.5
3. Risk: max drawdown in paper ≤ 8%
4. Correlation: pairwise ρ with any live strategy < 0.5
5. Slippage: realized vs. expected within 1.5×
6. Code review: 2 approvals (1 must be from Risk squad)
7. Capacity: estimated capacity ≥ 10× initial allocation
8. Kill switch tested: forced stop completes within 30 seconds

---

## On-call rotation (rotates weekly)

| Week | DevOps on-call | Risk on-call |
|------|----------------|--------------|
| 1 | Dr. Ship | Dr. Ruin |
| 2 | DevOps IC #1 | Risk IC #1 |
| 3 | DevOps IC #2 | Risk IC #2 |
| 4 | Dr. Ship | Dr. Ruin |

On-call carries the pager. Page on:
- P0 incident (P&L blocked, data feed down, risk breach)
- Failed deploy
- Strategy Sharpe drops below –0.5 over 5 days

---

## CTO weekly review (every Monday)

- Squad-by-squad summary: shipped vs. planned, blockers, headcount
- Top 3 winning strategies by P&L this week
- Top 3 losing strategies (kill or fix decision)
- ML model leaderboard
- Risk breaches and resolution
- Burn-down vs. sprint goal
- Next-week priorities — propagated to squad backlogs
