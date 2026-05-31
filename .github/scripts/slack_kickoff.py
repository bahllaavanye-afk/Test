"""
Slack kickoff — populate every channel with a realistic round of team chatter.

Posts as the QuantEdge bot, but each message is signed with the role/persona
that would have written it. Run this once after bootstrap, then run the
daily-chatter workflow on a schedule to keep channels active.

Required env:
    SLACK_BOT_TOKEN   xoxb-... with chat:write + channels:join + groups:write

Required scopes (also needed for posting in private channels):
    chat:write  channels:read  groups:read  channels:join
    (For private channels you also need to be invited OR have `groups:write`
     and post via conversations.join — we just send and let it fail soft.)
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.request
import urllib.error

# ── Realistic message scripts, by channel ─────────────────────────────────────
# Each script is a list of (persona, text) tuples. Personas mirror the
# 92-person org chart in CTO_ORG_FULL.md.

SCRIPTS: dict[str, list[tuple[str, str]]] = {
    # ── PUBLIC — engineering ops ──────────────────────────────────────────────
    "engineering-standup": [
        ("Maya Chen — VP Engineering",
         ":wave: standup thread — drop your yesterday/today/blockers in replies. Quiet days are fine; just say so."),
        ("Aarav Patel — Alpha Research Director",
         "yesterday: shipped TSMOM (#47), out-of-sample Sharpe 1.18 on SPY 2018-2024. today: starting Betting-Against-Beta. blockers: none."),
        ("Linh Tran — ML Modeling Lead",
         "yesterday: LSTM v3 trained on BTC/USDT 1h, val acc 0.612. today: SHAP analysis on top 20 features. blockers: need 30 GPU-hours for the TFT run, asking Kaggle."),
        ("Diego Ramírez — Execution Engineer",
         "yesterday: TWAP slicing fix landed — avg slippage down 3.4 bps on >$10k orders. today: VWAP participation-rate tuning. blockers: none."),
        ("Jian Wu — Risk Engineer",
         "yesterday: correlation cluster monitor live in prod. today: backtesting the new bucket allocation (70/30 → 60/30/10 with vol target). blockers: waiting on PnL attribution by bucket from data squad."),
        ("Priya Subramanian — Frontend Lead",
         "yesterday: Comparison page side-by-side chart shipped. today: code-splitting (505KB → target <300KB). blockers: none."),
    ],
    "alpha-research": [
        ("Aarav Patel — Alpha Research Director",
         ":memo: new strategy proposal: *Time-Series Momentum (Moskowitz/Ooi/Pedersen, JFE 2012)*. 12-month excess-return sign × inverse-vol sizing. Backtest 2018-2024 SPY-only: Sharpe 1.18, MaxDD 14%, t-stat 2.7."),
        ("Hugo Bernardes — Quant Researcher",
         "+1. Consider extending universe to QQQ, IWM, EFA, AGG, GLD — TSMOM original paper used 58 markets. The single-asset Sharpe drops in the cross section but the *diversified* Sharpe goes up."),
        ("Aarav Patel — Alpha Research Director",
         "good call. Adding to the queue. Hugo can you draft the multi-asset config? Use `experiments/configs/tsmom_multi_asset.yaml`."),
        ("Sofia Karlsson — VP Research",
         "Reminder: every alpha needs walk-forward validation BEFORE we merge. No in-sample-only results in `comparison_results`."),
        ("Hugo Bernardes — Quant Researcher",
         "Draft: <https://github.com/bahllaavanye-afk/QuantEdge/blob/main/experiments/configs/lstm_btc_1h.yaml|TSMOM multi-asset config>. WIP, will PR by Friday."),
    ],
    "pnl-daily": [
        ("CTO bot — automated",
         ":bar_chart: *EOD P&L — 2026-05-27*\n```\nstrategy                        trades  pnl       sharpe  dd\nmomentum (manual)                  14    +$  812.40   1.42  -2.1%\nml_momentum (LSTM-filtered)        11    +$1,204.10   2.06  -1.4%\npairs_trading                       6    +$  291.05   0.88  -1.9%\ntsmom (#47)                         3    +$  410.22   n/a    0%\ntriangular_arb (BTC/ETH/USDT)      42    +$   88.31   3.21  -0.3%\npoly_binary_arb                     8    +$  117.40   inf    0%\n────────────────────────────────────────────────────────────\nTOTAL                              84    +$2,923.48   1.91  -2.1%\n```"),
        ("Aarav Patel — Alpha Research Director",
         "ml_momentum > manual_momentum by 0.64 Sharpe today. 3-month rolling stays above. Recommend keeping ML filter on at 0.65 threshold."),
        ("Jian Wu — Risk Engineer",
         "DD on momentum bucket -2.1% — still well under -8% halt. No action."),
    ],
    "risk-alerts": [
        ("Risk bot — automated",
         ":warning: VaR (95%, 1-day) breach: portfolio level $-4,182 vs limit $-3,500. Cause: AAPL position size up 18% intraday on momentum signal. No circuit-breaker triggered (DD still inside -5% threshold)."),
        ("Jian Wu — Risk Engineer",
         "Investigating. The signal fired right before the FOMC announcement which inflated implied vol. Will rerun VaR with the higher vol regime model. Position remains open for now."),
        ("Marcus Olufemi — CRO",
         "Document the limit override in the audit log. If we override more than 2x/month we need to revisit the VaR window."),
    ],
    "incidents": [
        ("DevOps bot — automated",
         ":rotating_light: *P1 INCIDENT* — Render web service `quantedge-api` returned 503 for 47s starting 14:22 UTC. Recovered via automatic restart. Cause likely OOM."),
        ("Kenji Watanabe — Director of DevOps",
         "On it. Pulling logs. First-glance: the strategy_runner spawned 200+ asyncio tasks after a config reload; memory spiked from 380MB to 1.1GB."),
        ("Kenji Watanabe — Director of DevOps",
         "Fix in `342fe1d` — cap task spawning at 100 concurrent loops, add a semaphore. Postmortem doc started: <https://docs.google.com/document/d/incident-2026-05-27>. ETA: full RCA by EOD tomorrow."),
        ("Maya Chen — VP Engineering",
         "Thanks. Add this to the runbook and the chaos-test list."),
    ],
    "deploys": [
        ("Deploy bot — automated",
         ":rocket: *Backend deploy* `quantedge-api`\nRef: `da2f4cb` security: cross-user data leak fixes\nMigrations: none\nDuration: 1m 42s\nStatus: ✅ healthy"),
        ("Deploy bot — automated",
         ":rocket: *Frontend deploy* `quantedge-web`\nRef: `c7a346b` Slack diagnostic enhancements\nBundle: 487KB → 472KB (-3.1%)\nLighthouse perf: 91 → 93\nStatus: ✅ live at https://quantedge.vercel.app"),
    ],
    "ci-failures": [
        ("CI bot — automated",
         ":x: *CI failed* on `claude/advanced-trading-bot-d5Lmw` — commit `542bf1d`\nJob: bootstrap\nReason: Slack scope missing (`missing_scope`, needed `channels:read,groups:read`)\n<https://github.com/bahllaavanye-afk/QuantEdge/actions/runs/26572364509|View run>"),
        ("Aditi Sharma — Director of QA",
         "Not a code bug — Slack app needed scopes. Resolved at 11:48 with reinstall + new bot token. Tracking in #squad-security."),
    ],
    "ml-experiments": [
        ("Linh Tran — ML Modeling Lead",
         "LSTM v3 BTC/USDT 1h — done.\nval_acc: 0.612, val_sharpe: 1.84\ntest_sharpe (2024-Q4): 1.41\nlogged: `experiments/results/lstm_btc_1h_v3.json`"),
        ("Tomas Lindqvist — Research Scientist",
         "v3 is +0.27 Sharpe vs v2. Main delta was adding cross-asset features (btc_dom, eth_btc_ratio). SHAP shows btc_dom is the #2 most important after RSI-14."),
        ("Linh Tran — ML Modeling Lead",
         "Going to v4 with these features + a 3-layer attention block. Spinning up Kaggle T4 now. Should finish in ~25 min."),
        ("Sofia Karlsson — VP Research",
         "Make sure v4 retains the walk-forward 6-fold validation. Cherry-picking a single test split is how all academic papers lie. ;)"),
    ],
    "engineering": [
        ("Maya Chen — VP Engineering",
         ":wave: Welcome to QuantEdge engineering. We're a 92-person org running 47 strategies across equities, crypto, options, and prediction markets. Read the root `CLAUDE.md` and pick a squad."),
        ("Diego Ramírez — Execution Engineer",
         "PSA: Smart Order Router now defaults to LimitFirst for crypto buys with a 5bps offset and 30s timeout. Saving ~7bps per order vs market on average. Old behavior is still available via `execution_algo=market` override."),
        ("Priya Subramanian — Frontend Lead",
         "PSA #2: pages are React.lazy()-loaded now. First paint dropped from 1.4s → 0.6s. If you add a new page, follow the pattern in `App.tsx`."),
    ],
    "announcements": [
        ("Laavanye Bahl — CEO/Founder",
         ":mega: *Welcome to QuantEdge.* We just bootstrapped the entire org Slack — 60+ channels, 92 roles. The plan: ship a profitable, multi-broker, ML-enhanced quant platform that retail can trust and institutions will copy.\n\nWeekly all-hands: *Monday 14:00 UTC* in #leadership-summary (recap posted here).\nMonthly board: *first Tuesday* in #board.\nFounders day-1 principle: every strategy needs walk-forward proof. No exceptions."),
        ("Laavanye Bahl — CEO/Founder",
         "Public-relations note: we are *paper-first*. Every new strategy must run on Alpaca paper for 2 weeks before live capital is allocated. CFO + CRO sign-off required for live activation."),
    ],
    "wins": [
        ("Aarav Patel — Alpha Research Director",
         ":tada: TSMOM (#47) backtest came in: Sharpe 1.18, MaxDD 14%, t-stat 2.7 on SPY 2018-2024. Multi-asset version goes to walk-forward next week."),
        ("Linh Tran — ML Modeling Lead",
         ":tada: LSTM v3 +0.27 Sharpe over v2 on BTC. Cross-asset features (btc_dom, eth_btc_ratio) carried the lift."),
        ("Diego Ramírez — Execution Engineer",
         ":tada: TWAP slicing fix → -3.4 bps avg slippage on >$10k orders. Compounded over a year that's worth ~6% on the high-turnover bucket."),
        ("Kenji Watanabe — Director of DevOps",
         ":tada: zero unplanned downtime in last 14 days on the paper-trading stack."),
    ],
    "help": [
        ("Karl Nyström — Junior IC",
         "How do I add a new manual strategy? Do I need to write both `analyze()` and `backtest_signals()`?"),
        ("Aarav Patel — Alpha Research Director",
         "Yes — `analyze()` for live, `backtest_signals()` for VectorBT. Both should return signals at most once per bar and use `.shift(1)` to avoid lookahead. Template: `backend/app/strategies/manual/momentum.py` is the cleanest reference."),
        ("Karl Nyström — Junior IC",
         "Got it, thanks. Will follow that pattern."),
    ],

    # ── PUBLIC — market desks ─────────────────────────────────────────────────
    "desk-equities": [
        ("Aarav Patel — Alpha Research Director",
         ":chart_with_upwards_trend: Equity desk open. Today's universe: 50 top S&P names by volume + sector ETFs. Active strategies: momentum, low_vol, pairs_trading, tsmom, options_pcr_reversal."),
        ("Hugo Bernardes — Quant Researcher",
         "NVDA momentum signal fired at 14:32 — entry $1,142, stop $1,118. ML filter confidence 0.71, above threshold."),
        ("Diego Ramírez — Execution Engineer",
         "Filled at $1,141.83 via LimitFirst, 2 bps slippage. Decent."),
    ],
    "desk-crypto": [
        ("Linh Tran — ML Modeling Lead",
         ":chart: Crypto desk: BTC/USDT, ETH/USDT, SOL/USDT live on Alpaca crypto. Binance perps used only for funding-arb + liquidation-cascade fades."),
        ("Hugo Bernardes — Quant Researcher",
         "ETH funding turned negative (-0.0042% / 8h) — funding_rate_arb signal active, paper-long 0.5 ETH."),
        ("Jian Wu — Risk Engineer",
         "Cap the crypto bucket at 8% of NAV until walk-forward >6mo. Even arb has tail risk on exchange downtime."),
    ],
    "desk-options": [
        ("Yuki Mori — Options Researcher",
         ":eyes: PCR (put-call ratio) on QQQ at 1.23, 90th percentile. Mean-reversion signal armed. Will enter long QQQ delta-neutral via short ATM puts if PCR > 1.35 by close."),
        ("Aditi Sharma — Director of QA",
         "Reminder: backtest uses RSI(2) proxy for live PCR — make sure the live path is using `_fetch_alpaca_options_pcr()` and not the proxy."),
        ("Yuki Mori — Options Researcher",
         "Confirmed, live path verified. Proxy is backtest-only."),
    ],
    "desk-polymarket": [
        ("Lior Avraham — Polymarket Researcher",
         ":bar_chart: Today's high-volume markets: 2026-fed-cut-june, btc-100k-by-2027, recession-2026-q2. Scanning for YES+NO <$0.97 arb."),
        ("Lior Avraham — Polymarket Researcher",
         "Found: `will-fed-cut-by-q3` — YES $0.62 + NO $0.34 = $0.96. Filling both legs for $250."),
    ],
    "desk-fx-rates": [
        ("Tomas Lindqvist — Research Scientist",
         "FX + rates is paper-only until we have a real broker connection. For now we're just simulating EUR/USD + 2yr/10yr curve trades against FRED data."),
    ],

    # ── PUBLIC — feeds ────────────────────────────────────────────────────────
    "news-feed": [
        ("News bot — automated",
         ":newspaper: *Reuters* — Fed minutes signal pause through Q3. 2yr yield -3bps."),
        ("News bot — automated",
         ":newspaper: *Bloomberg* — NVDA reports Q1 earnings Thursday after close. Implied move 7.2%."),
        ("News bot — automated",
         ":newspaper: *CoinDesk* — Binance reports record perp open interest. Funding skewing short."),
    ],
    "earnings-watch": [
        ("Hugo Bernardes — Quant Researcher",
         "NVDA earnings Thursday 16:00 ET. Implied move 7.2% (straddle priced at $80). Historical avg actual move 5.4% — short premium edge if vol surface stays elevated."),
        ("Yuki Mori — Options Researcher",
         "I'd structure as a put credit spread, not a naked short straddle — earnings have tail risk that breaks short-vol PnL."),
    ],
    "fed-watch": [
        ("Tomas Lindqvist — Research Scientist",
         "Next FOMC: 2026-06-18. Fed futures pricing 84% no-change, 16% 25bp cut. Our macro regime classifier still in *late-cycle stable* — no bucket rebalance needed."),
    ],
    "papers": [
        ("Sofia Karlsson — VP Research",
         ":books: *new paper queue* — top 3 for this week:\n1. _Lim et al. 2021_ — Temporal Fusion Transformer (we have impl, want walk-forward results)\n2. _Frazzini & Pedersen 2014_ — Betting Against Beta (queued for Aarav)\n3. _Asness, Moskowitz, Pedersen 2013_ — Value & Momentum Everywhere (cross-asset extension to TSMOM)"),
        ("Aarav Patel — Alpha Research Director",
         "I'll take Frazzini-Pedersen for next sprint. Expected Sharpe 0.9 single-asset, 1.4 multi-asset per the paper."),
    ],
    "competitors": [
        ("Maya Chen — VP Engineering",
         "Notes from looking at Two Sigma's open positions filings + LinkedIn hiring patterns:\n- they're shifting headcount to alternative data (satellite, credit-card transaction)\n- their public funds underperformed QQQ in 2024 — quant winter narrative is real\n- we should NOT compete on alt-data spend; compete on agility + multi-broker coverage"),
    ],
    "external-research": [
        ("Sofia Karlsson — VP Research",
         "GS quant note: factor crowding in mom + value at decade-high. They recommend reducing factor exposure and rotating to single-stock alphas. Worth noting — our momentum book is 11% of NAV, on the lower end."),
    ],

    # ── PUBLIC — culture ──────────────────────────────────────────────────────
    "random": [
        ("Karl Nyström — Junior IC",
         "anyone else watch the F1 quali? Verstappen looked beatable."),
        ("Diego Ramírez — Execution Engineer",
         "the slippage on his last lap was wild though"),
        ("Priya Subramanian — Frontend Lead",
         "lol everything's a quant joke now"),
    ],
    "book-club": [
        ("Sofia Karlsson — VP Research",
         "This month: *Advances in Financial Machine Learning* by Marcos López de Prado. Discussion next Thursday 17:00 UTC. Focus: chapter 7 (cross-validation in finance, why k-fold fails)."),
        ("Aarav Patel — Alpha Research Director",
         "Chapter 7 is foundational. If you've ever done a backtest without purging the train/test boundary, this chapter will haunt you."),
    ],
    "culture": [
        ("Laavanye Bahl — CEO/Founder",
         "Core principles, in order of priority:\n1. *Risk-first*. No live capital without 2 weeks paper + CRO sign-off.\n2. *Walk-forward only*. In-sample backtests are not evidence.\n3. *No mock data*. Better crash than mock.\n4. *Show your work*. Every strategy ships with its config + backtest + paper trail.\n5. *Modular*. No cross-strategy or cross-broker coupling."),
    ],
    "show-and-tell": [
        ("Priya Subramanian — Frontend Lead",
         "Friday demo: I'll show the new Comparison page — manual vs ML vs SPY/BRK.B/All-Weather, side by side. 10 minutes, bring questions."),
    ],

    # ── PUBLIC — process ──────────────────────────────────────────────────────
    "okrs": [
        ("Maya Chen — VP Engineering",
         "Q2 OKRs locked:\n1. Ship 50 strategies live on paper (47/50, 3 left)\n2. Sharpe > 2.0 on the diversified portfolio (current: 1.91)\n3. Max drawdown < 15% (current: -8.4%)\n4. Frontend bundle <300KB gzipped (current: 472KB → working on it)\n5. Zero P0 incidents (current: 0)"),
    ],
    "hiring": [
        ("Maya Chen — VP Engineering",
         "Open roles posted to the careers page:\n- Senior Quant Researcher (Options) — 1\n- ML Infra Engineer (Ray/Lightning) — 1\n- DevOps SRE (Render + Vercel + Supabase) — 1\n- Compliance Engineer (KYC + audit trail) — 1"),
    ],
    "postmortems": [
        ("Kenji Watanabe — Director of DevOps",
         "Postmortem: *2026-05-27 OOM incident*. RCA in <https://docs.google.com/document/d/incident-2026-05-27|gdoc>. Root cause: unbounded asyncio task spawn on config reload. Fix landed in `342fe1d`. Action items: 1) chaos-test config reloads under load, 2) Render memory alert at 80%, 3) update runbook."),
    ],
    "security-alerts": [
        ("Sec bot — automated",
         ":closed_lock_with_key: GitHub secret scanning: 0 leaked secrets in last 24h."),
        ("Sec bot — automated",
         ":lock: Dependabot: 2 medium-severity advisories on `aiohttp`. Auto-PR opened."),
    ],
    "infra-alerts": [
        ("Infra bot — automated",
         ":green_heart: Render `quantedge-api` — up 14 days. p99 latency 73ms.\n:green_heart: Vercel `quantedge-web` — last deploy 2h ago, lighthouse 93/100.\n:green_heart: Supabase — connections 12/60, query p99 18ms.\n:green_heart: Upstash Redis — ops/sec 244, hit rate 96%."),
    ],

    # ── PRIVATE — squads (sample chatter — agents will post more later) ───────
    "squad-alpha-research": [
        ("Aarav Patel — Alpha Research Director",
         "Squad sync — agenda:\n1. TSMOM multi-asset config review (Hugo)\n2. BAB next-up assignment (Aarav)\n3. Lorentzian KNN paper-trading results (Tomas)"),
    ],
    "squad-microstructure": [
        ("Diego Ramírez — Execution Engineer",
         "TWAP slicing fix landed. Next: implement Almgren-Chriss optimal liquidation for orders >$50k."),
    ],
    "squad-ml-modeling": [
        ("Linh Tran — ML Modeling Lead",
         "LSTM v4 spinning up. TFT next sprint. Need someone to own the PPO RL execution model — Tomas, interested?"),
    ],
    "squad-ml-infra": [
        ("Ravi Iyer — ML Infra Engineer",
         "Kaggle GPU quota refreshed Monday — that's 30 T4 hours. I'll queue the LSTM + TFT + Lorentzian training runs in that order."),
    ],
    "squad-backend": [
        ("Anna Hoffmann — Backend Lead",
         "Security fixes landed in `da2f4cb` — cross-user leak on /trades, /analytics. CORS hardened. Rate-limit on /auth/register. Worth a re-audit of every endpoint that takes a query param."),
    ],
    "squad-frontend": [
        ("Priya Subramanian — Frontend Lead",
         "React.lazy() + Suspense for every route. Bundle 472KB → expected 280KB after split. PR coming today."),
    ],
    "squad-data": [
        ("Sina Hassani — Data Engineer",
         "OHLCV ingestion: 28 equity symbols + 12 crypto pairs flowing into Supabase. p95 ingestion lag 4.1s. Need to add ES futures next."),
    ],
    "squad-execution": [
        ("Diego Ramírez — Execution Engineer",
         "Smart router now defaults: LimitFirst for buys, TWAP for >$10k, market only with explicit `urgency=immediate`."),
    ],
    "squad-risk": [
        ("Jian Wu — Risk Engineer",
         "Correlation cluster live. Daily VaR computed at 23:55 UTC. Bucket allocation 70/30 stable."),
    ],
    "squad-security": [
        ("Cameron Park — Security Engineer",
         "Bot token rotation policy: bot tokens > 30 days old auto-flagged. Slack token from 11:48 today will be due for rotation by 2026-06-27."),
    ],
    "squad-devops": [
        ("Kenji Watanabe — Director of DevOps",
         "UptimeRobot pings the /health every 5min on the free Render tier — no more cold-start latency."),
    ],
    "squad-qa": [
        ("Aditi Sharma — Director of QA",
         "Test count: 283/283 passing. CI on PR #9 still flaky — investigating heavy-import startup time."),
    ],
    "squad-compliance": [
        ("Helena Voss — Compliance Engineer",
         "KYC docs templated for top 8 jurisdictions. Trading licenses tracker in <https://docs.google.com/document/d/trading-licenses|gdoc>."),
    ],
    "squad-finance-eng": [
        ("Wei Chang — Finance Engineer",
         "Cash burn modeling live. Runway calc: 17 months at current $7.2k/mo burn (Render+Vercel free, Supabase free, Upstash free, Alpaca commission-free)."),
    ],

    # ── PRIVATE — leadership ──────────────────────────────────────────────────
    "leadership": [
        ("Laavanye Bahl — CEO/Founder",
         "VP+ sync: priorities for the next 30 days:\n1. Get to live trading on paper-validated strategies (CTO + CRO own)\n2. First investor letter draft (CEO + CFO own)\n3. Hire 4 open roles (VP Eng owns)\n4. Compliance gap analysis for live trading (GC owns)"),
        ("Maya Chen — VP Engineering",
         "On 1: TSMOM + ml_momentum + options_pcr are paper-ready. Need CRO sign-off by Friday."),
        ("Marcus Olufemi — CRO",
         "Will review the 2-week paper trail this week and sign off Friday EOD."),
    ],
    "leadership-summary": [
        ("Maya Chen — VP Engineering",
         "*Daily summary — Engineering*\n- shipped: TSMOM strategy #47, security fixes (#9), Slack bootstrap (60+ channels)\n- in flight: code-splitting, LSTM v4 training, Render deploy\n- blocked: nothing today"),
        ("Sofia Karlsson — VP Research",
         "*Daily summary — Research*\n- shipped: TSMOM backtest, LSTM v3 results\n- in flight: Frazzini-Pedersen BAB implementation, TFT walk-forward\n- blocked: need 30 GPU-hours for TFT (Kaggle quota OK)"),
        ("Marcus Olufemi — CRO",
         "*Daily summary — Risk*\n- shipped: correlation cluster monitor\n- in flight: VaR review under elevated-vol regime\n- blocked: need bucket PnL attribution from data squad"),
    ],
    "board": [
        ("Laavanye Bahl — CEO/Founder",
         "Board, welcome to the comms channel. Monthly board deck drops here first Tuesday of every month. Real-time questions: tag @ceo + @cfo."),
    ],
    "pm-coordination": [
        ("Rohan Mehta — PM, Trading Surfaces",
         "Cross-PM sync — agenda:\n1. Comparison page launch — who owns the investor demo?\n2. Polymarket UI sprint sequencing\n3. Mobile-responsive cuts for Q3"),
    ],

    "cxo-direct": [
        ("Laavanye Bahl — CEO/Founder",
         "C-level direct. Use for time-sensitive decisions only — daily flow goes through the public squads."),
    ],
    "board-prep": [
        ("Laavanye Bahl — CEO/Founder",
         "Board deck draft v0.1 in <https://docs.google.com/presentation/d/2026-06-board-deck|gslides>. CFO own slides 4-7 (financials), CTO own slides 8-12 (engineering velocity)."),
    ],
    "investor-updates": [
        ("Laavanye Bahl — CEO/Founder",
         "May investor letter draft started. Hits: TSMOM walk-forward, ml_momentum +27% Sharpe over manual. Misses: live trading still gated on compliance review."),
    ],
    "legal-compliance": [
        ("Helena Voss — Compliance Engineer",
         "Audit-trail spec finalized. Every order, fill, login, risk event logged with user_id + timestamp in `audit_logs` table. 7-year retention via Supabase logical backup."),
    ],
    "finance-ops": [
        ("Wei Chang — Finance Engineer",
         "Monthly burn breakdown:\n- Render web + worker: $0 (free tier)\n- Vercel: $0 (Hobby)\n- Supabase: $0 (free tier)\n- Upstash: $0 (free tier)\n- Domain: $12/yr\n- One-off compute (Kaggle/Colab/Lightning): $0 (free)\nTotal: ~$1/mo. Adjust when AUM > $100k."),
    ],

    # ── PRIVATE — pods ────────────────────────────────────────────────────────
    "pod-equity-momentum": [
        ("Aarav Patel — Alpha Research Director",
         "Pod focus: 12-1 momentum (Jegadeesh-Titman), TSMOM (Moskowitz et al.), residual momentum (Blitz et al.). Next paper: Asness et al. 2013 — value & momentum everywhere."),
    ],
    "pod-equity-meanrev": [
        ("Hugo Bernardes — Quant Researcher",
         "Bollinger mean-rev on top 50 by volume. Current Sharpe 0.6, looking to add ML filter (XGBoost classifier on next-day return sign)."),
    ],
    "pod-options-vol": [
        ("Yuki Mori — Options Researcher",
         "PCR mean-reversion working. Next: realized vs implied vol cone, fit GARCH on realized + plot vs IV30."),
    ],
    "pod-crypto-perp": [
        ("Tomas Lindqvist — Research Scientist",
         "Funding-rate-arb live on Binance perps. Liquidation-cascade fade signal armed. Need to add Coinbase perps once API access lands."),
    ],
    "pod-poly-arb": [
        ("Lior Avraham — Polymarket Researcher",
         "YES+NO < $0.97 scanning. Need to add cross-market correlation pairs (e.g. recession-2026 ↔ fed-cut-q3)."),
    ],
    "pod-execution-tca": [
        ("Diego Ramírez — Execution Engineer",
         "TCA report v1 ships Friday. Will show realized vs expected, broken down by algo + size bucket + symbol bucket."),
    ],
    "pod-ml-features": [
        ("Linh Tran — ML Modeling Lead",
         "Cross-asset features (btc_dom, eth_btc_ratio, vix_term, dxy) boosted LSTM Sharpe by +0.27. Adding: yield curve slope, gold/copper ratio, breakeven inflation."),
    ],
    "pod-ml-models": [
        ("Linh Tran — ML Modeling Lead",
         "Roster: LSTM (live), XGBoost (live), Lorentzian KNN (live), TFT (training), PPO RL (training)."),
    ],
    "pod-ml-rl": [
        ("Tomas Lindqvist — Research Scientist",
         "PPO RL execution agent on 1h BTC. Reward = -slippage_bps - commission_bps. Training on Kaggle, ETA Friday."),
    ],
}


def slack_call(token: str, method: str, payload: dict) -> dict:
    url = f"https://slack.com/api/{method}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"http_{e.code}", "body": e.read().decode()[:200]}


def list_channels(token: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    cursor = ""
    while True:
        payload: dict = {"types": "public_channel,private_channel", "limit": 200}
        if cursor:
            payload["cursor"] = cursor
        data = slack_call(token, "conversations.list", payload)
        if not data.get("ok"):
            print(f"⚠ conversations.list failed: {data}")
            break
        for ch in data.get("channels", []):
            out[ch["name"]] = ch
        cursor = data.get("response_metadata", {}).get("next_cursor", "")
        if not cursor:
            break
    return out


def post(token: str, channel_id: str, persona: str, text: str) -> dict:
    return slack_call(token, "chat.postMessage", {
        "channel": channel_id,
        "text": f"*{persona}*\n{text}",
        "username": persona.split(" — ")[0] if " — " in persona else persona,
        "mrkdwn": True,
    })


def main() -> int:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token.startswith("xoxb-"):
        print("❌ SLACK_BOT_TOKEN missing or not xoxb-")
        return 1

    auth = slack_call(token, "auth.test", {})
    if not auth.get("ok"):
        print(f"❌ auth.test failed: {auth}")
        return 1
    print(f"✅ Authed as {auth.get('user')} in {auth.get('team')}")

    channels = list_channels(token)
    print(f"ℹ Found {len(channels)} channels in workspace")

    posted, missing, errors = 0, [], []

    for ch_name, script in SCRIPTS.items():
        ch = channels.get(ch_name)
        if not ch:
            missing.append(ch_name)
            continue
        ch_id = ch["id"]

        # Try to join public channels (bot must be a member to post)
        if not ch.get("is_private", False):
            join_result = slack_call(token, "conversations.join", {"channel": ch_id})
            if not join_result.get("ok") and join_result.get("error") not in (
                "already_in_channel", "method_not_supported_for_channel_type"
            ):
                print(f"  ⚠ #{ch_name}: join failed: {join_result.get('error')}")

        print(f"\n📨 #{ch_name} ({len(script)} messages)")
        for persona, text in script:
            r = post(token, ch_id, persona, text)
            if r.get("ok"):
                posted += 1
                print(f"   ✓ {persona[:40]}")
            else:
                err = r.get("error", "unknown")
                if err == "not_in_channel":
                    print(f"   ⚠ {persona[:40]} — not_in_channel (private, bot not invited)")
                else:
                    errors.append({"channel": ch_name, "persona": persona, "error": err})
                    print(f"   ✗ {persona[:40]} — {err}")
            # Light pacing — Slack rate limit is generous but be polite
            time.sleep(0.4)

    print(f"\n{'='*60}")
    print(f"✅ Posted:         {posted} messages")
    print(f"⚠ Missing chans:   {len(missing)}  ({', '.join(missing[:8])}{'…' if len(missing)>8 else ''})")
    print(f"❌ Errors:         {len(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
