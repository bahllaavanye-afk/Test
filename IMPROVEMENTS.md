# QuantEdge — Improvements & Task Tracker

## 🚨 MONDAY 2026-07-20 POST-MORTEM — "all desks bad, 0 trades, OA doesn't work" (diagnosed + fixed 2026-07-21)
Evidence-based, three stacked causes:
1. **[FIXED] Loss cap froze the entire book all session** — every market-hours desk run logged `🛑 DAILY LOSS CAP: equity down 2.72% vs prior close (cap 2%)`. The cap compares to Alpaca `last_equity` = FRIDAY's close, so weekend crypto drift on existing positions tripped it before Monday even opened — and it blocked ALL orders including exits (couldn't add, couldn't de-risk). Desks were otherwise perfect: 410 signals, DIA/JNJ/GLD/EWT at conf 1.00. FIX: under the cap, risk-REDUCING orders stay allowed (`is_risk_reducing` vs live Alpaca positions, fail-strict on fetch error); only new exposure is blocked; cap state surfaced in the run log + Discord funnel line. 6 tests.
2. **[FIXED] Bots never evaluated (all 61 last_run_at=None)** — APScheduler interval jobs wait one FULL interval before the first run, and every merge→deploy wipes the ephemeral SQLite + restarts the app, resetting that clock. 1h/1d bots never got to run. FIX: `next_run_time` = boot + 30–150s stagger, so every bot evaluates within ~2 min of every deploy.
3. **[USER ACTION — the remaining one] Bot trades are DB-only and die with every deploy** on the SQLite fallback (desk trades survive because they re-sync from Alpaca 30d). Durable bot P&L requires unpausing Supabase (schema-drift-gate is DONE — the catch-up migration `k6f7a8b9c0d1` for slippage_records IS fields shipped 2026-07-22 and applies automatically on the first boot that reaches Postgres). STATUS 2026-07-22: catch-up migration landed on main (PR #878, af42dc8 live — verified via the new `/health/detailed` scheduler job-table). BUT the restored-Supabase reconnection did NOT hold — the fresh af42dc8 boot STILL fell back to SQLite: `database_primary: (ENOTFOUND) tenant/user postgres.vexzwnfbmznvxoxxktax not found` (Supavisor "tenant not found" = the project is paused again / restore reverted). The Supabase MCP tools were NOT available this session to re-restore. **USER ACTION: supabase.com/dashboard → the `vexzwnfbmznvxoxxktax` project → Restore/Unpause.** The moment it's reachable, the next boot binds Postgres + applies `k6f7a8b9c0d1` cleanly (schema is now in sync). Until then bot activity is visible between deploys but resets on each merge.
- [ ] **[P2] Loss-cap window redesign** — `last_equity` spans the whole weekend for a 24/7 crypto book; measure vs a session anchor (portfolio-history API) instead. Needs validation — threshold semantics change.


## 🌡️ REGIME-AWARE STRATEGY SELECTION (user directive 2026-07-20: "different market conditions → different strategies")
EXISTS today: `_detect_regime_from_bars` (SPY trend → bear/sideways/bull) +
`_STRATEGY_REGIME_MAP` gating in desk_order_placer. Gaps: trend-only (no volatility
axis — the dimension that decides options premium selling vs buying), unmapped
strategies default to ALL regimes, and the active regime is invisible.
- [~] **[P1] Add the volatility axis** — SHIPPED (partial) 2026-07-20: `_detect_vol_regime_from_bars` (calm/stressed from SPY realized-vol ratio, free); premium sellers (all _MLEG_STRUCTURES + vol_carry) face a +0.08 confidence bar in CALM vol (thin premium — not a hard block, income desk still trades); vol shown in run log + funnel summary (`regime=trend/vol`). 6 tests. Remaining: full 6-cell trend×vol map for long-premium/trend/mean-reversion (needs backtest validation before hard gating). — realized-vol percentile on SPY bars (free, no new feed) → calm/stressed; 3×2 = 6 regimes. Map options income (condors/credit spreads/0DTE-VRP) to STRESSED-elevated-IV regimes, long-premium + trend-followers to CALM-trending, mean-reversion to SIDEWAYS-calm. Audit every default-mapped strategy into an explicit cell.
- [~] **[P1] Make the regime visible + felt** — bot `regime` condition SHIPPED 2026-07-20 (detect_regime helper: trend×vol; ConditionConfig.regimes; 7 tests). Remaining: Discord regime-transition post + funnel regime-gated line. — post regime transitions to Discord (with the SPY/vol chart via discord_post_chart); per-run funnel post shows which strategies were regime-gated; bot engine gets a `regime` condition type so OA-style bots can say "only in bull-calm" (same pattern as ml_signal).
- [ ] **[P2] Regime-aware exploration** — the 5% exploration budget (below) rotates strategies within their FITTING regime, so accrued evidence is regime-conditional (a mean-reverter judged in a trend is bad science).

## 🎯 STRATEGY DIVERSITY (user directive 2026-07-20: "trades across strategies irrespective of market confusion")
Evidence: dry run funnel 49 signals → conf≥0.6 gate → 34 conflict stand-asides → top-K
→ 5 survivors; live fills concentrate in 1–2 strategies. Consequence: the pruning/
promotion loop (needs ≥20 live trades per strategy) is EVIDENCE-STARVED for ~95% of the
book — most strategies can never be judged. Fix = exploration, not looser risk:
- [x] **[P1] Exploration allocation in desk_order_placer** — SHIPPED 2026-07-20 (regime-gated, 0.45 noise floor, min-notional clips, daily rotation, pruned strategies excluded; funnel now reports `explored=`; coid tagging deferred to keep attribution parsing intact) — reserve ~5% of each desk's budget for MIN-NOTIONAL clips ($10–25) allocated to wirable strategies with the FEWEST live fills in 30d (rotate round-robin; skip pruned-to-zero strategies; conflict stand-asides still apply within a symbol). Every strategy accrues real stats → pruning/promotion becomes decisive instead of starved. Tag fills `qe-explore-*` so attribution can report exploration P&L separately from conviction P&L.
- [x] **[P2] Funnel telemetry post** — SHIPPED 2026-07-20: desk run summary now carries `regime=` + `funnel: N generated → M survived gate+topK (K exploration) → P placed`, so "why so few trades" is visible in the #pnl-daily post (no extra message — enriched the existing one). — after each desk run, post the funnel (generated → gated → conflicted → topK → placed, per desk) as a Discord chart so "why no trades" is visible at a glance, not a mystery.

## 🔬 SOTA RESEARCH SWEEP 2026-07-20 (web-grounded; apply-not-cite items)
Multi-agent trading (arXiv 2412.20138 TradingAgents; FinCon; HedgeAgents; ContestTrade 2508.00554; eval taxonomy 2603.27539):
- [ ] **[P1] TradingAgents-style role separation for the debate gate** — our queued bull/bear/judge debate should mirror the proven analyst→researcher→trader→risk-manager pipeline: analysts summarize (cacheable), researchers debate, trader proposes, risk manager holds VETO. Fits llm_common cascade; run pre-order for large notionals only (cost-aware — the 2603.27539 finding: coordination structure matters more than model size).
- [ ] **[P2] ContestTrade-style internal contest** — score each desk-strategy's LIVE hit-rate weekly and allocate the top-K budget by contest rank (we already have perf weighting/pruning; add the contest layer to the top-K selection itself).
- [x] **[P2] Look-ahead hygiene** — AUDITED + GUARDED 2026-07-20. Finding: the vectorized engine is look-ahead-FREE (`position=signal.shift(1)`; P&L = position × forward bar-return, so a position is only earned the bar AFTER its signal); options_synthetic is clean too (entry priced off entry-bar data available at entry, exit re-priced at the actual future spot = the realized outcome). Locked in by test_backtest_no_lookahead.py (4 tests: same-bar spike NOT captured; pre-spike signal IS; flat→0; earns only bar-after). SEPARATE conservative finding (NOT shipped — needs A/B validation): strategies `.shift(1)` features AND the engine shifts → a 2-bar execution lag that UNDERSTATES returns; "fixing" it raises reported returns and could promote overfit strategies, so it stays as a validated-change item.
Options income (0DTE VRP evidence 2016→2026 positive+significant; deep-learning options 2407.21791):
- [ ] **[P1] 0DTE variance-risk-premium desk mode** — evidence: implied > realized variance holds for 0DTE SPX through 2026. Our iron-condor/credit-spread templates already exist; add a 0DTE defined-risk variant sized off expected-move, with the documented practice of LETTING expire (avoids 2–5% spread-crossing close cost) under the Expiration Protocol item (do that first).
- [ ] **[P2] End-to-end learned options signals** — 2407.21791 shows learning position sizing directly from option surfaces beats hand-crafted rules; feasible later via the CI-trained-GBM pipeline on Tradier chain snapshots (start LOGGING daily chain snapshots now — free, and the dataset compounds).

## 📣 WEAK DESKS + DEAD MESSAGES (user evidence 2026-07-20: #desk-fx-rates screenshot)
Screenshot shows the FX desk posting the IDENTICAL line ("10 signals ≥ 0.6, 3 orders
EUR_USD, GBP_USD, USD_JPY") ~8×/day: no direction, no prices, no fills, no P&L — and
nothing consumes these posts (peer_learnings only captures agent discussions, not desk
output). Write-only noise. Same weakness class hits Polymarket + arbitrage desks.
- [x] **[P1] Desk posts: dedupe + enrich + chart** — SHIPPED 2026-07-20: dedupe+enrich for FX desk (notify.post_dedup, stateless via bot API) AND an orders-by-desk bar chart on the desk_order_placer #pnl-daily summary (discord_post_chart, only on non-empty runs). 11 tests. Remaining P&L-bars-per-strategy chart folded into the daily P&L report (already charts net notional). (notify.post_dedup: stateless, reads channel history via bot API — no git state; FX post now shows side+entry px per order and suppresses identical repeats; 6 tests). Remaining: apply to desk_order_placer P&L posts + attach discord_post_chart P&L bars. — suppress consecutive identical posts (hash last message per channel in state); every desk post must carry direction, entry px, open-position count, running desk P&L; attach `discord_post_chart` P&L bars (helper shipped). Applies to fx_desk.py AND desk_order_placer P&L posts.
- [x] **[P1] Desk posts → shared brain** — SHIPPED 2026-07-22: the missing CONSUME side is wired. `notify.read_channel_recent` reads desk run summaries back from Discord (stateless, via the bot API — no git-state churn, same pattern as `post_dedup`); `company_brain.fetch_desk_knowledge()` pulls the newest substantive post per desk channel (pnl-daily, desk-fx-rates/crypto/equities/options/commodities/polymarket/kalshi, bot-fleet, trading-floor) into the brain as a `desk_outcomes` category AND feeds them to the CIO synthesis; `llm_common.get_company_context()` surfaces `Desk results:` into EVERY employee prompt. Desk outcomes are written directly (not only via LLM) so the brain stays grounded even when the free-LLM keys are unset. 13 tests (test_notify_channel_read.py + test_desk_brain_bridge.py). — pipe each desk's run summary into peer_learnings/company_brain so employees DISCUSS actual desk results in the morning loop (the missing consume-side; extends the queued outcome-linked-learning item).
- [~] **[P1] FX desk audit: same-3-orders monotony** — NO-REPEAT GUARD SHIPPED 2026-07-20: fx_desk fetches OANDA openPositions and skips signals already positioned in the SAME direction (opposite direction stays allowed — reduces/flips exposure; fetch failure fails OPEN so monitoring never blocks the desk); skips surface in the run log + Discord post ("N skipped (already positioned)"). 5 tests. Remaining: signal-variance audit (why the same 3 pairs always rank top — needs several live runs of funnel data to diagnose). — always EUR_USD/GBP_USD/USD_JPY means top-K by confidence is static → likely stale/constant signal inputs or too-narrow universe ranking. Audit signal variance; add no-repeat-position guard (skip if an equivalent open position exists) and log WHY each pair won.
- [ ] **[P1] Polymarket desk is signal-only** — signals flow (dry run: live markets, conf=1.00 ensembles) but NO order path: py-clob-client signing still unwired (POLYMARKET_PRIVATE_KEY is in the relay). Implement CLOB order placement with $1–5 clips + the same never-partial guard, or the desk stays a commentary bot.
- [ ] **[P2] Arbitrage-bucket audit** — 32 strategies in the arb bucket but near-zero desk fills attributed to them; verify their signals reach a desk with an order path and aren't all filtered at the confidence gate.

## 🤖 BOT FLEET = FULLY AUTOMATED MANAGEMENT (user directive 2026-07-20)
Owner of record for the 61 bots is the AUTOMATION, not the user. Already live:
`bot_lifecycle` scheduler job (disable proven losers, promote winners, grow fleet from
templates — deterministic policy over real closed trades), 5-min exit sweep (fixed
today), safeguards, additive seeding, reward-gated code changes. Gaps to close:
- [x] **[P1] Lifecycle decisions → Discord** — SHIPPED 2026-07-20: bot_lifecycle posts each enable/disable to Discord (slack.send fails over to Discord) WITH the stats that drove it — `name (N trades, win X%, P&L $Y)` — to #bot-fleet, no black-box decisions. Remaining: append to peer_learnings for employee discussion (folded into the desk-posts→shared-brain item). — every enable/disable/promote decision posts WHY (stats in hand) to #bot-fleet; decisions also append to peer_learnings so employees can veto/discuss.
- [ ] **[P2] Bot parameter tuner** — weekly: for bots with ≥30 closed trades, grid-walk TP/SL% against their own trade history (pure pandas, free) and open a reward-gated PR adjusting template params; never touches live config directly.
- [ ] **[P2] Weekly OA-comparison report** — auto-generate a positions/P&L table per bot (entry/exit/hold/P&L) as a Discord chart + markdown artifact so the user's "match vs Option Alpha" check is a 2-minute read, not manual data pulling.

## 🤖 OA-BOT THOROUGH TEST 2026-07-20 PM — found + fixed a live P0
- [x] **[P0 FOUND+FIXED] Bot positions NEVER closed on the live (SQLite-fallback) deploy** — the new `test_bot_lifecycle.py` (first-ever coverage of `check_bot_exits`, the OA profit-taking half) reproduced it: SQLite returns NAIVE datetimes even for `DateTime(timezone=True)` columns, so `now − order.created_at` raised `TypeError`, and the scheduler's catch-all silently killed the ENTIRE exit sweep every 5 min ("Bot exit checker failed"). Positions opened with TP/SL brackets and then sat there forever. FIX: normalize `created_at` to aware-UTC at both subtraction sites in `engine.py`. Lifecycle now pinned by 6 tests: bracket math on open (±TP%/−SL% both sides), profit-target close AT target with +P&L, stop-loss close AT stop with −P&L, short-side TP, inside-bracket stays open, 7-day safety expiry.
- [ ] **[P2] Test-isolation flake pattern** — two known cross-file flakes under `-n 4 --dist loadfile` (breakeven_inflation contract, tearsheet-on-sqlite): shared per-worker DB lets one file's rows leak into another's expectations. Fix: per-file DB fixtures or scoped assertions (lifecycle tests already scope per-bot).

## 🔍 DEEP REVIEW 2026-07-20 PM — findings + tech/scalability roadmap
> Full-repo sweep: security, schema, workflows, code health, infra. Secrets scan CLEAN
> (no real keys committed — only doc placeholders). Suite: 1,699 passed / 0 failed.

- [x] **[P0] Schema-drift landmine when Supabase unpauses** — GATE SHIPPED: schema-drift-gate.yml (ephemeral Postgres → alembic head → autogenerate diff, fails with the missing ops). Run it via workflow_dispatch BEFORE unpausing; write the catch-up revision it prints — the live schema has evolved via `create_all` on the SQLite fallback, but `create_all` NEVER adds columns to existing tables and only 6 alembic migrations exist for a fast-moving model layer (e.g. `strategy_name` referenced by 6 model files, covered by 1 migration). The moment the user unpauses Supabase, the backend binds to an OLD Postgres schema → column-not-found 500s on the endpoints we just fixed. FIX: CI job that spins ephemeral Postgres, runs `alembic upgrade head`, diffs against `Base.metadata` (alembic autogenerate), FAILS on drift and opens a catch-up-revision PR. Must land before the unpause.
- [ ] **[P1] agb8 double-execution hazard** — TWO backends run 24×7 against the SAME Alpaca paper account: the stale `quantedge-api-agb8` (old build, own working DB, own scheduler → places bot orders + runs desk sync) plus the keeper `9jz0`. Duplicate order placement and split-brain state. User action (30s): suspend/delete the agb8 service in the Render dashboard.
- [ ] **[P1] Slack long-tail removal (user directive, half-done)** — notify.py is Discord-first, but **65 of 105 workflows** still wire the dead `SLACK_BOT_TOKEN` env and pure `slack-*.yml` workflows still exist. Sweep: drop the env var everywhere, archive slack-only workflows, delete `app/integrations/slack*` after confirming no live caller.
- [ ] **[P2] Workflow consolidation** — 105 workflows with overlapping families (agent-health-check / agent-health-monitor / agent-heartbeat / agent-status-check; gemini-task-runner vs generic runners). Merge each family into one parameterized workflow; fewer schedules = less cron starvation.
- [ ] **[P2] Money-path exception audit** — 502 `except Exception` in backend/app is the intended fail-soft culture, but in `execution/`, `risk/`, `brokers/` a swallowed error can silently eat an exit or a risk check. Audit those three dirs only; escalate swallowed failures to Discord pages.
- [ ] **[P3] Test hygiene batch** — pytest warns "ignoring pytest config in pyproject.toml" (two config sources); Starlette TestClient deprecation; `HTTP_422_UNPROCESSABLE_ENTITY` rename; wavelet_features fragmentation warnings. One small PR.

### Tech roadmap (scalability assessment 2026-07-20 — architecture is sound, free-tier infra is the risk)
- [ ] **[P1] Database durability** — NOW: unpause Supabase + keep-alive ping (queued below). NEXT: migrate to Neon serverless Postgres (auto-wakes on connection — eliminates the pause failure class at $0) or Supabase Pro. The SQLite fallback stays as the last-resort guard.
- [ ] **[P0-GATE for live trading] Always-on execution worker** — GitHub Actions cadence (cron starvation, ~15-min floor, suppressed events) is acceptable for PAPER only. Before `TRADING_MODE=live` ever flips: move desk execution into an always-on worker (Fly.io/Railway/Render starter ~$7/mo) driving the existing APScheduler loop. Missed exits on live capital is not an acceptable failure mode.
- [ ] **[P2] Agent state out of git** — `.github/state/*.json` committed hundreds of times/day caused the improver-clobbering incident and pollutes history. Move agent memory/company brain to Postgres tables + the queued BM25 retrieval; keep git snapshots as daily backup only.
- [ ] **[P2] Single-origin serving** — drop Vercel: `static_server` already serves the built frontend from the backend. One origin kills the CORS, rewrite-target, VITE_API_URL-mismatch AND deploy-rate-limit failure classes (all four bit us this week).
- **Keep as-is (scales fine):** FastAPI/SQLAlchemy async, React+Vite, broker plugin layer, reward-gated PR loop, LLM cascade with paid backstop. No rewrites, no Kubernetes.

## OA UI/UX parity roadmap (from 10 live OA screenshots, 2026-07-19)
> Full setting inventory: `.github/state/oa_settings_catalog.json`. Study doc: `docs/research/OA_DOCS_STUDY.md`.
> Goal (user): make the QuantEdge dashboard look/work like Option Alpha — "lots of vizz,
> dashboard + graph in bot rows" — and use the Screener/Trade Ideas to find symbols and scale bots.
- [x] **Bots list = OA layout** — SHIPPED this session: `30D` sparkline column (real /performance
      series, dashed zero baseline), Total P/L, Return %, Win Rate, Allocation columns, AUTOS toggle,
      + aggregate top cards (Total P/L / Return % / Change / Change % / Allocation). `BotBuilder.tsx`.
- [x] **Per-bot dashboard = OA layout** — SHIPPED: filled Closed-P/L equity curve + Position Stats grid
      (Closed Positions, Closed P/L, Profit Factor, Max Drawdown, Win Rate, Wins, Losses, Avg P/L,
      Avg Win, Avg Loss, Streak, Sharpe) + Capital sidebar (Allocation/Net Liquid/At Risk/Available/
      Maintenance). Backend `/bots/{id}/performance` now returns all OA metrics + weekday/hour/symbol breakdowns.
- [x] **[P1] Analyze page** (screenshot 3) — SHIPPED (first pass): the expanded bot panel now has a
      **Dashboard / Analyze** tab toggle; Analyze renders Metrics (Positions/Wins/Losses/Sharpe/Sortino/
      Profit Factor) + signed bar charts for **P/L by Day-of-Week**, **by Hour-of-Day**, and horizontal
      **P/L by Symbol** — all from the real `/performance` breakdown. `BotAnalyzePanel` in BotBuilder.tsx.
      Follow-ups: donut cards, big area chart with Total/Daily/Calendar tabs, Averages (Return-on-Risk/
      Entry-POP/DTE/Days-in-Trade), Hindsight Report, Export Data.
- [ ] **[P1] Positions tables** (screenshots 5,6) — Open Positions (Bot icon, Description, Legs w/
      call/put chips, Last, DTE, Qty, P/L, ROR, Net Liq, Premium, Risk, DIT + aggregate cards) and
      Closed Positions (adds Exp, Status Expired/Closed, Trade Price, Close Price, Price at OI/CI).
      Needs per-leg option instrument fields on Order/Position (Desk-consolidation Stage 3).
- [ ] **[P2] Trade Log** (screenshot 4) — chronological grouped-by-day feed: Bot icon, Time, Action
      (open/close), Description (full legs), Pricing (entry→exit), Status (filled at $x / canceled).
- [x] **Backtest→bot generator** (screenshot 8, OA "Automate your strategy") — SHIPPED:
      `POST /bots/from-backtest/{run_id}` maps the backtest's strategy family to real engine
      conditions (mapped vs approx confidence), bakes provenance (source run + Sharpe/return/win)
      into the description, creates the bot **disabled** (paper-first). Frontend: "→ Create Bot"
      button on each done run in BacktestLab. `app/bots/backtest_to_bot.py` + 3 integration tests.
      Follow-ups: the full "Create Bot" side-panel (name/allocation/entry-time/trade-pricing inputs)
      and Update-existing mode.
- [ ] **[P1] Screener page → symbol-scaling feed** (screenshot 9) — Symbol, 1M sparkline, Last, Today%,
      Liquidity bars, Beta, IV Rank, RSI, Techs, 3M/6M/12M returns, Earnings-in-Nd. Wire the ranked
      output into Symbol Scout so bots auto-scale onto the best-ranked new symbols (user: "use this to
      find symbols and scaling up").
- [ ] **[P1] Trade Ideas / EV finder** (screenshot 10) — math/probability opportunity ranker: Position,
      Leg/Delta, DTE, Reward/Risk, POP, PMP, PML, Alpha, EV, Max Profit, Max Loss, Earnings, Traders;
      Credit/Debit + Bull/Bear/Neutral + DTE/RR/POP/OTM% filters. Compute from `options_synthetic` +
      probability (shares the EV/probability P2 item below).
- [ ] **[P1] Calendar + earnings strategies** (screenshot, user: "make calendars like this, add earnings
      strategies") — month grid of Ex-Dividend, Earnings AM/PM, Market Holiday, Nonfarm Payrolls / FOMC,
      End of Quarter/Month events (FF calendar feed + an earnings-date source). THEN earnings strategies:
      pre-earnings IV-crush short premium (iron condor/strangle sized off Earnings Edge expected move),
      post-earnings drift — gated by the calendar so they only fire around confirmed earnings dates.

## OA parity gaps (from docs.optionalpha.com audit, 2026-07-19 — see docs/research/OA_FEATURE_PARITY_2026.md)
- [ ] **[P1] Options Expiration Protocol** — auto-manage 0DTE/expiring ITM positions before the bell (close or flag); critical now that the desk places real mleg spreads and 3 user bots trade 0DTE into the close.
- [x] **[P1] Bot safeguard depth** — SHIPPED: `ActionConfig.max_open_positions` / `max_daily_positions` (OA "N at once / N per day"); the engine counts this bot's open paper positions + today's opens and refuses to open once a limit is hit ("Position limit reached"). Also wired `no_position`/`position_exists` conditions to real open-position state (they were stubs returning True). `app/bots/engine.py` + `test_bot_safeguards.py` (5 tests).
- [ ] **[P1] Decision-recipe catalog expansion** — condition types evidenced by the user's bots: FOMC-day gate (FF calendar feed exists), price-change-vs-N-min-ago, option-leg OI threshold, position-return%, touch-$ exit; then loops over watchlists and position tags.
- [ ] **[P2] Failsafe family** — per-bot excessive-errors auto-disable (10/day → off + alert), overlapping-strikes check, pricing-anomaly gate.
- [ ] **[P2] EV/probability metrics** — HV-based probability + EV per defined-risk trade (OA's Trade Ideas 2.0 'Alpha'), computable from options_synthetic.
- [x] **[P2] Backtest→bot generator** — SHIPPED (see the OA-UI roadmap above): `POST /bots/from-backtest/{run_id}` + BacktestLab "→ Create Bot" button.

- [ ] **[P1] Kalshi desk — mirror the Polymarket pattern** (2026-07-19: public API live-verified, KALSHI_KEY_ID/KEY_SECRET provided in relay): kalshi_data.py desk feed from the public /trade-api/v2/markets + candlesticks (top markets by volume, KX: symbol prefix, guarded from the Alpaca order path like PM:), poly_* strategies run on it signal-only into #desk-kalshi; then RSA-signed order placement with the provided key pair. Backend /market-data/kalshi browse endpoint already exists.

- [x] **[P0] Render deploys fail at STARTUP — ROOT-CAUSED + FIXED 2026-07-19.** The Render application-log dump (added to deploy-on-main.yml) gave the real traceback: `app/api/v1/pipeline.py:15` did `Path(__file__).resolve().parents[5] / "pipeline_runs.json"` -> **IndexError: 5** at IMPORT on Render, where the file is `/app/app/api/v1/pipeline.py` (only 5 parents, 0-4) -> crashed the whole app before uvicorn could bind (nonZeroExit:1). Classic works-locally-dies-in-prod (local repo path is deeper). FIX: `_resolve_state_file()` searches ancestors for the file, honours a `PIPELINE_STATE_FILE` override, and falls back to `parents[min(4,len-1)]` — never IndexErrors; the reader already guards `if not _STATE_FILE.exists()`. Verified: full `app.static_server` import clean + pipeline tests green. NOTE: the sibling `parents[4]` files (monitoring/agents/experiments) resolve to `/` on Render — import-safe but wrong path; queued a shared repo-root helper as cleanup. SECRET_KEY was NOT the cause (already valid, len 44); the idempotent ensure-step stays as a safety net.

## 🚨 DEEP REVIEW 2026-07-18 PM — why "nothing is working" was TRUE on the live site
Root cause of almost everything: **render.yaml `autoDeploy: false`** — merges never
deployed. The live backend was a weeks-old build (29/57 templates, new endpoints 404,
bot scheduler dead since Jul 5, only 2 bots ever ran). FIXED this session: CI now has a
`deploy` job (green main → Render deploy → poll to live → page on failure) and bot
seeding is ADDITIVE (new templates become site bots on next boot). Remaining from the
evidence:
- [ ] **[P0] Verify post-deploy revival chain** — after the first auto-deploy: /health shows the new build; 57 bots on site; bot last_run_at advancing (scheduler alive); paper orders → check_bot_exits → Trades rows → leaderboard non-empty → perf weighting/pruning engage. Each link was dead on the old build.
- [x] **[P0] Discord empty — agent chat posted only to Slack (dead token)** — DIAGNOSED 2026-07-19 from live run logs: `multi_agent_discussion.py` and the desk's `_post_slack` delivered ONLY to Slack (invalid_auth), never Discord — so every conversation and desk P&L vanished. FIXED: both now deliver via `notify.discord_post` (bot-token→channel routing, same helper other flows use). Discussions post the opening + real LLM replies to the matching #channel; the desk posts P&L/fills to #desk-*. Remaining: the ~1s discussion runtime shows the **free-LLM keys are unset** → no real replies; the script now posts ONE actionable Discord notice ("add GROQ_API_KEY_1/GEMINI_API_KEY_1/DEEPSEEK_API_KEY") instead of silence. **User action: add any one free-LLM key** to turn on real conversations. Same Slack-only audit still TODO for team-lead-issues + daily-employee-review + employee-conversations.

- [ ] **[P1] Move completely off Slack → Discord** (user directive 2026-07-19) — DONE so far: desk P&L/fills, multi-agent discussion, claude_conversations, team_lead_issues, daily_pnl_report, live_trading_reporter, fill_tracker, and the notify_slack CLI notifier all now deliver Discord-first via `notify.discord_post` (Slack only if a token exists, and it doesn't). REMAINING long-tail (still Slack-only, ~35 scripts): backend_team, frontend_team, standup_agent, employee_intros, gemini_multi_agent, collective_learner, okr_tracker, company_brain, run_experiments, strategy_ranker/trimmer/auto_tuner, session_handoff, system/heartbeat/p0 watchdogs, etc. Convert each poster the same one-liner way, then delete the pure-Slack workflows (slack-*.yml) and drop SLACK_BOT_TOKEN from the remaining workflow envs. Backend Slack notifier (`app/integrations/slack*`) → Discord too. Track as a sweep; the loop can drain it.
- [x] **[P0] Desk fills → backend Trades attribution** (this is why the website showed "no trades") — FIXED 2026-07-20. ROOT CAUSE found: `sync_desk_trades` (already wired into the scheduler every 15 min) filtered accounts on `encrypted_key IS NOT NULL`, but the ONLY Alpaca paper account is the seeded demo account (`demo@quantedge.app`) which stores NO key — the desks trade on the **env** `ALPACA_API_KEY` (GH-Actions secret relayed to Render), which had no DB Account. So the account filter matched nothing → **zero** `qe-*` desk fills ever became `Trade` rows (global leaderboard + every per-user view empty). FIX (`app/tasks/desk_trade_sync.py`): added an **env-keyed fallback** — when `settings.alpaca_api_key/secret` are set, fetch the desk account's closed `qe-*` orders directly and attribute the reconstructed round trips to the system/demo (keyless) paper account; idempotent by `close_order_id`; skips if a keyed account already covers that key (no double-count). 2 integration tests + refactor keeps the 13 unit tests green. REMAINING UX follow-up (P1, below): these are shared *house* desk trades attributed to the demo account, so they show under the demo login + the global `compute_live_strategy_performance` weighting; surfacing them to an arbitrary user's own login (or a guest view) is a separate product decision. Note on cadence: crypto trades 24×7, equities only weekdays 9:30–16:00 ET — low weekend volume is expected, not a bug.
- [ ] **[P1] Surface house/desk activity to the logged-in user / guest view** (follow-up to the P0 above; user: "My login shows 0 data … Guest doesn't work") — desk Trades are attributed to the demo account; `/trades` scopes to `Account.user_id == current_user.id`, so a fresh user login sees nothing. Decide + implement: either a read-only "Platform / House desk" activity surface visible to every login, or attach the demo account's activity to the guest/demo session so the site is never empty.
- [ ] **[P1] Employee individual memory depth** — employee_context exists in agent_memory.json and grows, but verify each named employee accrues per-conversation memory and that it's surfaced in their Discord posts (recall-before-post is wired; confirm on live runs).

## 🚨 LIVE OUTAGE 2026-07-20 — "everything seems broken" ROOT-CAUSED
- [x] **[P0] Live site dead: Supabase project PAUSED** — evidence from `/health/detailed` on the keeper backend (quantedge-api-9jz0.onrender.com): `database: (ENOTFOUND) tenant/user postgres.vexzwnfbmznvxoxxktax not found — SUPABASE PROJECT MAY BE PAUSED`. Every DB-touching endpoint 500'd (login, /auth/demo, register, trades, bots) while `/health` stayed green — the site looked completely broken. FIXED in code: `ensure_database_alive()` (app/database.py) probes the primary at boot and falls back to local SQLite (rebinds `AsyncSessionLocal` in place; creates schema; bots reseed; desk trades resync from Alpaca 30-day history). Surfaced as a failing `database_primary` check so status stays `degraded` and watchdogs keep paging. 4 tests (`test_db_fallback.py`). **USER ACTION for durable state: supabase.com/dashboard → find the project → Unpause** (free tier pauses after 7 idle days; ~90 days until data loss). Until then the fallback DB is ephemeral (resets each deploy).
- [x] **[P1] Supabase keep-alive** — SHIPPED: db-keepalive.yml, 3× daily — once unpaused, add a tiny scheduled workflow ping (one cheap SELECT via the pooler a few times/day) so the free-tier project never idles into a pause again.
- [x] **[P0] Frontend pointed at the WRONG backend** — root-caused 2026-07-20: `frontend/vercel.json` proxied `/api/*` to `quantedge-api-agb8.onrender.com`, a STALE deploy (29 bots = pre-07-18 build, 0 trades, no relayed env keys) — while every new merge deploys to the keeper `quantedge-api-9jz0.onrender.com`. So users saw an old, empty app no matter what shipped. FIXED: rewrite now targets 9jz0. Follow-up decision for user: delete/suspend the agb8 service (it costs a free-tier slot and confuses debugging).

## LLM/Discord silence — ROOT-CAUSED + FIXED 2026-07-20 (second pass)
- [x] **[P0] 29 workflows had NO paid-LLM backstop** — `ai-pr-review.yml` (the one flow that visibly produces real LLM output) passes `OPENROUTER_API_KEY` + `ANTHROPIC_API_KEY`; the other 29 LLM workflows (multi-agent-discussion, daily-standup, collective-learning, peer-review, team-lead-issues, employee reviews, watchdogs…) passed ONLY free-tier keys — when those rate-limit/fail the conversation dies silently. That asymmetry is why the AI review works while Discord stays quiet. FIXED: backstop keys added to all 28 consumer workflows (key-relay excluded — not an LLM consumer). The Anthropic Messages API path already existed in `llm_common` (`ANTHROPIC_API_KEY`/`_2`) — it was simply never fed in these jobs.
- [ ] **[P1] Everyday-improvement visibility loop** — now that desk fills become real `Trade` rows (PR #764) and conversations have a working LLM backstop, wire the daily P&L attribution INTO `peer_learnings` so agents discuss actual results each morning in #trading-floor (outcome-linked learning, not status theater).

## Session 2026-07-20 PM — shipped
- [x] **ml_signal bot condition (OA format)** — bots can now use the ML model as a decision recipe: `{"type":"ml_signal","direction":"up","min_confidence":0.65}` (engine runs inference once per tick, only when used; no trained model → condition is False, never an error). 8 tests. Combine with indicator conditions under ALL/ANY exactly like OA stacks decisions. TV webhooks already existed (`POST /api/v1/webhooks/tradingview` → Redis `tradingview:alerts`).
- [x] **Discord charts** — `notify.discord_post_chart()` renders QuickChart image embeds (green/red signed bars, no deps/keys); daily P&L report now posts a net-notional-by-symbol chart alongside the text. Roll out to desk P&L, leaderboard digest, bot lifecycle next.
- [x] **client.ts base-URL normalization** — a `VITE_API_URL` ending in `/api/v1` (the OLD documented value, likely still in the Vercel env) double-pathed every request (`…/api/v1/api/v1/auth/demo` → 404): reproduced in a real browser, this alone breaks guest login + all data even with a healthy backend. Client now strips the suffix. NOTE: Vercel hit its 100-deploys/day free limit (~05:11 UTC) — no frontend change goes live until it resets.

## ML experiments — why "not working at all" (diagnosed 2026-07-20)
Live evidence: `/health/detailed` → `ml_models: count 0`. Three stacked causes: (1) **torch is not installed on Render** (free tier: LSTM/PatchTST/SSM/Mamba log "unavailable" at every boot — by design, graceful degrade); (2) `models_dir` is **ephemeral** on Render — even a trained artifact vanishes on redeploy; (3) experiments that DO run in CI never ship artifacts anywhere the backend can load.
- [ ] **[P1] Scalable ML pipeline (the fix)**: train sklearn/GBM models (torch-free — this is also what most desks actually run in production) in a scheduled GitHub Action, commit versioned artifacts (small .pkl/.ubj) to the repo or a GitHub Release, and have the backend download/load at boot. Then `ml_signal` bots + `/ml` endpoints go live with real predictions. Walk-forward gate before any artifact is promoted.

## What top firms do that we don't (gap audit 2026-07-20, queued)
- [x] **[P1] Overfit killers in strategy promotion** — SHIPPED 2026-07-22: three Bailey & López de Prado stats now live in `backtest/cpcv.py` as pure, reusable functions — `deflated_sharpe_ratio` (rewritten to the correct PROBABILITY form: multiple-testing haircut over the trial Sharpes; the old dead z-score with a `sqrt(var+1)` bug is gone), `probabilistic_sharpe_ratio` (short-track-record + skew/kurtosis significance), and `probability_of_backtest_overfitting` (CSCV over an N-config returns matrix — noise→~0.5, genuine edge→~0). WIRED into the walk-forward path: `walk_forward()` now returns a `robustness_verdict` (`is_robust`/`deflated_sharpe`/`consistency`/`n_windows`/`verdict`) enforcing the module's own documented protocol (≥12 OOS windows, avg & majority of windows ≥0.7 Sharpe, DSR≥0.90) — previously that protocol was a comment, unenforced. The `/backtests/walk-forward` task persists the verdict into `BacktestRun.params.overfit` (no migration) so the leaderboard/promotion can refuse overfit runs. ALSO fixed a latent bug: `walk_forward()` was missing the `initial_equity` param the API always passed → the endpoint `TypeError`'d and silently failed every run. 20 tests. NEXT (optional): wire PBO into the multi-config leaderboard/auction selection (needs a per-config returns matrix); gate `backtest_to_bot` on PSR. — deflated Sharpe ratio + probability-of-backtest-overfitting (PBO) gates before a strategy is promoted/scaled (Bailey & López de Prado line of work). We only do walk-forward today.
- [ ] **[P1] GBM-on-features as the ML workhorse** — production quant desks overwhelmingly run gradient-boosted trees on engineered features with strict leakage control, not deep nets; aligns with the CI-trained-artifact pipeline above.
- [ ] **[P2] Portfolio-level risk (factor exposures + stress)** — we cap drawdown/Kelly per strategy; firms add net factor exposure limits (beta/sector/duration) + scenario stress across ALL desks and countries at once. Build on the desk-consolidated Trades now that attribution works.
- [ ] **[P2] Execution: queue-aware child orders** — SmartPricing ladder (queued) + participation-rate caps; measure realized slippage vs arrival in the slippage dashboard.
- [ ] **[P2] Cross-desk/country correlation watch** — international desk (19 ADRs) + FX + crypto share risk factors; add a rolling correlation matrix alert when diversification collapses (crisis mode → cut gross).
- [ ] **[P2] Agentic-trading literature sweep** — multi-agent debate (Du et al. '23), Reflexion (Shinn '23 — already used by the improver), skill libraries (Voyager '23), memory-augmented trading agents (FinMem '23), RL execution (FinRL). Apply-not-cite: debate gate + skill library are queued above.

## SOTA multi-agent 24×7 queue (user ask 2026-07-20)
- [ ] **[P1] Debate gate for large desk orders** — bull/bear/judge 3-role LLM debate (one `llm()` call each, shared context) before any paper order above a notional threshold; verdict + reasoning posted to the desk channel. (Du et al. multi-agent debate; cheap with the cascade.)
- [ ] **[P2] Market-based task allocation on AgentBus** — auction `agent-fix-needed` issues to employee agents by bid (past success rate per improvement_type from improvement_stats) instead of round-robin; the auction record IS the Discord conversation.
- [ ] **[P2] Verifier-role expansion** — extend ai-pr-review into a two-pass generate→verify pattern for Free-Agent Engineer PRs (verifier must reproduce the failure the PR claims to fix before automerge label is granted).
- [ ] **[P2] Shared-brain retrieval upgrade** — company_brain.json grows unbounded; add embedding-free BM25-style retrieval (pure-python) so `inject_company_context` pulls the 5 most relevant memories per prompt instead of the newest N.

## SOTA research + pipeline upgrades queue (user ask 2026-07-20)
- [ ] **[P2] SOTA sweep — execution**: SmartPricing-style laddered repricing (already queued below), plus survey adaptive limit-order placement literature (queue-position aware repricing) for _ensure_filled.
- [ ] **[P2] SOTA sweep — ML**: evaluate PatchTST/iTransformer-family for the existing feature pipeline vs the current LSTM/ensemble; only via walk-forward gate, paper-first.
- [ ] **[P2] SOTA sweep — portfolio**: compare current HRP against NCO (nested clustered optimization) and turnover-penalized variants on the desk universes.
- [ ] **[P2] Pipeline hardening**: CI: cache uv wheels for faster runs; deploy: post-deploy smoke now hits /health — extend to /health/detailed and fail deploy on `database.ok=false` (catches the next Supabase pause at deploy time instead of silently).

## Autonomy hardening (2026-07-20)
- [ ] **[P1] Improver PRs BYPASS CI entirely — the reward gate never runs on them** (found 2026-07-22). Evidence: PR #876 ("improve(strategy_logic)", bot-opened) reached `main` with ONLY a `Vercel` check — `test`/`test-agents`/`frontend-build` NEVER ran. Root cause: CI triggers on `pull_request`, but GitHub suppresses workflow runs for PRs opened by the bot's `GITHUB_TOKEN` (the same recursion guard that stops auto-merge.yml firing on bot PRs). So the "reward gate = full CI green" is a NO-OP for every improver PR — it merges unguarded. #876 shipped a nonsensical LLM hallucination (HTTP `X-Strategy-Entry/Confirmation` header validation wrapped around the WHOLE strategies router) that 400'd every `/api/v1/strategies/*` GET — dashboard strategy list + demo session dead. FIXED the live regression this session (revert in `router.py`, endpoint-smoke + demo-session tests green). **[MITIGATED 2026-07-24 — option (c) shipped]** `auto-merge.yml` had the exact hole: it filtered check-runs then checked `unfinished`/`failed` — but a bot PR with ZERO check-runs (CI never ran) gave empty arrays → "no checks" read as "all green" → merged unvalidated (this is literally how #876/#929 reached main and broke boot 3 sessions running). FIX: the gate now REQUIRES `test`/`test-agents`/`frontend-build` to be PRESENT and successful on the head sha; missing = refuse to merge. Improver PRs now pile up open (harmless) instead of landing unchecked. Takes effect immediately (workflow layer, no deploy). REMAINING (better, needs owner/PAT): (a) branch protection on `main` requiring those checks — belt-and-suspenders; (b) make the improver dispatch CI via `WORKFLOW_PAT` so its PRs actually get validated and can legitimately merge again.
- [x] **Continuous-improver safety rails** — FIXED (this session). Root-caused why the ~dozens of stuck `improver/run-*` PRs (backlog to 2026-07-05) could never be safely merged: (1) each PR bundled a stale snapshot of `.github/state/agent_memory.json` + `skill_library.json`, so merging ANY of them REVERTED newer live agent memory that other workflows write to main continuously; (2) the reward gate is full-CI-green, which does NOT cover the behavior of whole-file LLM rewrites — so green PRs still regressed the money path (e.g. `ml_breakout` stopped suppressing ML-unconfirmed signals; `ensemble`/`rsi2_pullback`/`hrp`/`iceberg` silently reworked). FIX (`continuous_improver.py`): (a) `_is_protected` bars the improver from `strategies/`, `execution/`, `risk/`, `ml/models/`, `bots/` — it may only touch non-core files (api, schemas, utils, tasks, integrations); (b) the run branch is reset so `.github/state/**` never enters the PR diff. 4 guard tests. Stale backlog to be closed (unmergeable, superseded).

## New queue (added 2026-07-18, OA-backend session)
- [x] **[P1] Tradier sandbox options-data adapter** — SHIPPED 2026-07-20 (PR #762): `.github/scripts/tradier_data.py` (fail-soft: quote/expirations/nearest_expiration/chain-with-greeks/pick_by_delta/atm_iv). Desk mleg spreads now pick short/long legs by target delta (~0.30Δ short / ~0.15Δ long) via one live chain fetch per underlying, falling back to spot*moneyness when the feed is down. `TRADIER_SANDBOX_TOKEN` wired through desk-trading.yml + test.yml relay. Live-verified: SPY 0.30Δ put→728 (δ −0.2947), ATM IV 0.1512; iron condor puts 728/699 calls 763/776. 14 tests. **User action to activate the real-greeks path: add the `TRADIER_SANDBOX_TOKEN` secret** (absent it, moneyness fallback runs — no regression).
- [ ] **[P1] SmartPricing-style laddered repricing** — extend _ensure_filled's one-shot cancel-replace into an OA-style ladder: post at mid, step limit toward market every ~7s (3 steps), then market out; measure in the slippage dashboard.

## New queue (added 2026-07-15, scale-up session)
- [x] **[P1] Per-desk performance attribution + auto-pruning** — DONE 2026-07-16: the live-leaderboard weighting now prunes proven losers to 0.0 (≥20 trades, negative P&L, sharpe<-0.5 → desk skips the strategy entirely; auto-revives when stats recover). 5 tests incl. missing-sharpe-never-prunes.
- [x] **[P1] Route Options-desk income structures through REAL multi-leg orders** — DONE 2026-07-18: wheel/condor/credit-spread/CSP/vol-carry signals now place actual defined-risk mleg spreads (moneyness-picked strikes ~35 DTE via /v2/options/contracts, 1 contract, day). Unresolvable legs place NOTHING and fall back to the underlying proxy. 9 tests incl. never-partial-spread.
- [x] **[P1] Symbol Scout** — DONE 2026-07-16: symbol_scout.py validates every desk symbol against /v2/assets (dead symbols get loud + queued), proposes unwired tradable crypto pairs + curated liquid ETFs; runs in the strategy-scout workflow; 6 tests.
- [x] **[P2] TV-desk hit-rate tracking** — DONE 2026-07-18 as a general pruning rule (applies to every strategy incl. TV): ≥100 trades + losing + win_rate<45% → weight 0.0, desk skips it. Profitable low-hit-rate trend riders are exempt by design (tested).
- [~] **[P1] Polymarket desk** — REAL DATA DONE 2026-07-18: desk now feeds the top-6 markets' hourly price bars from the public Gamma+CLOB APIs (live-verified) and logs real signals; PM: symbols are guarded from the Alpaca order path. REMAINING: order placement via py-clob-client signing — POLYMARKET_PRIVATE_KEY/KEY_ID/KEY_SECRET are provided in the secret relay, so this is implementable now.
- [x] **[P2] Research backlog → registry pipeline** — DONE 2026-07-18: Strategy Scout writes its top rotating idea to state/research_seed.json; strategy_generator injects it into the LLM prompt as PRIORITY RESEARCH DIRECTION (fail-open when absent). Verified end-to-end.

- [ ] **[P2] Unblock the 10 data-source-blocked strategies** (strategy-scout 2026-07-15; every WIRABLE strategy now trades — coverage 53→103): each needs a feed, not a desk: covered_call (share inventory), funding_rate_arb + crypto_basis_roll + dex_cex_arb (geo-permitted derivatives/DEX data), token_unlock_fade (unlock calendar), news_momentum (headline feed), earnings_accruals (fundamentals), micro_cap_momentum (small-cap universe), moc_auction_imbalance + order_flow_imbalance (intraday/L2 data).

> **Session 2026-07-15:** QUARANTINED emptied (hard-budget fail-soft); Commodities desk;
> FF red-folder gate for FX; **AlpacaBroker restored** (improver PR #420 had truncated
> 6 of 7 interface methods — improver now rejects shrunken/elided outputs, and
> test_broker_interface.py guards all brokers); employee-health hard gate pages Discord;
> tasks/ silent-except sweep; per-bot activity view (OA parity). Remaining open items
> below are the live queue.

- [x] **[P0] Fix 18 quarantined strategies** — DONE 2026-07-15: QUARANTINED is EMPTY. Final 6 (+5 latent) fixed via shared `app/strategies/_failsoft.apply_hard_budget` (detached-daemon-thread hard timeout; yfinance's curl_cffi bypasses socket kills, so this is the only real guard). Contract suite 115/115, 28s. ~~ (desk now guarded by a 10s per-strategy timeout — freezes impossible; remaining work is per-strategy fail-soft hygiene, remove from QUARANTINED as fixed) (contract-test audit 2026-07-11; the cause of desk freezes). Wrap fetches fail-soft w/ hard timeout, then remove from QUARANTINED in backend/tests/unit/test_strategy_contract.py. DESK-WIRED: ~~gamma_exposure, kalman_pairs, skew_arb, vrp_systematic, pead_sue~~ FIXED fail-soft 2026-07-11 (un-quarantined, contract-proven); credit_spread_income guarded but still slow offline (yfinance retries — stays quarantined); multi_factor_equity remains. Rest: lorentzian_knn, breakeven_inflation, dollar_carry, macro_risk_barometer, mvrv_zscore_timing, duration_momentum, yield_curve_momentum, pmi_sector_rotation, tlt_spy_rotation, yield_spread_reversion, basis_carry.


- [x] **[P0] scripts/import_oa_bots.py** + ~~scrape OA PUBLIC leaderboard~~ INVALID: 'optionsalpha.com' 200s were a wrong-domain false positive; real site (optionalpha.com) redirects all bot pages to /login (verified 2026-07-11). Cookie or Cowork are the only paths — both built
- [x] **[P0] OANDA FX desk** — fx_desk.py + fx-desk.yml, 7 majors 24/5, 7 tests; **[P1]** WORKFLOW_PAT fallback wiring; ADRs; EM cap 30%


> **How tasks are tracked (the answer to "where do tasks live"):**
> - **Canonical queue:** GitHub Issues labeled `agent-fix-needed` (the agents already
>   create/work these via `team_lead_issues.py` → `free_agent_engineer.py`).
> - **Human board:** [Notion — QuantEdge Tasks](https://app.notion.com/p/bec54f8a79444c2399316365a07e0291)
>   (seeded from this file; mirror via the *Notion ↔ GitHub Issues Sync* workflow).
> - **Cross-session continuity:** this file + `HANDOFF.md`, committed to the repo
>   (chat sessions are ephemeral — only what's committed survives).
> - **Slack:** notifications/visibility only — never the source of truth.

_Last updated: 2026-07-15_

---

## Session 2026-06-29 — review backlog (see `docs/REVIEW_2026-06-29.md`)
Queued for the autonomous loop / employees. Priority order top-to-bottom.
- [~] **[P0] OA Scout — auto-copy new Options Alpha bots daily** — BLOCKED on user unlock `OA_SESSION_COOKIE` (public pages auth-walled, verified 2026-07-11); importer + playbooks + issue template all built. — workflow fetches the
  public optionalpha.com template/library pages, diffs against .github/state/oa_library.json,
  LLM-parses any NEW bot into a BOT_TEMPLATES entry (delta/DTE/TP/entry window), opens a
  reward-gated PR, posts the find to #alpha-research. Runs daily + on CI events. Private
  account bots can't be scraped (auth) — screenshots remain the path for those.
- [x] **[P1] ForexFactory calendar feed** — DONE 2026-07-15: red-folder gate in fx_desk.py (±30min blackout per pair currency, fail-open, live-verified 99 events). — ingest the public ff_calendar_thisweek.json into
  /market-data/forex-calendar and gate Macro/FX desk entries around red-folder events.
- [x] **[P2] TradingView/FxReplay/Tradezilla** — DONE 2026-07-15: POST /webhooks/tradingview receiver (secret-gated, disabled-without-secret, normalizes alerts, /recent ring buffer, Redis fan-out; receives only — never trades). FxReplay/Tradezilla have no APIs — documented, closed. — no public trade APIs (manual UIs);
  TradingView useful as charts + webhook-IN alerts (receiver endpoint), not for dummy
  trading automation. Document + build the webhook receiver only.
- [x] **[P0] Options Alpha dashboard parity in the frontend** — DONE 2026-07-15: per-bot P&L graph (shipped earlier), + GET /bots/{id}/activity with open-positions & trade-history tables in BotBuilder expanded row; settings editor existed. — per-bot detail view in
  BotBuilder: cumulative P&L graph (endpoint /bots/{id}/performance is LIVE), open
  positions table (orders with bot_id in raw_payload), trade history (Trades by
  strategy_name == bot.name), settings editor (PATCH /bots/{id}). Use LWEquityCurve.
- [x] **[P0] Alpaca multi-leg options orders** — implementation existed (submit_alpaca_multileg_order, OCC symbols, delta picking, engine-wired) but had ZERO tests; 2026-07-15 added 8 tests which caught a real crash (structlog-style kwargs on a stdlib logger — a broker REJECTION crashed the caller instead of returning None). Fixed. — Alpaca paper supports options + multi-leg;
  extend brokers/alpaca_orders.py with the legs order shape so every oa_* bot fills REAL
  option legs with existing keys. Kills the TradeStation dependency.
- [x] **[P1] Synthetic options backtester** — DONE 2026-07-18: app/backtest/options_synthetic.py (BS pricer, realized-vol IV proxy, spread backtests mirroring the desk's mleg structures; limits stated honestly — no skew, conservative for short premium). 9 tests: put-call parity, defined-risk cap, theta harvest in flat tape, crash bleed. Original ask: Black-Scholes pricer over underlying OHLCV +
  realized vol to approximate premium-structure backtests (no chain history yet); gate
  bot enablement on a passing synthetic backtest (paper-first stays).
- [x] **[P0] Discord per-channel routing via bot token** — DONE 2026-07-12 (notify.py bot-token routing + DiscordBot UA + webhook fallback). — notify.py posts everything through ONE
  webhook into #general with a [#channel] prefix; channels exist now, so resolve channel name → id
  via the bot token (GET /guilds/{id}/channels) and POST /channels/{id}/messages, webhook fallback.
  Embed author = employee name for per-employee identity. Kills the "all channels empty" state.
- [ ] **[P1] LLM-brain employee personas on Discord** — desk/agent posts composed by llm_routed()
  with per-employee persona prompts (slack_agent_team.py personas exist), not fixed templates;
  numbers stay deterministic, only the commentary is generated. Two-way: reply when @mentioned
  via the interactions endpoint.
- [x] **[P0] Forex desk** — DONE 2026-07-13 as the OANDA FX desk (fx_desk.py, 7 majors 24/5, practice orders) rather than backend strategies; superset of the ask. — add `market_type="forex"` strategies (carry, trend/momentum), register,
      add "Forex" to `_MARKET_TYPE_DESK`, route data_loader to `EURUSD=X` etc., + scheduled desk.
- [x] **[P0] Commodities desk** — DONE 2026-07-15 via ETF proxies on the GitHub-Actions desk layer (GLD/SLV/USO/UNG/DBA/PDBC/GDX/CPER, TSMOM+Donchian+MR, regime-mapped, config guard tests). — add `market_type="commodity"` strategies (term-structure roll,
      momentum, gold/oil mean-reversion), register, add "Commodities" desk, route `GC=F`/`CL=F`.
- [x] **[P0] Render sleep** — keep-alive workflow chained to CI events pings /health 24x7 (event-driven, no cron starvation). — external uptime pinger (UptimeRobot) or paid tier so in-app employees
      don't halt (`/health` returned 000 — backend asleep).
- [~] **[P1] Audit & consolidate 86 workflows** — employee manifest shipped (`docs/WORKFLOWS.md`,
      via `scripts/gen_workflow_manifest.py`): 87 workflows / 70 scheduled, dup-families flagged
      (`slack-*`×10, `agent-*`×6, `render-*`×5, `strategy-*`×5). Next: actually dedupe the families.
- [x] **[P1] Durable auto-merge** — `auto-merge.yml` lands `automerge`-labeled PRs once all checks
      pass (no human merge). Removes the last manual step for the autonomous loops/employees.
- [x] **[P1] Employee-health hard gate** — DONE 2026-07-15: agent-health-check.yml no longer continue-on-error; critical findings fail the job AND page Discord #infra-alerts. — make the agent smoke test page on failure; verify
      `agent-health-*`/`system-status` actually alert when an employee is stale.
- [x] **[P1] Reward-gate self-improvement** — `continuous_improver.py` now pushes a throwaway
      `improver/run-*` branch and opens an `automerge` PR instead of pushing to `main`. The full CI
      suite must pass before changes land (auto-merge.yml). Stops the unvalidated direct-to-main
      commits that broke the app 3× (slots=True, @root_validator, dead scheduler) in one session.
- [x] **[P1] Wire Alpaca crypto into `price_feed`** — DONE 2026-07-15: root cause was improver PR #420 truncating brokers/alpaca.py (6 of 7 interface methods deleted → AlpacaBroker un-instantiable → silent yfinance fallback) + stale exception imports. Restored, guarded by test_broker_interface.py. for live quotes (Binance still geo-blocked for live).
- [~] **[P1] Narrow 435 broad `except Exception`** — tasks/ sweep DONE 2026-07-15 (every silent pass now logs); brokers/execution/risk money paths done earlier; remainder (llm, api) queued. — start with `tasks/`, `brokers/`, `llm`; add logging.
- [x] **[P1] Audit stale provider model IDs** — done: Cerebras gpt-oss-120b + NVIDIA deepseek slug live-verified, both env-overridable (CEREBRAS_MODEL/NVIDIA_MODEL).
- [x] **[P2] ML employees inert on prod** — already handled honestly: /health reports torch availability with non-critical 'degrades gracefully' status; ML strategies fail-soft. No further action. — run with `[ml]` extra on a worker, or mark degraded.

---

## Session 2026-06-24 — shipped (11 PRs merged to `main`)
- [x] **Options productization end-to-end** (#188): `OptionLeg` + `open_option_spread` schema,
      engine branch, TradeStation options API (chain + multi-leg order builders), 4 templates.
- [x] **Brain cascade fixed** — reasoning-model content extraction (Cerebras gpt-oss / R1) on
      `main` (#188) **and** on the default branch (#189: User-Agent, live model IDs) + in-call
      key fallthrough (#199). Verified live: groq/cerebras/nvidia answer.
- [x] **Backend-health banner + fresh-Render runbook** (#190) — `docs/RENDER_NEW_ACCOUNT.md`.
- [x] **TradeStation spread routing** (live-only, paper-first proven) + broker tests (#198).
- [x] **Kalshi public market reads** wired (#203) — matches the existing Polymarket endpoint.
- [x] **Tests added/guards:** income/macro strategy contracts (#202), TS options parsing (#198),
      pytest-asyncio deprecation removed (#206), **momentum lookahead causality guard** (#207),
      **cross-tenant isolation guard** (#208).
- [x] **Backlog hygiene:** closed 8 stale tsconfig issues + #193 (brain canary already exists).
- **Verified deploy-readiness:** booted backend locally → 158 routes, demo auth, seeds 29 bots
      /13 strategies/3 risk rules, Kalshi live. **Only blocker to going live = Render build-minute
      quota (#197) + default-branch flip to `main` (#196).**

---

## P0 — Reliability (the brain must never silently die)
- [x] **LLM cascade dead (Cloudflare 1010 / no User-Agent)** → fixed (#144).
- [x] **Cascade used only the primary key** → rotate across numbered variants (#145).
- [x] **Brain observability + always-on canary** → `cascade_status()`, `llm_metrics.jsonl`,
      `brain_health.py`, hourly `brain-health.yml` that alerts Slack #infra-alerts (this PR).
- [~] **Provider keys** — as of 2026-06-24, **3 work live** (Groq, Cerebras via gpt-oss-120b,
      NVIDIA via current model). Gemini=quota(429, recovers), DeepSeek=balance(402). Optional:
      add free SambaNova/Together/Hyperbolic keys to Doppler for more headroom. *(Drop key in Doppler; I wire the rest.)*
- [x] **Make the agent "smoke test" a hard gate** — smoke-test.yml pages Discord #ci-failures on failure; agent-health-check.yml hard-gated 2026-07-15 (fails + pages #infra-alerts on critical findings).

## P1 — Real bugs found this session
- [x] **`/ws/prices` all-symbols bug:** subscribed to literal topic `prices:*` but the feed
      broadcasts `prices:{symbol}`. Fixed: `ConnectionManager.broadcast` now fans concrete
      `prices:{symbol}` updates out to `prices:*` wildcard subscribers (+ regression tests).
- [x] **`test_realtime_endpoints.py` auth helper** — superseded by `test_realtime_live.py`,
      which authenticates with `email` + an `@example.com` address (no false-green skip).
- [x] **Redis default `localhost:6379`** spammed connection-refused. Fixed: prod default is
      now *unset* (`REDIS_URL` empty ⇒ clean no-op cache) **and** a connection-failure circuit
      breaker trips once, logs once, then no-ops for the rest of the process.
- [x] **3 broken workflows failing at YAML parse** (run name shown as the file path, 0 jobs):
      `slack-on-deploy.yml`, `agent-health-check.yml`, `gemini-ml-training.yml` — multi-line
      `run:` block scalars whose continuation lines lost their indentation. **Direct cause of
      "Slack dead except scheduled messages"** (deploy/health Slack posts never fired). Fixed;
      repo-wide workflow YAML lint now shows 0 broken.
- [x] **"TV Indicator SOTA" scheduled workflow** — investigated 2026-07-15: it was still cron-only (cron is starved on free tier → effectively never ran). Now event-chained to CI completions like the other 21 team workflows, with a ~6h cadence gate (git-log stamp) so the 20-min LLM job doesn't fire on every CI run.

## P1 — Issues the agents themselves flagged in Slack (live triage, 69/97 channels active)
- [x] `#deploys` — **cross-user data leak**: verified closed — all core routers scope to
      `current_user` (bots by user_id; orders/positions/trades by Account.user_id). Guard test (#208).
- [ ] `#leadership-summary` / risk — **VaR threshold exceeded**.
- [x] `#alpha-research` — **lookahead bias** in momentum strategies: verified — all 13 already
      `shift(1)`; causality regression guard added (#207).
- [ ] `#squad-qa` / `#ci-failures` — **test failures / bug** backlog.
- [ ] `#okrs` — **Sharpe-ratio shortfall** vs target.
- [ ] `#squad-backend` — **latency issues**; `#squad-frontend` — **screenshot upload failed**.
- [ ] `#finance-ops` — **upcoming paid triggers** (add spend caps before they fire).

## P2 — SOTA upgrades to make this a top-tier AI-first company
> Full durable research: `docs/research/AI_COMPANY_SOTA.md`,
> `docs/research/LLM_COST_OPTIMIZATION.md`, `docs/MODEL_ROUTING.md`.
1. **Observability + model routing** — Langfuse/OpenTelemetry traces on `llm_common`; route by
   task tier. *(Phase-1 metrics shipped; **cost-tiered `llm_routed()` ladder shipped** —
   free → OpenRouter open-mid → Claude backstop, env-configurable; Langfuse tracing next.)*
6. **Open-weight mid-tier so Claude is the rare backstop** — ✅ shipped in `llm_routed()` /
   `docs/MODEL_ROUTING.md`. DeepSeek/Qwen/Kimi/GLM/MiniMax via OpenRouter handle "hard" work at
   10–50× lower cost; Claude only on `tier="hard"` or last resort. Refresh `OPENROUTER_MODELS`
   to the exact current SOTA slugs as they rotate.
2. **Real memory layer** — replace flat `.github/state/*.json` with Mem0 or Letta backed by your
   existing **Supabase pgvector** (episodic + semantic recall).
3. **Outcome-driven self-improvement** — give the self-improver a *verifiable reward*
   (CI-green + coverage Δ + paper backtest Sharpe Δ); gate agent PRs behind an eval + LLM-judge.
   (DeepSWE / Darwin-Gödel-Machine pattern.)
4. **Durable, event-driven orchestration** — move the core loop (lead→engineer→reviewer) onto
   Temporal/Inngest/LangGraph durable execution instead of fire-and-forget cron.
5. **A2A agent protocol** — typed agent-to-agent coordination; demote Slack to a human digest
   (kills the repeated-message noise).

## Desk consolidation (staged — combine best of all desks/orders/tracking/risk)
> Execution (`execution/`) and risk (`risk/`) are already shared, desk-agnostic layers.
> `Bot` is already one unified JSON format across equity/crypto/polymarket. The work is
> consolidation, not a rewrite.
- [x] **Stage 1 — desk taxonomy (no migration):** `desk_of()` / `strategies_by_desk()` /
      `list_desks()` derive desks from existing attributes; `GET /strategies/desks` exposes the
      unified view (Equities 62 · Crypto 16 · TV 12 · Prediction Markets 8 · Options 7) + tests.
- [ ] **Stage 2 — extend the unified `Bot` format to all desks:** add `options`/`macro`/`rates`
      to `Bot.market_type`; tag finer desks via an explicit `desk` class attr (override hook
      already supported by `desk_of`).
- [ ] **Stage 3 — unified cross-desk tracking:** add `asset_class` + options instrument fields
      (strike/expiry/right/multiplier) to `Position`/`Order` (Alembic migration).
- [ ] **Options productization:** options desk is research-complete but not in the Bot builder
      (blocked on Stage 2/3); move scattered options strategies into `strategies/options/`.

## Housekeeping
- [ ] Deprecations: pytest-asyncio `event_loop_policy` fixture, Starlette `TestClient`+httpx,
      now-unused `passlib`.
- [x] Audit stale provider model IDs/endpoints in `llm_common` — duplicate of the 2026-06-29 item, done: live-verified + env-overridable (CEREBRAS_MODEL/NVIDIA_MODEL).
