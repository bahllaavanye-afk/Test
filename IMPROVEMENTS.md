# QuantEdge — Improvements & Task Tracker

- [ ] **[P0] Render deploys fail at STARTUP (update_failed)** — found 2026-07-18: the deploy pipeline works end-to-end (build succeeds), but the new app fails to boot, so Render keeps the weeks-old build live — THIS is the true staleness root cause (autoDeploy:false only hid it). deploy-on-main.yml now dumps Render service events on failure; read the next failed run's CI log for the boot error (likely alembic migration vs Supabase, a missing prod dep, or startup-timeout on free tier) and fix start.sh/boot path. The 30-min smoke stays red until fixed.

## 🚨 DEEP REVIEW 2026-07-18 PM — why "nothing is working" was TRUE on the live site
Root cause of almost everything: **render.yaml `autoDeploy: false`** — merges never
deployed. The live backend was a weeks-old build (29/57 templates, new endpoints 404,
bot scheduler dead since Jul 5, only 2 bots ever ran). FIXED this session: CI now has a
`deploy` job (green main → Render deploy → poll to live → page on failure) and bot
seeding is ADDITIVE (new templates become site bots on next boot). Remaining from the
evidence:
- [ ] **[P0] Verify post-deploy revival chain** — after the first auto-deploy: /health shows the new build; 57 bots on site; bot last_run_at advancing (scheduler alive); paper orders → check_bot_exits → Trades rows → leaderboard non-empty → perf weighting/pruning engage. Each link was dead on the old build.
- [ ] **[P0] Multi-agent discussions are 2-second canned exchanges** — 3 "messages" in 2s means no real LLM calls (log evidence: run 29652961586 references lib/api/json.rs, a Rust path not in this repo); no visible Discord delivery line. Make discussions call llm_routed() for real, post to Discord via notify, and FAIL LOUD when providers are down instead of emitting filler. Same audit for team-lead-issues + daily-employee-review.
- [ ] **[P0] Desk fills → backend Trades attribution** — the GitHub-Actions desks trade on Alpaca but the backend DB had zero closed trades; order_sync/fill attribution must ingest desk orders (client_order_id prefix qe-*) so dashboards/leaderboard reflect REAL desk activity, not just bot paper-sims.
- [ ] **[P1] Employee individual memory depth** — employee_context exists in agent_memory.json and grows, but verify each named employee accrues per-conversation memory and that it's surfaced in their Discord posts (recall-before-post is wired; confirm on live runs).

## New queue (added 2026-07-18, OA-backend session)
- [ ] **[P1] Tradier sandbox options-data adapter** — free real chains WITH ORATS-computed greeks/IV (see docs/research/OA_BACKEND_STACK_2026.md): real delta-based strikes for desk mleg spreads, real IV rank replacing the HV proxy. Requires free TRADIER_SANDBOX_TOKEN secret (user signup, no card).
- [ ] **[P1] SmartPricing-style laddered repricing** — extend _ensure_filled's one-shot cancel-replace into an OA-style ladder: post at mid, step limit toward market every ~7s (3 steps), then market out; measure in the slippage dashboard.

## New queue (added 2026-07-15, scale-up session)
- [x] **[P1] Per-desk performance attribution + auto-pruning** — DONE 2026-07-16: the live-leaderboard weighting now prunes proven losers to 0.0 (≥20 trades, negative P&L, sharpe<-0.5 → desk skips the strategy entirely; auto-revives when stats recover). 5 tests incl. missing-sharpe-never-prunes.
- [x] **[P1] Route Options-desk income structures through REAL multi-leg orders** — DONE 2026-07-18: wheel/condor/credit-spread/CSP/vol-carry signals now place actual defined-risk mleg spreads (moneyness-picked strikes ~35 DTE via /v2/options/contracts, 1 contract, day). Unresolvable legs place NOTHING and fall back to the underlying proxy. 9 tests incl. never-partial-spread.
- [x] **[P1] Symbol Scout** — DONE 2026-07-16: symbol_scout.py validates every desk symbol against /v2/assets (dead symbols get loud + queued), proposes unwired tradable crypto pairs + curated liquid ETFs; runs in the strategy-scout workflow; 6 tests.
- [x] **[P2] TV-desk hit-rate tracking** — DONE 2026-07-18 as a general pruning rule (applies to every strategy incl. TV): ≥100 trades + losing + win_rate<45% → weight 0.0, desk skips it. Profitable low-hit-rate trend riders are exempt by design (tested).
- [ ] **[P2] Polymarket desk trades a SPY proxy** — wire py-clob-client paper flow so poly_* strategies act on real prediction-market prices instead.
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
