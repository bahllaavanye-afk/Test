# CONTINUITY — read me first, every session

> **Purpose:** chat sessions are ephemeral and context resets when tokens run out. This
> file (committed to the repo) + the `SessionStart` hook in `.claude/settings.json` make
> every new/resumed session **auto-load the current state** so no memory or progress is
> lost. Keep it current: when you finish or start something material, update this file in
> the same commit.

_Last updated: 2026-07-15._

## ⚡ STATE AS OF 2026-07-18 PM (read this first)
**THE QUEUE NOW DRAINS ITSELF**: improvements-worker.yml (event-chained, ~24h gate)
converts top unchecked IMPROVEMENTS.md items into deduped `agent-fix-needed` issues →
Free-Agent Engineer implements → reward-gated PR. Scouts append, worker drains — the
full loop runs with no session. Blocked items (user unlocks/missing feeds) are skipped,
never filed. OA backend analysis in docs/research/OA_BACKEND_STACK_2026.md: OA runs on
broker APIs (Tradier flagship; greeks are ORATS-via-Tradier) + a rules engine QuantEdge
already mirrors; queued the two real gaps — Tradier sandbox data adapter (free real
greeks/IV; needs TRADIER_SANDBOX_TOKEN) and SmartPricing laddered repricing.
Synthetic options backtester landed (app/backtest/options_synthetic.py, 9 tests).

## Earlier 2026-07-18 state
All queue P1s+P2s from the scale-up session are DONE except Polymarket CLOB (needs
POLYMARKET_PRIVATE_KEY wiring — queued). Landed since 07-15: auto-pruning (leaderboard
weight 0.0 at ≥20 trades/neg P&L/sharpe<-0.5, PLUS hit-rate rule ≥100 trades/losing/
win<45%); Symbol Scout (validates desk universes vs /v2/assets, proposes new pairs/ETFs,
runs in strategy-scout workflow); Options desk places REAL defined-risk mleg spreads for
wheel/condor/credit-spread/CSP/vol-carry (moneyness strikes ~35 DTE, never-partial-legs,
proxy fallback); research→registry pipeline (scout seeds strategy_generator's prompt via
state/research_seed.json).

## Previous state (2026-07-15 PM)
**SCALE-UP LANDED: 53→103 wired strategies, 9 desks.** New TV Indicators desk (12 TV
strategies, #desk-tv-indicators). Universes: Equities 30 syms, Crypto 20 pairs, Options 8,
Macro/FX 14 (full rates/credit complex), StatArb 12, Commodities 12, International 19.
Every wirable registry strategy now trades; the 10 leftovers are data-source-blocked
(documented in strategy_scout.KNOWN_EXCLUSIONS). **Strategy Scout** (workflow, event-
chained, 3-day gate) re-audits registry-vs-desks every run, posts to #desk-research,
appends to IMPROVEMENTS.md when new strategies appear, and rotates build-next ideas from
a 15-item research backlog — "always finding new strategies" is now automated.
**BOT_TEMPLATES 49→57**: added OA's flagship public bots (0DTE breakeven IC, short put
ladder, Black Swan hedge, 1-1-2, Rhino BWB, weekly diagonal, 16Δ strangle defined-risk,
Slow&Steady IC). Private leaderboard copies still blocked on OA_SESSION_COOKIE.

## Earlier 2026-07-15 state
**8 desks** (Commodities added: 8 ETF proxies, TSMOM/Donchian/MR) + OANDA FX desk with a
ForexFactory red-folder gate (stands aside ±30min around High-impact prints, fail-open).
**QUARANTINED is EMPTY** — all 115 registry strategies pass the contract suite via the
shared hard-budget fail-soft (`app/strategies/_failsoft.py`; yfinance uses curl_cffi, so
socket kills don't reach it — the daemon-thread budget is the only real guard).
**AlpacaBroker RESTORED**: improver PR #420 (Jul 9) truncated brokers/alpaca.py to a
"(truncated for brevity)" stub — 6 of 7 interface methods deleted, class un-instantiable,
backend silently on yfinance fallback. Also fixed the never-valid exception imports
(alpaca.common.exceptions). Improver now rejects shrunken/elided LLM outputs and skips
>8k-char files; `test_broker_interface.py` fails loudly if any broker goes abstract.
Cash-aware sizing live (#578): orders capped to 95% of relevant buying power, $25 floor.
Health hard gate: agent-health-check fails + pages #infra-alerts on critical findings.
Per-bot detail view (OA parity): P&L graph + open positions + trade history in BotBuilder.

## Previous state (2026-07-11)
**Everything runs event-driven 24x7** — 21 team/research workflows + keep-alive chained to CI completions (cron starved; that was THE autonomy bug). 7 desks (incl. new International, 13 country ETFs). Signal ensembling live (agree→boost, conflict→stand aside). 8 advanced strategies deployed. Claude backstop capped $1/day (state/claude_budget.json). Per-employee brains in agent_memory.json employee_context.

**🎉 FIRST TRADES PLACED 2026-07-12 07:59 UTC** — recovery flattened 22 orphaned positions (cash was -27,476) and the SAME run placed 2 real paper orders (UNI/USD + AAVE/USD, $755 notional, avellaneda_stoikov_mm conf 0.90/0.74). Desks are TRADING. Watch fill-tracking → Trades table → dashboard P&L now populate.

**Blocker history:** the 403 was 'insufficient balance for USD' — paper account overdrawn -$25,207, $0 available (orphaned notional buys nothing tracked). recover_negative_cash() in the desk now auto-flattens the orphaned PAPER book once (triple-guarded: paper URL assert, not-live, AUTO_FLATTEN_ON_NEGATIVE_CASH kill switch) — cash frees, next run trades. If Alpaca still refuses: reset the paper account in the dashboard (instant $100k).

**USER UNLOCKS (only manual steps left):** WORKFLOW_PAT secret (playbook exists) · OA_SESSION_COOKIE · Alpaca crypto check.

**DONE 2026-07-11 session 2 (7 queue items, PRs 488-494):** import_oa_bots.py · vol-targeted sizing (20% budget, 0.5-2x clamp) · money-path tests (9, mock Alpaca) · strategy contract test (115 audited, 18 violators found) · desk 10s per-strategy timeout (freezes impossible) · 5 violators fixed fail-soft + un-quarantined · OA public-scrape item was a wrong-domain false positive (optionalpha.com is auth-walled everywhere — cookie/Cowork only).
**DONE 2026-07-12 session 3 (PRs 496-517):** FX desk LIVE (OANDA practice, 7 majors 24/5, 8th desk) · fail-soft guards on 18 strategies, quarantine 18->7 · frontend-build CI gate (truncations now fail PRs, no Vercel dependency) · daily loss circuit breaker (2% vs prior close, stateless) · money-path silent-except sweep (trailing stops, HRP fallback, OCO cancels now logged) · basis_carry/funding_settlement_timer contract updates.
**TOP QUEUE NOW:** fix remaining 13 quarantined strategies (see backend/tests/unit/test_strategy_contract.py) · 887 broad-except sweep (brokers/execution/risk first) · OANDA FX desk · frontend component tests · workflow consolidation.
**Playbooks:** docs/playbooks/ (OA copy, bookmarklet, PAT, Alpaca). Verify a desk log + live-stats before claiming anything works.


## Mission
QuantEdge is an AI-first quant-trading company that must run **24/7**, cheaply, and
**self-improve**. `TRADING_MODE` stays `paper`. Never print secret values. Secrets live in
**Doppler** (single source of truth) and are injected as env vars.

## How tasks/memory persist (the source-of-truth map)
- **Canonical task queue:** GitHub Issues labeled `agent-fix-needed`.
- **Human board:** [Notion — QuantEdge Tasks](https://app.notion.com/p/bec54f8a79444c2399316365a07e0291)
  (mirrors GitHub Issues / `IMPROVEMENTS.md`).
- **Cross-session continuity:** **this file** + `IMPROVEMENTS.md` + `HANDOFF.md`.
- **Research (durable):** `docs/research/AI_COMPANY_SOTA.md`,
  `docs/research/LLM_COST_OPTIMIZATION.md`, `docs/MODEL_ROUTING.md`.
- **Slack:** notifications only — never the source of truth.
- **Agent runtime memory:** `.github/state/company_brain.json` (+ `llm_metrics.jsonl`).

## Resume procedure (what a fresh session should do)
1. The `SessionStart` hook prints this file + open `IMPROVEMENTS.md` items automatically.
2. `git log --oneline -10` to see what already landed.
3. Pick the top unchecked item in `IMPROVEMENTS.md` and continue.
4. Work on a branch off `main`; open a **draft PR**; never force-push.

## DONE (landed on `main`)
- ✅ LLM cascade revived — browser User-Agent fixes Cloudflare 1010 (#144).
- ✅ Key rotation across numbered free-provider keys (#145).
- ✅ Brain observability: `llm_metrics.jsonl`, `cascade_status()`, hourly `brain-health.yml`
  canary → Slack `#infra-alerts` (#146).
- ✅ Doppler single-source secrets (#139); Bot Archiver soft-delete/restore (#137).
- ✅ **2026-06-24 (11 PRs):** options productization (#188); brain cascade fixed on `main`+default
  branch incl. reasoning-model extraction + in-call key fallthrough (#189/#199); backend-health
  banner + Render runbook (#190); TradeStation spread routing (#198); Kalshi public reads (#203);
  guards for income/macro strategies (#202), momentum lookahead (#207), cross-tenant isolation (#208);
  pytest-asyncio deprecation removed (#206). Backend verified deploy-ready locally (158 routes, seeds
  29 bots). **Live blockers are human-only: Render build-minutes (#197) + default branch → `main` (#196).**

## 2026-07-07 — THE no-trades root cause found & fixed
- **Zero trades for days despite open markets** traced to ONE bug: 3 crypto
  strategies (basis_carry, funding_settlement_timer, mvrv_zscore_timing) did a
  module-level `import aiohttp`, which isn't in the desk workflow pip env. That
  crashed `from app.strategies import STRATEGY_REGISTRY` → desk_order_placer
  FATAL'd at line 690 on EVERY run (scheduled + event-driven), before placing a
  single order. One optional dep took down all 6 desks. Fixed: lazy aiohttp
  import at call sites + aiohttp added to desk deps + regression test
  (test_registry_import_safe). Merged #348.
- Gate race fixed (#324): auto-merge ignored ops-sync/Vercel non-gating checks;
  green PRs had stranded overnight. Improver now dispatches CI on its own PR
  branch (GITHUB_TOKEN fires no events) — closes the org perpetual-motion loop.
- 5 evidence-based hybrid strategies live (#303); 15 OA clones + factory + ML
  variant generation + synthetic-BS backtester scoring every options template.
- STILL MANUAL (1 paste): DISCORD_WEBHOOK_URL secret — the webhook works (204),
  it's just not saved as a GitHub secret, so automated Discord senders have no pipe.

## 2026-07-06 — org layer + honest data + the cron-starvation fix
- **Fake data purged:** live-stats hardcoded Sharpe 2.1/68%/14.7% → computed from real
  trades (null until data); tearsheet 500 (Postgres-only date_trunc) → DB-agnostic;
  Landing hero shows "—" until real metrics. Smoke gate (smoke-test.yml) verifies the
  DEPLOYED api twice hourly + post-merge and pages #ci-failures.
- **Google login 404 root cause:** OAuth callback redirected to cors_origins[0] = the
  dead quantedge.vercel.app stub. Now FRONTEND_URL-aware; render.yaml reordered.
- **Org layer:** team_lead.py (reviews automerge PRs — revokes label on protected-file
  edits (the #298 failure), assigns issues, reports to #leadership-summary) +
  data_team.py (bars coverage/freshness/ML-readiness for every desk symbol) +
  bot lifecycle manager (disable losers / promote winners / instantiate templates).
- **P&L loop closed:** desk_trade_sync (Alpaca fills → Trade rows, FIFO, idempotent);
  /leaderboard/live; perf-weighted desk sizing (winners 1.3x, losers 0.6x);
  /bots/{id}/performance (per-bot cum-P&L series).
- **Options desk:** wheel/iron_condor/credit_spread wired with iv_rank injection
  (verified firing); crypto desk 4 → 12 Alpaca pairs; India ETF sleeve (INDA/EPI/SMIN).
- **CRON STARVATION (critical infra fact):** GitHub schedules barely fire in this repo
  (keep-alive cron */5 = 162 runs EVER; discord-sync daily NEVER ran). Anything critical
  must be EVENT-driven → new ops-sync job in CI (test.yml) creates Discord channels,
  registers slash commands, and relays keys GitHub→Render on every CI run.
  VERIFIED 08:25Z: commands HTTP 200; DISCORD_BOT_TOKEN/GOOGLE_SERVICE_ACCOUNT_JSON/
  ALPACA keys → Render all HTTP 200. Bot was in NO server (root cause of empty
  Discord) — user re-invited it 08:2x.
- Hourly standup → Discord #engineering + self-provisioned "QuantEdge Standups" Google
  Doc (ensure_doc finds-or-creates + silent share; link posted in Discord, not email).

## 2026-07-05 — Discord two-way + pipeline self-heal
- Discord fully wired: webhook delivery (verified 204), per-employee bot profiles,
  per-channel routing, Discord-primary employee runs, shared `notify.py` chokepoint.
- **Slash commands** (`/status /pnl /health /run-bot`): Ed25519-verified endpoint at
  `/api/v1/discord/interactions` + `discord-commands-sync.yml` (needs DISCORD_BOT_TOKEN).
  Caught+fixed a real merged bug: equity lives on AccountSnapshot, not Account.
- **`auto-pr.yml`**: pushes to `claude/**` now auto-open an `automerge` PR + dispatch CI —
  closes the stranded-branch gap (needs repo setting "Allow Actions to create PRs" ON).
- Bars-fetch root cause fixed (#286): Alpaca defaulted `start` to today → every desk ran
  on an empty cache. Now 200d of real bars; 3 crypto strategies pruned (Binance 451 /
  CoinGecko 401 unreachable from US runners → issue #289 to reroute via Bybit/OKX).
- ML experiments VERIFIED working: Sunday run, real Alpaca data, honest OOS results (#288).

## 2026-07-04 — the big landing (12 PRs merged in one session)
- ✅ #270 commodity_reversion strategy + commodities research; #271 stale backend URLs → agb8;
  #272 keep-alive hardening; #273 website fixes (Google OAuth, WS revival, cold-start retry);
  #274 Slack fail-loud pipeline; #275 pydantic v2; #276 reward-gate all editors; #277 broker
  logging; #278 **ANTHROPIC_API_KEY wiring** (haiku default backstop, sonnet PR reviews,
  @claude mention agent, key-sync workflow); #279 **Discord failover** at all 3 posting
  chokepoints + models:read + weekly claude-queue-worker; #280 Option Alpha parity research.
- ✅ **Key sync VERIFIED live**: self-trigger hack fired on #278's merge — 1-token Haiku call OK,
  Doppler upserted, Render env var set (deploy call was 429 but the merge train's auto-deploy
  covers it). The sync re-runs whenever `sync-anthropic-key.yml` itself changes on main.
- ✅ #281 (`claude/stoic-johnson-7z4wtz`) **why-no-trades root fix**: BotRunner was NEVER
  instantiated (29 bots at runs=0) → ignition one-shot job in `tasks/scheduler.py`;
  `check_bot_exits` was never scheduled (positions would never close) → 5-min interval job;
  self-ping keeps the Render dyno awake; Option Alpha endpoints the frontend 404'd on
  (`/options/rules/validate`, `/flow`, `/put-call-ratio`, `/wheel`, `/macro-calendar`,
  `/next-fomc`); real order submit in Options.tsx; `automerge` drafts now auto-ready+merge.

## NEXT (see `IMPROVEMENTS.md` for the full list)
- Verify on prod after redeploy: bots show `last_run`/`run_count` > 0, `/api/v1/scheduler/jobs`
  exists, trades appear once markets open (Jul 6 Mon).
- ML experiments: torch can't run on the 512MB Render dyno — run training in GitHub Actions
  (CPU-friendly models) or a worker with the `[ml]` extra; nightly retrain job now actually
  fires (scheduler starts) but degrades without torch.
- Human-only: DISCORD_WEBHOOK_URL secret (GitHub + Render), UptimeRobot pinger, Alpaca keys
  rotation, delete the Vercel stub project (`quantedge.vercel.app`), Slack plan decision.
- Refresh dead free-provider keys in Doppler (only Groq works) → multi-provider resilience.
- Langfuse/OTel tracing on `llm_common`; pgvector memory (Mem0/Letta); verifiable-reward
  self-improvement gate; durable orchestration (Temporal); A2A protocol.

## Gotchas (don't relearn these the hard way)
- Cloudflare blocks default urllib UA with error 1010 → always send a browser `User-Agent`.
- pytest-xdist shares one DB → per-worker DB via `PYTEST_XDIST_WORKER` (already handled).
- bcrypt 5.x ≠ passlib 1.7.4 → use bcrypt directly (already handled).
- Scheduled workflows must `checkout` with `ref: main` (a CI test enforces this).
- Binance is geo-blocked (451) and Stooq unreachable in this env; yfinance is rate-limited.
