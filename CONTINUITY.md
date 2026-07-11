# CONTINUITY — read me first, every session

> **Purpose:** chat sessions are ephemeral and context resets when tokens run out. This
> file (committed to the repo) + the `SessionStart` hook in `.claude/settings.json` make
> every new/resumed session **auto-load the current state** so no memory or progress is
> lost. Keep it current: when you finish or start something material, update this file in
> the same commit.

_Last updated: 2026-07-11._

## ⚡ STATE AS OF 2026-07-11 (read this first)
**Everything runs event-driven 24x7** — 21 team/research workflows + keep-alive chained to CI completions (cron starved; that was THE autonomy bug). 7 desks (incl. new International, 13 country ETFs). Signal ensembling live (agree→boost, conflict→stand aside). 8 advanced strategies deployed. Claude backstop capped $1/day (state/claude_budget.json). Per-employee brains in agent_memory.json employee_context.

**FIRST-TRADE BLOCKER:** crypto desk signals fire (conf 0.90) but Alpaca order POST → 403. Order errors now print the response body — read latest desk-trading-crypto log for the exact reason; playbook: docs/playbooks/ALPACA_CRYPTO_CHECK.md.

**USER UNLOCKS (only manual steps left):** WORKFLOW_PAT secret (playbook exists) · OA_SESSION_COOKIE · Alpaca crypto check.

**TOP QUEUE (IMPROVEMENTS.md):** import_oa_bots.py · public-leaderboard scout · vol-targeted sizing · money-path tests · 887 broad-except sweep · OANDA FX desk.
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
