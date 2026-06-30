# QuantEdge — Improvements & Task Tracker

> **How tasks are tracked (the answer to "where do tasks live"):**
> - **Canonical queue:** GitHub Issues labeled `agent-fix-needed` (the agents already
>   create/work these via `team_lead_issues.py` → `free_agent_engineer.py`).
> - **Human board:** [Notion — QuantEdge Tasks](https://app.notion.com/p/bec54f8a79444c2399316365a07e0291)
>   (seeded from this file; mirror via the *Notion ↔ GitHub Issues Sync* workflow).
> - **Cross-session continuity:** this file + `HANDOFF.md`, committed to the repo
>   (chat sessions are ephemeral — only what's committed survives).
> - **Slack:** notifications/visibility only — never the source of truth.

_Last updated: 2026-06-29_

---

## Session 2026-06-29 — review backlog (see `docs/REVIEW_2026-06-29.md`)
Queued for the autonomous loop / employees. Priority order top-to-bottom.
- [ ] **[P0] Forex desk** — add `market_type="forex"` strategies (carry, trend/momentum), register,
      add "Forex" to `_MARKET_TYPE_DESK`, route data_loader to `EURUSD=X` etc., + scheduled desk.
- [ ] **[P0] Commodities desk** — add `market_type="commodity"` strategies (term-structure roll,
      momentum, gold/oil mean-reversion), register, add "Commodities" desk, route `GC=F`/`CL=F`.
- [ ] **[P0] Render sleep** — external uptime pinger (UptimeRobot) or paid tier so in-app employees
      don't halt (`/health` returned 000 — backend asleep).
- [~] **[P1] Audit & consolidate 86 workflows** — employee manifest shipped (`docs/WORKFLOWS.md`,
      via `scripts/gen_workflow_manifest.py`): 87 workflows / 70 scheduled, dup-families flagged
      (`slack-*`×10, `agent-*`×6, `render-*`×5, `strategy-*`×5). Next: actually dedupe the families.
- [x] **[P1] Durable auto-merge** — `auto-merge.yml` lands `automerge`-labeled PRs once all checks
      pass (no human merge). Removes the last manual step for the autonomous loops/employees.
- [ ] **[P1] Employee-health hard gate** — make the agent smoke test page on failure; verify
      `agent-health-*`/`system-status` actually alert when an employee is stale.
- [x] **[P1] Reward-gate self-improvement** — `continuous_improver.py` now pushes a throwaway
      `improver/run-*` branch and opens an `automerge` PR instead of pushing to `main`. The full CI
      suite must pass before changes land (auto-merge.yml). Stops the unvalidated direct-to-main
      commits that broke the app 3× (slots=True, @root_validator, dead scheduler) in one session.
- [ ] **[P1] Wire Alpaca crypto into `price_feed`** for live quotes (Binance still geo-blocked for live).
- [ ] **[P1] Narrow 435 broad `except Exception`** — start with `tasks/`, `brokers/`, `llm`; add logging.
- [ ] **[P1] Audit stale provider model IDs** in `llm_common` (Cerebras/NVIDIA).
- [ ] **[P2] ML employees inert on prod** — run with `[ml]` extra on a worker, or mark degraded.

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
- [ ] **Make the agent "smoke test" a hard gate** that pages on failure (it was red but unwatched).

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
- [ ] **"TV Indicator SOTA" scheduled workflow** — still to investigate.

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
- [ ] Audit stale provider model IDs/endpoints in `llm_common` (Cerebras/NVIDIA).
