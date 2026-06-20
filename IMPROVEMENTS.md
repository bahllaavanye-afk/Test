# QuantEdge — Improvements & Task Tracker

> **How tasks are tracked (the answer to "where do tasks live"):**
> - **Canonical queue:** GitHub Issues labeled `agent-fix-needed` (the agents already
>   create/work these via `team_lead_issues.py` → `free_agent_engineer.py`).
> - **Human board:** Notion, auto-mirrored by the *Notion ↔ GitHub Issues Sync* workflow.
> - **Cross-session continuity:** this file + `HANDOFF.md`, committed to the repo
>   (chat sessions are ephemeral — only what's committed survives).
> - **Slack:** notifications/visibility only — never the source of truth.

_Last updated: 2026-06-20_

---

## P0 — Reliability (the brain must never silently die)
- [x] **LLM cascade dead (Cloudflare 1010 / no User-Agent)** → fixed (#144).
- [x] **Cascade used only the primary key** → rotate across numbered variants (#145).
- [x] **Brain observability + always-on canary** → `cascade_status()`, `llm_metrics.jsonl`,
      `brain_health.py`, hourly `brain-health.yml` that alerts Slack #infra-alerts (this PR).
- [ ] **Refresh dead provider keys** (only Groq works): Gemini=quota(429), DeepSeek=balance(402),
      Cerebras=no-access, NVIDIA=404. Add free SambaNova/OpenRouter/Together/Hyperbolic keys to
      Doppler → multi-provider resilience. *(Drop key in Doppler; I wire the rest.)*
- [ ] **Make the agent "smoke test" a hard gate** that pages on failure (it was red but unwatched).

## P1 — Real bugs found this session (not yet fixed)
- [ ] **`/ws/prices` all-symbols bug:** subscribes to literal topic `prices:*` but the feed
      broadcasts `prices:{symbol}` → dashboard `LivePriceTicker.tsx` gets nothing.
- [ ] **`test_realtime_endpoints.py` auth helper** posts `username` (endpoint wants `email`) +
      a `.test` email → protected tests silently **skip** (false green). Fix → real E2E coverage.
- [ ] **Redis default `localhost:6379`** spams connection-refused; memory fallback only triggers
      on empty URL → fall back on connect failure.
- [ ] **2 failing scheduled workflows:** "Render — Sync LLM/Slack secrets" (superseded by Doppler —
      delete) and "TV Indicator SOTA" (investigate).

## P1 — Issues the agents themselves flagged in Slack (live triage, 69/97 channels active)
- [ ] `#deploys` — **cross-user data leak fixes** (security; verify it's actually closed).
- [ ] `#leadership-summary` / risk — **VaR threshold exceeded**.
- [ ] `#alpha-research` — **lookahead bias** in momentum strategies.
- [ ] `#squad-qa` / `#ci-failures` — **test failures / bug** backlog.
- [ ] `#okrs` — **Sharpe-ratio shortfall** vs target.
- [ ] `#squad-backend` — **latency issues**; `#squad-frontend` — **screenshot upload failed**.
- [ ] `#finance-ops` — **upcoming paid triggers** (add spend caps before they fire).

## P2 — SOTA upgrades to make this a top-tier AI-first company
1. **Observability + model routing** — Langfuse/OpenTelemetry traces on `llm_common`; route by
   task tier (cheap for bulk classify, strong for plan/code). *(Phase-1 metrics shipped; tracing next.)*
2. **Real memory layer** — replace flat `.github/state/*.json` with Mem0 or Letta backed by your
   existing **Supabase pgvector** (episodic + semantic recall).
3. **Outcome-driven self-improvement** — give the self-improver a *verifiable reward*
   (CI-green + coverage Δ + paper backtest Sharpe Δ); gate agent PRs behind an eval + LLM-judge.
   (DeepSWE / Darwin-Gödel-Machine pattern.)
4. **Durable, event-driven orchestration** — move the core loop (lead→engineer→reviewer) onto
   Temporal/Inngest/LangGraph durable execution instead of fire-and-forget cron.
5. **A2A agent protocol** — typed agent-to-agent coordination; demote Slack to a human digest
   (kills the repeated-message noise).

## Housekeeping
- [ ] Deprecations: pytest-asyncio `event_loop_policy` fixture, Starlette `TestClient`+httpx,
      now-unused `passlib`.
- [ ] Audit stale provider model IDs/endpoints in `llm_common` (Cerebras/NVIDIA).
