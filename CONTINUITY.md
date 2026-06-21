# CONTINUITY — read me first, every session

> **Purpose:** chat sessions are ephemeral and context resets when tokens run out. This
> file (committed to the repo) + the `SessionStart` hook in `.claude/settings.json` make
> every new/resumed session **auto-load the current state** so no memory or progress is
> lost. Keep it current: when you finish or start something material, update this file in
> the same commit.

_Last updated: 2026-06-21._

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

## SESSION 2026-06-21 — deploy + income (landed on `main`)
- ✅ Deploy-critical (#178): `vercel.json` `/api` proxy → real service `quantedge-api-9jz0`;
  `render.yaml` dropped torch from build (free-tier OOM) + `autoDeploy: false`; CI torch
  tests guarded with `importorskip` (884 collect clean); `scripts/verify_live.py` added.
- ✅ Bot seeder (#179): `app/bots/seed.py` → demo user + paper account + one enabled bot per
  template on boot (idempotent, DEMO_MODE-gated, wired into `start.sh`).
- ✅ Ops (#180): throttle `*/5`→`*/20` crons; fix `HMMRegimeModel` import (class is
  `RegimeDetector`); screenshots default to prod URL.
- ✅ Income (#184, #185): wheel / iron condor / bull-put credit spread / funding carry +
  **cash-sweep SGOV** (~risk-free floor). **25 templates** total.

## BLOCKERS — exact operator actions (this sandbox can ONLY reach Doppler `quantedge/dev` + repo)
It CANNOT write GitHub Secrets, read the Render dashboard, or provision external services.
1. **Render deploy** → add `RENDER_API_KEY` to **Doppler `quantedge/dev`**
   (`doppler secrets set RENDER_API_KEY=... -p quantedge -c dev`). Then an agent can create a
   native-Python service from `main` + deploy via the Render API. Live svc `quantedge-api-9jz0`
   is Docker + out of **pipeline minutes** (account quota — reset/upgrade only).
2. **Agent brain in CI** → the 49 agent workflows read LLM keys from **GitHub Actions Secrets**
   (not Doppler) and they're empty → "all providers failed". Add the LLM keys as GitHub Secrets,
   OR add `DOPPLER_TOKEN` as a GitHub Secret and rewire workflows to `doppler run`.
3. **Shared memory** → add `REDIS_URL` (free Upstash) to Doppler `quantedge/dev`. Non-blocking.
Frontend (Vercel) is live; once the backend is up it serves the real app and all desks self-seed.

## IN THIS BRANCH (`claude/sota-docs-and-fixes`)
- ✅ **Cost-tiered routing** `llm_routed()`: free → OpenRouter open-mid → Claude backstop.
- ✅ `/ws/prices` wildcard delivery fix (all-symbols ticker now receives per-symbol ticks).
- ✅ Redis connection-failure **circuit breaker** + prod default (`REDIS_URL` unset ⇒ no-op,
  no localhost spam).
- ✅ Research docs + `MODEL_ROUTING.md` + this continuity system + tests.

## NEXT (see `IMPROVEMENTS.md` for the full list)
- Refresh dead free-provider keys in Doppler (only Groq works) → multi-provider resilience.
- Langfuse/OTel tracing on `llm_common`; pgvector memory (Mem0/Letta); verifiable-reward
  self-improvement gate; durable orchestration (Temporal); A2A protocol.

## Gotchas (don't relearn these the hard way)
- Cloudflare blocks default urllib UA with error 1010 → always send a browser `User-Agent`.
- pytest-xdist shares one DB → per-worker DB via `PYTEST_XDIST_WORKER` (already handled).
- bcrypt 5.x ≠ passlib 1.7.4 → use bcrypt directly (already handled).
- Scheduled workflows must `checkout` with `ref: main` (a CI test enforces this).
- Binance is geo-blocked (451) and Stooq unreachable in this env; yfinance is rate-limited.
