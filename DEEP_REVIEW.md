# QuantEdge — Deep Review (state of everything)

> Live audit, 2026-06-21. Each area: **🔴 what's breaking** and **🟢 improvements**.
> The single thing blocking "nothing works": the **deployment is disconnected from this repo**
> (see Infrastructure). Fix that first; most "nothing works" symptoms disappear with it.

---

## 0. Infrastructure / Deployment — 🔴 THE root cause
- **Render backend is the wrong app / down.** Account has 2 services (`quantedge-api`
  → `quantedge-api-6orc.onrender.com`, and duplicate `quantedge-api-9jz0`), **both deploying
  from the stale branch `claude/advanced-trading-bot-d5Lmw`, both currently 503/000 (down)**.
- **Frontend calls `quantedge-api.onrender.com`** (bare) — an **orphan 3-route stub** not in
  the account (`/`, `/api/analyze/{ticker}`, `/api/download/{ticker}`). Every real API call
  404s → blank data, dead buttons. This is why the site has "never worked."
- 🟢 **Fix (3 steps):** (1) repoint `quantedge-api` Branch → `main`, set env
  (`DATABASE_URL, SECRET_KEY, ALLOWED_ORIGINS=https://quantedge.vercel.app, DEMO_MODE=true`),
  clear-cache deploy; (2) **delete** the duplicate `quantedge-api-9jz0`; (3) set Vercel
  `VITE_API_URL=https://quantedge-api-6orc.onrender.com/api/v1` (+ `VITE_WS_URL=wss://…/ws`),
  redeploy. Verify `/health` → ok and `/openapi.json` shows ~100 routes (not 3).
- 🟢 Add `RENDER_API_KEY` + `RENDER_SERVICE_ID` to Doppler so the repo's render-monitor
  automation can self-heal future deploy drift.

## 1. Website (Vercel frontend)
- 🔴 Non-functional downstream of #0 (backend down + was pointing at the stub).
- 🔴 **Not mobile-friendly** — 0 `@media` queries, only 27 breakpoint utilities across the app;
  fixed-px widths can overflow on iPhone (390px). `AppShell` uses `bg-[${theme...}]` template
  literals that Tailwind JIT can't see (background may not apply).
- 🔴 Public production was serving a **generic template** ("Unlock the Power…", fake S&P 500)
  that isn't in the repo — confirm Vercel prod builds `frontend/` on `main`.
- 🟢 Modern, on-brand redesign (Bloomberg-dark, real copy) + responsive pass (issue #167).
- 🟢 Demo session (#169, merged) auto-logs-in so the login-free app has a token → data loads
  once the backend is real.

## 2. Brain (free-LLM cascade)
- 🟢 **Healthy now:** 3 live providers — `groq`, `cerebras`, `nvidia` (stale Cerebras/NVIDIA
  model IDs fixed in #163). `llm()` auto-escalates free → OpenRouter → Claude (#155/#159).
- 🔴 Gemini = 429 (quota), DeepSeek = 402 (account balance) — **account-side**, not code.
- 🔴 `OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY` absent from Doppler **and** GitHub secrets, so
  the paid escalation tiers are inert. Add them to Doppler for real resilience.
- 🟢 Hourly `brain-health.yml` canary now alerts `#infra-alerts` when the cascade dies.

## 3. Agents / employees / multi-agent loop
- 🟢 Run 24/7 on GitHub Actions cron (~87–90% success). Loop:
  `team_lead_issues → free_agent_engineer → peer_reviewer`, **plus** the new per-PR AI review
  (#158) and reward-gated auto-merge (#160).
- 🔴 When the brain was down they produced **empty output but green workflows** (silent
  degradation) — mitigated by the canary; make the agent smoke test a **hard, paging gate**.
- 🔴 **AI PR reviewer sees only the diff**, so it raises false positives (flagged
  `OPEN_ACCESS`/`uuid` as "undefined" when they exist). Feed it the full changed files.
- 🟢 Reward gate is opt-in via `agent-auto-merge` label — start labeling agent PRs so the
  self-improvement loop actually merges verified work.

## 4. Trading / desks
- 🔴 **Barely trading** — `#pnl-daily` shows paper equity flat (~$25,001, +$1.75 lifetime,
  3 positions, **0 fills**); ~1 BTC trade where hundreds were expected. Root cause = the
  Render scheduler isn't running 24/7 (service down / free-tier sleep). Keep-alive (#165)
  helps once the service is real. (Issue #168.)
- 🔴 **Options-alpha desk** is research-complete but not productized — no options market-data
  source wired; bots can't be built for it in the UI yet (Bot.market_type added in #153,
  Position/Order options fields in #154 — needs the data feed + seeding).
- 🟢 Desks unified: taxonomy (#152) → bot market-types (#153) → cross-desk tracking (#154).
  7 desks: Equities 53 · Crypto 16 · TradingView 12 · Prediction Markets 8 · Options 7 ·
  Macro 5 · Rates 4.
- 🟢 Seed default enabled bots per desk + a heartbeat that screams when the trade loop is idle.

## 5. Slack (97 channels)
- 🟢 Token works; `#pnl-daily` posts real reports; page audits + PnL reporter active.
- 🔴 **Repeated / templated, low-signal messages** across many channels (long, duplicated).
  Move to concise, visually-summarized digests; demote Slack to notifications (A2A for
  agent-to-agent), per the SOTA roadmap.
- 🔴 Screenshots posted to Slack were of **localhost** (fixed #164 → now the prod site URL).
- 🔴 `/slack/events` had **no signature verification** (regressed; restored in #161).
- 🟢 Per-channel concise summary job exists; wire it to the now-healthy brain.

## 6. GitHub codebase / CI
- 🟢 **825 tests pass**, CI green; brain-independent security guards (#162); Alembic two-head
  migration bug fixed (#154); 3 parse-broken workflows fixed (#150).
- 🔴 **Deprecations:** pytest-asyncio `event_loop_policy` fixture, Starlette `TestClient`+httpx,
  now-unused `passlib` removed (#151) — finish the asyncio/Starlette ones.
- 🔴 Torch-gated ML models (LSTM/Mamba/PatchTST/SSM) unavailable without torch — expected, but
  the `gemini-ml-training` path needs torch + data to actually train.
- 🟢 Continuity system (`CONTINUITY.md` + SessionStart hook) + Notion board keep state across
  sessions.

## 7. Google Drive / Docs / Notion
- 🔴 **Google Drive & Docs are NOT wired** — no Google service-account / OAuth credential in
  Doppler. Reports-to-Drive and meeting-notes-to-Docs can't run until that lands.
- 🔴 **Google login** is fully coded (backend `/auth/google` + callback, frontend button) but
  inert — needs `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET`.
- 🟢 **Notion** is wired — the *QuantEdge Tasks* board exists and is seeded.

## Priority order (what to do first)
1. **Fix the Render deployment + repoint the frontend** (#0) — unblocks literally everything.
2. Seed/enable bots + verify the scheduler runs (#168) → real trades.
3. Add `OPENROUTER_API_KEY` (+ `ANTHROPIC_API_KEY`) and the missing Doppler keys.
4. UI redesign + mobile (#167).
5. Wire Google Drive/Docs (service-account) + Google login creds.
6. Slack digest cleanup; AI-reviewer full-file context; finish deprecations.
