# QuantEdge — Session Handoff

> Living handoff doc. A new chat session starts with a **fresh container and no memory
> of previous chats** — only what is committed to the repo survives. This file + the
> auto-loaded `CLAUDE.md` files are how context carries over. Read this first.

_Last updated: 2026-06-19 • Branch: `claude/advanced-trading-bot-d5Lmw`_

---

## 0. How to carry context into a new chat

Chat memory does **not** transfer automatically between sessions — each session is a
clean container that re-clones the repo. To preserve context, it must be **committed**:

- **This file (`HANDOFF.md`)** — running status + next tasks. Update it before ending a session.
- **`CLAUDE.md` (root + per-module)** — auto-loaded into every session's context.
- The git history and branch state.

So: "copy context/memory" = commit it here. There is no hidden cross-chat memory.

---

## 1. Environment / network (READ BEFORE DEBUGGING "nothing works")

This repo runs in a sandboxed container with a **network egress allowlist**.

- A running container keeps whatever policy it was **started** with. Changing the
  environment's network setting (e.g. to "full") does **not** affect an already-running
  container — you must **start a new session** to get a container under the new policy.
- As of this session the allowlist permitted **dev/package hosts** (PyPI, GitHub, npm,
  Ubuntu → HTTP 200) but **blocked runtime/data hosts**: `paper-api.alpaca.markets`,
  `data.alpaca.markets`, `query1/query2.finance.yahoo.com`, `slack.com`,
  `api.render.com`, even `example.com` → **HTTP 403 "Host not in allowlist."**

**Consequence:** with data hosts blocked, the price feed gets nothing, strategies skip
("market closed / no OHLCV"), Slack MCP tools fail, and the UI shows empty/synthetic
data. This is an **environment limitation, not necessarily app breakage.** First thing
in a new session — confirm what's actually reachable:

```bash
for h in paper-api.alpaca.markets data.alpaca.markets query1.finance.yahoo.com \
         slack.com api.render.com example.com; do
  echo "$h -> $(curl -sS -m 8 -o /dev/null -w '%{http_code}' https://$h 2>/dev/null)"
done
```

If these are still 403, live data/Slack will not work regardless of code changes.

---

## 2. Running the stack locally (verified working this session)

Backend (FastAPI, paper mode, in-process SQLite) — **must run from `backend/`**:
```bash
cd /home/user/Test/backend && \
DATABASE_URL=sqlite+aiosqlite:///./dev.db SECRET_KEY=dev-only-secret-key-32-bytes-hex-x \
TRADING_MODE=paper uvicorn app.main:app --host 0.0.0.0 --port 8000
# health: curl http://localhost:8000/health  -> {"status":"ok",...,"mode":"paper"}
```

Frontend (Vite/React):
```bash
cd /home/user/Test/frontend && npm run dev -- --port 5173 --host 0.0.0.0
```

Screenshots of all pages (Playwright browser lives at a non-default path):
```bash
PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers NODE_PATH=/opt/node22/lib/node_modules \
  node /tmp/shot.js     # see script template in prior session; chromium-cli not installed
```
All 12 frontend pages render (Landing, Login, Dashboard, EquityTrading, CryptoTrading,
Polymarket, Comparison, BacktestLab, Experiments, MLInsights, Analytics, RiskManager).

---

## 3. Test-coverage gap (IMPORTANT — explains "tests pass but nothing is live")

`backend/tests/integration/test_realtime_endpoints.py` is named like an E2E real-time
suite but **only asserts liveness, not real data**:

- Dominant assertion is `status_code < 500` or `in (200, 404)` → a **404 or empty `[]`
  passes**. The quote test comment even says "404 if Alpaca not connected is OK."
- The `_shape()` helper returns early on empty lists ("empty list is OK").
- **No** assertion anywhere on a timestamp, non-zero price, or data freshness.
- WebSocket "tests" are unit tests of `ConnectionManager` (broadcast-with-no-subscribers);
  the real chain **price_feed → Redis → WebSocket → browser is untested end-to-end**.
- In-process ASGI + SQLite + `DEMO_MODE`/synthetic fallback ⇒ "broker down / no
  real-time" is structurally invisible to CI.

**To actually catch real-time breakage, add tests that:**
1. Assert market-data responses contain a **price and a recent timestamp** (fail on
   empty/stale), gated behind a `@pytest.mark.live` marker so they only run when data
   hosts are reachable.
2. Open a **real WebSocket** to `/ws/prices`, subscribe, and assert a tick arrives within
   N seconds — a true E2E real-time test.
3. Run a smoke test against a **live-network** container in CI (separate job), not just
   the mocked in-process one.

---

## 4. Render deploy `build_failed` — diagnosis so far

`render-sync-secrets.yml` last run: secrets synced OK (17 keys), but the triggered Render
deploy returned `build_failed`. Findings:

- **Not a dependency problem.** Reproduced the Docker build's install step locally with
  PyPI reachable: `uv pip install --system --no-cache -e .` from `backend/pyproject.toml`
  → **EXIT 0**, all deps (incl. xgboost, lightgbm, scipy, scikit-learn, pandas) resolved
  and installed cleanly. So `pyproject.toml` is fine.
- **Still unknown / next steps** (need `api.render.com` reachable OR the Render dashboard):
  pull the actual Render **build logs** for the failed deploy id to see the real error.
  Likely candidates: free-tier **build OOM** on heavy wheels, a `COPY`/context issue in
  `backend/Dockerfile`, or a healthcheck/start failure being reported as build failure.
- `backend/render.yaml` confirms `TRADING_MODE: paper` (keep it that way) and Docker build
  via `./backend/Dockerfile`.

---

## 5. "Option Alpha copy" features — status (audited this session, evidence-based)

| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 1 | Visual Bot Builder | ✅ DONE | conditions/actions/exits + templates + JSON persistence + CRUD wired (`frontend/src/pages/BotBuilder.tsx`, `backend/app/api/v1/bots.py`, `models/bot.py`) |
| 2 | Bot Desk / Dashboard | ✅ DONE | desk grouping + live P&L/win-rate from DB (`/bots/summary/all`) |
| 3 | Copy Trading | ✅ DONE | Sharpe leaderboard, follow/unfollow, size multiplier (`backend/app/api/v1/copy_trading.py`) |
| 4 | Bot Archiver | ⚠️ PARTIAL | `app/archive/trade_archiver.py` logs **trade events** to JSONL, but bots are **hard-deleted** (`bots.py` `db.delete`). No `is_archived`/`archived_at` on `Bot`, no retire+restore, no preserved config/P&L history. |
| 5 | Multi-broker "desks" (same bot across accounts) | ⚠️ PARTIAL | schema allows many `Account`s, but a bot has one optional `account_id`; **no fan-out execution** to mirror a bot across accounts. |
| 6 | TradeStation options | ⛔ STUB | `app/brokers/tradestation.py` has OAuth + equity orders only. **No option chain, no multi-leg orders, no Greeks.** |

### Remaining work to "complete Option Alpha copy" (prioritized)

1. **Bot Archiver (do first — self-contained, fully testable on local SQLite, no network):**
   - Add `is_archived: bool` + `archived_at: datetime|None` to `backend/app/models/bot.py`.
   - Replace hard `DELETE /bots/{id}` with **archive** (soft-delete, preserves row +
     config + linked trades); add `POST /bots/{id}/restore`; add `?archived=true` filter
     to the list endpoint. Exclude archived bots from the runner/summary.
   - Frontend `Archive.tsx`: add an "Archived Bots" view with Restore; change the bot
     delete action to "Archive."
   - Tests: archive → bot hidden from active list but present in archived list + trades
     retained → restore → active again.
2. **Multi-account fan-out:** let a bot target a list of accounts (or "all"); in
   `backend/app/bots/engine.py` loop accounts and route one order per account. Test with
   2 paper accounts.
3. **TradeStation options:** `get_option_chain`, `get_greeks`, multi-leg
   (`place_spread_order`) on `tradestation.py`. **Cannot be live-tested without TS creds +
   network** — write against API docs, unit-test the request-building, mark live paths skip.

---

## 6. Other standing/pending items (from earlier sessions — verify before assuming)

- **Slack channel analysis** the user originally asked for — blocked until Slack host is
  reachable (new full-network session). Re-run `mcp__slack__slack_list_channels` first.
- **Frontend redesign mockups** (2–3 options, terminal-dense vs consumer-fintech) — user
  wanted to "see options first." Not started.
- **15-feature ML/strategy autopilot backlog** (HRP/CVaR, Binance funding features, PPO RL
  exec, SSM/Mamba, cross-sectional momentum, HMM regime, ensemble weight opt, LOB features,
  LLM alpha mining, intraday strategies, investor tearsheet) — was dispatched via the
  autopilot workflow in an earlier session; **status unconfirmed.** Note: backend startup
  log shows `SSMPredictor` and `HMMRegimeModel` **failing to import** ("cannot import
  name ... from app.ml.models.*") — these two are at minimum broken/incomplete and worth
  checking against that backlog.

---

## 7. Hard constraints (do not violate)

- `TRADING_MODE` stays `paper` everywhere. No live trading.
- No paid APIs in autonomous/LLM loops. Free providers only.
- Never print actual secret/API-key values (masked/presence checks only).
- No `git add -A`/`git add .` — stage files by name. No force-push. No `reset --hard`
  without explicit OK. Only commit when asked. Push only to
  `claude/advanced-trading-bot-d5Lmw`.
- GitHub only via `mcp__github__*` MCP tools (repo `bahllaavanye-afk/test`); no `gh` CLI.
- Ground claims in verified evidence (logs/tools), not assumption.
