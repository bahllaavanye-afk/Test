# QuantEdge — Session Handoff

> Living handoff doc. A new chat session starts with a **fresh container and no memory
> of previous chats** — only what is committed to the repo survives. This file + the
> auto-loaded `CLAUDE.md` files are how context carries over. Read this first.

_Last updated: 2026-06-20 • Branch: `claude/stoic-johnson-7z4wtz`_

> **Session 2026-06-20 summary (what changed this session):**
> - **Bot Archiver shipped** (§5 item 1 → ✅): soft-delete + restore + archived filter,
>   backend + frontend + tests. Details below.
> - **Real-time E2E tests added** (§3): `@pytest.mark.live` quote-freshness + a *real*
>   WebSocket tick test, plus deterministic `ConnectionManager` delivery tests.
> - **Fixed a real auth bug:** `passlib 1.7.4` + `bcrypt 5.x` were incompatible and
>   **password hashing crashed** (register/login broken). `app/utils/security.py` now uses
>   `bcrypt` directly (same `$2b$` format → existing hashes still verify).
> - **Fixed ML registry import without torch** (§6): `app/ml/models/__init__.py` now degrades
>   on any import error, and `transformer.py`/`itransformer.py` use a guarded `nn.Module`
>   base, so `TransformerPredictor`/`iTransformerPredictor` import (non-None) without torch.
> - **Fixed a workflow secret-guard:** `member-standup.yml` set `ANTHROPIC_API_KEY: ""`
>   (tripped `test_no_anthropic_key_leaked_in_workflows`) → now `"disabled"`.
> - **Test suite: 918 passed, 95 skipped, 0 failed** (excluding 3 torch-only unit modules
>   that `import torch` at top-level and can't collect without the optional `[ml]` extra).
> - **Slack STILL blocked:** the MCP Slack **token is `invalid_auth`** (host reachable, token
>   dead). Channel analysis remains impossible until the token is refreshed.

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
- **2026-06-20 update — this session ran on a MUCH more permissive network** (no 403s):
  `slack.com` 200, `api.render.com` 200, `example.com` 200, `paper-api.alpaca.markets` 404,
  `data.alpaca.markets` 401 (needs keys), `query1.finance.yahoo.com` 429 (rate-limited).
  **Still restricted:** `api.binance.com` **451** (geo-blocked), `stooq.com` **000**
  (unreachable). Net effect: **equity** real-time data flows via the free **yfinance** path
  (verified — a real SPY tick arrived through `/ws/prices/SPY` in the live WS test), but
  **crypto (Binance) is blocked** and the REST `/market-data/quote` is Alpaca-only so it
  returns `source: "unavailable"` without Alpaca keys. Local boots also have **no Redis**
  (`localhost:6379` refused) → price caching degraded (set `REDIS_URL=""` to use the
  in-process memory cache).
- (Earlier sessions saw the opposite — dev hosts 200 but all runtime/data hosts
  **403 "Host not in allowlist."** The policy is per-container; always re-run the probe.)

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

**ADDRESSED (2026-06-20):** `tests/integration/test_realtime_live.py` adds
`@pytest.mark.live` tests that assert a **non-zero price + recent, parseable timestamp**
(skips without real Alpaca creds) and open a **real WebSocket** to `/ws/prices/{symbol}`
asserting a real-data tick arrives (passed here via yfinance). `pytest.ini` registers the
`live` marker; run them with `-m live` (item 3, a separate live-network CI job, is still
TODO). Also note: the previous "WebSocket tests" only covered broadcast-with-no-subscribers
— `test_websocket.py` now also asserts a subscriber actually **receives** the broadcast and
that dead sockets are pruned. **Two pre-existing bugs that made the realtime suite silently
useless were found and fixed:** auth login took `username` (endpoint wants `email`) and used
a `.test` email the validator rejects — so every protected test **skipped**; and password
hashing crashed under bcrypt 5.x. With those fixed the protected endpoints actually run.

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
| 4 | Bot Archiver | ✅ DONE | `Bot` now has `is_archived`/`archived_at`; `DELETE /bots/{id}` **soft-archives** (preserves row + config + linked trades), `POST /bots/{id}/restore` brings it back, `GET /bots/?archived=true` lists archived. Archived bots excluded from `/bots/summary/all` and the scheduler. Frontend: "Archive" action + "Archived" tab w/ Restore in `BotBuilder.tsx`. Alembic `i4d5e6f7a8b9`. Tests: `tests/integration/test_bot_archive.py`. |
| 5 | Multi-broker "desks" (same bot across accounts) | ⚠️ PARTIAL | schema allows many `Account`s, but a bot has one optional `account_id`; **no fan-out execution** to mirror a bot across accounts. |
| 6 | TradeStation options | ⛔ STUB | `app/brokers/tradestation.py` has OAuth + equity orders only. **No option chain, no multi-leg orders, no Greeks.** |

### Remaining work to "complete Option Alpha copy" (prioritized)

1. **Bot Archiver — ✅ DONE (2026-06-20).** Implemented exactly as scoped:
   `is_archived`/`archived_at` on `Bot`; `DELETE /bots/{id}` soft-archives;
   `POST /bots/{id}/restore`; `?archived=true` list filter; excluded from
   `/bots/summary/all` + `BotRunner.start()`. Frontend archived-bots view + Restore lives
   in `BotBuilder.tsx` (NOT the trade-event `Archive.tsx`, which is unrelated). Alembic
   migration `i4d5e6f7a8b9` for prod Postgres. Tests in `test_bot_archive.py` (4 passing).
2. **Multi-account fan-out:** let a bot target a list of accounts (or "all"); in
   `backend/app/bots/engine.py` loop accounts and route one order per account. Test with
   2 paper accounts.
3. **TradeStation options:** `get_option_chain`, `get_greeks`, multi-leg
   (`place_spread_order`) on `tradestation.py`. **Cannot be live-tested without TS creds +
   network** — write against API docs, unit-test the request-building, mark live paths skip.

---

## 6. Other standing/pending items (from earlier sessions — verify before assuming)

- **Slack channel analysis** — STILL BLOCKED, but now for a different reason: `slack.com`
  is reachable (200), yet the **Slack MCP token is invalid** — every call returns
  `{"ok":false,"error":"invalid_auth"}` (tried repeatedly 2026-06-20). This needs a valid
  `SLACK_BOT_TOKEN` in the Slack MCP server config; it cannot be worked around from code.
- **Frontend redesign mockups** (2–3 options, terminal-dense vs consumer-fintech) — user
  wanted to "see options first." Not started.
- **15-feature ML/strategy autopilot backlog** (HRP/CVaR, Binance funding features, PPO RL
  exec, SSM/Mamba, cross-sectional momentum, HMM regime, ensemble weight opt, LOB features,
  LLM alpha mining, intraday strategies, investor tearsheet) — was dispatched via the
  autopilot workflow in an earlier session; **status unconfirmed.**
- **ML models / torch (2026-06-20):** the model **registry now imports cleanly without
  torch** (`app/ml/models/__init__.py` catches any import error; `transformer.py` +
  `itransformer.py` use a guarded `nn.Module` base). **But ML experiments still need torch**
  — install `pip install -e ".[ml]"` to actually train/run LSTM/Transformer/SSM/Mamba.
  Without it, those classes import as inert placeholders / `None`, and the 3 unit modules
  that `import torch` at top (`test_a3c_lstm.py`, `test_ml_models.py`, `test_ssm_model.py`)
  cannot collect. Still-broken regardless of torch: **`HMMRegimeModel`** —
  `cannot import name 'HMMRegimeModel' from app.ml.models.hmm_regime` (the class is
  missing/misnamed in that file); needs a real fix, not just a torch install.

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
