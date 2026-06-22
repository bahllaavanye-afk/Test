# QuantEdge — Operations & the Right Long-Term Fixes

> Why this exists: the month of "nothing works" had three structural causes —
> (1) a paid-grade always-on system on free tiers, (2) config spread across 4 places
> that drifts, (3) autonomous agents auto-deploying on every commit. These are the
> durable fixes. Items marked ✅ are already in the repo; ⚙️ need a one-time
> dashboard/credential action (noted exactly).

## 1. One config source that auto-syncs (kills "key missing in X")
Secrets currently live in Doppler **and** GitHub Actions Secrets **and** Render env
**and** Render env-groups — and they drift (we hit missing `SECRET_KEY`,
`ALPACA_API_KEY`, LLM keys absent in CI, env-groups not linking).

**Fix — make Doppler the single source and let it push everywhere:**
- ⚙️ Doppler dashboard → project `quantedge` / config `dev` → **Integrations**:
  - Add the **Render** integration → select service `quantedge-api-9jz0` → auto-syncs
    every secret to the service's env (no env-groups, no hand-copying).
  - Add the **GitHub Actions** integration → repo `bahllaavanye-afk/test` → syncs the
    LLM/Slack keys into Actions Secrets so the agent brain stops failing.
- After this, you only ever edit secrets in Doppler; Render + GitHub stay in sync.
- ✅ `scripts/verify_live.py` can read Render state via `doppler run` once
  `RENDER_API_KEY` is in Doppler (it is).

## 2. Deterministic, verified deploys (kills "a stub ran for months")
- ✅ Deploy strictly from **`main`** via `render.yaml` (IaC). Service branch = `main`.
- ✅ **Drift guard**: `.github/workflows/deploy-verify.yml` runs `scripts/verify_live.py`
  on push to main + every 6h and alerts `#infra-alerts` if the live backend isn't the
  real ~200-route API. (Flip its final step to `exit 1` after go-live to make it block.)
- ✅ `backend/tests/test_prod_deployment_smoke.py` (`-m live`) asserts the deployed app
  exposes the real routes, not a 3-route stub.

## 3. Decouple the agent workforce from production (kills minute/token burn)
The agents committing every few minutes + auto-deploy is what exhausted Render
pipeline minutes and caused most breakage.
- ✅ `autoDeploy: false` in `render.yaml` (deploys are intentional only).
- ✅ Over-frequent crons throttled `*/5 → */20`.
- ⚙️ Policy: agents commit to **short-lived branches → PRs**, never push the deploy
  branch directly. Production deploys happen only on a deliberate merge to `main` (or
  the deploy hook). Keep "agents working 24/7" separate from "production deploying".

## 4. One trunk (kills branch chaos)
`main` vs `claude/advanced-trading-bot-d5Lmw` vs `claude/stoic-johnson-7z4wtz` diverged
into a 512-file, unmergeable mess.
- ⚙️ Make `main` the only long-lived branch. Point Render at `main` (done). Retire the
  long-lived agent branches; agents branch off `main` and PR back. Cherry-pick anything
  still needed from the old branches, then delete them.

## 5. The infra decision (the root fragility)
Running a 24/7 trading backend + DB + Redis + ~60 workflows on **all-free** tiers is the
underlying cause (free Render sleeps + exhausts build minutes; no Redis).
- **Option A (recommended for a real product):** ~$7–25/mo for **one** always-on backend
  (Render Starter / Fly / Railway) + Doppler sync → removes ~90% of the firefighting.
- **Option B (strictly free):** one always-on free host (Koyeb — no sleep, no build-minute
  wall), Doppler auto-sync (#1), deploy-from-main-only (#2), agents throttled hard (#3).
  Workable, but always near the edges.

## Deploy runbook (free tier, today)
1. Branch = `main`, `autoDeploy: false` (done).
2. Build minutes reset monthly (Render → My Workspace → Billing shows the date).
3. On reset: fire the deploy hook (or Manual Deploy → Clear cache). `deploy-verify.yml`
   confirms the live app is real and alerts on drift.
4. `python scripts/verify_live.py` (or `doppler run -- …`) verifies from anywhere.
