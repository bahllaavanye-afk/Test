# Restarting the backend on a fresh Render account

**Why this works:** the current backend isn't broken — every deploy fails with
`pipeline_minutes_exhausted` (the free build-minute quota is spent). A *new*
Render account has a fresh quota, so deploys run again. The repo is already
Infrastructure-as-Code (`render.yaml`), so setup is ~5 minutes.

## Steps (one-time, ~5 min)

1. **New account** → https://render.com (sign in with GitHub).
2. **New → Blueprint** → connect this repo → it reads `render.yaml` and proposes:
   - `quantedge-api` (web service, free, branch `main`)
   - `quantedge-db` (free Postgres)
   Click **Apply**.
3. **Fill the `sync: false` secrets** when prompted (these are NOT in the blueprint
   on purpose). Minimum to boot + trade on paper:
   - `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` (paper keys)
   - `SLACK_BOT_TOKEN` (for alerts)
   - `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` (only if using Google login)
   - `SECRET_KEY` is auto-generated; `DATABASE_URL` is auto-wired from the blueprint DB.
   > Tip: avoid hand-copying — add Doppler's **Render integration** (project
   > `quantedge`/config `dev`) and it pushes every secret automatically. See
   > `docs/OPERATIONS.md` §1.
4. **Note the new service URL** — it will be `https://quantedge-api-XXXX.onrender.com`
   (the suffix differs from the old `-9jz0`).

## Point the frontend at the new backend (1 env var, no code change)

`frontend/src/api/client.ts` already reads `VITE_API_URL`. So:

1. Vercel → project → **Settings → Environment Variables** →
   set `VITE_API_URL = https://quantedge-api-XXXX.onrender.com`
2. **Redeploy** the Vercel project.

That's it — the app calls the new backend directly (bypassing the stale
`vercel.json` proxy). The `BackendHealthBanner` will clear once `/api` responds.

## Two stale URLs to update if you keep the proxy / Google login

- `render.yaml` → `GOOGLE_REDIRECT_URI` still points at the old `-9jz0` host.
  Update it to the new URL **and** add the new callback URL in the Google OAuth
  console. (Skip entirely if you only use open-access/demo mode.)
- `frontend/vercel.json` → the `/api/*` rewrite destination is the old `-9jz0`
  host. Only used when `VITE_API_URL` is unset — setting that env var (above)
  makes it irrelevant.

## Reality check on the free tier

- ✅ Fresh build minutes → deploys succeed.
- ⚠️ Free web services **sleep after ~15 min idle** (first request ~50s cold start).
  `keep-alive.yml` pings it; or upgrade to Starter (~$7) for always-on.
- ⚠️ The monthly build-minute cap still exists — but with `autoDeploy: false`
  (already set) deploys are intentional only, so minutes last far longer.
- 💡 If you want truly set-and-forget free hosting with **no sleep and no
  build-minute wall**, Koyeb is the better target (`docs/OPERATIONS.md` §5,
  Option B) — same blueprint-style setup.
