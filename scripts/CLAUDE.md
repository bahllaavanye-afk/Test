# DevOps / Infrastructure Engineer — Guide

## Your Role
You keep QuantEdge running 24/7 across Render (backend), Vercel (frontend), Supabase (DB), and Upstash (Redis). You own deployment, CI/CD, monitoring, and scaling. Your job is zero-downtime delivery.

## Owned Files (safe to modify)
```
scripts/
  launch.sh          # Master launcher: dev|paper|live|backtest|train|compare
  dev.sh             # docker-compose dev stack
  paper.sh           # Paper trading mode
  live.sh            # Live trading mode (CONFIRM gate)
  backtest.sh        # Strategy backtest runner
  train.sh           # Local CPU model training
  compare.sh         # Manual vs ML comparison
  migrate.sh         # alembic upgrade head
  seed.sh            # Seed default strategies + risk rules
  download_data.sh   # Download historical OHLCV for training
  agents/            # Autonomous agent scripts

.github/workflows/
  test.yml           # CI: unit tests on every push
  e2e-demo.yml       # E2E: Playwright demo with live Alpaca data

backend/
  render.yaml        # Render deployment config
  Dockerfile         # Backend container

frontend/
  vercel.json        # Vercel deployment config
  vite.config.ts     # Build config
```

## Do NOT Modify
- `backend/app/**/*.py` — application code
- `backend/alembic/versions/` — migrations (create new ones, never edit existing)
- `.env` — secrets file; never committed to git

## Deployment Architecture
```
GitHub push → CI (test.yml) → merge to main → auto-deploy

Backend:  Render free web service
  URL:   https://quantedge-api.onrender.com
  Env:   DATABASE_URL, REDIS_URL, SECRET_KEY, ALPACA_*, TRADING_MODE
  Keep-alive: UptimeRobot pings /health every 5min

Frontend: Vercel (auto-deploy on push to main)
  URL:    https://quantedge.vercel.app
  Env:    VITE_API_URL=https://quantedge-api.onrender.com/api/v1
          VITE_WS_URL=wss://quantedge-api.onrender.com/ws

Database: Supabase PostgreSQL
  Apply migrations: ALEMBIC_DATABASE_URL=<psycopg2 URL> alembic upgrade head

Cache: Upstash Redis TLS
  URL format: rediss://default:<token>@<host>:6379
```

## CI/CD Pipeline (`.github/workflows/test.yml`)
```yaml
# Critical: always use set -eo pipefail + no pipes to head/tail
run: |
  set -eo pipefail
  pytest tests/unit/ -x -v --tb=short
```
**Never** use `cmd 2>&1 | head -N` in CI — it masks non-zero exit codes.

## Secrets Management
All secrets live in GitHub Actions Secrets and Render environment variables:
| Secret                | Where Set       | Used By           |
|-----------------------|-----------------|-------------------|
| `SECRET_KEY`          | Render + GH     | JWT, encryption   |
| `DATABASE_URL`        | Render + GH     | asyncpg ORM       |
| `ALEMBIC_DATABASE_URL`| GH only         | psycopg2 migrate  |
| `REDIS_URL`           | Render + GH     | Upstash TLS       |
| `ALPACA_API_KEY`      | Render + GH     | Alpaca paper API  |
| `ALPACA_SECRET_KEY`   | Render + GH     | Alpaca paper API  |

## Adding a New GitHub Secret (Render)
```bash
# In Render dashboard → Environment → Add variable
# In GitHub: Settings → Secrets and variables → Actions → New repository secret
```

## Database Migrations (never edit existing migrations)
```bash
# Generate new migration
cd backend
alembic revision --autogenerate -m "add_my_new_table"

# Apply to Supabase (from local machine with network access)
ALEMBIC_DATABASE_URL="postgresql+psycopg2://postgres:<pwd>@db.<project>.supabase.co:5432/postgres" \
  alembic upgrade head
```

## Monitoring
- UptimeRobot: monitors `/health` endpoint every 5min → Slack alert on failure
- Render: built-in log streaming + deploy notifications
- Supabase: query performance + connection pool monitoring in dashboard

## Keep-Alive Strategy (Render free tier sleeps after 15min idle)
- UptimeRobot free tier: pings `/health` every 5min → prevents sleep
- `/health` endpoint returns `{"status": "ok"}` with no DB query (fast)
- Worker service (separate Render free service) runs APScheduler tasks 24/7

## Running E2E Demo
```bash
# Trigger the workflow from GitHub Actions UI or:
gh workflow run e2e-demo.yml
# Artifacts (videos + logs) uploaded to GitHub Actions → 14 day retention
```
