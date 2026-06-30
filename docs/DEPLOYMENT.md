# Deployment Guide

## Architecture

```
Users → Vercel (frontend) → Render (backend API) → Supabase (Postgres)
                                                  → Upstash (Redis)
                                                  → Broker APIs (Alpaca etc.)
UptimeRobot → /health every 5 min → keeps Render alive
```

All services have free tiers and require no credit card.

## 1. Supabase (Database)

1. Sign up: https://supabase.com (free 500 MB)
2. Create new project — note the password
3. Copy the **Pooler URL** (port 6543, IPv4-safe) from Project Settings → Database
4. Set as `DATABASE_URL` in Render env vars (convert prefix to `postgresql+asyncpg://`)
5. Set as `ALEMBIC_DATABASE_URL` with `postgresql+psycopg2://` prefix

## 2. Upstash (Redis)

1. Sign up: https://upstash.com (free 10K commands/day)
2. Create a Redis database (any region)
3. Copy the `redis://...` URL → set as `REDIS_URL` in Render

## 3. Render (Backend)

1. Sign up: https://render.com (free 750 hrs/month)
2. Connect your GitHub repo
3. New → Web Service → select repo
4. Render auto-detects `backend/render.yaml`
5. Set environment variables:
   - `TRADING_MODE=paper`
   - `SECRET_KEY` (generate with `python -c "import secrets; print(secrets.token_hex(32))"`)
   - `DATABASE_URL` (from Supabase)
   - `REDIS_URL` (from Upstash)
   - `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
   - `ALLOWED_ORIGINS=https://your-app.vercel.app`
6. Deploy
7. Note the URL: `https://quantedge-api-agb8.onrender.com`

## 4. UptimeRobot (Keep-Alive)

Render free tier spins down after 15 min of inactivity. UptimeRobot prevents this.

1. Sign up: https://uptimerobot.com (free)
2. Add new monitor:
   - Type: HTTP(s)
   - URL: `https://quantedge-api-agb8.onrender.com/health`
   - Interval: 5 minutes
3. Save

## 5. Vercel (Frontend)

1. Sign up: https://vercel.com (free unlimited static)
2. Import your repo
3. Set:
   - **Framework**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Update `frontend/vercel.json` to point at your Render backend:
   ```json
   {
     "rewrites": [
       { "source": "/api/:path*", "destination": "https://quantedge-api-agb8.onrender.com/api/:path*" }
     ]
   }
   ```
5. Deploy

## 6. Run Migrations on Production DB

Locally (one time after first deploy):
```bash
DATABASE_URL="postgresql+psycopg2://...supabase..." \
ALEMBIC_DATABASE_URL="postgresql+psycopg2://...supabase..." \
cd backend && alembic upgrade head
```

## 7. Custom Domain (Optional)

- Vercel: Project Settings → Domains → add your domain
- Update DNS to Vercel's nameservers

## Switching to Live Trading

1. **Paper-test 2 weeks minimum** (per platform policy)
2. Generate Alpaca live API keys at https://alpaca.markets
3. In Render env vars:
   - `TRADING_MODE=live`
   - `ALPACA_API_KEY` = live key
   - `ALPACA_SECRET_KEY` = live secret
   - `ALPACA_BASE_URL=https://api.alpaca.markets`
4. Restart Render service
5. Verify in dashboard top bar: should show `● LIVE TRADING`

## Monitoring

- **Render logs**: dashboard.render.com → service → Logs tab
- **Supabase queries**: dashboard.supabase.com → SQL Editor
- **Upstash commands**: dashboard.upstash.com
- **Slack notifications**: configure all 5 webhook URLs in Render env

## Backup & Disaster Recovery

- **Supabase**: free tier has 7-day point-in-time recovery
- **Model artifacts** (`models_artifacts/`): backup with `aws s3 cp --recursive` to free S3 tier
- **Experiment results** (`experiments/results/`): commit to git

## Cost Summary

| Service | Free Tier | Estimated Use | Cost |
|---------|-----------|---------------|------|
| Vercel | Unlimited static | Frontend hosting | $0 |
| Render | 750 hrs/month | One web service | $0 |
| Supabase | 500 MB | DB + auth | $0 |
| Upstash | 10K cmd/day | Redis | $0 |
| UptimeRobot | 50 monitors | 1 ping | $0 |
| **Total** | | | **$0/month** |

Upgrade triggers:
- Supabase 500 MB → $25/month for 8 GB
- Render 750 hrs → $7/month for always-on (no spin-down)
- Upstash 10K cmd/day → $0.20 per 100K commands
