# QuantEdge — Live System Status

> Last manual edit: 2026-05-28. The CTO agent updates this on deploys.

## Production deployment

| Component | Status | URL / location |
|---|---|---|
| Frontend (Vercel) | ❌ **NOT DEPLOYED** | needs Vercel → import repo → set `VITE_API_URL` + `VITE_WS_URL` |
| Backend (Render) | ❌ **NOT DEPLOYED** | needs Render → New Blueprint → repo → set env vars |
| Database (Supabase) | ❌ **schema not applied** | trigger `migrate.yml` workflow OR paste SQL at supabase.com SQL editor |
| Redis (Upstash) | ✅ provisioned | `right-yak-96501.upstash.io:6379` |
| Alpaca paper API | ✅ creds in `.env` (not in GH secrets yet) | account: `PKYZZEPIFF…` |
| Slack workspace | ✅ 64 channels bootstrapped | `QuantEdge.slack.com` |

## Where do trades happen? Where do I see them?

**They don't happen yet.** Until backend is deployed to Render, no strategy code runs against live data. After Render deploy:
1. `backend/app/main.py:lifespan()` boots — starts `StrategyRunner` (24/7 loop, one task per strategy×symbol)
2. `PriceFeed` polls Alpaca paper for OHLCV, pushes to Redis
3. Each strategy's `analyze()` runs every `tick_interval_seconds` (default 60s)
4. Signals → `SmartOrderRouter` → Alpaca paper trade
5. Fills flow back to DB → P&L attribution → Slack `#pnl-daily`
6. Frontend pulls from REST `/api/v1/analytics/*` + WebSocket `/ws/orders`

## Slack agent team (the live "company")

| What | Where | Owner |
|---|---|---|
| 5 asset-class teams compete on Sharpe | `#desk-equities`, `#desk-crypto`, `#desk-options`, `#desk-polymarket`, `#desk-fx-rates` | Aarav, Linh, Yuki, Lior, Tomas |
| Daily scoreboard | `#pnl-daily` (4×/day) | Scoreboard bot |
| Friday presentation by winning team | `#leadership-summary` | Winner team lead |
| 25 agents post real findings | `#engineering`, `#alpha-research`, `#ml-experiments`, others | various |
| Auto-issues for untested strategies | GitHub Issues | Aditi (QA) agent |
| Real Alpaca P&L | `#pnl-daily` | PnL bot |

Agent scheduling: `cron: "15 9,13,17,21 * * *"` UTC (4 waves/day).

## Repo state today

| Metric | Count |
|---|---|
| Manual strategies | 41 |
| ML-enhanced strategies | 7 |
| Total strategies registered | 48 |
| Test files | 27 |
| Unit tests passing | 258 (lightweight suite, ~8s) |
| Strategies without a unit test | ~24 (Aditi agent auto-opens tracker issues) |
| Backtest result JSONs in `experiments/results/` | 173 entries across N files |
| Frontend pages | 20 (added MLInsights) |
| Backend routes | 90 |
| Frontend bundle (gzip) | 97 KB shared + per-page lazy chunks (2–8 KB each) |

## What to do next (in this order)

1. **Add 7 secrets** at https://github.com/bahllaavanye-afk/Test/settings/secrets/actions
   - `SLACK_BOT_TOKEN` — your `xoxb-…` (lets agent team run on schedule)
   - `ALPACA_API_KEY` — `PKYZZEPIFF25TUFHQVAA522VWN`
   - `ALPACA_SECRET_KEY` — `9BkJdQkLbaZR99uBM1Xwv5Z7VWapPZkq57YSqas2SkoG`
   - `DATABASE_URL`, `ALEMBIC_DATABASE_URL`, `REDIS_URL`, `SECRET_KEY` (from `.env`)
2. **Deploy backend** at https://dashboard.render.com → New Blueprint → connect repo → set the 4 env vars
3. **Deploy frontend** at https://vercel.com/new → import repo → root: `frontend/` → set `VITE_API_URL` + `VITE_WS_URL`
4. **Apply DB schema** via `migrate.yml` workflow dispatch (or paste SQL manually)

After step 1, the Slack `#pnl-daily` channel starts showing real Alpaca paper P&L every 4 hours.
After steps 2-4, strategies actually execute and the dashboard at the Vercel URL becomes the live demo.
