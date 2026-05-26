# Backend Agent Guide

## Architecture
FastAPI (async) + SQLAlchemy (async) + PostgreSQL + Redis

## Key Entry Points
- `app/main.py` — FastAPI app factory
- `app/strategies/` — all trading strategies (manual + ML)
- `app/ml/` — ML models, features, training
- `app/execution/` — order routing and slippage minimization
- `app/risk/` — Kelly sizing, circuit breakers, correlation limits
- `app/tasks/` — always-running background tasks

## Always-Running Tasks
```
AlgoAgent       → UCB1 exploration/exploitation, tests strategies every 5min
StrategyRunner  → One asyncio task per (strategy, symbol), runs 24/7
PriceFeed       → Polls broker quotes, publishes to Redis + WebSocket
Scheduler       → Hourly snapshots, nightly retraining (02:00 UTC)
```

## Adding a Strategy
See `app/strategies/CLAUDE.md`

## Running Tests
```bash
cd backend && pytest tests/ -x -v
```

## Deployment
- Render (free web service) — `render.yaml`
- Supabase PostgreSQL — set DATABASE_URL
- Upstash Redis — set REDIS_URL
