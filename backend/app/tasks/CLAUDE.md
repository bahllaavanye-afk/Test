# Data Engineer — Module Guide

## Your Role
You own the data pipeline: real-time price ingestion, OHLCV storage, Redis caching, and the always-on background task system. Clean, timely data is the foundation of every alpha signal.

## Owned Files (safe to modify)
```
backend/app/tasks/
  price_feed.py        # Continuous price ingestion → Redis + WebSocket
  strategy_runner.py   # Event loop: poll strategies → signals → orders
  scheduler.py         # APScheduler: hourly snapshots, nightly retraining
  order_sync.py        # Broker order status → DB sync
  snapshot.py          # Hourly account snapshots
  ml_retrain.py        # Nightly model retraining at 02:00 UTC

backend/app/redis_client.py   # Upstash REST Redis wrapper
backend/app/database.py       # Async SQLAlchemy engine + session factory
```

## Do NOT Modify
- `backend/app/brokers/*.py` — data pipeline reads from brokers but does not change broker logic
- `backend/app/strategies/` — strategy code; you only call `analyze()` from strategy_runner.py
- `backend/alembic/` — DB schema is owned by the Platform Engineer

## Data Flow Architecture
```
Broker WebSocket / REST API
         │
         ▼
   price_feed.py
   ├── Redis HSET prices:<symbol>  (TTL 60s)
   ├── Redis LPUSH ohlcv:<symbol>  (ring buffer, 2000 bars)
   └── WebSocket broadcast → frontend

scheduler.py (APScheduler)
   ├── Every 60s   → strategy_runner.py checks Redis for fresh prices
   ├── Every 1h    → snapshot.py captures account NAV
   ├── Every 5m    → order_sync.py pulls broker fills → DB
   └── 02:00 UTC  → ml_retrain.py triggers model retraining
```

## Redis Key Schema
| Key Pattern                  | Type   | TTL  | Contents                          |
|------------------------------|--------|------|-----------------------------------|
| `prices:<SYMBOL>`            | HASH   | 60s  | {last, bid, ask, volume, ts}      |
| `ohlcv:<SYMBOL>:<interval>`  | LIST   | 7d   | OHLCV JSON rows (ring buf 2000)   |
| `signal:<strategy>:<symbol>` | STRING | 5m   | Signal JSON (direction, size, ts) |
| `account:<user_id>:nav`      | STRING | 1h   | Current NAV float                 |
| `positions:<user_id>`        | HASH   | 5m   | {symbol: qty} map                 |

## Adding a New Scheduled Task
```python
# In scheduler.py, inside setup_scheduler():
scheduler.add_job(
    my_task_fn,
    trigger="interval",
    seconds=300,
    id="my_task",
    replace_existing=True,
    max_instances=1,       # never let two copies run simultaneously
)
```

## Performance Requirements
| Metric                    | Target     |
|---------------------------|------------|
| Price feed latency        | < 200ms    |
| Signal computation        | < 500ms    |
| Redis write throughput    | > 1000/s   |
| Strategy runner cycle     | < 60s      |
| DB write batch size       | 500 rows   |

## Running Locally (no broker connection needed)
```bash
# Start with mock price feed (reads from Redis if prices exist, otherwise skips)
cd backend
TRADING_MODE=paper REDIS_URL=... python -m app.tasks.scheduler
```

## Common Failure Modes
- **Stale prices**: Redis TTL expired but strategy_runner still consumes old data → add freshness check
- **Missing symbols**: broker doesn't have a ticker → strategy_runner must log and skip gracefully
- **Redis OOM**: ohlcv list grows unboundedly → use LTRIM after LPUSH to cap at 2000 entries
- **Duplicate signals**: two runner tasks for same (strategy, symbol) → use `max_instances=1` in APScheduler
