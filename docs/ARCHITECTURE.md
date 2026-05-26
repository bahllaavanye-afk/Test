# Architecture

## Overview

QuantEdge is a multi-process, async-first trading platform built on FastAPI and SQLAlchemy 2.0. The backend runs four concurrent always-on systems plus REST/WebSocket API serving.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FastAPI ASGI App                              │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ REST routes  │  │ WebSocket    │  │ AlgoAgent    │  │ Scheduler   │ │
│  │ /api/v1/*    │  │ /ws/*        │  │ (UCB1 loop)  │  │ (APScheduler│ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  StrategyRunner: one asyncio task per (strategy, symbol) pair    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  PriceFeed: polls broker quotes → Redis cache → WebSocket fan-out│  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
            │              │              │              │
            ▼              ▼              ▼              ▼
        ┌──────┐      ┌──────┐      ┌──────────┐    ┌──────────┐
        │ PG   │      │Redis │      │ Brokers  │    │ yfinance │
        │      │      │      │      │ (4)      │    │          │
        └──────┘      └──────┘      └──────────┘    └──────────┘
```

## Layered Modules

### `app/strategies/` — Trading Logic
- `base.py` — `AbstractStrategy` interface
- `manual/` — indicator-only strategies (9 files)
- `ml_enhanced/` — same logic + ML filter (5 files)
- `__init__.py` — `STRATEGY_REGISTRY` mapping name → class

### `app/ml/` — Machine Learning
- `models/` — `LSTMPredictor`, `XGBoostClassifier`, `LorentzianKNN`, `EnsembleModel`
- `features/` — `engineer.py` (master pipeline), `technical.py` (pandas-ta), `sentiment.py` (Fear & Greed), `multi_timeframe.py`
- `training/` — `train_lstm.py`, `trainer.py` (Lightning + MLflow)
- `inference.py` — singleton `InferenceService`

### `app/execution/` — Order Routing
Smart router decides between TWAP / VWAP / LimitFirst / Iceberg / Market based on size and urgency.

### `app/risk/` — Risk Engine
- `kelly.py` — fractional Kelly sizing
- `correlation.py` — cluster detection + allocation limits
- `circuit_breaker.py` — drawdown halt logic
- `manager.py` — combines all three; gates all orders

### `app/brokers/` — Multi-Broker Abstraction
All brokers implement `AbstractBroker`:
```python
async def place_order(req) -> OrderResult
async def cancel_order(id) -> bool
async def get_quote(symbol) -> QuoteResult
async def get_positions() -> list[dict]
async def get_historical(symbol, interval, start, end) -> list[dict]
```

### `app/tasks/` — Background Tasks (Always-On)
- `algo_agent.py` — UCB1 exploration/exploitation
- `strategy_runner.py` — per-(strategy, symbol) signal loops
- `price_feed.py` — broker polling
- `ml_retrain.py` — nightly model retraining
- `scheduler.py` — APScheduler setup

### `app/ws/` — WebSocket Endpoints
- `manager.py` — topic-based pub/sub
- `prices.py`, `orders.py`, `alerts.py` — endpoint handlers

### `app/notifications/` — Slack + Activity Tracking
- `slack.py` — multi-channel webhook client
- `tracker.py` — in-memory bounded event log
- `screenshot.py` — Playwright dashboard capture

## Data Flow: From Signal to Fill

```
1. StrategyRunner pulls OHLCV from Redis (or broker)
2. strategy.analyze() returns Signal | None
3. If confidence > threshold → publish to "alerts" topic + Slack
4. (Manual trader / future auto-trader) submits order via REST
5. POST /orders/ → SmartOrderRouter chooses algo (e.g. TWAP)
6. RiskManager.check_order() → Kelly sizing + cluster check + breaker check
7. Algorithm slices and submits via broker.place_order()
8. SlippageTracker records signal_price vs fill_price
9. Order events broadcast via /ws/orders + Slack notification
10. Position update via /ws/positions
```

## Database Schema (Key Tables)

```sql
users (id, email, hashed_password, ...)
accounts (id, user_id, broker, mode, encrypted_key, encrypted_secret)
orders (id, account_id, strategy_id, symbol, side, status, ...)
positions (id, account_id, symbol UNIQUE, quantity, avg_cost, ...)
trades (id, account_id, strategy_id, realized_pnl, opened_at, closed_at)
strategies (id, account_id, name, is_enabled, params, symbols, ...)
backtest_runs + backtest_results
experiments (id, name UNIQUE, config, val_sharpe, test_sharpe, ...)
ml_models (id, model_type, artifact_path, is_active)
slippage_records (id, order_id, signal_price, fill_price, slippage_bps, execution_algo)
comparison_results (id, strategy_name, manual_sharpe, ml_sharpe, p_value, winner, ...)
risk_rules + risk_events
```

## Security Boundaries

1. **JWT** at FastAPI level — all endpoints except `/health`, `/auth/*` require valid token
2. **AES-256** at storage — `Account.encrypted_key`, `Account.encrypted_secret` via Fernet
3. **Pydantic strict** at request level — all bodies validated before reaching handlers
4. **ORM-only** — zero raw SQL, no injection surface
5. **CORS allowlist** — production restricts to Vercel domain
6. **Rate limiting** — 100 req/min per user via slowapi
7. **Server-side gates** — paper/live mode enforced in DB, position caps enforced in RiskManager
