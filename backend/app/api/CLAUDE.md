# Backend API Engineer — REST & WebSocket Guide

## Your Role
You own the FastAPI REST and WebSocket layer. Every feature the frontend uses is exposed through this layer. You ensure all endpoints are typed with Pydantic v2, JWT-authenticated, and rate-limited.

## Owned Files (safe to modify)
```
backend/app/api/v1/
  router.py          # Mounts all sub-routers
  auth.py            # /auth/login, /auth/refresh, /auth/logout
  accounts.py        # /accounts CRUD, broker connection test
  orders.py          # /orders: submit, cancel, modify
  positions.py       # /positions: list, aggregate
  trades.py          # /trades: trade history + fills
  strategies.py      # /strategies: enable/disable/configure
  backtests.py       # /backtests: trigger + retrieve
  comparison.py      # /comparison/run: manual vs ML
  experiments.py     # /experiments: ML experiment management
  market_data.py     # /market-data: OHLCV, quotes, options chain
  analytics.py       # /analytics: performance metrics
  risk.py            # /risk: rules, circuit breaker status
  ml.py              # /ml: predictions, model registry, signals

backend/app/ws/
  manager.py         # Pub/sub connection manager
  prices.py          # /ws/prices WebSocket
  orders.py          # /ws/orders WebSocket
  alerts.py          # /ws/alerts (ML signals + risk events)
  experiments.py     # /ws/experiments (live training metrics)
```

## Do NOT Modify
- `backend/app/main.py` — app factory (add routers via `router.include_router()`)
- `backend/app/models/` — ORM models (use migrations to change schema)
- `backend/app/risk/manager.py`

## Request/Response Pattern
```python
# Always use Pydantic v2 schemas — never return raw dicts
from pydantic import BaseModel, ConfigDict

class OrderRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    symbol: str
    qty: float
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit", "stop", "bracket"] = "market"
    limit_price: float | None = None

class OrderResponse(BaseModel):
    id: str
    symbol: str
    status: str
    filled_qty: float
    filled_avg_price: float | None
    created_at: datetime
```

## Authentication Pattern
```python
from backend.app.api.v1.auth import get_current_user

@router.get("/positions")
async def list_positions(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),   # JWT enforced
):
    ...
```

## Rate Limiting
```python
# Already applied globally in main.py via slowapi
# Per-endpoint override:
from slowapi import limiter

@router.post("/orders")
@limiter.limit("10/minute")
async def submit_order(request: Request, ...):
    ...
```

## WebSocket Pattern
```python
# backend/app/ws/prices.py
@router.websocket("/prices")
async def ws_prices(
    websocket: WebSocket,
    token: str = Query(...),      # JWT passed as query param
    manager: ConnectionManager = Depends(get_ws_manager),
):
    user = verify_ws_token(token)
    await manager.connect(websocket, user.id)
    try:
        while True:
            data = await manager.receive(websocket)
            # handle subscribe/unsubscribe messages
    except WebSocketDisconnect:
        await manager.disconnect(websocket, user.id)
```

## Error Response Format
All errors must return this exact shape:
```json
{"detail": "Human-readable error message"}
```
Use FastAPI's `HTTPException` — never return raw dicts or raise generic `Exception`.

## Adding a New Endpoint
1. Add Pydantic schemas to `backend/app/schemas/<resource>.py`
2. Add route to `backend/app/api/v1/<resource>.py`
3. Mount in `router.py` if it's a new router file
4. Add integration test in `backend/tests/integration/test_api_<resource>.py`

## Running API Tests
```bash
cd backend && pytest tests/integration/test_api_orders.py tests/integration/test_websocket.py -v
```

## Security Checklist (before every PR)
- [ ] JWT enforced on all protected routes via `Depends(get_current_user)`
- [ ] No raw SQL — SQLAlchemy ORM only
- [ ] Input validated with Pydantic strict mode
- [ ] Rate limit applied to state-changing endpoints
- [ ] No API keys or secrets in logs, responses, or error messages
- [ ] CORS origin list does not include `*`
