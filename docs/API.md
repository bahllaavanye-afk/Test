# API Reference

Base URL: `http://localhost:8000/api/v1`  
Authentication: JWT Bearer token (`Authorization: Bearer <token>`)

## Auth

### POST `/auth/register`
Register a new user.
```json
{ "email": "you@example.com", "password": "secure-pw-here" }
```
Response: `{ "id": "...", "email": "..." }`

### POST `/auth/login`
Exchange credentials for JWT tokens.
```json
{ "username": "you@example.com", "password": "secure-pw-here" }
```
Response:
```json
{
  "access_token": "eyJ...",   // 15-min expiry
  "refresh_token": "eyJ...",  // 7-day expiry
  "token_type": "bearer"
}
```

### POST `/auth/refresh`
Exchange a refresh token for a fresh access token.

## Accounts (broker connections)

### GET `/accounts/`
List all broker accounts for current user.

### POST `/accounts/`
Add a new broker account. Encrypted with AES-256 at rest.
```json
{
  "broker": "alpaca",
  "label": "Alpaca Paper",
  "mode": "paper",
  "api_key": "PKAB...",
  "api_secret": "..."
}
```

### DELETE `/accounts/{account_id}`

## Orders

### GET `/orders/?limit=50`
List recent orders.

### POST `/orders/`
Submit an order. Goes through SmartOrderRouter → risk checks → broker.
```json
{
  "symbol": "AAPL",
  "side": "buy",
  "order_type": "market",
  "quantity": 10,
  "limit_price": null,
  "execution_algo": "auto",
  "account_id": "<account-uuid>"
}
```

### DELETE `/orders/{order_id}`
Cancel an open order.

## Positions

### GET `/positions/`
Current open positions (qty != 0).

## Strategies

### GET `/strategies/available`
List all registered strategy classes.

### GET `/strategies/`
List user's configured strategies.

### PATCH `/strategies/{strategy_id}/toggle`
Enable/disable a strategy.
```json
{ "is_enabled": true }
```

## Backtests

### POST `/backtests/`
Trigger a new backtest run.
```json
{
  "strategy_name": "momentum",
  "symbol": "SPY",
  "interval": "1d",
  "start_date": "2021-01-01",
  "end_date": "2024-01-01",
  "initial_equity": 100000
}
```

### GET `/backtests/`
List recent backtest runs.

## Comparison

### GET `/comparison/`
Recent manual-vs-ML comparison results.

### GET `/comparison/benchmarks`
Static SPY/QQQ/BRK.B/All Weather reference stats.

## Experiments (ML training runs)

### GET `/experiments/`
List ML experiments.

### GET `/experiments/{experiment_id}`
Full experiment detail including config and metrics history.

## AlgoAgent (UCB1 monitor)

### GET `/agents/leaderboard`
Current UCB1 leaderboard — strategies ranked by avg Sharpe.

### GET `/agents/results?limit=50`
Recent backtest results from the always-running AlgoAgent.

### GET `/agents/status`
```json
{ "running": true, "total_runs": 142, "candidates": 13, "top_3": [...] }
```

## ML Models

### GET `/ml/models`
List trained ML models with val/test metrics.

### GET `/ml/models/{model_id}/activate`
Activate a model for inference.

## Risk

### GET `/risk/rules`
List active risk rules.

### POST `/risk/rules`
Add a new risk rule.
```json
{ "rule_type": "max_drawdown", "threshold": 0.10, "action": "halt_all" }
```

### GET `/risk/events`
Recent risk events (circuit breakers tripped, position caps hit).

## Market Data

### GET `/market-data/quote/{symbol}`
Current quote via yfinance.

### GET `/market-data/history/{symbol}?interval=1d&period=1y`
OHLCV history.

## Analytics

### GET `/analytics/performance`
Aggregate trade performance.
```json
{ "total_trades": 142, "avg_pnl": 28.50, "total_pnl": 4047.0 }
```

### GET `/analytics/slippage`
Average slippage by execution algorithm.

## Notifications

### GET `/notifications/activity?limit=100&category=order`
In-memory activity tracker — recent events.

### GET `/notifications/stats`

### POST `/notifications/slack/test`
Send a test message via configured Slack webhook.

## WebSockets

### `WS /ws/prices/{symbol}`
Subscribe to real-time price updates for a symbol.

### `WS /ws/orders`
Order fill/cancel/reject stream.

### `WS /ws/alerts`
Strategy signals and risk events stream.
