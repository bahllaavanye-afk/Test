# QuantEdge — Root Agent Guide

## What This Is
QuantEdge is an institutional-grade quantitative trading platform. It is a startup product.
It connects to Alpaca (equities), TradeStation (equities), Binance (crypto), and Polymarket (prediction markets).
It runs ML-enhanced trading strategies 24/7 across hundreds of tickers and coins simultaneously.

## Quick Start
```bash
./scripts/launch.sh dev        # Full local dev stack (Docker)
./scripts/launch.sh paper      # Paper trading mode
./scripts/launch.sh backtest momentum SPY 1d 2021-01-01 2024-01-01
./scripts/launch.sh compare momentum SPY
./scripts/launch.sh train --config experiments/configs/lstm_btc_1h.yaml
```

## Repository Layout
```
quantedge/
├── backend/        Python FastAPI backend (strategies, ML, brokers, risk)
├── frontend/       React 18 + Vite + TypeScript dashboard
├── scripts/        One-command launch scripts for every mode
└── CLAUDE.md       This file
```

## Key Architectural Principles
1. **Paper-first**: every strategy must run 2 weeks on paper before live activation
2. **Risk buckets**: 70% capital to arbitrage strategies, 30% to ML/directional
3. **Modular**: each strategy/model/broker is a plugin — zero cross-coupling
4. **Agent-ready**: every module has its own CLAUDE.md with safe modification zones
5. **Walk-forward only**: no in-sample-only backtests are accepted as valid

## Development Workflow
1. Add/modify a strategy → `backend/app/strategies/CLAUDE.md`
2. Tune an ML model → `backend/app/ml/CLAUDE.md`
3. Change execution logic → `backend/app/execution/CLAUDE.md`
4. Update the UI → `frontend/CLAUDE.md`

## Environment Variables
Copy `.env.example` to `.env` and fill in:
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` — from alpaca.markets
- `TRADESTATION_CLIENT_ID` / `TRADESTATION_SECRET` — from developer.tradestation.com
- `BINANCE_API_KEY` / `BINANCE_SECRET` — from binance.com
- `POLYMARKET_PRIVATE_KEY` — Polygon wallet private key
- `DATABASE_URL` — Supabase connection string (pooler URL port 6543)
- `REDIS_URL` — Upstash Redis REST URL
- `SECRET_KEY` — random 32-byte hex string

## Running Tests
```bash
cd backend && pytest tests/ -x -v
```

## Branch Strategy
Main branch: `main`
Feature branches: `feature/<name>`
Always push to the designated branch in the task description.
