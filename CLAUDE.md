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

## CTO Agent Protocol — Token Conservation (OKR1)

**Rule:** Claude Code context is the scarcest resource in any session. Never waste it on raw code.

### Delegation Rules
| Task | Action |
|------|--------|
| Read any file >100 lines | Delegate to Explore agent |
| Implement a new feature | Delegate to general-purpose agent with worktree isolation |
| Search the codebase | Delegate to Explore agent |
| Run tests + get results | Delegate to general-purpose agent (return: pass/fail + first failure only) |
| Debug CI logs | Delegate to general-purpose agent (return: root cause only) |
| Plan architecture | Delegate to Plan agent |
| CTO does inline | Routing decisions, commit messages, user responses, final verification |

### Parallel Dispatch Pattern
Always send independent tasks in a single message with multiple Agent tool calls.
Never chain sequentially what can fan out.

### Agent Prompt Template
Every agent delegation must specify:
1. The exact task (file paths, line numbers if known)
2. What to return (diff summary, test result, file list) — NOT full file contents
3. Word limit on response (under 200 words for research, under 400 for implementations)

### Anti-Patterns (banned)
- Reading >100-line files inline with Read tool
- Letting CI log output appear in CTO context
- Re-reading files already explored
- Getting agent responses with raw code blocks (ask for diffs/summaries)
- Sequential agent calls for independent tasks

### Free Agents Available (GitHub Actions)
5-tier free cascade: Groq (Llama 3.3 70B, 3 accounts × 1.5M tok/day) → Cerebras (Qwen3 32B, 1M tok/day) → GitHub Models (GPT-4o-mini, free in Actions) → OpenRouter (Llama 3.3 70B :free, 50 req/day) → Gemini Flash (1500 req/day, 3 accounts)

All orchestrated via `.github/scripts/slack_agent_team.py`. ALLOW_PAID_APIS=False enforced in code.
