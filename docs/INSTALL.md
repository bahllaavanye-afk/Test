# Installation Guide

## 1. Clone the Repository

```bash
git clone <repo-url> quantedge
cd quantedge
```

## 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your broker API keys (start with Alpaca paper):
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` — from https://alpaca.markets
- `SECRET_KEY` — generate with `python -c "import secrets; print(secrets.token_hex(32))"`

Leave others blank for now if you only want to test equities.

## 3. Choose Your Install Path

### Option A: Docker (Recommended)

Requires Docker 24+ and docker-compose.

```bash
./scripts/launch.sh dev
```

This starts:
- Backend at `http://localhost:8000` (docs at `/docs`)
- Frontend at `http://localhost:5173`
- PostgreSQL at port 5432
- Redis at port 6379

### Option B: Local Python (no Docker)

```bash
# Backend
cd backend
pip install uv
uv pip install --system -e .
uvicorn app.main:app --reload

# Frontend (in another terminal)
cd ../frontend
npm install
npm run dev
```

You'll need PostgreSQL and Redis running separately, OR use the SQLite fallback by setting `DATABASE_URL=sqlite+aiosqlite:///./quantedge_dev.db` in `.env`.

## 4. Run Database Migrations

```bash
./scripts/migrate.sh
# or: cd backend && alembic upgrade head
```

## 5. Seed Default Strategies (optional)

```bash
./scripts/seed.sh
```

Creates 3 default strategies (momentum, mean_reversion, triangular_arb) and 3 risk rules. All start disabled — toggle them on from the dashboard.

## 6. Register Your First User

Open `http://localhost:5173` → click "Sign In" → use any email/password. The first registration creates a new account.

## 7. Verify Installation

```bash
curl http://localhost:8000/health      # → {"status":"ok","mode":"paper"}
cd backend && pytest tests/unit/ -v    # all unit tests should pass
```

## 8. Run a Backtest

```bash
./scripts/launch.sh backtest momentum SPY 1d 2021-01-01 2024-01-01
```

You should see Sharpe > 0.7 on out-of-sample data.

## Troubleshooting

### Port 8000 already in use
```bash
lsof -ti:8000 | xargs kill -9
```

### "Module not found" errors
```bash
cd backend && uv pip install --system -e . --reinstall
```

### Redis connection refused
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

### Postgres connection refused
```bash
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:16-alpine
```

### LSTM training "out of memory"
Reduce batch_size in `experiments/configs/lstm_btc_1h.yaml` from 256 to 64, or use free GPU notebooks.
