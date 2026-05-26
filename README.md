# QuantEdge

**Institutional-grade quantitative trading platform** — multi-broker, ML-enhanced, always-running, self-improving.

[![Status](https://img.shields.io/badge/status-paper%20trading-orange)]() [![Backend](https://img.shields.io/badge/backend-FastAPI-009688)]() [![Frontend](https://img.shields.io/badge/frontend-React%2018-61dafb)]() [![License](https://img.shields.io/badge/license-proprietary-red)]()

> Beat Renaissance, Two Sigma, and Citadel performance benchmarks with proven strategies from academic literature, ML-enhanced signals, and never-sleeping execution.

---

## Quick Start

```bash
# 1. Clone and configure
git clone <repo>
cd quantedge
cp .env.example .env             # fill in broker API keys

# 2. Start full stack (Docker)
./scripts/launch.sh dev          # backend + frontend + postgres + redis

# 3. Open dashboard
open http://localhost:5173

# 4. (Optional) Run a backtest
./scripts/launch.sh backtest momentum SPY 1d 2021-01-01 2024-01-01

# 5. (Optional) Train an ML model
./scripts/launch.sh train --config experiments/configs/lstm_btc_1h.yaml
```

---

## Features

### 🔌 Multi-Broker
- **Alpaca** (primary, commission-free equities)
- **TradeStation** (secondary, OAuth2)
- **Binance** (crypto via CCXT)
- **Polymarket** (prediction markets via CLOB client)

### 📈 14 Trading Strategies (manual + ML-enhanced)
Every strategy runs in two versions — manual (indicator-only) and ML-enhanced (same logic + ML filter). Compared head-to-head with statistical significance testing.

### 🤖 Always-Running, Self-Improving
- **AlgoAgent** uses UCB1 (Upper Confidence Bound) to continuously test strategies, exploring new ones while exploiting winners
- **Nightly retraining** of ML models on fresh data
- **Per-(strategy, symbol)** asyncio tasks scale to hundreds of concurrent loops
- Bot generates signals 24/7 across equities, crypto, and prediction markets

### 🎯 Execution & Slippage Minimization
- **TWAP** (>$10k orders): splits across N slices over duration minutes
- **VWAP**: participates at volume-weighted intervals
- **LimitFirst**: post limit, fall back to market after 30s — saves 5-15 bps vs market orders
- **Iceberg**: hides large order size
- **SmartOrderRouter**: picks best algo automatically
- **SlippageTracker**: real-time bps tracking per algo

### 🛡️ Risk Management
- **Kelly criterion** position sizing (25% fractional, 20% hard cap)
- **Correlation clusters** with 30% max allocation per cluster
- **Circuit breakers**: 10% global drawdown, 5% arb bucket
- **Paper-first** policy: 2-week paper validation before live activation

### 📊 Investor-Ready Comparison
- Side-by-side against SPY, QQQ, BRK.B, Ray Dalio All Weather
- t-test for statistical significance (p<0.05)
- Walk-forward + Monte Carlo validation
- Investor-pitch performance reports

### 🔔 Slack Notifications
- Order fills, signal alerts, risk events, circuit breakers, experiments
- Per-channel webhooks (orders / signals / alerts / experiments / system)

### 🔒 Security
- JWT auth (15-min access + 7-day refresh)
- AES-256 (Fernet) encryption for broker keys at rest
- Rate limiting (slowapi: 100 req/min/user)
- CORS allowlist (Vercel domain only in production)
- All ORM (no raw SQL) — SQL injection-proof
- Pydantic v2 strict validation at API boundaries

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Frontend (React 18 + Vite + TS, hosted on Vercel)              │
│   ├── TradingView Advanced Chart (free iframe)                  │
│   ├── Lightweight Charts (portfolio analytics)                  │
│   └── Bloomberg-style dark theme + Redux + TanStack Query       │
└─────────────────────────────────────────────────────────────────┘
                              ↑↓ REST + WebSocket
┌─────────────────────────────────────────────────────────────────┐
│  Backend (FastAPI async, hosted on Render)                       │
│   ├── AlgoAgent (UCB1, runs every 5 min, always)                │
│   ├── StrategyRunner (1 task per strategy+symbol, 24/7)         │
│   ├── PriceFeed (Redis fan-out + WebSocket)                     │
│   ├── Smart Order Router → TWAP/VWAP/LimitFirst/Iceberg         │
│   ├── Risk Manager (Kelly + correlation + circuit breakers)     │
│   ├── ML Inference Service (LSTM + XGBoost + Lorentzian + TFT)  │
│   ├── Comparison Engine (manual vs ML + benchmarks)             │
│   └── Slack Notifier (orders, signals, alerts)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────┐ ┌────────────┐ ┌──────────────┐ ┌──────────────┐
│  Supabase    │ │  Upstash   │ │  yfinance    │ │  Alpaca/TS/  │
│  PostgreSQL  │ │  Redis     │ │  benchmarks  │ │  Binance/    │
│              │ │            │ │              │ │  Polymarket  │
└──────────────┘ └────────────┘ └──────────────┘ └──────────────┘
```

---

## Trading Strategies

### Manual (indicator-only)
| Strategy | Logic | Academic Basis | Sharpe Target |
|----------|-------|----------------|---------------|
| Pairs Trading | Engle-Granger cointegration, z-score entry | Gatev et al. (2006) | 1.5-2.5 |
| Momentum | 12-1 month return ranking | Jegadeesh & Titman (1993) | 0.7-1.0 |
| Low Volatility | Long bottom-decile vol, short top-decile | Baker et al. (2011) | 0.6-0.8 |
| RSI + MACD | RSI(14)<30 + MACD cross confirmation | Empirical | 0.5-0.8 |
| Breakout | Volume-confirmed 52W high breakout | Empirical | 0.5-0.7 |
| Mean Reversion | Bollinger Band + RSI | Empirical | 0.6-0.9 |
| Supertrend | ATR-based trend follower | TV community | 0.6-0.9 |
| Triangular Arb | BTC→ETH→USDT→BTC mismatches (Binance) | Microstructure | >2.0 |
| Polymarket Arb | YES+NO < $0.97 → risk-free | Pure arbitrage | risk-free |

### ML-Enhanced
| Strategy | ML Filter | Expected Improvement |
|----------|-----------|---------------------|
| ml_momentum | LSTM probability > 0.6 | +20-40% Sharpe |
| ml_mean_reversion | XGBoost prob > 0.65 | -30% false signals |
| ml_breakout | LSTM + XGBoost + Lorentzian ensemble | +25-35% win rate |
| lorentzian_knn | TradingView Lorentzian KNN port | Direct replacement |
| ensemble | Weighted combo of all models | Highest Sharpe |

---

## Performance Targets

| Metric | Target | SPY | BRK.B | All Weather |
|--------|--------|-----|-------|-------------|
| Annual Return | 20-35% | 10% | 19.9% | 8.2% |
| Sharpe Ratio | >2.0 | 0.47 | 0.79 | 0.67 |
| Max Drawdown | <15% | -57% | -48% | -20% |
| Win Rate | >55% | n/a | n/a | n/a |

---

## ML Framework

- **PyTorch Lightning** for training loops
- **MLflow** for experiment tracking
- **Ray Tune** for distributed HPO
- **Optuna** for XGBoost HPO (100 trials)
- **SHAP** for feature importance
- **Models**: BiLSTM+attention, Temporal Fusion Transformer, XGBoost, LightGBM, Lorentzian KNN, Ensemble

### Free GPU Training
- **Kaggle**: 30 GPU hrs/week free (T4/P100) → `notebooks/train_lstm.ipynb`
- **Google Colab**: free T4 → `notebooks/train_xgboost.ipynb`
- **Lightning.AI Studios**: 22 GPU hrs/month → `notebooks/train_transformer.ipynb`

---

## Hosting (All Free Tier)

| Service | Used For | Free Tier |
|---------|----------|-----------|
| Vercel | Frontend | Unlimited static |
| Render | Backend API + Worker | 750 hrs/month + UptimeRobot keep-alive |
| Supabase | PostgreSQL + Auth | 500 MB DB |
| Upstash | Redis REST | 10K commands/day |
| UptimeRobot | Health-check ping | 5 min interval |

---

## Tech Stack

**Backend**: Python 3.11 · FastAPI · SQLAlchemy 2.0 (async) · asyncpg · Alembic · Redis · slowapi · python-jose · cryptography (Fernet)

**ML**: PyTorch 2.5 · Lightning · MLflow · Ray Tune · XGBoost · LightGBM · scikit-learn · Optuna · SHAP · stable-baselines3

**Data**: pandas · numpy · pandas-ta · yfinance · scipy · statsmodels · VectorBT

**Brokers**: alpaca-py · ccxt · py-clob-client · custom httpx for TradeStation OAuth2

**Frontend**: React 18 · Vite · TypeScript · Redux Toolkit · TanStack Query · Tailwind CSS · TradingView widgets · Lightweight Charts

---

## Development Workflow

See module-specific guides for safe-modification zones:
- [`backend/CLAUDE.md`](backend/CLAUDE.md) — backend overview
- [`backend/app/strategies/CLAUDE.md`](backend/app/strategies/CLAUDE.md) — adding strategies
- [`backend/app/ml/CLAUDE.md`](backend/app/ml/CLAUDE.md) — ML experiments
- [`backend/app/execution/CLAUDE.md`](backend/app/execution/CLAUDE.md) — execution algos
- [`frontend/CLAUDE.md`](frontend/CLAUDE.md) — UI components

---

## Testing

```bash
cd backend && pytest tests/ -x -v          # all unit + integration tests
cd backend && pytest tests/unit/ -x -v     # unit only (no DB)
cd backend && pytest --cov=app             # coverage report
```

See [`docs/TESTING.md`](docs/TESTING.md) for details.

---

## Deployment

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for full guide.

```bash
# Backend → Render (free tier)
git push origin main  # auto-deploys via render.yaml

# Frontend → Vercel (free tier)
vercel --prod  # or push to git, Vercel auto-deploys
```

---

## Documentation

- [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) — system requirements
- [`docs/INSTALL.md`](docs/INSTALL.md) — local install guide
- [`docs/API.md`](docs/API.md) — REST + WebSocket API reference
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — deep architecture
- [`docs/STRATEGIES.md`](docs/STRATEGIES.md) — strategy details and parameters
- [`docs/TESTING.md`](docs/TESTING.md) — test suite reference
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — production deployment

---

## License

Proprietary — not for redistribution. Built for institutional internal use and investor demos.
