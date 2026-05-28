# System Requirements

## Minimum (Development)
- **OS**: Linux / macOS / Windows (WSL2)
- **CPU**: 2 cores
- **RAM**: 4 GB
- **Disk**: 5 GB
- **Python**: 3.11+
- **Node.js**: 20+
- **Docker**: 24+ (for full-stack via docker-compose)

## Recommended (Production)
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Disk**: 20 GB SSD (for OHLCV cache + model artifacts)
- **Network**: Stable broadband — broker connections require <100ms latency for execution

## For ML Training (optional, use free GPU clouds)
- **Local CPU training**: 8 GB RAM, ~30 min for LSTM on 2yr 1h BTC data
- **Kaggle GPU (free)**: T4/P100 — 30 hrs/week
- **Colab GPU (free)**: T4 — limited runtime
- **Lightning.AI (free)**: 22 hrs/month, A10G available

## Broker Accounts (paper recommended for first month)
| Broker | Required | Where to sign up | Free? |
|--------|----------|-------------------|-------|
| Alpaca | yes (primary) | https://alpaca.markets | Yes (paper + free live) |
| TradeStation | optional | https://developer.tradestation.com | Yes (paper) |
| Binance | for crypto | https://binance.com | Yes (testnet) |
| Polymarket | for prediction markets | https://polymarket.com | Yes (mainnet, real money) |

## API Keys (all free except where noted)
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` — from Alpaca dashboard
- `TRADESTATION_CLIENT_ID`, `TRADESTATION_SECRET` — from TradeStation developer portal
- `BINANCE_API_KEY`, `BINANCE_SECRET` — from Binance API management
- `POLYMARKET_PRIVATE_KEY` — Polygon wallet private key (for trading)
- `NEWSAPI_KEY` (optional) — for sentiment features, free at https://newsapi.org

## Hosting (all free tier — no card required)
- **Vercel** (frontend): https://vercel.com
- **Render** (backend): https://render.com — needs UptimeRobot keep-alive
- **Supabase** (Postgres): https://supabase.com — 500 MB free
- **Upstash** (Redis): https://upstash.com — 10K commands/day free
- **UptimeRobot**: https://uptimerobot.com — pings /health every 5 min

## Optional Tools
- **Slack workspace** with incoming webhooks — for trade notifications (free)
- **MLflow tracking server** — local file-based by default, no setup needed
- **Playwright** — for automated screenshot capture (`pip install playwright`)
