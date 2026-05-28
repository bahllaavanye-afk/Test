# Infrastructure Guide

## Current Free Stack (Recommended for < $50k revenue)
- Vercel: frontend hosting, CDN, serverless functions — $0
- Render: backend API + worker — $0 (sleep after 15min inactivity — use UptimeRobot to keep alive)
- Supabase: PostgreSQL 500MB, auth, realtime — $0
- Upstash: Redis 10K req/day — $0
- Alpaca: paper + live trading, market data — $0 commissions
Total: $0/month

## When to Switch to AWS
| Need | AWS Service | Monthly Cost | Trigger |
|------|-------------|-------------|---------|
| No cold starts (always-on API) | EC2 t3.micro + EIP | ~$10 | > 100 active users |
| GPU model training | g4dn.xlarge Spot | ~$0.15-0.50/hr | Need LSTM/TFT training |
| Low-latency execution (<1ms) | EC2 c5n us-east-1 (same DC as Alpaca) | ~$50-100/mo | Live trading > $100k |
| More DB | RDS PostgreSQL | ~$15/mo | > 500MB data |
| Redis cache | ElastiCache | ~$15/mo | > 10K req/day |

## ML Training Strategy (Free → Paid)
1. **Free first** (recommended): Kaggle (30 GPU hrs/week T4), Google Colab (free T4), Lightning.AI (22 hrs/mo A10G)
2. **AWS Spot** when needed: `aws ec2 run-instances --instance-type g4dn.xlarge --spot-price 0.50` — gets ~$0.15-0.30/hr
3. **SageMaker** only if you need MLOps pipelines — overkill for a startup

## Execution Speed Optimization
Alpaca's infrastructure is co-located in us-east-1. For fastest execution:
1. Deploy backend to AWS EC2 us-east-1 (same region as Alpaca) — saves 5-20ms round-trip vs Render
2. Use Alpaca WebSocket for order updates instead of polling
3. Pre-validate orders (compute all fields client-side before hitting the API)
4. Use httpx with connection pooling (already done) — reuses TCP connections

## Current Bottlenecks
- Render free tier sleeps after 15 min idle → 30s cold start. Fix: UptimeRobot pings /health every 5 min (free)
- SQLite in dev → switch to Supabase PostgreSQL for production (better concurrent write performance)
