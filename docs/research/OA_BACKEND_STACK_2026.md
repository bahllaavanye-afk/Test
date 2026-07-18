# What Would Options Alpha Be Running? — Backend Analysis (2026-07)

Honest sourcing note: Options Alpha has never published an engineering blog of
their stack. What follows separates **publicly verifiable facts** about how
their product works from **inference** about what that implies, then maps it
to concrete QuantEdge upgrades.

## Publicly verifiable facts about OA's platform

1. **Broker connections, not a broker.** OA executes through users' accounts
   at **Tradier, TradeStation, Tastytrade, and Schwab** via those brokers'
   REST/streaming APIs. Tradier is their flagship integration (the
   "commission-free automated options" partner).
2. **Tradier's option greeks/IV are ORATS-computed.** Tradier's own docs state
   greeks & IV on their chains endpoint are supplied by ORATS, refreshed
   roughly hourly. So an OA bot picking a "16Δ put" via Tradier is standing on
   ORATS analytics — without paying for ORATS directly.
3. **A no-code rules engine.** Bots are trigger → decision (yes/no recipe
   trees) → action, scanned on schedules (e.g. every 15 min in market hours).
   This is exactly the shape QuantEdge's Bot Builder already mirrors.
4. **SmartPricing.** OA's published order logic: start at/near mid, then step
   the limit price toward the market on a timer until filled or bailed —
   patient price improvement, not market orders.
5. **Position/decision state per bot** with full decision logs, tags, and
   portfolio-level guardrails (max allocation, position counts).

## What that implies about their backend (inference, labeled as such)

- A **job scheduler** ticking thousands of bot scans (any of Sidekiq/Celery/
  Temporal-class queueing would do; nothing exotic is required at their scale).
- **Broker-normalizing adapter layer** (four brokers, one internal order/
  position model) — the same AbstractBroker pattern QuantEdge has.
- **Vendor market data** for chains/quotes where broker feeds are thin
  (ORATS-via-Tradier for greeks; likely Polygon/dxFeed-class equities data).
- Web SaaS on commodity cloud. There is no evidence of HFT-class infra — OA
  is minutes-cadence automation, which is exactly GitHub-Actions-compatible.

**Key takeaway: QuantEdge's architecture is already OA-shaped.** Bot Builder ≈
their rules engine; desks ≈ scheduled scans; AbstractBroker ≈ their adapter
layer. The two genuine gaps are **data quality** (real greeks/IV vs our
HV-proxy + moneyness strikes) and **execution polish** (SmartPricing vs our
single cancel-replace).

## Concrete upgrades (queued in IMPROVEMENTS.md)

1. **Tradier sandbox as the options-data backbone** — free developer sandbox,
   real option chains WITH ORATS greeks/IV. Wire a `tradier.py` data adapter
   (chains + greeks only; execution stays Alpaca paper). Unlocks: real
   delta-based strike selection for the desk's mleg spreads (replacing
   moneyness approximation), real IV rank (replacing the HV proxy), and
   0DTE-capable chain lookups. Needs a free `TRADIER_SANDBOX_TOKEN` secret.
2. **SmartPricing-style laddered repricing** — extend `_ensure_filled`'s
   one-shot cancel-replace into a ladder: post at mid, step toward market
   every N seconds (e.g. 3 steps), only then market out. Directly measurable
   in the slippage dashboard.
3. (Later, live-money era) TradeStation/Tastytrade execution adapters — the
   TradeStation OAuth shell already exists in `brokers/tradestation.py`.

## What we deliberately DON'T copy

- Paying for ORATS/dxFeed directly (free tiers first; Tradier sandbox gives
  the greeks free).
- Multi-tenant SaaS concerns (auth tiers, billing) — QuantEdge is one fund,
  not a bot marketplace.
