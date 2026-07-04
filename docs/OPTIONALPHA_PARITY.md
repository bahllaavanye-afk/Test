# Option Alpha Feature-Parity Deep Research

Research date: 2026-07-04. This document compares [Option Alpha](https://optionalpha.com) — a
no-code options-trading automation platform — against QuantEdge's current options/bot surface,
based on a page-by-page review of Option Alpha's public site/docs and a direct read of the
QuantEdge codebase (backend `app/api/v1/`, `app/strategies/`, `app/bots/`, `app/execution/`, and
frontend `src/pages/`).

Legend: ✅ have · 🟡 partial · ❌ missing

## 1. Feature Matrix

### Automation / Bot Builder

| Option Alpha feature | QuantEdge status | Where in the codebase |
|---|---|---|
| No-code bot builder (visual, form-based) | ✅ have | `frontend/src/pages/BotBuilder.tsx`, `backend/app/api/v1/bots.py` |
| Decision recipes (technical indicators, AND/OR logic) | ✅ have | `backend/app/bots/engine.py` — 20+ indicators (RSI, MACD, BB, ADX, Stoch, CCI, Ichimoku, Supertrend, MFI, PPO, pivots) with `condition_logic: ALL/ANY` |
| Triggers: schedule / price-cross / indicator / manual button | ✅ have | `bots.py` `TriggerConfig`; `POST /bots/{id}/run` = OA's "Buttons" |
| Scanners (continuous, find new positions) vs. Monitors (manage open ones) | 🟡 partial | Bots run on a fixed schedule/trigger (`bots/engine.py`); there is no distinct "scanner cadence vs. monitor cadence" split like OA's 15-min scanner / continuous monitor model |
| Actions: open/close position, alert, reduce, tag | 🟡 partial | `ActionConfig` supports `open_long/short`, `close_position`, `send_alert`, `reduce_position`, `open_option_spread` (`bots/engine.py:826-877`); no "tag" action |
| Multi-leg options spread bots | 🟡 partial | `open_option_spread` action + 4 options bot templates exist (`bots/templates.py:441-529`); delta-based legs only route to a live TradeStation account, otherwise degrade to an alert (`bots/engine.py:895-958`) — frontend `BotBuilder.tsx` market-type dropdown doesn't even list "options" (only Equity/Crypto/Polymarket, `BotBuilder.tsx:585-589`), so building one from scratch in the UI isn't possible today (only via template load) |
| Bot templates library (pre-built strategies) | ✅ have | `backend/app/bots/templates.py` — 25 templates incl. RSI, momentum, wheel, iron condor, bull put spread |
| Auto-generate a bot directly from a winning backtest | ❌ missing | No such glue between `backtests.py` results and `bots.py` create |
| Bot cloning / sharing / public community library / leaderboard of shared bots | ❌ missing | `leaderboard.py` ranks only the current user's own strategies (backtest/paper/live Sharpe etc.) — there is no cross-user sharing, cloning, or community |
| SmartPricing (adaptive multi-step limit-order walk, Normal/Fast/Patient modes) | 🟡 partial | `backend/app/execution/limit_first.py` posts one limit order offset from the quote and falls back to market after a fixed timeout — a single-step analog, not OA's configurable 3-5 step price ladder with nickel rounding |

### Backtesting

| Option Alpha feature | QuantEdge status | Where in the codebase |
|---|---|---|
| Vectorized single-strategy backtester | ✅ have | `backend/app/api/v1/backtests.py` `POST /backtests/run`, `backend/app/backtest/engine.py` |
| Walk-forward validation (train/test rolling windows) | ✅ have (exceeds OA) | `POST /backtests/walk-forward`, `backend/app/backtest/walk_forward.py` — OA does not publicly document walk-forward testing |
| Historical stress-test scenarios | ✅ have (exceeds OA) | `GET /backtests/scenarios`, `backend/app/backtest/stress_test.py` |
| 1-minute options-chain-accurate backtesting (0DTE/next-day, real historical option prices) | ❌ missing | `options_strategies.py` backtests use a realized-volatility (HV20) percentile as an **IV-rank proxy** because no historical options-chain dataset is wired in (see docstrings, e.g. `options_strategies.py:112-135`) |
| Combine multiple backtests into one portfolio P&L | ❌ missing | No endpoint sums/combines arbitrary `BacktestRun` equity curves; `comparison.py` only does manual-vs-ML head-to-head |
| Detailed trade log with entry/exit times & pricing | 🟡 partial | `BacktestResult.trades_log` exists for walk-forward runs (`backtests.py:268`) but the standard vectorized run doesn't persist a per-trade log, only equity curve + summary metrics |

### Scanners / Market Data

| Option Alpha feature | QuantEdge status | Where in the codebase |
|---|---|---|
| IV Rank scanner (single symbol) | ✅ have | `GET /market-data/iv-rank/{symbol}`, `market_data.py:571` |
| IV Rank batch scanner (curated watchlist) | ✅ have | `GET /market-data/iv-rank-scan`, `market_data.py:590` |
| Earnings calendar w/ EPS estimates | 🟡 partial | `GET /market-data/earnings` proxies Alpaca corporate-actions; returns empty when Alpaca's premium tier isn't enabled (`market_data.py:666-708`) |
| Put/Call ratio | 🟡 partial | Real implementation exists at `GET /market-data/pcr` (`market_data.py:713`), but the options desk frontend (`OptionsFlow.tsx:15-19`) calls a different, **non-existent** `GET /options/put-call-ratio` — that panel is broken |
| Options unusual-activity / flow scanner | ❌ broken | `frontend/src/pages/OptionsFlow.tsx:9-13` calls `GET /options/flow`, which does not exist anywhere in `backend/app/api/v1/options.py` or elsewhere — confirmed via full-repo search. The page renders but the query 404s |
| Wheel (CSP) opportunity scanner across tickers, with annualized yield | ❌ broken | `OptionsFlow.tsx:21-25` calls `GET /options/wheel` — no such route exists. The `wheel` **strategy** exists as a signal generator (`options_strategies.py:574-756`) and bot template, but there's no scanner endpoint that surfaces candidate tickers/strikes |
| Macro/earnings event calendar (FOMC, CPI, PPI, NFP) | ❌ broken | `OptionsFlow.tsx:27-37` calls `GET /options/macro-calendar` and `GET /options/next-fomc` — neither exists in the backend |
| Curated watchlists (Momentum / IV Rank / Earnings, one screen) | 🟡 partial | Individual pieces exist (`iv-rank-scan`, `earnings`) but nothing composes them into one OA-style curated watchlist screen; `scanners.py` desks are `equity`/`crypto`/`polymarket` only — no `options` desk |

### Options Trading Desk (chain, Greeks, order ticket)

| Option Alpha feature | QuantEdge status | Where in the codebase |
|---|---|---|
| Options chain viewer with bid/ask/Greeks/IV | ✅ have | `frontend/src/pages/Options.tsx`, `backend/app/api/v1/options.py` (`/options/chain/{symbol}`, `/options/snapshot/{symbol}`), Alpaca-backed |
| Strategy quick-select (covered call, CSP, iron condor, long call/put) | ✅ have | `Options.tsx:132-138` `STRATEGIES` quick-select buttons |
| Straddle/strangle cost & breakeven calculator | ✅ have | `Options.tsx:760-769` |
| Portfolio Greeks aggregation | ✅ have | `frontend/src/components/options/PortfolioGreeks.tsx`, embedded in `Options.tsx` |
| Order-rules guardrails (DTE/delta/IV-rank/size checks before submit) | ❌ broken | `Options.tsx` `RulesPanel`/`OrderPanel` call `POST /options/rules/validate` (`Options.tsx:353,500`) — this endpoint does not exist in the backend; the "Rules Check" panel silently fails to load |
| Actual order submission from the options ticket | ❌ missing | `OrderPanel.handleSubmit()` (`Options.tsx:508-519`) only fakes a 300ms delay and shows a "preview" string — it never calls a real order-submission endpoint. Only bot-driven `open_option_spread` actions can route a real (live, TradeStation-only) order |

### Paper Trading & Risk

| Option Alpha feature | QuantEdge status | Where in the codebase |
|---|---|---|
| Built-in broker-free paper trading engine | ✅ have (exceeds OA in policy) | Paper-first is a hard platform rule (root `CLAUDE.md`: "every strategy must run 2 weeks on paper before live activation"); `Order.status == "paper"`, `bots/engine.py:_create_paper_order` |
| Automatic TP/SL/time-based exit monitoring | ✅ have | `bots/engine.py:check_bot_exits` — scheduler-driven, closes paper positions and records `Trade` rows (OA-style trade history) |
| Greeks-based risk limits (delta/gamma/theta/vega caps) | 🟡 partial | Documented and intended per `strategies/options/CLAUDE.md:59-65` (±0.15 NAV delta, etc.) but not confirmed enforced in `risk/manager.py` for options positions specifically — only documented as a target |

### Community, Alerts, Brokers

| Option Alpha feature | QuantEdge status | Where in the codebase |
|---|---|---|
| Community: share/clone bot templates, public leaderboard | ❌ missing | No multi-tenant sharing model anywhere in `models/` or `api/v1/`; `leaderboard.py` is single-account |
| Email/SMS alerts on bot triggers | 🟡 partial | `send_alert` action logs the event and is surfaced via `notifications.py`, but that module is a Slack-integration + in-app activity tracker (`notifications.py:1-20`), not an email/SMS delivery pipeline like OA's |
| Broker integrations | 🟡 partial (different set) | QuantEdge: Alpaca (equities+options), TradeStation (equities+options), Binance (crypto), Polymarket (prediction markets) — `backend/app/brokers/`. OA: tastytrade, TradeStation, Tradier, Charles Schwab. No overlap except TradeStation; QuantEdge has crypto/prediction-market coverage OA lacks, OA has tastytrade/Tradier/Schwab QuantEdge lacks |
| Multiple accounts / sub-accounts per broker | ✅ have | `accounts.py`, `models/account.py` supports multiple `Account` rows per user |
| SaaS pricing tiers (Free/Pro/Elite, broker-linked free upgrades) | N/A | QuantEdge is a proprietary single-tenant platform, not a subscription SaaS — no equivalent concept |

**Matrix coverage: 30 Option Alpha feature rows assessed** (11 automation/bot, 5 backtesting, 8
scanner/market-data, 6 options-desk, 3 paper/risk, 5 community/alerts/broker — some rows bundle
closely related sub-features, e.g. TP/SL/time exits).

## 2. Prioritized Gap List

### P0 — Broken or actively misleading (frontend calls endpoints that don't exist)

1. **Wire up `/options/rules/validate`.** `Options.tsx` ships a full "Rules Check" UI
   (DTE/delta/IV-rank/position-size) that calls a POST endpoint absent from `options.py`. Add a
   `POST /options/rules/validate` handler that reuses the DTE/delta/IV-rank thresholds already
   documented in `strategies/options/CLAUDE.md` (e.g. IVR > 50 → sell, 21-DTE hard exit) and
   returns the `RulesValidationResponse` shape the frontend already expects.
2. **Wire up `/options/flow`, `/options/put-call-ratio`, `/options/wheel`, `/options/macro-calendar`,
   `/options/next-fomc`.** `OptionsFlow.tsx` is an entire page (unusual-activity feed, PCR tile,
   wheel-scanner tile, macro calendar) built against five endpoints that 404 today. Cheapest fix:
   alias `/options/put-call-ratio` to the existing `/market-data/pcr` logic, add a thin `/options/wheel`
   scanner that runs the existing `WheelStrategy.analyze()` signal across a ticker list and returns
   candidates with annualized yield, and add a small static/curated FOMC+CPI+PPI+NFP calendar
   endpoint (`macro-calendar`, `next-fomc`) — this is mostly static data, low effort, high payoff.
3. **Real order submission from the options ticket.** `OrderPanel.handleSubmit()` in `Options.tsx`
   fakes success after a `setTimeout`; no order is ever placed. Wire it to a real
   `POST /orders` (or a new `/options/orders`) call through the existing Alpaca/TradeStation broker
   layer so manual options trades actually execute (paper first, per platform policy).

### P1 — Real gaps vs. Option Alpha's core value proposition

4. **True SmartPricing (adaptive multi-step limit ladder).** Extend `execution/limit_first.py`
   (or add `execution/smart_pricing.py`) to walk N prices from mid toward bid/ask over configurable
   intervals (mirroring OA's Normal/Fast/Patient: 3-5 steps, 5-20s each, nickel rounding) instead of
   one limit attempt + market fallback. Wire it into `smart_router.py` as an option-order-specific
   execution mode.
5. **Options-chain-accurate backtesting.** Today `options_strategies.py` backtests substitute a
   realized-volatility percentile for IV rank because no historical options-chain data source is
   wired in. Add a historical options data loader (even a coarser daily IV/Greeks snapshot store)
   so `IronCondorStrategy`/`WheelStrategy`/etc. can backtest against real IV rank and real spread
   pricing instead of an HV proxy — this is the single biggest fidelity gap versus OA's advertised
   "3 years of 1-minute historical options data."
6. **Expose "options" as a first-class market type in BotBuilder.tsx.** The backend already
   supports `market_type: "options"` and `open_option_spread` actions (see the 4 templates in
   `bots/templates.py:441-529`), but the visual builder's Market Type `<Select>` only offers
   Equity/Crypto/Polymarket (`BotBuilder.tsx:585-589`). Add the option and a leg-builder sub-form
   (delta/DTE/side per leg) so users can build multi-leg options bots from scratch, not just via
   template load.
7. **Auto-generate a bot from a backtest result.** Add a "Create Bot from this Backtest" action on
   `BacktestLab.tsx` that maps a `BacktestRun`'s strategy/symbol/params into a `BotCreate` payload
   and calls `POST /bots/`, closing the backtest→automation loop OA advertises as a headline
   feature.
8. **Email/SMS alert delivery.** `send_alert` bot actions currently only log + surface in-app;
   there's no outbound email/SMS. Add a lightweight notification channel (e.g. SES/Twilio) fed by
   the same `BotResult`/`send_alert` path in `bots/engine.py`.

### P2 — Nice-to-have / lower urgency

9. **Combine multiple backtests into one portfolio equity curve**, mirroring OA's "run concurrent
   strategies, see combined P/L." Add an endpoint that sums selected `BacktestResult.equity_curve`
   series with normalized weights.
10. **Community layer: shareable/cloneable bot templates + public leaderboard.** Would require new
    multi-tenant data model (`shared_templates`, visibility flags) — a bigger lift, lower priority
    since QuantEdge is presently single-tenant per the CLAUDE.md description.
11. **Dedicated "options" scanner desk** in `scanners.py` (alongside equity/crypto/polymarket) that
    composes `iv-rank-scan` + `earnings` + the new wheel scanner into one OA-style curated watchlist
    screen, rather than three separate ad hoc calls.
12. **Persisted per-trade log for standard (non-walk-forward) backtests**, matching OA's "every
    backtest includes detailed trade logs with exact entry/exit times."

## 3. Sources

- [Bots | Automated Trading from Option Alpha](https://optionalpha.com/bots)
- [Automations | Option Alpha](https://docs.optionalpha.com/tools/bots/automations)
- [Bot Automation Basics | Option Alpha](https://optionalpha.com/help/automation-basics)
- [Bots 101 | The Ultimate Guide to Option Alpha Bots](https://optionalpha.com/bots-101)
- [Decisions | Option Alpha](https://docs.optionalpha.com/platform/bots/decisions)
- [Decision actions | Option Alpha](https://docs.optionalpha.com/tools/bots/decision-actions)
- [Automation Structure | Option Alpha](https://optionalpha.com/help/automation-structure)
- [SmartPricing | Getting Started | Option Alpha](https://optionalpha.com/help/smartpricing)
- [SmartPricing | Option Alpha Docs](https://docs.optionalpha.com/platform/bots/smartpricing)
- [Option Alpha's SmartPricing Technology](https://optionalpha.com/tools/smartpricing)
- [Backtest 0DTE & Next-Day Options Strategies with Option Alpha](https://optionalpha.com/backtester)
- [Backtesting Metrics | Option Alpha](https://optionalpha.com/help/backtesting-metrics)
- [Automatically Generate Bots From Any Backtest](https://optionalpha.com/blog/automatically-generate-bots-from-any-backtest)
- [Introduction to Strategy Backtesting | Option Alpha](https://optionalpha.com/learn/strategy-backtesting)
- [Watchlist | Option Alpha](https://optionalpha.com/watchlist)
- [Options Trading Scanner & Stock Watch List | Option Alpha](https://optionalpha.com/members/watch-list)
- [Scanning For Trades | Option Alpha](https://optionalpha.com/lessons/scanning-for-trades)
- [Bot Templates & Cloning Strategies | Option Alpha](https://optionalpha.com/help/templates-and-cloning)
- [Share Bot Automations Just as Easily as Bot Templates](https://optionalpha.com/blog/share-bot-automations-just-as-easily-as-bot-templates)
- [Community | Option Alpha](https://optionalpha.com/community)
- [Bot Templates | Option Alpha](https://optionalpha.com/templates)
- [Broker Integrations | Option Alpha](https://optionalpha.com/integrations)
- [Connecting your brokerage account | Option Alpha](https://docs.optionalpha.com/getting-started/connecting-your-brokerage-account)
- [Trading Accounts | Option Alpha Help](https://optionalpha.com/help/trading-accounts)
- [Tradier and Option Alpha | Partner Offer](https://trade.tradier.com/option-alpha/)
- [Creating Automatic Alerts | Option Alpha](https://optionalpha.com/lessons/creating-automatic-alerts)
- [Platform Tour | Option Alpha Automated Trading](https://optionalpha.com/help/platform-tour)
- [Pricing | Option Alpha](https://optionalpha.com/pricing)
- [Option Alpha FAQs | Get Answers to Common Questions](https://optionalpha.com/faqs)
- [Option Alpha Review 2026 — Pricing, Features, Pros & Cons](https://tradingtoolshub.com/review/option-alpha/)
