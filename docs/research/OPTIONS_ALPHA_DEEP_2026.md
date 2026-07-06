# Options Alpha — Deep Feature Teardown & Strategy Backlog (2026)

_Research doc. Source: Options Alpha product (optionalpha.com) + the two dashboard
screenshots shared 2026-07-06. Purpose: catalog every Options-Alpha feature, map
each to what QuantEdge already has, and give a prioritized build backlog + the top
options strategies to run. Companion to `STRATEGY_SOTA_2026.md` and `DESK_SOTA_2026.md`._

## TL;DR
QuantEdge already has the hard part — a **Bot** model, a visual **BotBuilder**,
scheduled triggers, TP/SL/expiry exits, per-bot trade history, and ~13 options
strategy plugins. What's missing vs Options Alpha is mostly **productization**:
(1) the per-bot **equity/P&L graph** (screenshot parity), (2) **backtest-before-activate**
gating, (3) a **decision-recipe** layer, and (4) wiring the core **income strategies**
(wheel, iron condor, credit spreads, covered call) into the live options desk — they
exist in the registry but aren't enabled (live `system-status` showed `options: 0`).

---

## 1. Feature catalog — Options Alpha → QuantEdge

| # | Options Alpha feature | What it does | QuantEdge status | Gap / action |
|---|---|---|---|---|
| 1 | **Bots** (no-code automated bots) | Each bot = symbols + triggers + decisions + positions | ✅ `Bot` model + `BotBuilder.tsx` + `bot_runner` | Parity on structure |
| 2 | **Automations / decision flow** | Scanner → decision recipe → action, as a visual graph | ⚠️ conditions/triggers exist as flat rules | **Add a "decision recipe" block** (reusable named condition sets) |
| 3 | **Decision recipes** (reusable logic) | "SPY IV Rank > 50" saved + shared across bots | ❌ | New: `bot_recipes` — named, reusable condition groups |
| 4 | **Bot templates** | Prebuilt Iron Condor / Strangle / Wheel bots | ⚠️ `bots/templates` endpoint exists | Populate with the top-strategy presets (§3) |
| 5 | **Triggers** (scheduled + event) | Run every N min, or on price/indicator/earnings event | ✅ schedule + indicator + time-window | Add **event triggers**: earnings, FOMC, IV-rank cross |
| 6 | **Position management** | Profit-target %, stop %, DTE exit, trailing | ✅ `check_bot_exits` (TP/SL/expiry) | Add **DTE-based exit** + **trailing %** for options |
| 7 | **SmartPricing** | Walk the spread for best fill | ⚠️ limit-first in `desk_order_placer` | Add mid-price → walk-to-fill for multi-leg |
| 8 | **Backtesting** | Backtest a bot before going live | ✅ backtest engine exists | **Gate activation on a passing backtest** (Principle: paper-first) |
| 9 | **Autotrading / broker link** | Tradier / TastyTrade execution | ⚠️ TradeStation multi-leg routing only | Real options fills need TradeStation live creds |
| 10 | **Bot visualization** (the screenshots) | Equity curve + open positions with a P&L graph + trade log | ⚠️ trade history exists; **no per-bot equity sparkline** | **Build `/bots/{id}/performance` (30D P&L series) + sparkline** |
| 11 | **Tags / grouping** | Organize bots by tag/strategy | ❌ | Add `tags: list[str]` to Bot (migration) |
| 12 | **Paper trading** | Run bots on paper first | ✅ TRADING_MODE=paper platform-wide | Parity |
| 13 | **Toolbox** (calculators) | Expected move, IV rank, prob-of-profit, Greeks | ⚠️ IV-rank + PCR endpoints exist | Add **expected-move** + **prob-of-profit** + **Greeks** endpoints |
| 14 | **Economic/earnings calendar** | Gate entries around events | ⚠️ `/options/macro-calendar` + `/next-fomc` | Add **earnings calendar** gating hook |
| 15 | **Alerts / notifications** | Bot fired / position closed / target hit | ✅ Slack→Discord notify layer | Route per-bot fills to a `#bots` channel |
| 16 | **Track record / signals** | Public performance of each bot | ⚠️ `/leaderboard/live` (new) ranks by realized P&L | Surface per-bot in the UI |
| 17 | **Watchlists** | Symbol universes per bot | ✅ bot.symbol(s) | Extend to multi-symbol baskets |

**Screenshot parity (item 10) is the single most visible gap** — the user explicitly
asked for the per-bot P&L graph twice. It's a backend `/bots/{id}/performance`
endpoint (30-day realized-P&L series from `Trade` rows) + a Lightweight-Charts
sparkline in `BotBuilder.tsx`. It only shows real data once trades exist (Alpaca keys + DB).

---

## 2. Position P&L graph — the screenshot feature (spec)

**Backend** `GET /bots/{bot_id}/performance?days=30`:
- Query `Trade` where `strategy_name == bot.name`, `closed_at >= now-days`, ordered.
- Build a cumulative realized-P&L series `[{date, cum_pnl}]` + summary
  (total_pnl, win_rate, trades, avg_hold_h, max_drawdown of the cum curve).
- Reuse the exact accounting in `leaderboard.compute_live_strategy_performance`.

**Frontend** in `BotBuilder.tsx`: a compact `LWEquityCurve` sparkline per bot row +
a stat strip (P&L, win-rate, # trades) — matches the Options-Alpha bot card.

---

## 3. Top options strategies — status & priority

Options Alpha's bread-and-butter are **premium-selling / income** structures. Many
already exist as plugins but are **not wired into the live options desk** (which today
runs only the vol-arb set). Priority = expected value × how close it is to shippable.

| Strategy | Market condition | In registry? | On live desk? | Priority |
|---|---|---|---|---|
| **The Wheel** (CSP → CC) | Neutral/bullish, high IV | ✅ `wheel` | ❌ | **P0 — wire in** |
| **Iron Condor** | Range-bound, high IV rank | ✅ `iron_condor` | ❌ | **P0 — wire in** |
| **Put Credit Spread** | Bullish, elevated IV | ✅ `credit_spread_income` | ❌ | **P0 — wire in** |
| **Covered Call** | Mildly bullish, own shares | ✅ `covered_call` | ❌ | **P0 — wire in** |
| **Short Strangle/Straddle** | High IV, mean-revert | ⚠️ via `vrp_systematic` | partial | P1 |
| **VRP / vol-carry short** | IV > realized | ✅ `vrp_systematic`, `vol_carry_short` | ✅ | done |
| **Gamma scalp** | Long-gamma hedging | ✅ `options_gamma_scalp` | ⚠️ | P1 |
| **Dispersion** | Index IV > component IV | ✅ `dispersion_trading` | ✅ | done |
| **Skew arb** | Put/call skew rich | ✅ `skew_arb` | ✅ | done |
| **0DTE / weekly income** | Intraday theta | ❌ | ❌ | P2 (needs 0DTE data + tight risk) |
| **Calendar / diagonal** | Term-structure roll | ⚠️ `vol_term_structure` signal only | ⚠️ | P2 |
| **Broken-wing butterfly** | Directional + defined risk | ❌ | ❌ | P3 |
| **Earnings strangle** | Pre-earnings IV ramp | ❌ | ❌ | P2 (needs earnings calendar, item 14) |

### The P0 action ("add top strategies")
Add `wheel`, `iron_condor`, `credit_spread_income`, `covered_call` to the **Options
desk** `strategy_names` in `.github/scripts/desk_order_placer.py`, and register/enable
them as backend `Strategy` rows so `system-status` stops reporting `options: 0`.

**Caveat (be honest):** the desk currently places **underlying-equity proxy orders**
on Alpaca, so these income structures trade as directional proxies until real
multi-leg routing (TradeStation live, or Tradier/Tastytrade) is connected. Wiring them
in makes them *run and get ranked by the P&L loop*; true options fills are a broker task.

---

## 4. Prioritized backlog (do in this order)
1. **P0 — per-bot P&L graph** (`/bots/{id}/performance` + sparkline) — the requested screenshot parity.
2. **P0 — wire the 4 income strategies into the options desk** + enable as Strategy rows.
3. **P1 — DTE-based + trailing exits** in `check_bot_exits` (options need time-based exits).
4. **P1 — decision recipes** (reusable named condition groups) + populate bot templates.
5. **P1 — Toolbox endpoints**: expected move, prob-of-profit, Greeks.
6. **P2 — earnings-calendar event trigger** + earnings-strangle + 0DTE income.
7. **P2 — real multi-leg execution** via TradeStation live / Tradier (unblocks true options).

## 5. Dependencies / honest blockers
- Real options fills need a live options broker (TradeStation creds or Tradier/Tastytrade).
- Per-bot graphs stay empty until Alpaca keys (GitHub secrets) + a persistent DB exist.
- 0DTE/earnings strategies need an options-chain + earnings data feed the desk doesn't fetch yet.
