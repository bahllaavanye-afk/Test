# Option Alpha — Full Docs Study (2026-07-19)

Studied **all 100 doc pages** at `docs.optionalpha.com` (they publish `llms.txt` + raw `.md`
for every page — fully public, no login). This is the durable reference for building QuantEdge
to OA fidelity. Machine-readable setting inventory lives next to it in
`.github/state/oa_settings_catalog.json` (consumed by the bot importer + template schema).

## Answering the two questions the user asked

**"Can all features / bots be copied the exact same way?"**
Yes for the *rules and settings* — every OA bot is a set of automations (scanners/monitors/events)
made of decision recipes over a security object, with safeguards, SmartPricing, Exit Options, loops,
tags and inputs. All of that is enumerated in the catalog and captured either as native
`BOT_TEMPLATES` fields or verbatim under a bot's `oa_meta` key. What can't be "copied" is OA's
**data** (Tradier/ORATS real chains + greeks) and their **historical options backtester** — those
need a data source (Tradier sandbox token is the cheapest unlock; see `OA_BACKEND_STACK_2026.md`).

**"Need lots of vizz, dashboard + graph in bot rows."**
Shipped this session (see below). The rest of OA's UI is captured as a spec-backed roadmap in
`IMPROVEMENTS.md` ("OA UI/UX parity roadmap").

## What OA actually is (architecture)
- **Bot** = allocation + safeguards + one or more **automations**.
- **Automation** = an ordered tree of **actions** (decision, open, close, notify, tag, loop) on a
  **schedule**. Execution order across a bot is always: Exit Options → Scheduled Events → Monitors → Scanners.
- **Decision recipes** evaluate a JSON **security object** (real-time price/greeks/IV-rank) as a
  binary Yes/No tree; group with AND/OR, negate with NOT.
- **Scanners** open positions (stop at position limits); **Monitors** manage/close them.
- Everything is checked at **intervals** (15/5/1-min) — the bot does not see between-interval spikes
  except Smart-Stop high-water tracking (every minute).

## Settings inventory (summary — full detail in the catalog JSON)
| Area | Key settings |
|---|---|
| Safeguards | allocation (required), daily-position-limit (10), max-position-limit (10), risk-defined-only |
| Scan speed | 15m / 5m / 1m, window 9:31–3:59 ET |
| Triggers | scanner, monitor, date, repeating, market-open (9:40), market-close (3:50), position-opened/closed, webhook, button |
| SmartPricing | normal (4×10s) / fast (3×5s) / patient (5×20s) / off / market; final price = % of bid-ask + $ slippage-from-mid; math operators in Close; SPX nickel rounding |
| Exit Options | profit-target, price-target, stop-loss, trailing-stop, **touch (ITM)**, expiration, earnings; + bid/ask guard, PDT box, presets; checked every 1 min, 2-min order TTL |
| **Options Expiration Protocol (the "3 overrides")** | (1) calc P/L from underlying close [default, no order] · (2) close with market order · (3) override + manual entry (auto-override 3:50pm) |
| Loops | position (oldest→newest), symbol, bot-symbol (watchlist) |
| Tags | bot / position / symbol; tag / untag / reset; used in decisions |
| Inputs | decision → automation → bot hierarchy; bot-input wins; defaults only when link broken; NOT updated on upgrade |

## P&L / metrics math (so our numbers match OA)
- **Valuation**: whole-position **mid price**, not per-leg last; **excludes** commissions/fees.
- **Total P/L** = open + closed. **Return %** = P/L / allocation. **Day P/L** = intraday total − prior close.
- **Win Rate** = wins / closed. **Profit Factor** = gross win $ / gross loss $.
- **Max Drawdown** = greatest peak→trough (anchored at 0 if never negative).
- **Avg P/L / Avg Win / Avg Loss** = per-trade means. EOD settle 4:15pm ET.
- **Analyze** adds Sharpe / Sortino / Return-on-Risk / Entry POP / DTE / Days-in-Trade + Day-of-Week /
  Hour-of-Day / By-Strategy / By-Symbol breakdowns + Hindsight (hold-to-expiration) report.

## Backtester (for "scale to more symbols + better backtesting")
- 1-minute historical, up to 3-year window, compare 4 / combine into one curve.
- Entry time in 5-min increments or custom 9:35–3:55 ET; entry filters (change% vs prev close, IV rank,
  indicators, min/max position criteria).
- Stats: Total/High/Low P/L, Max Risk, Max Drawdown, RDD%, Profit Factor, Count, Win Rate, Best Win,
  Worst Loss, Win/Loss Streak, Max Profit/Loss %.
- **Automate**: any backtest → generate a bot carrying all its settings/filters (our backtest→bot item).

## Order handling & failsafes (correctness guards worth mirroring)
- Position states: open / opening / closed / closing (released to broker mid-transition).
- Partial fills: timed limit orders, 2-min timeout resets per partial; leftover canceled → stays open at partial qty.
- Bots are unaware of assignment — must manually override; BTC/STC errors if position absent.
- **Leg enforcement**: credit spread short-delta > long-delta; iron condor = 4 strikes; iron butterfly = equal short strikes.
- **Excessive-errors failsafe**: automations auto-disable after too many errors (warnings like
  not-enough-capital / pricing-anomaly / position-limit don't count).

## Shipped this session (viz)
- **Backend** `/bots/{id}/performance`: added total_pnl_pct, day_pnl, change_pnl/pct, wins/losses,
  profit_factor, avg_win/avg_loss/avg_pnl, high/low_pnl, streak(+kind), Sharpe/Sortino, capital block
  (allocation/net_liquid/at_risk/available/maintenance), and weekday/hour/symbol breakdowns.
- **Frontend** `BotBuilder.tsx`: OA bots-list layout (30D sparkline column + P/L columns + aggregate
  cards) and OA per-bot dashboard (filled equity curve + Position Stats grid + Capital sidebar).

## Not copyable without a data unlock
- Real options chains + ORATS greeks/IV → **Tradier sandbox token** (free, cheapest unlock).
- Historical-options backtester → needs 1-min historical options data (paid); our `options_synthetic`
  (Black-Scholes) is the honest stand-in until then.
