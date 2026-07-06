# Options Alpha Bot Fleet — Performance Analysis & Testing-at-Scale (2026-07)

_Research doc. Sources: the operator's live OA account (2 screenshots, 2026-07-06,
"10 of 49 active bots"), Option Alpha's public template library, and our own
clone fleet (15 templates in `app/bots/templates.py`, tag `oa_*`)._

## 1. The observed fleet and its live results (30D window)

| Bot | 30D P&L | Return | Archetype |
|---|---|---|---|
| CCS FRIDAY 15D | $1,382 | +27% | call credit spread, time-boxed |
| NCR_0DTE_SPX (A) | $1,330 | +26.6% | 0DTE index iron condor |
| Friday 14 DTE BWB | $845 | +33.8% | broken-wing put butterfly |
| IB @9:45 | $685 | +27.4% | timed ATM iron butterfly |
| Trendy Short Put | $425 | +17% | trend-gated put credit spread |
| NCR_SPX_0DTE (B) | $205/$140 | +13%/+5.6% | 0DTE condor, tighter deltas |
| Vix range | $185 | +7.4% | regime-gated condor |
| Steamrolled | $77 | +5% | laddered put spreads |
| Iron butter Clone / Short Put Spread | $25–$50 | +2.5% | newer, few closes |
| Delta Adjusting Strangle | **–$37** | –2.5% | undefined-risk short strangle |
| Long Calls / TD Trend / new bots | — | — | directional or too new |

## 2. Why did *these* bots win? The pattern is unambiguous

Every outperformer shares four properties:

1. **Short premium on index underlyings (SPX/SPY).** They harvest the variance
   risk premium — index options are persistently overpriced because the world
   pays for crash protection. This is the best-documented edge in options.
2. **Defined risk.** Condors/butterflies/spreads — the long wings cap the tail.
   The one *undefined*-risk bot (Delta Adjusting Strangle) is the one **loser**.
3. **Time-boxed, mechanical entries** (9:45 ET, Fridays, fixed windows).
   Option Alpha's own research shows entry time alone produced a **$15,000 P/L
   difference** in backtested 0DTE SPX condors — consistent sampling beats
   discretion.
4. **High-probability short strikes (10–30Δ) + profit-taking at 25–50%.**
   Many small wins, step-function equity curves (visible in every sparkline).
   OA's published 0DTE iron-butterfly study reports a **95.4% win rate** with
   half the trades closed before noon.

**The honest caveats (do not skip):**
- **Selection bias:** we see the top 10 *of 49 active bots*, sorted by return.
  The fleet's aggregate (+12.1%, $2,259) is far below the visible winners —
  most of the 49 are mediocre. Judge the *fleet*, not the leaders.
- **Regime dependence:** a 30-day calm/grind-up window is the best possible
  weather for short premium. These curves price **no vol event**. The BWB's
  +33.8% and the 0DTE condors' +26% are regime returns, not expectancy. The
  Vix-range bot's mid-month drawdown spike shows exactly what one vol pop does.
- **Directional bots underperform in the same window** (Long Calls, TD Trend
  flat) — not because they're bad, but because chop kills momentum entries.
  Pattern: **theta wins calm months; gamma wins violent ones.** A real fleet
  holds both and sizes by regime (our regime gate + perf-weighted sizing).

## 3. Can we test *every possible* bot template on Options Alpha? No — but we can here

**On OA itself: no.** There is no public API for programmatic bot creation or
backtesting; plans cap active bots (the account shows "1 left in your plan" at
49); backtests are manual per-bot. Exhaustive search on their platform is
structurally impossible.

**On QuantEdge: yes, two ways —**
1. **Grid generation (built: `app/bots/factory.py`).** The template space that
   matters is small and enumerable: structure {condor, butterfly, BWB, put
   spread, call spread} × short-delta {0.10, 0.16, 0.20, 0.30} × DTE
   {0, 7, 14, 30} × profit-target {25, 50}. The factory generates bounded
   variant sets programmatically; every variant is a normal Bot the engine runs.
2. **Evolutionary paper selection (already live).** Variants trade paper at
   small size → desk→Trades ingestion records real results → `/leaderboard/live`
   ranks them → the **lifecycle manager disables losers and keeps winners** →
   perf-weighted sizing scales survivors. This is survival-of-the-fittest over
   template space, with real (paper) fills instead of backtest overfitting —
   and it runs 24/7 without a human.
3. **Synthetic options backtester (queued):** historical options chains are the
   missing dataset; until then a Black-Scholes synthetic pricer over underlying
   OHLCV + realized vol gives approximate backtests for premium structures.
   Filed in IMPROVEMENTS.md for the autonomous workers.

## 4. TradeStation alternatives for real options fills

| Broker | API | Paper | Multi-leg | Verdict |
|---|---|---|---|---|
| **Alpaca (options)** | ✅ same API we already use | ✅ | ✅ (multi-leg orders supported on paper) | **Best: zero new integration — our keys already work. Enable options on the paper account and the existing broker layer extends.** |
| Tradier | ✅ clean REST, sandbox | ✅ | ✅ | Strong #2; what OA itself uses for autotrading |
| Tastytrade | ✅ (open API) | ✅ cert env | ✅ | Options-first broker, good fills |
| IBKR | ✅ (heavy) | ✅ | ✅ | Most capable, most complex |
| Webull/Robinhood | ❌ no real public API | — | — | Not viable |

**Action queued:** extend `brokers/alpaca_orders.py` with the options order
shape (legs array) — Alpaca paper supports multi-leg; that makes every `oa_*`
bot fill real option legs with **no new broker and no new keys**.

## 5. Automating the remaining Options Alpha features (status)
- ✅ Bots, templates, triggers, conditions, TP/SL/time exits, paper-first,
  per-bot performance endpoint, lifecycle (create/disable/promote), 15 clones
- ✅ NEW: variant factory (this commit) — employees generate/scale bot variants
- 🔜 queued for the autonomous workers: bot-level synthetic backtest before
  enable, decision recipes, SmartPricing (mid→walk), frontend sparkline,
  Alpaca multi-leg fills
