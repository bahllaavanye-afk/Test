# Commodities Desk — Deep Analysis (2026-06-30)

> Why the Commodities desk is ranking low, and the concrete fix shipped this session.
> Diagnosis is grounded in the actual desk code, not vibes.

## TL;DR
The Commodities desk is weak for two structural reasons, both fixable:

1. **No edge diversity.** Until this session the desk had exactly **two** strategies —
   `commodity_momentum` and `commodity_trend` — and they are the *same factor*:
   **long-only trend-following.** In the range-bound, mean-reverting regimes that
   dominate commodity tapes between trends, both sit flat or get whipsawed, so the
   desk's blended Sharpe and trade count collapse. This session adds the missing
   **two-sided mean-reversion** complement (`commodity_reversion`).
2. **Fragile, single-source data.** Commodity prices come *only* from yfinance
   continuous futures (`GC=F`, `CL=F`, …). There is no Alpaca/Binance commodity
   coverage, and in this dev container yfinance is TLS-broken → it silently falls
   back to **synthetic** prices. A desk backtested on noise cannot rank well. This
   is the #1 reliability fix still open (see Recommendations).

---

## 1. What the desk actually is (code-grounded)

Registry (`backend/app/strategies/__init__.py`) → `_MARKET_TYPE_DESK["commodity"] = "Commodities"`.

| Strategy | File | Edge | Side | Verdict |
|---|---|---|---|---|
| `commodity_momentum` | `manual/commodity_momentum.py` | 60-bar time-series momentum (long while trailing return > 0) | **long-only** | Classic managed-futures TSMOM, but only half of it (no short leg). |
| `commodity_trend` | `manual/commodity_trend.py` | Fast/slow SMA cross + Donchian breakout | **long-only** | Same directional/trend factor as above → highly correlated. |
| `commodity_reversion` *(new)* | `manual/commodity_reversion.py` | Z-score fade vs rolling mean | **two-sided** | The missing decorrelated edge. |

**The problem in one sentence:** two long-only trend-followers are ~the same bet, so
the desk had no way to (a) profit from commodity *downtrends* or (b) harvest the
mean-reversion that occurs the rest of the time. The engine already supports shorts
(`BacktestSignals.short_entries / short_exits`) — the desk just wasn't using them.

## 2. Live market view (why two-sided matters right now)

As of late June 2026 commodities are in a **high-volatility, geopolitically-driven,
two-sided** regime — relief rallies that fade into well-defined ceilings — the exact
tape where long-only trend gets chopped:

- **Gold** ~$4,050/oz — peaked above $5,500 in January, fell ~15% in March, held above
  ~$4,400 on central-bank/ETF demand, now pulling back. Round-trips, not a clean trend.
- **Silver** ~ $67 (testing resistance).
- **Crude** — Brent ~$95, WTI capped at $90.50–$92.50 resistance; swinging on
  US–Iran / Strait of Hormuz headlines and peace-deal expectations.
- **Copper** < $6/lb, near 7-week lows on a stronger USD / Fed-hike expectations.
- **Natural gas** ~$3.35/MMBtu — highest since early February, bid on LNG export flows
  and warm weather (the one level independently confirmable from this environment).

A z-score fade is built for exactly this: long the oversold flushes, short the
exhausted relief rallies, flatten back to the mean.

## 3. The fix shipped this session — `commodity_reversion`

`backend/app/strategies/manual/commodity_reversion.py`, registered as
`commodity_reversion` on the Commodities desk.

- **Signal:** z-score of close vs a rolling mean/std (`window=20`). Long when
  `z ≤ -entry_z` (oversold), short when `z ≥ +entry_z` (overbought, default `entry_z=2.0`),
  flatten as `|z|` reverts inside `exit_z=0.5`.
- **Two-sided:** populates `short_entries`/`short_exits` — the first commodity strategy
  on the desk that can be net short.
- **Causal:** `backtest_signals` shifts the z-score by one bar (decide at *t* from *t-1*).
  Verified against the registry-wide causality guard (truncation invariance) and the
  per-strategy contract guards: **431 passed** in `test_all_strategies_contract.py`.
- **Decorrelated:** on a random-walk smoke path it fired 19 long + 10 short entries while
  the trend strategies sat flat — different exposure, by construction.

This takes the desk from 2 → 3 strategies and, more importantly, from **1 factor → 2
factors** (trend + reversion), which is what actually lifts a desk's blended Sharpe.

## 4. Recommendations (priority order)

- **[P0] Wire a real commodity data feed.** This is the single biggest lever. yfinance
  continuous futures is one fragile, rate-limited/TLS-fragile source that degrades to
  synthetic silently. Options: a keyed EOD vendor (Stooq/Nasdaq Data Link/Tiingo) or
  Alpaca's metal/commodity-tracking ETFs (GLD/SLV/USO/CPER/UNG) as liquid proxies that
  *do* come over the working Alpaca pipe. Until this lands, every commodity backtest
  number should be treated as unverified.
- **[P1] Add term-structure / roll-yield carry.** The largest *real* commodity alpha is
  contango/backwardation carry, which the desk has zero exposure to. Needs front/second
  contract or an ETF pair (e.g., USO vs USL) to proxy the curve.
- **[P1] Cross-commodity relative value.** Gold/oil and gold/silver ratio reversion are
  market-neutral-ish and decorrelate further from the directional book. Requires the
  engine to feed two symbols to one strategy (small contract extension).
- **[P2] Two-sided TSMOM.** Give `commodity_momentum` a short leg (short when trailing
  return < 0) so the desk's trend factor is also symmetric.
- **[P2] Regime filter.** Gate reversion vs trend on ADX so they don't fight each other
  in strong trends.

---

### Data-integrity caveat
Live commodity prices are **not reachable** from this execution environment (yfinance
TLS-broken here; Alpaca/Binance don't cover commodities). The market levels in §2 come
from web sources, not the platform's own feed — which is itself the strongest evidence
for the P0 data recommendation.

**Sources:**
- [TradingEconomics — Gold](https://tradingeconomics.com/commodity/gold)
- [TradingEconomics — Copper](https://tradingeconomics.com/commodity/copper)
- [S&P Global — Copper & Gold Market Outlook 2026](https://www.spglobal.com/market-intelligence/en/news-insights/research/2026/04/copper-gold-market-outlook-2026-prices-supply-mining-costs)
- [TD Securities — Oil & Precious Metals Projections](https://www.tdsecurities.com/ca/en/commodities-projections-adjusted-post-conflict)
- [FXEmpire — Live Commodity Prices](https://www.fxempire.com/commodities)
- [Barchart — WTI Crude Futures](https://www.barchart.com/futures/quotes/WSM26)
