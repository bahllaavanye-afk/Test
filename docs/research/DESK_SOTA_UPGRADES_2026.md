# Desk SOTA Upgrades — prioritized, implementable queue (2026-07)

Evidence-backed upgrades per desk, sized for the autonomous improver.
Each item: what, why (citation-class evidence), where, acceptance test.

## Cross-desk (highest impact first)
1. **Volatility-targeted sizing** — scale each order's notional by
   `target_vol / realized_vol20` (cap 2x). Vol targeting is the most robust
   documented Sharpe improvement across asset classes (Moreira-Muir 2017).
   Where: `_kelly_notional()` in desk_order_placer.py. Test: sizing halves
   when HV20 doubles.
2. **Signal ensembling per symbol** — when >=2 same-direction signals on one
   symbol, boost confidence (`1-(1-c1)(1-c2)`), opposite-direction signals
   cancel. Stacked/ensemble signals beat single-strategy entries.
   Where: post-signal-generation aggregation. Test: two 0.55s -> one ~0.80.
3. **Meta-labeling gate (López de Prado)** — train weekly triple-barrier
   classifier on desk signals vs realized outcome; multiply confidence by
   P(win). `triple_barrier_momentum` already has the labeling core.
4. **Execution: replace market-at-open with adaptive limit** — post at
   mid±k*spread, cancel-replace after N min (already "limit-first"; add the
   cancel-replace loop). Cuts slippage vs marketable orders.

## Equities
5. Cross-sectional ranking portfolio (long top-N / short bottom-N of the 10
   symbols by composite momentum+low-vol+reversal score) — market-neutral,
   replaces per-symbol one-shots.
## Crypto
6. Perp-funding harvest via exchange-neutral public funding feeds mirrored
   through a GH-Action proxy (Binance 451-safe). Flat-market income 24/7.
7. Intraday bars (15Min) for avellaneda_stoikov_mm — MM logic on daily bars
   underuses it; Alpaca crypto bars support minute granularity.
## Options
8. Real IV rank from Alpaca options chain snapshots (replace HV proxy);
   delta-targeted strike selection already in alpaca_orders helpers.
9. 0DTE risk guard: block premium-selling entries within 30min of close.
## Macro/FX + StatArb
10. Johansen cointegration refresh weekly for pairs universe (replaces
    static pairs); half-life filter < 20d.
11. Cross-asset TSMOM overlay (Moskowitz-Ooi-Pedersen) sizing tilt.

## Risk
12. Desk-level daily loss cap (halt desk after -2% day) + portfolio
    correlation cap already partially in risk manager — enforce in CI desks.
