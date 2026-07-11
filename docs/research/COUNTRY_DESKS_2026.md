# Country Desks — research + rollout (2026-07)

## Tradable TODAY (US-listed ETFs on Alpaca, no new broker/data)
Country momentum & cross-country rotation are documented premia (Asness-
Moskowitz-Pedersen "Value and Momentum Everywhere"; country TSMOM in
Moskowitz-Ooi-Pedersen). Universe (liquid, 1990s-2000s inception):
- Asia: EWJ (Japan), FXI/MCHI (China), EWY (Korea), EWT (Taiwan), INDA/EPI/SMIN (India — already live)
- Americas: EWZ (Brazil), EWW (Mexico), EWC (Canada)
- Europe: EWU (UK), EWG (Germany), EWQ (France), EWL (Switzerland)
- Frontier/other: EZA (S.Africa), EIDO (Indonesia), VNM (Vietnam)
Strategies that transfer as-is: cross_sectional_momentum (rank countries,
long winners), time_series_momentum, mean_reversion (country reversals),
low_volatility, sector_rotation. All already in the registry and desk-tested.

## Deeper work (queued)
1. FX majors desk via OANDA practice API (keys already relayed) — carry
   (interest_rate_differential), fx_trend, fx_reversion are written but idle.
2. Timezone-aware market-hours per desk — the US equity clock gates all
   non-crypto desks; country ETFs trade US hours so OK today, but direct
   int'l venues (NSE, LSE) need per-desk calendars.
3. India direct (NSE) — largest derivatives venue globally; needs a broker
   (Zerodha Kite/Dhan) — cost/KYC decision for the user.
4. Japan/Korea single names via ADRs (TM, SONY, etc.) — momentum universe
   expansion with zero new infra.
5. Country risk overlay — cap aggregate EM exposure at 30% of desk notional.
