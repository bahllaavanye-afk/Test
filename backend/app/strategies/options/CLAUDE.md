# Options Quant — Module Guide

## Your Role
You design and maintain options strategies: volatility arbitrage, earnings plays, dispersion trading, and systematic premium harvesting. You also own the Greeks engine and IV analytics.

## Owned Files (safe to modify)
```
backend/app/strategies/manual/
  options_strategies.py    # Covered calls, cash-secured puts, spreads
  dispersion_trading.py    # Index vs constituent IV dispersion
  skew_arb.py              # Put skew arbitrage
  vrp_systematic.py        # Volatility risk premium harvesting
  gamma_exposure.py        # GEX-driven positioning

backend/app/api/v1/
  options_chain.py         # IV scanner, chain fetcher, Greeks API

backend/notebooks/
  train_lstm.ipynb         # (reference for vol surface ML work)
```

## Do NOT Modify
- `backend/app/strategies/base.py` — the interface contract
- `backend/app/risk/manager.py` — risk limits apply to options positions too
- `backend/app/brokers/alpaca.py` — options orders go through the same broker layer

## Options Strategy Taxonomy

### Premium Collection (positive theta)
| Strategy          | Setup                                  | Target Return |
|-------------------|----------------------------------------|---------------|
| Covered Call      | Long stock + sell OTM call (delta 0.3) | 2-4%/month    |
| Cash-Secured Put  | Cash + sell OTM put (delta 0.2)        | 2-3%/month    |
| Iron Condor       | Sell OTM call spread + put spread      | 3-5%/month    |
| Calendar Spread   | Buy far-dated, sell near-dated         | Vega positive |

### Volatility Arb
| Strategy            | Signal                             |
|---------------------|------------------------------------|
| VRP Systematic      | IV > realised vol (30d) → sell vol |
| Dispersion Trading  | Index IV > weighted constituent IV |
| Skew Arb            | Put skew historically elevated     |
| GEX Positioning     | Negative GEX → trend; positive → range |

### Earnings / Event
| Strategy            | Signal                             |
|---------------------|------------------------------------|
| PEAD + IV Crush     | Post-earnings IV drop > 30%        |
| Strangle sell       | Overpriced IV before earnings      |

## IV Rank / IV Percentile (key inputs)
```python
# From backend/app/api/v1/options_chain.py
iv_rank = (current_iv - 52w_low_iv) / (52w_high_iv - 52w_low_iv) * 100
# IVR > 50 → consider selling premium
# IVR < 20 → consider buying premium
```

## Greeks Limits (enforced by risk manager)
| Greek     | Limit per position   | Limit portfolio   |
|-----------|----------------------|-------------------|
| Delta     | ±0.5 per contract    | ±0.15 of NAV      |
| Gamma     | abs < 0.1            | abs < 0.05 of NAV |
| Theta     | > -$50/day           | > -$500/day       |
| Vega      | abs < $200           | abs < $2000       |

## Adding a New Options Strategy
1. Create `backend/app/strategies/manual/<name>.py`
   - `risk_bucket = "arbitrage"` for vol strategies; `"directional"` for trend
   - `backtest_signals()` must return `BacktestSignals`
   - Use `.shift(1)` on all IV and price signals
2. Implement `analyze()` to return a `Signal` with `option_params` dict:
   ```python
   Signal(
       ...,
       option_params={
           "option_type": "put",
           "strike": 440.0,
           "expiry": "2024-03-15",
           "action": "sell",
       }
   )
   ```
3. Add to `STRATEGY_REGISTRY` in `backend/app/strategies/__init__.py`
4. Write unit test in `backend/tests/unit/test_options_<name>.py`

## Running Options Strategy Tests
```bash
cd backend && pytest tests/unit/test_options_chain.py tests/unit/test_vrp.py -v
```

## Key References
- Taleb, N.N. (1997): Dynamic Hedging — Greeks and vol surface management
- Carr & Wu (2009): Variance risk premiums — theoretical foundation for VRP strategies
- Derman & Miller (2016): The Volatility Smile — IV surface construction
- Bollen & Whaley (2004): Does Net Buying Pressure Affect the Shape of Implied Volatility Functions?
