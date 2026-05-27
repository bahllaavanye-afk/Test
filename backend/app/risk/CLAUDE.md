# Risk Engineer — Module Guide

## Your Role
You own the risk management layer. Your job is to prevent catastrophic losses while maximising the capital deployed in high-conviction strategies. Every order in QuantEdge passes through the risk manager before it reaches a broker.

## Owned Files (safe to modify)
```
backend/app/risk/
  manager.py          # Central gatekeeper — RiskManager.check_order()
  kelly.py            # Kelly criterion + fractional Kelly sizing
  circuit_breaker.py  # Drawdown halts per bucket + global halt
  correlation.py      # Position correlation limits
  factor_exposure.py  # Beta, sector, factor exposure caps
  var.py              # Value-at-Risk engine (parametric + historical)
  correlation_monitor.py
  drawdown_recovery.py
```

## Do NOT Modify
- `backend/app/strategies/base.py` — changing the interface breaks all 38 strategies
- `backend/app/brokers/*.py` — broker-level safety is a separate concern
- Any DB migration file unless you have created a new column

## Risk Architecture

```
Order Request
     │
     ▼
RiskManager.check_order()
     ├─ kelly.py          → compute max position size
     ├─ circuit_breaker   → halt if daily/bucket drawdown exceeded
     ├─ correlation.py    → block if corr to existing positions > 0.85
     ├─ factor_exposure   → block if beta to SPY > 1.5
     └─ var.py            → block if 1-day 99% VaR > 2% of NAV
          │
          ▼ (approved)
     Execution Layer
```

## Capital Allocation Rules (ENFORCED, do not change limits without sign-off)
| Bucket         | Capital % | Max Single Position | Max Drawdown |
|----------------|-----------|---------------------|--------------|
| arbitrage      | 70%       | 5%                  | 8%           |
| directional    | 30%       | 3%                  | 12%          |
| global         | 100%      | —                   | 20%          |

## Kelly Sizing Formula
```python
# backend/app/risk/kelly.py
# Full Kelly: f* = (p * b - q) / b  where b = win/loss ratio
# We use 25% fractional Kelly for safety.
KELLY_FRACTION = 0.25
MAX_KELLY_POSITION = 0.05   # never exceed 5% of NAV from Kelly
```

## Adding a New Risk Rule
1. Add method `_check_<rule_name>(self, order, portfolio) -> bool` to `manager.py`
2. Call it inside `check_order()` before the approval return
3. Add a unit test to `tests/unit/test_risk_engine.py`
4. Document the academic/empirical basis in a comment above the check

## Key Principles
- Every check must be O(1) or O(n positions) — never O(n²)
- Circuit breakers reset at 00:01 UTC daily (scheduler.py calls `manager.reset_daily()`)
- Risk limits are hardcoded; they cannot be relaxed via API calls (by design)
- Paper mode uses the same risk engine as live — never bypass for paper

## Running Risk Tests
```bash
cd backend && pytest tests/unit/test_risk_engine.py tests/unit/test_kelly.py -v
```

## Relevant Papers
- Kelly (1956): A New Interpretation of Information Rate → `kelly.py`
- Avellaneda & Jeanblanc (2013): Portfolio VaR decomposition → `var.py`
- Jegadeesh & Titman (1993): Correlation clustering → `correlation.py`
