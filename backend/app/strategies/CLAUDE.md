# Strategy Agent Guide

## Your Role
You add and improve trading strategies. Each strategy has a **manual** version (indicators only) and an **ML-enhanced** version (same logic + ML filter).

## How to Add a New Strategy

### 1. Create the manual version
```python
# backend/app/strategies/manual/my_strategy.py
from app.strategies.base import AbstractStrategy, Signal
import pandas as pd

class MyStrategy(AbstractStrategy):
    name = "my_strategy"
    market_type = "equity"      # equity | crypto | polymarket
    strategy_type = "manual"
    risk_bucket = "directional" # directional | arbitrage
    tick_interval_seconds = 3600
    confidence_threshold = 0.60

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        # Your signal logic here
        # Return Signal(...) or None
        pass

    def backtest_signals(self, df: pd.DataFrame):
        # Return pd.Series of -1/0/1 with shift(1) to prevent lookahead
        return signals.shift(1)
```

### 2. Register it
Edit `backend/app/strategies/__init__.py` and add:
```python
from app.strategies.manual.my_strategy import MyStrategy
STRATEGY_REGISTRY["my_strategy"] = MyStrategy
```

### 3. Backtest it
```bash
./scripts/backtest.sh my_strategy SPY 1d 2021-01-01 2024-01-01
```

### 4. If Sharpe > 1.0 on out-of-sample, create ML version
Copy to `backend/app/strategies/ml_enhanced/ml_my_strategy.py`, add ML filter.

## Rules
- **ALWAYS** use `.shift(1)` in `backtest_signals()` — no lookahead
- **NEVER** modify `strategies/base.py` — interface change breaks everything
- **NEVER** modify `risk/manager.py` — risk rules are safety-critical

## Files Safe to Modify
- `strategies/manual/*.py`
- `strategies/ml_enhanced/*.py`
- `strategies/__init__.py` (to register)
