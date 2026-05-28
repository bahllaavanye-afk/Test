# Testing Guide

## Test Suite Structure

```
backend/tests/
├── conftest.py              # Shared fixtures (DB, HTTP client)
├── unit/                    # Pure unit tests (no DB, no network)
│   ├── test_kelly.py
│   ├── test_circuit_breaker.py
│   ├── test_correlation.py
│   ├── test_features.py
│   ├── test_strategies.py
│   ├── test_execution.py
│   ├── test_slippage.py
│   ├── test_security.py
│   ├── test_walk_forward.py
│   ├── test_monte_carlo.py
│   ├── test_algo_agent.py
│   ├── test_smart_router.py
│   ├── test_benchmarks.py
│   ├── test_risk.py
│   └── test_backtest.py
└── integration/             # API + DB tests (SQLite in-memory)
    ├── test_api_health.py
    └── test_websocket.py
```

## Running Tests

```bash
# All tests
cd backend && pytest

# Verbose, stop on first failure
cd backend && pytest -x -v

# Just unit tests (fast, no DB)
cd backend && pytest tests/unit/ -v

# Coverage report
cd backend && pytest --cov=app --cov-report=html
open htmlcov/index.html

# Single file
cd backend && pytest tests/unit/test_kelly.py -v

# Filter by name
cd backend && pytest -k "test_kelly_fraction"
```

## Test Configuration

`backend/pytest.ini`:
```ini
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
```

`asyncio_mode = auto` means all `async def` tests are auto-decorated with `@pytest.mark.asyncio`.

## Fixtures

`tests/conftest.py` provides:
- `test_db` — SQLite in-memory async engine, session-scoped
- `client` — `httpx.AsyncClient` against the FastAPI app with overridden DB dependency

## Adding a Test

1. Pick the right directory (`unit/` if no DB/network, `integration/` otherwise)
2. Name the file `test_<module>.py`
3. Name test functions `test_<behavior>()`
4. Use `pytest.mark.asyncio` only if NOT using auto mode

Example:
```python
# tests/unit/test_my_module.py
import pytest
from app.my_module import my_function

def test_my_function_basic():
    assert my_function(1, 2) == 3
```

## What's Tested

### Risk
- Kelly fraction with various win rates / loss ratios
- Kelly position sizing capped at max_pct
- Circuit breaker trip / no-trip / reset
- Correlation cluster detection (correlated vs independent assets)
- Cluster allocation limits

### Strategies
- All registered strategies have required attributes
- `backtest_signals()` returns valid -1/0/1 series

### Features
- No lookahead bias (lag-0 correlation < lag-1 correlation with future returns)
- Sequence shape correctness for LSTM input
- Binary label assignment

### Execution
- TWAP slices order correctly
- LimitFirst posts limit first, fills, falls back to market on timeout
- Smart router algorithm selection
- Slippage tracker records signal_price vs fill_price

### Backtesting
- Buy-and-hold reference Sharpe
- Walk-forward window aggregation
- Monte Carlo confidence intervals

### Security
- bcrypt password hashing roundtrip
- JWT access/refresh token roundtrip
- Fernet encrypt/decrypt with different IV each time

### API
- `/health` returns ok
- Auth register → login flow
- WebSocket connection manager broadcasts

## Continuous Integration

GitHub Actions config (`.github/workflows/test.yml` — create if needed):
```yaml
name: test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install uv && cd backend && uv pip install --system -e .
      - run: cd backend && pytest tests/unit/ -v
```

## Coverage Goals

| Module | Target | Current |
|--------|--------|---------|
| app/risk/ | 90% | check via `pytest --cov` |
| app/backtest/ | 80% | |
| app/strategies/ | 70% (signals + interface) | |
| app/execution/ | 80% | |
| app/ml/features/ | 70% (lookahead detection critical) | |

## What's NOT Tested (and why)

- **Live broker calls** — would require API keys, paid in some cases, and is non-deterministic. Use paper trading for integration.
- **Real ML training** — too slow for CI. Test feature engineering + dataloader shapes instead.
- **WebSocket fan-out under load** — separate load test, not in unit suite.
