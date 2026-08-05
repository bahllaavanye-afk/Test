# Strategies — Employee Deep Review
**Date:** 2026-08-05  |  **Employee:** `alpha_dir` (the Alpha Research Director leading the equities desk at QuantEdge (momentum, mean-reversion, pairs/Kalman, breakout, idio-vol, ML directional))  |  **LLM:** nvidia_nim  |  **Grade:** ?

_This review was written by alpha_dir (the Alpha Research Director leading the equities desk at QuantEdge (momentum, mean-reversion, pairs/Kalman, breakout, idio-vol, ML directional)) using nvidia_nim, independently of all other employees' reports._

---

### Critical Issues (P0/P1)

1. **`backend/app/strategies/_failsoft.py:30-35`** — `apply_hard_budget()` uses `threading.Thread(daemon=True)` but never joins or tracks the thread. If `analyze()` holds a GIL-blocking C extension (e.g., numpy/pandas), the daemon thread can outlive the main process, causing zombie threads and memory leaks. **Fix**: Add `thread.join(timeout=0)` in a `finally` block after the budget expires, or use `concurrent.futures.ThreadPoolExecutor` with explicit shutdown.

2. **`backend/app/strategies/base.py:45-50`** — `BacktestSignals` uses `pd.Series` for entries/exits but no validation that indices are aligned. If a strategy returns misaligned timestamps (e.g., daily vs. intraday), the backtest engine silently computes wrong PnL. **Fix**: Add `@dataclass` validator to check `entries.index.equals(exits.index)`.

### Performance & Reliability Improvements

1. **`backend/app/strategies/__init__.py:3-20`** — All 16 strategies are imported eagerly on module load. This adds ~200ms startup time and wastes memory. **Fix**: Use lazy imports inside `analyze()` or a factory function.

2. **`backend/app/strategies/_failsoft.py:25`** — `STRATEGY_ANALYZE_BUDGET_S` default is 3.5s, but yfinance fetches can take 10s+ on congested networks. **Fix**: Increase default to 8.0s and add exponential backoff retry logic.

### Alpha / Signal Quality

1. **`backend/app/strategies/base.py:30-35`** — `Signal.confidence` is a float 0.0-1.0 but no strategy defines how it's calculated. Without a consistent calibration (e.g., z-score of recent win rate), confidence is meaningless. **Fix**: Add a `calibrate_confidence()` method that maps historical Sharpe to confidence.

2. **`backend/app/strategies/__init__.py`** — No cross-symbol interaction checks. If `MomentumStrategy` and `MeanReversionStrategy` both fire on the same symbol, the desk gets conflicting signals. **Fix**: Add a `SignalConflictResolver` that merges or suppresses overlapping signals.

### Security & Safety

1. **`backend/app/strategies/_failsoft.py:28`** — `os.environ` is read for `STRATEGY_ANALYZE_BUDGET_S` without validation. An attacker who sets this to `0` or negative could cause immediate timeout of all strategies. **Fix**: Add `max(1.0, float(os.getenv(...)))` guard.

### Implementation Priority Queue

1. **Fix zombie threads in `_failsoft.py`** — P0, high impact (prevents memory leaks in production)
2. **Add `BacktestSignals` index validation** — P1, medium impact (catches silent PnL errors)
3. **Lazy import strategies** — P2, low impact (reduces startup time)
4. **Calibrate confidence calculation** — P2, medium impact (improves signal quality)
5. **Add cross-symbol conflict resolver** — P3, low impact (prevents contradictory orders)

### Overall Grade
**C+** — Core signal infrastructure is sound but has critical thread-safety bugs and missing validation that could cause silent failures in production. Immediate fixes needed for P0 items.