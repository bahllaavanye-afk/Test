# Risk — Employee Deep Review
**Date:** 2026-08-05  |  **Employee:** `risk_eng` (the Risk Engineer at QuantEdge)  |  **LLM:** nvidia_nim  |  **Grade:** ?

_This review was written by risk_eng (the Risk Engineer at QuantEdge) using nvidia_nim, independently of all other employees' reports._

---

### Critical Issues (P0/P1)

1. **`backend/app/risk/correlation.py:34`** — `union(s_a, s_b)` uses `abs(corr) > threshold` but never resets parent after union, causing incorrect cluster assignments when multiple pairs are processed. Fix: add `parent[find(x)] = find(y)` inside union and ensure path compression is applied after all unions.

2. **`backend/app/risk/circuit_breaker.py:28`** — `confirmation_period` is defined but never implemented in the breach logic. The breaker halts on first breach regardless of `confirmation_period`. Fix: add a counter `consecutive_breaches` in `check_drawdown()` and only halt when `consecutive_breaches >= confirmation_period`.

3. **`backend/app/risk/drawdown_recovery.py:48`** — `estimate_recovery()` is incomplete (truncated code). Missing Monte Carlo simulation logic. This will crash at runtime. Fix: implement full Monte Carlo with 10,000 paths using `np.random.normal(avg_daily_return, std_dev, days)`.

### Performance & Reliability Improvements

1. **`backend/app/risk/correlation.py:12`** — `returns.tail(60)` hardcoded window. Replace with configurable `lookback_window` parameter to adapt to different market regimes (e.g., 20 for high-frequency, 120 for long-term).

2. **`backend/app/risk/correlation_monitor.py:45`** — Missing `asyncio.sleep()` in monitoring loop. Add `await asyncio.sleep(300)` to prevent CPU spin on every iteration.

### Alpha / Signal Quality

1. **`backend/app/risk/correlation.py:20`** — Using absolute correlation threshold ignores negative correlations that can be equally dangerous in tail events. Add `abs(corr) > threshold` is correct but should also flag `corr < -0.70` as inverse clusters.

### Security & Safety

1. **`backend/app/risk/circuit_breaker.py:15`** — No input validation on `max_drawdown_pct`. A value of `0.0` or negative would never trigger. Add `assert 0 < max_drawdown_pct <= 1.0`.

### Implementation Priority Queue

1. Fix circuit breaker `confirmation_period` logic (P0, high impact)
2. Complete `estimate_recovery()` Monte Carlo implementation (P0, crash risk)
3. Fix correlation cluster union-find path compression (P1, accuracy)
4. Add configurable lookback window to correlation clustering (P2, adaptability)
5. Add input validation on `max_drawdown_pct` (P2, safety)

### Overall Grade
**C** — Core risk functions have critical bugs (unimplemented confirmation logic, incomplete Monte Carlo) that could cause false halts or crashes. Correlation clustering is functional but lacks path compression. Needs immediate fixes before production use.