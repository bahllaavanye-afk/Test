# Tasks Scheduler — Employee Deep Review
**Date:** 2026-08-05  |  **Employee:** `devops_dir` (the Director of DevOps at QuantEdge)  |  **LLM:** nvidia_nim  |  **Grade:** ?

_This review was written by devops_dir (the Director of DevOps at QuantEdge) using nvidia_nim, independently of all other employees' reports._

---

### Critical Issues (P0/P1)

1. **backend/app/tasks/agent_bus.py:1-30** — `DISPATC` is truncated mid-function name, likely `DISPATCHER`. This breaks the entire agent bus initialization. **Fix**: Complete the function definition or remove the incomplete line. **Impact**: All event-driven agent communication fails silently.

2. **backend/app/tasks/agent_memory.py:38** — `await self._r.lpush(key, payload)` without `MAX_LIST_LEN` enforcement. The docstring claims a cap of 500, but no trim logic exists. **Fix**: Add `await self._r.ltrim(key, 0, _MAX_LIST_LEN - 1)` after lpush. **Impact**: Unbounded Redis memory growth, eventual OOM.

3. **backend/app/tasks/algo_agent.py:45** — `EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)` runs at import time, not async-safe. If multiple workers import simultaneously, race condition on directory creation. **Fix**: Move to `async def start()` with `await asyncio.to_thread(pathlib.Path.mkdir)`. **Impact**: File system corruption under concurrent deployment.

### Performance & Reliability Improvements

1. **backend/app/tasks/agent_memory.py:38** — Replace `lpush` with `rpush` + `ltrim` for O(1) writes. Current `lpush` requires O(n) for trim. **Fix**: Use `rpush` then `ltrim(key, -_MAX_LIST_LEN, -1)`.

2. **backend/app/tasks/ai_strategy_generator.py:15** — `STAGING_DIR` uses `Path(__file__).parent.parent` which breaks in Docker containers with symlinked paths. **Fix**: Use `os.path.realpath(__file__)` or `Path(__file__).resolve()`.

3. **backend/app/tasks/algo_agent.py:50** — `@property avg_sharpe` divides by `n_runs` without zero check. **Fix**: Add `if self.n_runs == 0: return 0.0`.

### Alpha / Signal Quality

1. **backend/app/tasks/algo_agent.py:55** — UCB1 uses hardcoded `c=1.414`. In bull market, this over-explores. **Fix**: Make `c` dynamic based on market regime (e.g., `c=0.5` in bull, `c=2.0` in bear).

2. **backend/app/tasks/ai_strategy_generator.py:20** — No validation that generated strategies pass basic sanity checks (e.g., Sharpe > 0, max drawdown < 50%). **Fix**: Add `validate_proposal()` method before writing to staging.

3. **backend/app/tasks/agent_memory.py:25** — `write()` overwrites existing data with `**data` spread. If two agents write same topic simultaneously, data loss. **Fix**: Use Redis `HSET` for atomic field updates.

### Security & Safety

1. **backend/app/tasks/ai_strategy_generator.py:25** — `call_consensus()` sends strategy parameters to external LLM APIs. No sanitization of output before writing to filesystem. **Fix**: Add `json.loads()` validation and reject any non-JSON output.

2. **backend/app/tasks/algo_agent.py:30** — `EXPERIMENTS_DIR` is world-writable. If compromised, attacker can inject malicious strategy files. **Fix**: Set `mode=0o750` and verify ownership.

### Implementation Priority Queue

1. **Fix agent_bus.py truncated function** — P0, blocks all event-driven architecture. Impact: Critical.
2. **Add Redis ltrim after lpush** — P0, prevents memory leak. Impact: High.
3. **Make UCB1 exploration dynamic** — P1, improves alpha in current bull market. Impact: Medium.
4. **Add strategy proposal validation** — P1, prevents garbage strategies. Impact: Medium.
5. **Secure EXPERIMENTS_DIR permissions** — P1, prevents supply chain attack. Impact: