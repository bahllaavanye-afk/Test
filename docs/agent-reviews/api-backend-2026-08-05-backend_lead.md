# Api Backend — Employee Deep Review
**Date:** 2026-08-05  |  **Employee:** `backend_lead` (the Backend Lead at QuantEdge)  |  **LLM:** nvidia_nim  |  **Grade:** ?

_This review was written by backend_lead (the Backend Lead at QuantEdge) using nvidia_nim, independently of all other employees' reports._

---

### Critical Issues (P0/P1)

1. **backend/app/api/deps.py:28-30** — `db is None: ra` is a truncated line that will cause a `SyntaxError` at import time. This breaks ALL authenticated endpoints. Fix: replace with `if db is None: raise UnauthorizedError()`.

2. **backend/app/main.py:25-27** — `configure_logging()` is defined but never called. Production logs use structlog defaults, losing custom formatting and correlation IDs. Fix: add `configure_logging()` at line 30 before `logger` usage.

3. **backend/app/config.py:38-40** — `secret_key` default is `"change-me-in-production-32-byte-hex"` with no validation. If `.env` is missing, JWT signing uses this weak key. Fix: add `@field_validator("secret_key")` that raises `ValueError` if value equals the default.

### Performance & Reliability Improvements

1. **backend/app/api/limiter.py:4** — `Limiter(key_func=get_remote_address)` uses IP-based rate limiting behind proxies. Add `key_func=lambda: request.headers.get("X-Forwarded-For", get_remote_address())` to handle reverse proxies.

2. **backend/app/main.py:35-38** — All WebSocket routers (`prices_router`, `orders_router`, `alerts_router`) are mounted without connection limits. Add `max_size=1024` and `ping_interval=20` to each WebSocket endpoint to prevent memory leaks.

### Alpha / Signal Quality

1. **No signal validation middleware** — All strategy signals from `mean_rev_20_1.5` are accepted without checking for stale data (>5min old). Add a Pydantic validator in `app/schemas/signal.py` that rejects signals with `timestamp < utcnow() - timedelta(minutes=5)`.

### Security & Safety

1. **backend/app/config.py:15** — `trading_mode: str = "paper"` with comment "live trading permanently disabled" but no runtime guard. Add `@model_validator(mode="after")` that raises `RuntimeError("Live trading disabled")` if `trading_mode == "live"`.

2. **backend/app/api/deps.py:12** — `bearer_scheme = HTTPBearer(auto_error=False)` allows missing tokens to pass through to `get_current_user`. Change to `auto_error=True` and remove the manual `if not credentials` check.

### Implementation Priority Queue

1. Fix `deps.py` truncated line (P0) — 5 min, unblocks all auth
2. Add `configure_logging()` call in `main.py` (P1) — 2 min, fixes production logging
3. Add `secret_key` validation in `config.py` (P1) — 5 min, prevents weak JWT keys
4. Add `trading_mode` runtime guard (P1) — 3 min, prevents accidental live trading
5. Add signal staleness validator (P2) — 30 min, improves signal quality

### Overall Grade
**D** — Two P0 bugs (truncated code, missing logging init) make the app non-functional in production, and critical security defaults are unvalidated.