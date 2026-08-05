# Api Backend — Employee Deep Review
**Date:** 2026-08-05  |  **Employee:** `backend_lead` (the Backend Lead at QuantEdge)  |  **LLM:** nvidia_nim  |  **Grade:** ?

_This review was written by backend_lead (the Backend Lead at QuantEdge) using nvidia_nim, independently of all other employees' reports._

---

### Critical Issues (P0/P1)

1. **backend/app/api/deps.py:28** — `db is None` check raises `ra` (incomplete statement). This will cause a `NameError` at runtime on any authenticated request. Fix: replace with `raise UnauthorizedError()`.

2. **backend/app/config.py:33** — `secret_key` defaults to `"change-me-in-production-32-byte-hex"`. In paper mode this is acceptable for dev, but if `trading_mode` ever flips to `"live"` (line 28), JWT tokens become trivially forgeable. Fix: add a `model_validator` that raises `ValueError` if `trading_mode == "live"` and `secret_key` equals the default.

3. **backend/app/main.py:22** — `configure_logging()` is defined but never called. Production logs use structlog defaults, losing structured context. Fix: add `configure_logging()` at line 1 of the lifespan context manager.

### Performance & Reliability Improvements

1. **backend/app/api/limiter.py:4** — `get_remote_address` uses client IP, which behind a reverse proxy will be the proxy IP (e.g., `127.0.0.1`). All users share one rate limit bucket. Fix: use `request.headers.get("X-Forwarded-For", "").split(",")[0]` as key function.

2. **backend/app/main.py:28** — `Base.metadata.create_all` runs on every startup. For 50+ models this adds ~2s latency. Fix: use Alembic migrations and remove `create_all` from production code.

### Alpha / Signal Quality

1. **No signal validation in api-backend** — The Polymarket desk failure (3 signals, 0 orders) suggests missing order path validation. Add a Pydantic model `SignalOrderPath` that validates broker adapter exists before signal dispatch.

### Security & Safety

1. **backend/app/config.py:30** — `allowed_origins: str = "http://localhost:5173"` is a single string, not a list. FastAPI's `CORSMiddleware` expects `List[str]`. This will silently fail to allow any origin, blocking all frontend requests. Fix: change type to `list[str]` and parse comma-separated env var.

2. **backend/app/api/deps.py:15** — `HTTPBearer(auto_error=False)` means missing tokens return `None` instead of 403. Combined with the broken `ra` statement, unauthenticated requests may pass through. Fix: set `auto_error=True` and remove the manual `None` check.

### Implementation Priority Queue

1. Fix `deps.py:28` incomplete statement (P0, blocks all auth)
2. Fix `config.py:30` CORS type (P0, blocks all frontend)
3. Add `configure_logging()` call in `main.py` (P1, production observability)
4. Add `secret_key` validator for live mode (P1, prevents catastrophic key leak)
5. Replace `get_remote_address` with proxy-aware IP extraction (P2, rate limiting effectiveness)

### Overall Grade
**D** — Two P0 bugs (broken auth, broken CORS) make the API non-functional in production; the incomplete statement suggests code was never tested end-to-end.