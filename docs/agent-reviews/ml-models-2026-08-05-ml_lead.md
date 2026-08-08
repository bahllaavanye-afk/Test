# Ml Models — Employee Deep Review
**Date:** 2026-08-05  |  **Employee:** `ml_lead` (the ML Modeling Lead and crypto desk lead at QuantEdge)  |  **LLM:** nvidia_nim  |  **Grade:** ?

_This review was written by ml_lead (the ML Modeling Lead and crypto desk lead at QuantEdge) using nvidia_nim, independently of all other employees' reports._

---

### Critical Issues (P0/P1)

1. **`backend/app/ml/inference.py:38`** — `self.models["lstm"]` is incomplete; the line ends abruptly without loading the model. This will cause a `KeyError` at inference time. Fix: complete the assignment, e.g., `self.models["lstm"] = LSTMPredictor.load(lstm_path)`.

2. **`backend/app/ml/registry.py:38`** — `_load_index()` is called in `__init__` but not defined in the snippet. If missing, the registry will silently fail to load existing records, causing duplicate registrations. Fix: implement `_load_index()` to read JSON from `index_path`.

3. **`backend/app/ml/inference.py:20`** — Hardcoded `weights = {"lstm": 0.50, "xgboost": 0.35, "lorentzian": 0.15}`. No validation that these models are loaded; if one fails, ensemble will silently use partial weights. Fix: normalize weights dynamically based on loaded models.

### Performance & Reliability Improvements

1. **`backend/app/ml/inference.py:15`** — `_inference_service` global singleton is not thread-safe. Use `threading.Lock` or `asyncio.Lock` when loading models to prevent race conditions on startup.

2. **`backend/app/ml/registry.py:25`** — JSON index file is written without atomicity. Use `tempfile.NamedTemporaryFile` + `os.replace` to avoid corruption on crash.

### Alpha / Signal Quality

1. **`backend/app/ml/inference.py:20`** — Static ensemble weights ignore regime changes. In a bull market, momentum models should be upweighted. Implement regime-aware weighting using the `market_regime` from company context.

2. **`backend/app/ml/registry.py:30`** — No decay mechanism for old models. Models trained >30 days ago should be downweighted or retired to avoid stale signals.

### Security & Safety

1. **`backend/app/ml/inference.py:10`** — `from app.config import settings` — if `settings.models_dir` is user-controllable, an attacker could load arbitrary `.pt` files. Validate path is within allowed directory.

2. **`backend/app/ml/registry.py:20`** — JSON index is loaded with `json.load()` without schema validation. Malformed or malicious index could cause arbitrary code execution via `__reduce__` in pickle artifacts. Use `pydantic` models for validation.

### Implementation Priority Queue

1. **Fix incomplete LSTM load** (`inference.py:38`) — P0, blocks all LSTM predictions. Impact: high.
2. **Implement `_load_index()`** (`registry.py:38`) — P1, prevents registry corruption. Impact: high.
3. **Dynamic ensemble weights** (`inference.py:20`) — P1, improves alpha capture in current bull regime. Impact: medium.
4. **Atomic JSON writes** (`registry.py:25`) — P2, prevents data loss. Impact: medium.
5. **Path traversal protection** (`inference.py:10`) — P2, security hardening. Impact: low.

### Overall Grade
**D** — Critical incomplete code and missing registry functionality make the ML pipeline non-functional for LSTM and prone to data corruption.