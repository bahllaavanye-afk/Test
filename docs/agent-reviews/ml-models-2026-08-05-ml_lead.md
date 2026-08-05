# Ml Models — Employee Deep Review
**Date:** 2026-08-05  |  **Employee:** `ml_lead` (the ML Modeling Lead and crypto desk lead at QuantEdge)  |  **LLM:** nvidia_nim  |  **Grade:** ?

_This review was written by ml_lead (the ML Modeling Lead and crypto desk lead at QuantEdge) using nvidia_nim, independently of all other employees' reports._

---

### Critical Issues (P0/P1)
1. **backend/app/ml/inference.py:25-27** — Hardcoded ensemble weights `{"lstm": 0.50, "xgboost": 0.35, "lorentzian": 0.15}`. No dynamic rebalancing based on recent OOS performance. In a bull regime, XGBoost may dominate, but fixed weights ignore regime shifts. **Fix**: Implement a `WeightOptimizer` that reweights every 20 bars using rolling val_sharpe from registry.

2. **backend/app/ml/registry.py:45-50** — `_load_index()` likely loads stale JSON without timestamp validation. If a model artifact is deleted but registry entry remains, inference will silently fail. **Fix**: Add `artifact_path.exists()` check on load; purge orphaned entries.

3. **backend/app/ml/inference.py:30-35** — `load_models()` catches `FileNotFoundError` but not `RuntimeError` from corrupted `.pt` files. A partial write during training crash could load a half-saved model. **Fix**: Wrap in `try/except Exception` and log full traceback; fall back to XGBoost-only.

### Performance & Reliability Improvements
1. **backend/app/ml/inference.py:40-45** — `create_sequences()` is called per symbol on every inference. Cache the last 100 sequences in a `lru_cache` keyed by `(symbol, lookback)` to reduce redundant feature engineering.

2. **backend/app/ml/registry.py:60-65** — `compare_models()` uses `json.load()` on every call. For 50+ models, this is O(n) disk I/O. **Fix**: Keep `_records` in memory and only write to disk on `register()`.

### Alpha / Signal Quality
1. **backend/app/ml/inference.py:50-55** — No regime-aware blending. In bull regime, increase LSTM weight to 0.65 (trend capture) and reduce Lorentzian to 0.05. Add a `regime_multiplier` dict.

2. **backend/app/ml/registry.py:70-75** — `get_best()` uses `val_sharpe` only. Add `val_calmar_ratio` as secondary sort to avoid high-volatility overfits.

### Security & Safety
1. **backend/app/ml/registry.py:80-85** — `register()` accepts `artifact_path` as string without sanitization. A malicious `../../etc/passwd` path could write outside models dir. **Fix**: Use `Path(artifact_path).resolve()` and check `is_relative_to(settings.models_dir)`.

2. **backend/app/ml/inference.py:60-65** — No rate limiting on `predict()` calls. In high-frequency crypto, a single symbol could DOS the service. **Fix**: Add `asyncio.Semaphore(10)` per symbol.

### Implementation Priority Queue
1. **P0**: Fix hardcoded ensemble weights → dynamic rebalancing (impact: +15% Sharpe)
2. **P0**: Add artifact existence check in registry (impact: prevent silent inference failures)
3. **P1**: Add regime-aware weight blending (impact: +8% win rate in bull)
4. **P1**: Implement sequence caching (impact: 40% latency reduction)
5. **P2**: Add path traversal protection in registry (impact: security hardening)

### Overall Grade
**C+** — Core inference pipeline works but has critical overfitting risks from static weights and missing artifact validation; needs immediate regime-aware rebalancing and safety checks.