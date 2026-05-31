# ML Model Serving Layer — Agent Guide

## Overview

This module handles everything between a trained model artifact on disk and a
live trading signal in production.  It implements the **champion/challenger
pattern** with full A/B testing, inference logging, and outcome tracking.

---

## Release Lifecycle

```
    ┌────────────┐     POST /shadow     ┌────────────┐
    │ registered │ ──────────────────► │   shadow   │
    └────────────┘                     └─────┬──────┘
                                             │ POST /challenge
                                             ▼
    ┌────────────┐     POST /promote    ┌────────────┐
    │  champion  │ ◄──────────────────  │ challenger │
    └────────────┘                     └─────┬──────┘
          │                                  │ POST /archive
          │ (old champion auto-archived)      ▼
          │                            ┌────────────┐
          └──────────────────────────► │  archived  │
                POST /promote
```

**Status definitions:**

| Status       | Traffic | Description |
|-------------|---------|-------------|
| `registered` | 0%      | Artifact known to system, not yet serving |
| `shadow`     | 0%      | Receiving silent predictions for logging only |
| `challenger` | 1–50%   | Active A/B test vs champion |
| `champion`   | 50–100% | Primary model in production |
| `archived`   | 0%      | Retired; kept for audit trail |

---

## A/B Testing Guide

### Starting a test

```bash
# 1. Register a newly trained artifact
curl -X POST /api/v1/releases/ \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"model_name": "lstm_momentum", "version": "v2.1.0",
       "artifact_path": "models/lstm_momentum_v21.pt",
       "framework": "pytorch",
       "train_metrics": {"val_sharpe": 1.92, "val_accuracy": 0.64}}'

# 2. Shadow-deploy (optional) — logs silent predictions for 24h before A/B
curl -X POST /api/v1/releases/{id}/shadow

# 3. Start A/B test with 10% challenger traffic
curl -X POST /api/v1/releases/{id}/challenge \
  -d '{"traffic_pct": 10}'
```

### Monitoring the test

```bash
# All active tests + metrics
curl /api/v1/releases/ab-tests/active

# Specific test vs champion
curl /api/v1/releases/{challenger_id}/metrics
```

Response includes:
- `n_predictions` per group
- `accuracy` (fraction of is_correct predictions, once outcomes recorded)
- `avg_confidence`
- `recommendation`: "promote_challenger" | "keep_champion" | "insufficient_data"
- `min_samples_needed`: 30 (configurable in `releases.py:_MIN_SAMPLES`)

### Promoting the winner

```bash
# Challenger wins → promote (old champion auto-archived)
curl -X POST /api/v1/releases/{challenger_id}/promote

# Or stop the test and keep champion
curl -X POST /api/v1/releases/{challenger_id}/archive
```

---

## Traffic Routing Algorithm

`ABRouter` (in `ab_router.py`) uses a **lazy-refresh in-memory snapshot**:

1. On each inference call, `route(model_name)` is called.
2. If the snapshot is stale (> 60s), it re-reads from DB — but only one coroutine
   does the refresh at a time (asyncio.Lock prevents thundering herd).
3. Routing decision uses `random.random() * 100 < challenger.traffic_pct`.
   - If challenger wins the coin flip → inference goes to challenger.
   - Otherwise → champion.
4. After any promote/archive API call, `invalidate(model_name)` is called so
   the next request picks up the new state within milliseconds.

**Key property:** The routing is stateless and requires no coordination between
worker processes — each process independently rolls dice and routes to the
correct artifact.

---

## Adding a New Framework

To support a new serialization format, add a branch in `serve.py:_load_artifact`:

```python
if framework == "my_new_framework":
    from app.ml.models.my_model import MyModel
    return MyModel.load(path)
```

Then set `framework="my_new_framework"` when registering the release.

---

## Inference Logging Schema

Every served prediction is written to `inference_logs` asynchronously.  The
write is fire-and-forget — a DB failure never blocks a trading signal.

| Column         | Type    | Description |
|----------------|---------|-------------|
| `release_id`   | FK      | Which model release served this request |
| `model_name`   | str     | Logical model name (e.g. "lstm_momentum") |
| `version`      | str     | Artifact version |
| `symbol`       | str     | Ticker (e.g. "SPY") |
| `ts`           | datetime | UTC timestamp of inference |
| `prediction`   | float   | Raw model output [0, 1] |
| `signal`       | str     | "buy" / "sell" / "hold" |
| `confidence`   | float   | abs(prediction - 0.5) * 2 |
| `latency_ms`   | float   | Time from tensor in to logit out |
| `ab_group`     | str     | "champion" or "challenger" |
| `actual_return`| float?  | Filled ex-post via `/record-outcome` |
| `is_correct`   | bool?   | True if predicted direction matched actual |

### Recording outcomes

After the next bar closes, record the actual return so accuracy can be computed:

```bash
curl -X POST /api/v1/releases/{id}/record-outcome \
  -d '{"symbol": "SPY", "actual_return": 0.0023}'
```

This fills the most recent unresolved log entry for that (release, symbol) pair.

---

## Model Cache

`ModelServingLayer` keeps a process-local LRU cache of at most 16 loaded model
objects (`_CACHE_MAX`).  Cache entries are keyed by `release_id`.

The cache is automatically invalidated when:
- A release is promoted to champion (old champion evicted)
- A release is archived

To manually evict:
```python
from app.ml.serving.serve import get_serving_layer
get_serving_layer().invalidate_model(release_id)
```

---

## Using from a Strategy

```python
from app.ml.serving.serve import get_serving_layer

class MLMomentumStrategy(AbstractStrategy):
    async def analyze(self, df: pd.DataFrame) -> Signal | None:
        features = engineer_features(df)
        X, _ = create_sequences(features, seq_len=64)
        if len(X) == 0:
            return None

        serving = get_serving_layer()
        result = await serving.predict("lstm_momentum", X[-1], symbol=self.symbol)
        if result is None or result.signal == "hold":
            return None

        return Signal(side=result.signal, confidence=result.confidence)
```

---

## Common Commands

```bash
# List all releases
GET /api/v1/releases/

# Get champion for a model
GET /api/v1/releases/champion/lstm_momentum

# Update notes or metrics on a release
PATCH /api/v1/releases/{id}  -d '{"notes": "Trained on 2 years of 1h BTC data"}'

# View inference logs for a release
GET /api/v1/releases/{id}/inferences?limit=100&symbol=SPY

# Record outcome for accuracy tracking
POST /api/v1/releases/{id}/record-outcome  -d '{"symbol": "SPY", "actual_return": 0.005}'
```

---

## Champion/Challenger Best Practices

1. **Always shadow before challenging** — let the model accumulate silent
   predictions for ≥24h to catch bugs before it touches live traffic.

2. **Keep challenger traffic ≤ 20%** — protects against a bad model causing
   significant P&L damage before you detect it.

3. **Wait for 30+ samples before interpreting accuracy** — the `insufficient_data`
   recommendation means exactly this.

4. **Record outcomes religiously** — without actual returns, accuracy is always
   null and you cannot make a data-driven promotion decision.

5. **One challenger at a time** — the API enforces this, but operationally it
   also makes interpretation cleaner.

6. **Promote on weekends** — traffic is lower on weekends, reducing the blast
   radius of a bad promotion.

---

## Files in This Module

| File | Purpose |
|------|---------|
| `ab_router.py` | Traffic routing logic, DB snapshot cache |
| `serve.py` | Model loading, inference execution, async logging |
| `__init__.py` | Package marker |
| `CLAUDE.md` | This file |

Related files outside this directory:
| File | Purpose |
|------|---------|
| `backend/app/models/model_release.py` | ORM model for releases |
| `backend/app/models/inference_log.py` | ORM model for inference logs |
| `backend/app/api/v1/releases.py` | REST API (12 endpoints) |
| `frontend/src/pages/Releases.tsx` | Dashboard UI |
