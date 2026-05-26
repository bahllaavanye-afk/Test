# ML Agent Guide

## Your Role
You improve ML models and run experiments. Every change must be tracked in `experiments/`.

## How to Add a New Model

1. Create `backend/app/ml/models/my_model.py` implementing `AbstractModel`
2. Add training script `backend/app/ml/training/train_my_model.py`
3. Create experiment config `experiments/configs/my_model_btc_1h.yaml`
4. Run: `python experiments/run_experiment.py --config my_model_btc_1h.yaml`
5. If test_sharpe > existing best, update InferenceService to load it

## Experiment Config Fields
See `experiments/configs/lstm_btc_1h.yaml` for the full schema.

## Running Experiments
```bash
python experiments/run_experiment.py --config lstm_btc_1h.yaml
python experiments/run_experiment.py --config lstm_btc_1h.yaml --sweep hidden_size=64,128,256
```

## Checking for Lookahead Bias
```bash
python experiments/debug/debug_feature_leak.py --config lstm_btc_1h.yaml
```

## Files Safe to Modify
- `ml/models/*.py` (except base_model.py interface)
- `ml/features/*.py` (add new features, keep shift(1) rule)
- `ml/training/*.py`
- `experiments/configs/*.yaml`

## Files to AVOID
- `ml/models/base_model.py` — interface change breaks all models
- `ml/inference.py` — only update after model is validated
