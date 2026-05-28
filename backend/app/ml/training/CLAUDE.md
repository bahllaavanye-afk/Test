# ML Research Engineer — Training Guide

## Your Role
You design, train, and evaluate ML models that improve on pure indicator-based strategies. Every model you ship must demonstrably add alpha — not just improve accuracy metrics.

## Owned Files (safe to modify)
```
backend/app/ml/
  models/
    lstm.py              # PyTorch BiLSTM with attention
    transformer.py       # Temporal Fusion Transformer
    xgboost_model.py     # XGBoost with Optuna HPO
    lightgbm_model.py    # LightGBM alternative
    ensemble_model.py    # Weighted ensemble
    lorentzian_knn.py    # Lorentzian KNN (TV port)
  training/
    trainer.py           # Generic Trainer (early stop, LR scheduler)
    train_lstm.py        # LSTM entry point
    train_transformer.py
    train_xgboost.py     # Optuna HPO + training
    walk_forward.py      # Walk-forward CV for all models
  features/
    engineer.py          # Master feature pipeline
    technical.py         # RSI, MACD, BB, ATR, OBV, etc.
    microstructure.py    # Bid-ask spread, OFI
    cross_asset.py       # VIX, yield curve, dollar index
    normalization.py     # Scaler fit/transform/save/load
  datasets/
    sequence_dataset.py  # PyTorch Dataset for LSTM
    flat_dataset.py      # Flat Dataset for XGBoost
    rl_env.py            # Custom gym.Env for PPO
  registry.py            # Model artifact store
  inference.py           # Unified inference (ensemble + threshold)

experiments/
  configs/*.yaml         # Experiment definitions (model, data, HPO)
  results/               # Auto-generated JSON per run
  run_experiment.py      # CLI: python run_experiment.py --config <yaml>
```

## Do NOT Modify
- `backend/app/strategies/` — ML strategy wrappers call `inference.py`, not model files directly
- `backend/app/ml/models/base_model.py` — interface contract; all models must implement it
- `backend/app/risk/manager.py`

## Model Quality Gates (ALL must pass before deployment)
| Gate                            | Threshold            |
|---------------------------------|----------------------|
| OOS Sharpe (walk-forward)       | ≥ 0.8                |
| OOS Accuracy (directional)      | ≥ 55%                |
| Calibration error (ECE)         | ≤ 0.10               |
| IS Sharpe / OOS Sharpe ratio    | ≤ 2.0 (overfitting)  |
| Information Coefficient (IC)    | ≥ 0.05               |
| Max OOS Drawdown                | ≤ 20%                |

## Experiment Config Schema
```yaml
experiment:
  name: "lstm_spy_1d_v3"
  model: "lstm"           # lstm | transformer | xgboost | lorentzian | ensemble
  symbol: "SPY"
  interval: "1d"

data:
  train_start: "2018-01-01"
  train_end:   "2022-12-31"
  val_start:   "2023-01-01"
  val_end:     "2023-06-30"
  test_start:  "2023-07-01"
  test_end:    "2024-12-31"

features:
  technical:   ["rsi_14", "macd", "bb_width", "atr_14", "obv"]
  cross_asset: ["vix", "yield_curve_10y2y"]
  lookback:    60

model_params:
  hidden_size: 128
  num_layers: 2
  dropout: 0.3
  bidirectional: true

training:
  epochs: 100
  batch_size: 256
  lr: 0.001
  optimizer: "adamw"
  early_stopping_patience: 10
```

## Running on Free GPU (Kaggle)
```bash
# 1. Upload OHLCV CSV to Kaggle dataset
# 2. Open notebooks/train_lstm.ipynb
# 3. Run all cells (T4 GPU, ~20min for 2yr of 1h BTC data)
# 4. Download model.pt + scaler.pkl
# 5. Place in backend/models_artifacts/<experiment_name>/
```

## Adding a New Model Architecture
1. Create `backend/app/ml/models/<name>.py` implementing `AbstractModel`
2. Add to `MODEL_REGISTRY` dict in `backend/app/ml/registry.py`
3. Add a `train_<name>.py` entry point in `ml/training/`
4. Add an experiment config in `experiments/configs/`
5. Run walk-forward validation: pass all quality gates above before opening PR

## Debugging Overfitting
```bash
python experiments/debug/debug_overfitting.py --config lstm_spy_1d_v3.yaml
# → plots train/val loss curves; if val diverges after epoch 20, reduce capacity
```

## Feature Importance (post-training)
```bash
# For XGBoost: SHAP values built in
python experiments/debug/debug_signal_quality.py --model xgb_spy_daily_v2
# Output: IC per feature, top-10 SHAP importances
```
