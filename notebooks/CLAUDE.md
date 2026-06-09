# QuantEdge Notebooks — Training & Analysis

## Environment Options

### Kaggle
1. Go to kaggle.com/code → New Notebook
2. Upload the .ipynb file or paste cells manually
3. Enable GPU: Settings → Accelerator → GPU T4 x2
4. Internet must be enabled for `pip install` and `yfinance` downloads

### Google Colab
1. Go to colab.research.google.com → Upload → select .ipynb
2. Runtime → Change runtime type → GPU (A100 or T4)
3. All notebooks use `!pip install` cells at the top — run them first

### Lightning.AI (recommended for production training)
1. Create a new Studio at lightning.ai
2. Upload notebooks or clone the repo
3. Select a machine with GPU (T4, A100, or L4)
4. Open Jupyter Lab and run notebooks in order

## Notebook Descriptions

| Notebook | Purpose | Estimated Runtime |
|---|---|---|
| `train_lstm.ipynb` | Bidirectional LSTM with attention on BTC hourly data | 20–40 min (GPU) |
| `train_xgboost.ipynb` | XGBoost with Optuna hyperparameter tuning on SPY | 10–20 min (CPU) |
| `train_transformer.ipynb` | Temporal Fusion Transformer on BTC data | 30–60 min (GPU) |
| `train_ppo_rl.ipynb` | PPO RL agent for execution optimization | 1–2 hrs (GPU) |
| `compare_strategies.ipynb` | Backtest result comparison with charts | 2–5 min |
| `feature_analysis.ipynb` | Feature IC/IR analysis and correlation matrix | 2–5 min |

## Output Files
Models are saved to `../models_artifacts/` by default (relative to this directory).
Update paths if running on Kaggle/Colab — use `/kaggle/working/` or `/content/` respectively.

## Data
All notebooks download data via `yfinance`. No local data files are required.
For Binance data, `python-binance` or `ccxt` may be used as alternatives.

## Notes
- Always run the `!pip install` cell first in a fresh environment
- Notebooks are designed to be run top-to-bottom; avoid running cells out of order
- The `train_lstm.ipynb` saves `lstm_btc_1h.pt` — copy this to the backend `models_artifacts/` directory
