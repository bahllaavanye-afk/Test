#!/usr/bin/env python3
"""Generate all 50 ML experiment YAML configs with systematic ablations.

Run: python scripts/generate_experiments.py
Creates files in experiments/configs/
"""

import os
import yaml
from pathlib import Path

CONFIGS_DIR = Path(__file__).parent.parent / "experiments" / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


def write(name: str, config: dict) -> None:
    path = CONFIGS_DIR / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  wrote {path.name}")


def base_dates(asset: str = "equity") -> dict:
    if asset == "crypto":
        return {
            "train_start": "2019-01-01",
            "train_end": "2024-06-30",
            "val_start": "2023-01-01",
            "val_end": "2024-06-30",
            "test_start": "2024-07-01",
            "test_end": "2026-01-01",
        }
    return {
        "train_start": "2015-01-01",
        "train_end": "2024-06-30",
        "val_start": "2023-01-01",
        "val_end": "2024-06-30",
        "test_start": "2024-07-01",
        "test_end": "2026-01-01",
    }


def patch_tst_config(name, symbol, interval, seq_len=64, patch_len=16, d_model=128,
                     n_heads=4, n_layers=3, dropout=0.2, channel_independent=True,
                     features=None, epochs=150, lr=5e-4, batch=128, asset="equity"):
    if features is None:
        features = {"technical": True, "advanced": True, "multi_timeframe": True, "lookback": seq_len}
    return {
        "experiment": {"name": name, "model": "patch_tst", "symbol": symbol, "interval": interval},
        "data": base_dates(asset),
        "features": features,
        "model_params": {
            "seq_len": seq_len, "patch_len": patch_len, "d_model": d_model,
            "n_heads": n_heads, "n_layers": n_layers, "dropout": dropout,
            "channel_independent": channel_independent,
        },
        "training": {
            "epochs": epochs, "batch_size": batch, "lr": lr,
            "optimizer": "adamw", "early_stopping_patience": 20, "scheduler": "cosine",
        },
        "ablation_group": "patch_tst",
        "ablation_var": f"patch_len={patch_len},d_model={d_model},n_layers={n_layers},seq_len={seq_len},dropout={dropout},ci={channel_independent}",
    }


def mst_config(name, symbol, interval, d_model=128, n_heads=4, n_layers=2, dropout=0.2,
               n_streams=3, features=None, epochs=150, lr=5e-4, batch=128, asset="equity"):
    if features is None:
        features = {"technical": True, "advanced": True, "multi_timeframe": True, "lookback": 60}
    return {
        "experiment": {"name": name, "model": "multiscale_transformer",
                       "symbol": symbol, "interval": interval},
        "data": base_dates(asset),
        "features": features,
        "model_params": {
            "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
            "dropout": dropout, "n_streams": n_streams,
        },
        "training": {
            "epochs": epochs, "batch_size": batch, "lr": lr,
            "optimizer": "adamw", "early_stopping_patience": 20, "scheduler": "cosine",
        },
        "ablation_group": "multiscale_transformer",
        "ablation_var": f"d_model={d_model},n_layers={n_layers},n_streams={n_streams}",
    }


def lstm_config(name, symbol, interval, hidden=128, n_layers=2, dropout=0.3,
                bidirectional=True, features=None, epochs=100, lr=1e-3, batch=256, asset="equity"):
    if features is None:
        features = {"technical": True, "advanced": True, "multi_timeframe": True, "lookback": 60}
    return {
        "experiment": {"name": name, "model": "lstm", "symbol": symbol, "interval": interval},
        "data": base_dates(asset),
        "features": features,
        "model_params": {
            "hidden_size": hidden, "num_layers": n_layers,
            "dropout": dropout, "bidirectional": bidirectional,
        },
        "training": {
            "epochs": epochs, "batch_size": batch, "lr": lr,
            "optimizer": "adamw", "early_stopping_patience": 15, "scheduler": "cosine",
        },
        "ablation_group": "lstm",
        "ablation_var": f"hidden={hidden},layers={n_layers},bidirectional={bidirectional},dropout={dropout}",
    }


def xgb_config(name, symbol, interval, n_estimators=500, max_depth=6, lr=0.05,
               subsample=0.8, colsample=0.8, features=None, asset="equity"):
    if features is None:
        features = {"technical": True, "advanced": True, "multi_timeframe": True}
    return {
        "experiment": {"name": name, "model": "xgboost", "symbol": symbol, "interval": interval},
        "data": base_dates(asset),
        "features": features,
        "model_params": {
            "n_estimators": n_estimators, "max_depth": max_depth,
            "learning_rate": lr, "subsample": subsample,
            "colsample_bytree": colsample, "use_gpu": False,
        },
        "training": {"early_stopping_rounds": 50, "eval_metric": "auc"},
        "ablation_group": "xgboost",
        "ablation_var": f"n_est={n_estimators},depth={max_depth},lr={lr}",
    }


def ensemble_config(name, symbol, interval, components=None, asset="equity"):
    if components is None:
        components = ["patch_tst", "multiscale_transformer", "xgboost", "lorentzian_knn"]
    return {
        "experiment": {"name": name, "model": "ensemble", "symbol": symbol, "interval": interval},
        "data": base_dates(asset),
        "features": {"technical": True, "advanced": True, "multi_timeframe": True, "lookback": 64},
        "ensemble": {
            "components": components,
            "optimize_weights": True,
            "weight_optimization": "sharpe",
            "min_component_weight": 0.05,
        },
        "training": {
            "epochs": 100, "batch_size": 128, "lr": 5e-4,
            "optimizer": "adamw", "early_stopping_patience": 15,
        },
        "ablation_group": "ensemble",
        "ablation_var": f"components={'+'.join(components)}",
    }


def feat_flags(technical=True, advanced=True, mtf=True, lookback=60):
    return {"technical": technical, "advanced": advanced, "multi_timeframe": mtf, "lookback": lookback}


print("Generating 50 ML experiment configs with ablations...")

# ─── GROUP A: PatchTST Architecture Ablations (SPY 1d) ─────────────────────
print("\n[A] PatchTST architecture ablations")

write("patch_tst_spy_p8",    patch_tst_config("patch_tst_spy_p8",    "SPY", "1d", patch_len=8))
write("patch_tst_spy_p32",   patch_tst_config("patch_tst_spy_p32",   "SPY", "1d", patch_len=32))
write("patch_tst_spy_p64",   patch_tst_config("patch_tst_spy_p64",   "SPY", "1d", seq_len=128, patch_len=64))
write("patch_tst_spy_d64",   patch_tst_config("patch_tst_spy_d64",   "SPY", "1d", d_model=64, n_heads=4))
write("patch_tst_spy_d256",  patch_tst_config("patch_tst_spy_d256",  "SPY", "1d", d_model=256, n_heads=8))
write("patch_tst_spy_l2",    patch_tst_config("patch_tst_spy_l2",    "SPY", "1d", n_layers=2))
write("patch_tst_spy_l6",    patch_tst_config("patch_tst_spy_l6",    "SPY", "1d", n_layers=6))
write("patch_tst_spy_s32",   patch_tst_config("patch_tst_spy_s32",   "SPY", "1d", seq_len=32, patch_len=8))
write("patch_tst_spy_s128",  patch_tst_config("patch_tst_spy_s128",  "SPY", "1d", seq_len=128, patch_len=16))
write("patch_tst_spy_do5",   patch_tst_config("patch_tst_spy_do5",   "SPY", "1d", dropout=0.5))
write("patch_tst_spy_ci_off",patch_tst_config("patch_tst_spy_ci_off","SPY", "1d", channel_independent=False))

# ─── GROUP B: PatchTST Cross-Asset ─────────────────────────────────────────
print("\n[B] PatchTST cross-asset")

write("patch_tst_btc_1h",    patch_tst_config("patch_tst_btc_1h",  "BTC-USD", "1h",  seq_len=96,  patch_len=16, asset="crypto"))
write("patch_tst_eth_1h",    patch_tst_config("patch_tst_eth_1h",  "ETH-USD", "1h",  seq_len=96,  patch_len=16, asset="crypto"))
write("patch_tst_qqq_1d",    patch_tst_config("patch_tst_qqq_1d",  "QQQ",     "1d"))
write("patch_tst_aapl_1d",   patch_tst_config("patch_tst_aapl_1d", "AAPL",    "1d"))
write("patch_tst_gld_1d",    patch_tst_config("patch_tst_gld_1d",  "GLD",     "1d"))
write("patch_tst_tlt_1d",    patch_tst_config("patch_tst_tlt_1d",  "TLT",     "1d"))

# ─── GROUP C: MultiScaleTransformer Ablations ──────────────────────────────
print("\n[C] MultiScaleTransformer ablations")

write("mst_btc_1stream",  mst_config("mst_btc_1stream", "BTC-USD", "1h", n_streams=1, asset="crypto"))
write("mst_btc_2stream",  mst_config("mst_btc_2stream", "BTC-USD", "1h", n_streams=2, asset="crypto"))
write("mst_btc_d64",      mst_config("mst_btc_d64",     "BTC-USD", "1h", d_model=64,  n_heads=4, asset="crypto"))
write("mst_btc_d256",     mst_config("mst_btc_d256",    "BTC-USD", "1h", d_model=256, n_heads=8, asset="crypto"))
write("mst_spy_1d",       mst_config("mst_spy_1d",      "SPY",     "1d"))
write("mst_eth_1h",       mst_config("mst_eth_1h",      "ETH-USD", "1h", asset="crypto"))
write("mst_qqq_1d",       mst_config("mst_qqq_1d",      "QQQ",     "1d"))

# ─── GROUP D: Feature Engineering Ablations ────────────────────────────────
print("\n[D] Feature engineering ablations")

for model, model_fn in [("lstm", lstm_config), ("patch_tst", patch_tst_config)]:
    write(f"{model}_spy_tech_only",
          model_fn(f"{model}_spy_tech_only", "SPY", "1d",
                   features=feat_flags(advanced=False, mtf=False)))
    write(f"{model}_spy_tech_adv",
          model_fn(f"{model}_spy_tech_adv",  "SPY", "1d",
                   features=feat_flags(mtf=False)))
    write(f"{model}_spy_tech_mtf",
          model_fn(f"{model}_spy_tech_mtf",  "SPY", "1d",
                   features=feat_flags(advanced=False)))

write("xgb_spy_tech_only", xgb_config("xgb_spy_tech_only", "SPY", "1d",
                                       features=feat_flags(advanced=False, mtf=False)))

# ─── GROUP E: LSTM Ablations ────────────────────────────────────────────────
print("\n[E] LSTM ablations")

write("lstm_spy_1d",         lstm_config("lstm_spy_1d",         "SPY",     "1d"))
write("lstm_btc_1h_v2",      lstm_config("lstm_btc_1h_v2",      "BTC-USD", "1h", asset="crypto"))
write("lstm_eth_1h",         lstm_config("lstm_eth_1h",         "ETH-USD", "1h", asset="crypto"))
write("lstm_qqq_1d",         lstm_config("lstm_qqq_1d",         "QQQ",     "1d"))
write("lstm_spy_h256",       lstm_config("lstm_spy_h256",       "SPY",     "1d", hidden=256))
write("lstm_spy_unidirect",  lstm_config("lstm_spy_unidirect",  "SPY",     "1d", bidirectional=False))
write("lstm_spy_l3",         lstm_config("lstm_spy_l3",         "SPY",     "1d", n_layers=3))

# ─── GROUP F: XGBoost Ablations ─────────────────────────────────────────────
print("\n[F] XGBoost ablations")

write("xgb_btc_1h",   xgb_config("xgb_btc_1h",   "BTC-USD", "1h",  asset="crypto"))
write("xgb_eth_1h",   xgb_config("xgb_eth_1h",   "ETH-USD", "1h",  asset="crypto"))
write("xgb_qqq_1d",   xgb_config("xgb_qqq_1d",   "QQQ",     "1d"))
write("xgb_spy_d10",  xgb_config("xgb_spy_d10",  "SPY",     "1d",  max_depth=10))
write("xgb_spy_n1k",  xgb_config("xgb_spy_n1k",  "SPY",     "1d",  n_estimators=1000))

# ─── GROUP G: Ensemble Ablations ────────────────────────────────────────────
print("\n[G] Ensemble ablations")

write("ensemble_btc_1h",     ensemble_config("ensemble_btc_1h",     "BTC-USD", "1h",  asset="crypto"))
write("ensemble_qqq_1d",     ensemble_config("ensemble_qqq_1d",     "QQQ",     "1d"))
write("ensemble_no_mst",     ensemble_config("ensemble_no_mst",     "SPY",     "1d",
                                              components=["patch_tst", "xgboost", "lorentzian_knn"]))
write("ensemble_2comp",      ensemble_config("ensemble_2comp",      "SPY",     "1d",
                                              components=["patch_tst", "xgboost"]))
write("ensemble_3comp_lstm", ensemble_config("ensemble_3comp_lstm", "SPY",     "1d",
                                              components=["patch_tst", "lstm", "xgboost"]))

print("\nDone!")

# Count total configs
all_configs = list(CONFIGS_DIR.glob("*.yaml"))
print(f"Total experiment configs in {CONFIGS_DIR.relative_to(Path.cwd())}: {len(all_configs)}")
for c in sorted(all_configs):
    print(f"  {c.stem}")
