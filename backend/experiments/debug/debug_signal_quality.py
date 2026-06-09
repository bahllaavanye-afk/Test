"""
Signal quality diagnostics: IC, IR, and SHAP feature importance for ML models.
Usage: python experiments/debug/debug_signal_quality.py --model lstm_btc_1h_v3
       python experiments/debug/debug_signal_quality.py --model xgb_spy_daily_v2 --shap
Output: IC/IR per feature printed to stdout; optional SHAP bar chart saved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT       = Path(__file__).parents[2]
ARTIFACTS  = ROOT.parent / "models_artifacts"
RESULTS    = ROOT / "experiments" / "results"
DEBUG_DIR  = Path(__file__).parent


# ─── IC / IR ──────────────────────────────────────────────────────────────────

def compute_ic(predictions: np.ndarray, returns: np.ndarray) -> float:
    """Pearson Information Coefficient between predicted probability and next-bar return."""
    if len(predictions) != len(returns) or len(predictions) < 2:
        return float("nan")
    corr = np.corrcoef(predictions, returns)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_rolling_ic(
    predictions: np.ndarray,
    returns: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Rolling IC with given window."""
    n = len(predictions)
    ics = []
    for i in range(window, n):
        p = predictions[i - window : i]
        r = returns[i - window : i]
        if np.std(p) < 1e-9 or np.std(r) < 1e-9:
            ics.append(0.0)
        else:
            ics.append(np.corrcoef(p, r)[0, 1])
    return np.array(ics)


def print_ic_report(result: dict) -> None:
    """Print IC/IR statistics from a saved result file."""
    predictions = result.get("test_predictions")
    returns     = result.get("test_returns")

    if not predictions or not returns:
        print("No test_predictions / test_returns in result file — skipping IC analysis.")
        return

    preds = np.array(predictions)
    rets  = np.array(returns)

    ic = compute_ic(preds, rets)
    rolling = compute_rolling_ic(preds, rets, window=20)
    ir = rolling.mean() / (rolling.std() + 1e-9)

    print(f"\n{'─'*50}")
    print(f"  IC  (Pearson, full test set) : {ic:+.4f}")
    print(f"  IC  (rolling 20, mean)        : {rolling.mean():+.4f}")
    print(f"  IC  (rolling 20, std)         : {rolling.std():.4f}")
    print(f"  IR  (IC mean / IC std)        : {ir:+.4f}")
    print(f"{'─'*50}")

    if abs(ic) < 0.02:
        print("⚠  IC < 0.02 — signal has weak predictive value. Consider retraining.")
    elif abs(ic) >= 0.05:
        print("✓  IC ≥ 0.05 — signal passes the quality gate.")
    else:
        print("○  IC in 0.02–0.05 range — borderline; monitor live performance.")

    if ir < 0.3:
        print("⚠  IR < 0.3 — inconsistent IC across time. May be regime-dependent.")
    else:
        print("✓  IR ≥ 0.3 — consistent signal quality.")


# ─── Feature importance (SHAP or built-in) ───────────────────────────────────

def plot_shap(model_name: str, artifact_dir: Path, out_path: Path) -> None:
    """Load XGBoost model and run SHAP on saved test data if available."""
    try:
        import xgboost as xgb
    except ImportError:
        print("xgboost not installed — skipping SHAP analysis.")
        return

    model_path = artifact_dir / f"{model_name}.json"
    meta_path  = artifact_dir / f"{model_name}_meta.json"
    data_path  = artifact_dir / f"{model_name}_test_X.npy"

    if not model_path.exists():
        print(f"Model file not found: {model_path} — skipping SHAP.")
        return
    if not data_path.exists():
        print(f"Test data not found: {data_path} — skipping SHAP (no X_test saved).")
        return

    try:
        import shap
    except ImportError:
        print("shap not installed — pip install shap — skipping SHAP.")
        return

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    X_test = np.load(str(data_path))
    features = (
        json.loads(meta_path.read_text()).get("features", [f"f{i}" for i in range(X_test.shape[1])])
        if meta_path.exists()
        else [f"f{i}" for i in range(X_test.shape[1])]
    )

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:500])

    mean_abs     = np.abs(shap_values).mean(axis=0)
    fi_df        = pd.DataFrame({"feature": features, "importance": mean_abs})
    fi_df        = fi_df.sort_values("importance", ascending=True)

    print("\nTop-10 SHAP Feature Importances:")
    print(fi_df.tail(10).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.4)))
        fig.patch.set_facecolor("#131722")
        ax.set_facecolor("#1e2433")
        ax.barh(fi_df["feature"], fi_df["importance"], color="#2196F3")
        ax.set_title(f"SHAP Feature Importance — {model_name}", color="white")
        ax.set_xlabel("Mean |SHAP|", color="#888")
        ax.tick_params(colors="#888")
        ax.spines[:].set_color("#2a2a2a")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"SHAP plot saved: {out_path}")
    except Exception as e:
        print(f"Matplotlib error ({e}) — skipping plot, printed table above.")


def print_feature_ic(model_name: str, artifact_dir: Path) -> None:
    """Compute per-feature IC using saved test features and returns."""
    X_path = artifact_dir / f"{model_name}_test_X.npy"
    y_path = artifact_dir / f"{model_name}_test_y.npy"
    meta_p = artifact_dir / f"{model_name}_meta.json"

    if not X_path.exists() or not y_path.exists():
        print(f"Test arrays not found in {artifact_dir} — skipping per-feature IC.")
        return

    X_test   = np.load(str(X_path))
    y_test   = np.load(str(y_path))
    features = (
        json.loads(meta_p.read_text()).get("features", [f"f{i}" for i in range(X_test.shape[1])])
        if meta_p.exists()
        else [f"f{i}" for i in range(X_test.shape[1])]
    )

    print("\nPer-feature IC (correlation with next-bar return):")
    print(f"{'Feature':<30} {'IC':>8}")
    print("─" * 40)
    ics = []
    for i, feat in enumerate(features):
        ic = compute_ic(X_test[:, i], y_test.astype(float))
        ics.append((feat, ic))

    for feat, ic in sorted(ics, key=lambda x: abs(x[1]), reverse=True):
        flag = " ✓" if abs(ic) >= 0.02 else ""
        print(f"  {feat:<28} {ic:+.4f}{flag}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="IC/IR and SHAP signal quality diagnostics.")
    ap.add_argument("--model",  required=True, help="Experiment / model name (e.g. lstm_btc_1h_v3)")
    ap.add_argument("--shap",   action="store_true", help="Run SHAP analysis (XGBoost only)")
    ap.add_argument("--out",    default=None, help="Output PNG path for SHAP chart")
    args = ap.parse_args()

    model_name   = args.model
    artifact_dir = ARTIFACTS / model_name
    result_path  = RESULTS / f"{model_name}.json"
    out_path     = Path(args.out) if args.out else DEBUG_DIR / f"shap_{model_name}.png"

    print(f"Signal quality diagnostics — model: {model_name}")

    # IC/IR from result file
    if result_path.exists():
        result = json.loads(result_path.read_text())
        print_ic_report(result)
    else:
        print(f"No result file at {result_path} — skipping IC report.")

    # Per-feature IC from saved test arrays
    if artifact_dir.exists():
        print_feature_ic(model_name, artifact_dir)
        if args.shap:
            plot_shap(model_name, artifact_dir, out_path)
    else:
        print(f"Artifact directory {artifact_dir} not found — no feature analysis.")


if __name__ == "__main__":
    main()
