"""
Debug overfitting: plot train vs val loss curves for a given experiment.
Usage: python experiments/debug/debug_overfitting.py --config lstm_btc_1h.yaml [--epochs 50]
Output: experiments/debug/overfitting_<name>.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[2]
RESULTS_DIR = ROOT / "experiments" / "results"
DEBUG_DIR = Path(__file__).parent


def _load_history(experiment_name: str) -> list[dict]:
    """Load metrics_history from experiments/results/<name>.json."""
    result_path = RESULTS_DIR / f"{experiment_name}.json"
    if not result_path.exists():
        sys.exit(f"No results file found at {result_path}. Run the experiment first.")
    data = json.loads(result_path.read_text())
    history = data.get("metrics_history", [])
    if not history:
        sys.exit(
            f"No metrics_history in {result_path}. "
            "Ensure the trainer writes per-epoch metrics."
        )
    return history


def plot_loss_curves(history: list[dict], experiment_name: str, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")

    epochs     = [h["epoch"]      for h in history]
    train_loss = [h.get("train_loss", h.get("loss", None)) for h in history]
    val_loss   = [h.get("val_loss",   h.get("val_loss", None)) for h in history]

    if not any(v is not None for v in train_loss):
        sys.exit("No 'train_loss' key found in metrics_history entries.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#131722")

    # --- Loss curve ---
    ax = axes[0]
    ax.set_facecolor("#1e2433")
    if any(v is not None for v in train_loss):
        ax.plot(epochs, train_loss, color="#f5a623", linewidth=2, label="Train loss")
    if any(v is not None for v in val_loss):
        ax.plot(epochs, val_loss, color="#2196F3", linewidth=2, label="Val loss")
    ax.set_title(f"Loss — {experiment_name}", color="white", fontsize=13)
    ax.set_xlabel("Epoch", color="#888")
    ax.set_ylabel("Loss", color="#888")
    ax.tick_params(colors="#888")
    ax.legend(facecolor="#1e2433", labelcolor="white")
    ax.spines[:].set_color("#2a2a2a")

    # Annotate divergence if val_loss > 2× min val_loss after first 20% of epochs
    if val_loss and any(v is not None for v in val_loss):
        valid_val = [(e, v) for e, v in zip(epochs, val_loss) if v is not None]
        if valid_val:
            min_val    = min(v for _, v in valid_val)
            cutoff_ep  = epochs[len(epochs) // 5]
            diverged   = [(e, v) for e, v in valid_val if e > cutoff_ep and v > min_val * 1.5]
            if diverged:
                ax.axvline(x=diverged[0][0], color="#ff1744", linestyle="--", alpha=0.7)
                ax.text(
                    diverged[0][0], ax.get_ylim()[1] * 0.9,
                    "Diverge", color="#ff1744", fontsize=9,
                )

    # --- Accuracy / AUC curve ---
    ax2 = axes[1]
    ax2.set_facecolor("#1e2433")
    train_acc = [h.get("train_acc", h.get("train_auc", None)) for h in history]
    val_acc   = [h.get("val_acc",   h.get("val_auc",   None)) for h in history]
    has_acc   = any(v is not None for v in train_acc) or any(v is not None for v in val_acc)

    if has_acc:
        if any(v is not None for v in train_acc):
            ax2.plot(epochs, train_acc, color="#f5a623", linewidth=2, label="Train acc/AUC")
        if any(v is not None for v in val_acc):
            ax2.plot(epochs, val_acc, color="#2196F3", linewidth=2, label="Val acc/AUC")
        ax2.set_title(f"Accuracy/AUC — {experiment_name}", color="white", fontsize=13)
        ax2.legend(facecolor="#1e2433", labelcolor="white")
    else:
        # Compute IS/OOS loss ratio as overfitting indicator
        ratio = [
            tl / (vl + 1e-9) if tl is not None and vl is not None else None
            for tl, vl in zip(train_loss, val_loss)
        ]
        ax2.plot(
            epochs, [r for r in ratio if r is not None],
            color="#00c853", linewidth=2, label="Train/Val loss ratio",
        )
        ax2.axhline(y=1.0, color="#888", linestyle="--", alpha=0.5)
        ax2.set_title(f"IS/OOS ratio — {experiment_name}", color="white", fontsize=13)
        ax2.legend(facecolor="#1e2433", labelcolor="white")

    ax2.set_xlabel("Epoch", color="#888")
    ax2.tick_params(colors="#888")
    ax2.spines[:].set_color("#2a2a2a")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")

    # Print summary
    if val_loss and any(v is not None for v in val_loss):
        valid_vals = [v for v in val_loss if v is not None]
        min_val    = min(valid_vals)
        last_val   = valid_vals[-1]
        gap        = (last_val - min_val) / (min_val + 1e-9)
        print(f"\nMin val loss : {min_val:.4f}")
        print(f"Final val loss: {last_val:.4f}")
        print(f"Gap (final / min - 1): {gap:.2%}")
        if gap > 0.15:
            print("⚠  Val loss rose >15% above minimum — likely overfitting.")
            print("   Suggestions: increase dropout, reduce capacity, add L2, or use early stopping.")
        else:
            print("✓  No significant overfitting detected.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot train/val loss curves for overfitting diagnosis.")
    ap.add_argument("--config",     required=True, help="Experiment name or config filename (without .yaml)")
    ap.add_argument("--out",        default=None,  help="Output PNG path (default: debug/overfitting_<name>.png)")
    args = ap.parse_args()

    experiment_name = args.config.replace(".yaml", "").replace("configs/", "")
    out_path = Path(args.out) if args.out else DEBUG_DIR / f"overfitting_{experiment_name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    history = _load_history(experiment_name)
    print(f"Loaded {len(history)} epochs of metrics for '{experiment_name}'")
    plot_loss_curves(history, experiment_name, out_path)


if __name__ == "__main__":
    main()
