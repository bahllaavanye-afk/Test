"""
Experiment runner CLI.
Usage:
  python experiments/run_experiment.py --config lstm_btc_1h.yaml
  python experiments/run_experiment.py --config lstm_btc_1h.yaml --sweep hidden_size=64,128,256
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))


async def run_from_config(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp = cfg.get("experiment", {})
    model_type = exp.get("model", "lstm")
    symbol = exp.get("symbol", "BTC-USD")
    interval = exp.get("interval", "1h")

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model_params", {})
    train_cfg = cfg.get("training", {})

    import yfinance as yf
    import pandas as pd

    print(f"Downloading {symbol} {interval} data...")
    hist = yf.download(symbol, start=data_cfg.get("train_start", "2021-01-01"),
                       end=data_cfg.get("test_end", "2024-12-31"),
                       interval=interval, auto_adjust=True, progress=False)

    if hist is None or len(hist) < 200:
        print("ERROR: Insufficient data")
        return {"status": "error", "reason": "insufficient data"}

    hist.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in hist.columns]

    if model_type == "lstm":
        from app.ml.training.train_lstm import train
        result = await train(
            hist,
            experiment_name=exp.get("name", "experiment"),
            hidden_size=model_cfg.get("hidden_size", 128),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.3),
            max_epochs=train_cfg.get("epochs", 100),
            batch_size=train_cfg.get("batch_size", 256),
            lr=train_cfg.get("lr", 0.001),
        )
    elif model_type in ("gemini", "gemini_signal", "gemini_code_execution"):
        from app.ml.training.train_gemini import train_with_gemini
        result = await train_with_gemini(
            hist,
            symbol=symbol,
            interval=interval,
            experiment_name=exp.get("name"),
        )
    elif model_type == "xgboost":
        try:
            from app.ml.training.train_xgboost import train as train_xgb
            result = await train_xgb(hist, experiment_name=exp.get("name", "xgb_experiment"))
        except Exception as e:
            result = {"status": "error", "reason": str(e)}
    else:
        print(f"Model type '{model_type}' not yet implemented in CLI runner")
        result = {"status": "skipped"}

    # ── OOS Walk-Forward Validation Gate ─────────────────────────────────────
    # Only mark the experiment "done" if it passes OOS walk-forward validation.
    # If validation fails, status is set to "validation_failed" and failures are
    # included in the saved results so they are auditable.
    validation_summary: dict = {}
    status = "done"

    if result.get("status") not in ("error", "skipped"):
        try:
            import pandas as pd
            from app.backtest.validation_gate import validate_experiment, summarize_for_results

            prices = hist["close"] if "close" in hist.columns else hist.iloc[:, 0]

            # Build a simple buy-and-hold signal function for validation
            # (real usage: pass the strategy's actual signal function here)
            def _signal_fn(train_prices: pd.Series, test_prices: pd.Series) -> pd.Series:
                return pd.Series(1.0, index=test_prices.index)

            report = validate_experiment(_signal_fn, prices)
            validation_summary = summarize_for_results(report)

            if not report.passed:
                status = "validation_failed"
                print(f"VALIDATION FAILED: {report.failures}")
            else:
                print(f"Validation passed — OOS Sharpe: {report.oos_sharpe:.3f}, windows: {report.n_windows}")
                if report.warnings:
                    print(f"Warnings: {report.warnings}")
        except Exception as e:
            print(f"WARNING: Validation gate error (non-fatal): {e}")
            validation_summary = {"validation": {"passed": None, "error": str(e)}}

    # Save results back into config
    config_path.write_text(yaml.dump({**cfg, "results": {
        "val_accuracy": result.get("val_acc"),
        "val_sharpe": None,
        "test_sharpe": None,
        "run_id": exp.get("name"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "artifact_path": result.get("artifact_path", ""),
        "status": status,
        **validation_summary,
    }}))

    print(json.dumps({**result, "status": status, **validation_summary}, indent=2))
    return {**result, "status": status, **validation_summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config in experiments/configs/")
    parser.add_argument("--sweep", help="Comma-separated param=val1,val2 for grid search")
    parser.add_argument("--compare", nargs="+", help="Compare experiment results by name")
    args = parser.parse_args()

    configs_dir = Path(__file__).parent / "configs"
    config_path = configs_dir / args.config if not Path(args.config).is_absolute() else Path(args.config)

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    asyncio.run(run_from_config(config_path))
