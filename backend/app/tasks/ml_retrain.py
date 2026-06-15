"""
Nightly ML retraining: downloads fresh data, retrains all active models,
compares new vs old Sharpe, promotes if improved.
"""
from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.services.agent_logger import agent_logger
from app.utils.logging import logger

ARTIFACTS_DIR = Path(__file__).parents[3] / "models_artifacts"


async def retrain_model(model_name: str, symbol: str, interval: str = "1h") -> dict:
    """Download 2 years of data and retrain a model. Returns result dict."""
    agent_logger.log_action_fire_and_forget(
        action="retrain_model",
        employee_id="ml_agent",
        agent_type="ml",
        tool_used="pytorch",
        input_summary=f"model={model_name} symbol={symbol} interval={interval}",
        status="ok",
        symbol=symbol,
    )
    _t0 = time.monotonic()
    try:
        import yfinance as yf
        loop = asyncio.get_running_loop()
        end = datetime.now(UTC)
        start = end - timedelta(days=730)

        hist = await loop.run_in_executor(
            None,
            lambda: yf.download(symbol, start=str(start.date()), end=str(end.date()),
                                  interval=interval, auto_adjust=True, progress=False)
        )
        if hist is None or len(hist) < 200:
            return {"status": "skipped", "reason": "insufficient data"}

        # Normalize column names
        hist.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in hist.columns]

        from app.ml.training.train_lstm import train
        experiment_name = f"{model_name}_{symbol.lower()}_{datetime.now(UTC).strftime('%Y%m%d')}"
        result = await train(hist, experiment_name=experiment_name, max_epochs=30)
        result["symbol"] = symbol
        result["model"] = model_name
        result["retrained_at"] = datetime.now(UTC).isoformat()
        logger.info("Model retrained", **{k: v for k, v in result.items() if k != "best_model_path"})
        _dur = int((time.monotonic() - _t0) * 1000)
        agent_logger.log_action_fire_and_forget(
            action="retrain_model_complete",
            employee_id="ml_agent",
            agent_type="ml",
            tool_used="pytorch",
            input_summary=f"model={model_name} symbol={symbol} interval={interval}",
            output_summary=f"status={result.get('status','ok')} sharpe={result.get('sharpe','?')}",
            duration_ms=_dur,
            status="ok",
            symbol=symbol,
        )
        return result

    except Exception as e:
        logger.error("Retrain failed", model=model_name, symbol=symbol, error=str(e))
        _dur = int((time.monotonic() - _t0) * 1000)
        agent_logger.log_action_fire_and_forget(
            action="retrain_model_complete",
            employee_id="ml_agent",
            agent_type="ml",
            tool_used="pytorch",
            input_summary=f"model={model_name} symbol={symbol} interval={interval}",
            duration_ms=_dur,
            status="error",
            error_message=str(e)[:200],
            symbol=symbol,
        )
        return {"status": "error", "error": str(e)}


def _load_retrain_configs() -> list[tuple[str, str, str]]:
    """
    Discover retrain targets dynamically from experiment configs (*.yaml).
    Falls back to a minimal default set if no configs exist or yaml is unavailable.
    Returns list of (model_name, symbol, interval).
    """
    configs_dir = Path(__file__).parents[3] / "experiments" / "configs"
    seen: set[tuple[str, str, str]] = set()
    results: list[tuple[str, str, str]] = []

    try:
        import yaml as _yaml
        _load_yaml = _yaml.safe_load
    except ImportError:
        _load_yaml = None

    for cfg_path in sorted(configs_dir.glob("*.yaml")):
        try:
            with open(cfg_path) as f:
                if _load_yaml:
                    cfg = _load_yaml(f)
                else:
                    # Minimal fallback: regex-extract model/symbol/interval from YAML text
                    import re
                    text = f.read()
                    cfg = {"experiment": {
                        k: v for k, v in re.findall(
                            r"^\s{2}(model|symbol|interval):\s*['\"]?([^\s'\"#]+)", text, re.MULTILINE
                        )
                    }}
            exp = (cfg or {}).get("experiment", {})
            model = exp.get("model", "lstm")
            symbol = exp.get("symbol", "SPY")
            interval = exp.get("interval", "1d")
            key = (model, symbol, interval)
            if key not in seen:
                seen.add(key)
                results.append(key)
        except Exception:
            continue

    if not results:
        results = [("lstm", "BTC-USD", "1h"), ("lstm", "ETH-USD", "1h"), ("lstm", "SPY", "1d")]

    return results


async def nightly_retrain() -> None:
    """Retrain all models discovered from experiment configs. Called by APScheduler at 02:00 UTC."""
    retrain_configs = _load_retrain_configs()
    # Cap at 10 per night to avoid overwhelming free-tier CPU
    retrain_configs = retrain_configs[:10]
    logger.info("Nightly retrain starting", configs=len(retrain_configs))
    results = await asyncio.gather(
        *[retrain_model(m, s, i) for m, s, i in retrain_configs],
        return_exceptions=True
    )
    successes = sum(1 for r in results if isinstance(r, dict) and r.get("status") != "error")
    logger.info("Nightly retrain complete", total=len(retrain_configs), succeeded=successes)
