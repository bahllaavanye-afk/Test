"""Rule-based reasoning engine — zero-API-key autonomous analysis.

When no free LLM provider is configured this engine reads the numeric/status
context produced by the agent dispatcher and applies domain heuristics to emit
structured {agent, analysis, recommendations} output — the same shape the
gateway returns, so callers need no special-casing.

This makes all autonomous loops functional by default with no configuration.
LLM providers (Groq/DeepSeek/Gemini) are treated as an *enhancement* that
adds language-model depth when keys are present; the core reasoning always runs.
"""
from __future__ import annotations


def rule_reason(task_type: str, context: dict) -> dict:
    """Produce a structured analysis dict from task context using only heuristics.

    Returns {"agent", "analysis", "recommendations", "source": "rule_engine"}.
    """
    handler = _HANDLERS.get(task_type, _generic_handler)
    return handler(context)


# ── Task handlers ──────────────────────────────────────────────────────────


def _evaluate_strategies(ctx: dict) -> dict:
    poor = ctx.get("poor_performers", [])
    total = ctx.get("total_strategies_evaluated", 0)
    threshold = 0.3

    if not poor:
        analysis = (
            f"All {total} evaluated strategies are above the Sharpe threshold of {threshold}. "
            "Portfolio health looks acceptable; continue monitoring for regime changes."
        )
        recs = [
            "Continue paper-trading all active strategies for another week.",
            "Re-evaluate once daily trade count exceeds 20 per strategy.",
        ]
    else:
        names = ", ".join(p.get("strategy_id", "?") for p in poor[:3])
        worst = min(poor, key=lambda p: p.get("sharpe", 0))
        analysis = (
            f"{len(poor)} of {total} strategies underperform (Sharpe < {threshold}): {names}. "
            f"Worst offender: {worst.get('strategy_id', '?')} at Sharpe {worst.get('sharpe', 0):.2f} "
            f"over {worst.get('trades', 0)} trades. Underperformers drag portfolio Sharpe."
        )
        recs = [
            f"Disable or reduce allocation to {worst.get('strategy_id', 'worst strategy')} immediately.",
            "Inspect signal quality for underperformers — check for feature staleness.",
            "Run walk-forward backtest on remaining strategies to confirm out-of-sample validity.",
        ]
        if len(poor) >= 3:
            recs.append("Consider switching underperforming slots to arbitrage bucket to preserve capital.")

    return {
        "agent": "strategy_agent",
        "analysis": analysis,
        "recommendations": recs[:4],
        "source": "rule_engine",
    }


def _risk_check(ctx: dict) -> dict:
    regime = ctx.get("regime")
    status = ctx.get("status", "ok")

    if regime and isinstance(regime, dict):
        state = regime.get("state", regime.get("regime", "unknown"))
        label = {0: "bear", 1: "sideways", 2: "bull"}.get(state, str(state))
    elif isinstance(regime, (int, float)):
        label = {0: "bear", 1: "sideways", 2: "bull"}.get(int(regime), "unknown")
    elif isinstance(regime, str):
        label = regime
    else:
        label = "unknown"

    if label == "bear":
        analysis = (
            "HMM regime detector signals BEAR market. High-volatility, negative-drift environment. "
            "Directional strategies carry elevated drawdown risk; arbitrage bucket should dominate."
        )
        recs = [
            "Suspend all directional (momentum/breakout) strategies immediately.",
            "Shift 100% of new capital to arbitrage bucket until regime flips to sideways or bull.",
            "Tighten stop-losses on any remaining directional positions to 1.5×ATR.",
            "Review circuit-breaker thresholds — reduce max drawdown tolerance to 5%.",
        ]
    elif label == "bull":
        analysis = (
            "Regime is BULL: low volatility, positive drift. Momentum and breakout strategies "
            "historically outperform in this regime. Risk posture can be standard."
        )
        recs = [
            "Allow full 30/70 capital split (directional / arbitrage).",
            "Activate cross-sectional momentum strategy across top-50 equities.",
            "Monitor for regime flip — re-evaluate each 5-minute cycle.",
        ]
    elif label == "sideways":
        analysis = (
            "Regime is SIDEWAYS: low volatility, near-zero drift. Mean-reversion and stat-arb "
            "strategies are favoured. Momentum strategies should be underweighted."
        )
        recs = [
            "Increase allocation to mean-reversion and pairs-trading strategies.",
            "Reduce momentum strategy size to 25% of normal allocation.",
            "VWAP-reversion and intraday mean-reversion are best suited for this regime.",
        ]
    else:
        analysis = (
            f"Regime is UNKNOWN (no HMM signal yet, Redis key 'market:regime' may be unset). "
            "Defaulting to conservative posture until regime detection completes."
        )
        recs = [
            "Run RegimeMonitor.run() to fit HMM on SPY returns and seed Redis key.",
            "Default to 80/20 arb/directional split until regime is established.",
        ]

    return {
        "agent": "risk_agent",
        "analysis": analysis,
        "recommendations": recs[:4],
        "source": "rule_engine",
    }


def _alpha_mining(ctx: dict) -> dict:
    factors_found = ctx.get("factors_found", 0)
    symbols = ctx.get("symbols", ["SPY"])

    if factors_found > 0:
        analysis = (
            f"Alpha miner produced {factors_found} passing factor(s) across {len(symbols)} symbol(s). "
            "New formulaic signals have cleared IC > 0.02 and IR > 0.30 validation gates. "
            "Add to the ML feature pipeline for live testing."
        )
        recs = [
            "Incorporate passing factors into ml/features/engineer.py for next retraining run.",
            "Run walk-forward validation on factors before live use — confirm IC is stable.",
            "Compare IC decay curve: if IC drops to zero within 5 days, factor may be overfit.",
        ]
    else:
        analysis = (
            "Alpha mining cycle ran using built-in factor library (no external LLM call needed). "
            f"No new factors passed validation on {', '.join(symbols[:3])}. "
            "Built-in factors remain the baseline signal set."
        )
        recs = [
            "Set GROQ_API_KEY or GEMINI_API_KEY to enable LLM-proposed factor generation.",
            "Try different symbols or extend the data window (currently 2 years) for more signal.",
            "Review IC/IR of existing built-in factors — disable any with |IC| < 0.01.",
        ]

    return {
        "agent": "research_agent",
        "analysis": analysis,
        "recommendations": recs[:4],
        "source": "rule_engine",
    }


def _slippage_analysis(ctx: dict) -> dict:
    avg_slippage_bps = ctx.get("avg_slippage_bps")
    algo_breakdown = ctx.get("algo_breakdown", {})

    if avg_slippage_bps is None:
        analysis = (
            "No slippage data available yet — platform is in early paper-trading phase "
            "with insufficient fills to compute realized implementation shortfall. "
            "This is expected in the first weeks of operation."
        )
        recs = [
            "Ensure SlippageTracker records are written on every fill via order_sync.py.",
            "Once 50+ fills are recorded, slippage analysis will produce actionable numbers.",
            "Default to LimitFirst execution to minimise early paper-trading slippage.",
        ]
    elif float(avg_slippage_bps) > 15:
        analysis = (
            f"Average slippage of {avg_slippage_bps:.1f} bps is HIGH (target < 10 bps). "
            "This level of implementation shortfall significantly erodes strategy returns. "
            "Execution algorithm selection appears suboptimal for current order sizes."
        )
        recs = [
            "Switch to TWAP for any order > $5,000 to reduce market impact.",
            "Increase LimitFirst timeout from 30s to 60s before falling back to market.",
            "Route crypto orders through DEX-CEX arb path if spread allows.",
            "Audit broker routing — PFOF brokers may be internalising against your flow.",
        ]
    elif float(avg_slippage_bps) > 8:
        analysis = (
            f"Average slippage of {avg_slippage_bps:.1f} bps is MODERATE. "
            "Performance is acceptable but there is room for improvement, especially "
            f"in {max(algo_breakdown, key=lambda k: algo_breakdown[k], default='market')} orders."
        )
        recs = [
            "Test VWAP participation rate of 8% (down from 10%) for mid-size orders.",
            "Increase iceberg slice size for large orders to reduce queue priority loss.",
        ]
    else:
        analysis = (
            f"Average slippage of {avg_slippage_bps:.1f} bps is within target (< 8 bps). "
            "Execution quality is strong. LimitFirst and TWAP are working as designed."
        )
        recs = [
            "Maintain current execution algorithm mix.",
            "Continue tracking slippage by broker to detect PFOF degradation.",
        ]

    return {
        "agent": "execution_agent",
        "analysis": analysis,
        "recommendations": recs[:4],
        "source": "rule_engine",
    }


def _fetch_ohlcv(ctx: dict) -> dict:
    cached = ctx.get("symbols_in_cache", [])
    total_syms = ctx.get("total_symbols", max(len(cached), 4))
    coverage = len(cached) / max(total_syms, 1)

    if coverage >= 0.9:
        analysis = (
            f"Price cache coverage is healthy: {len(cached)}/{total_syms} symbols cached in Redis. "
            "Data latency is within normal bounds for strategy execution."
        )
        recs = [
            "Data pipeline healthy — no action required.",
            "Verify TTL on Redis keys matches strategy tick interval to avoid stale reads.",
        ]
    elif coverage >= 0.5:
        missing = total_syms - len(cached)
        analysis = (
            f"Partial price cache: {len(cached)}/{total_syms} symbols available. "
            f"{missing} symbols are missing — strategies that depend on them will skip this cycle."
        )
        recs = [
            "Check broker connection status — Alpaca/Binance WebSocket may have disconnected.",
            "Restart price_feed.py task if symbols are consistently missing.",
            "Add missing symbols to the fallback yfinance poller as backup data source.",
        ]
    else:
        analysis = (
            f"Price cache is critically sparse: only {len(cached)}/{total_syms} symbols available. "
            "Most strategy cycles will be skipped due to missing data. Immediate attention required."
        )
        recs = [
            "Check ALPACA_API_KEY and BINANCE_API_KEY are set in Render environment.",
            "Verify Redis connection (REDIS_URL) — all cached prices may have expired.",
            "Restart the quantedge-api service to reinitialise the price feed.",
        ]

    return {
        "agent": "data_agent",
        "analysis": analysis,
        "recommendations": recs[:4],
        "source": "rule_engine",
    }


def _evaluate_models(ctx: dict) -> dict:
    accuracy = ctx.get("accuracy")
    models_found = ctx.get("models_found", 0)
    stale_models = ctx.get("stale_models", [])

    if models_found == 0:
        analysis = (
            "No trained ML models found in models_artifacts/. The platform is running "
            "in rule-based mode only. ML-enhanced strategies will fall back to their "
            "manual counterparts until models are trained and registered."
        )
        recs = [
            "Run experiments/run_experiment.py --config lstm_btc_1h.yaml to train first model.",
            "Upload training notebooks to Kaggle/Colab for free GPU training.",
            "Use XGBoost model first — fastest to train, no GPU required.",
        ]
    elif stale_models:
        analysis = (
            f"Found {models_found} model(s), but {len(stale_models)} are stale (trained > 30 days ago). "
            "Stale models carry increased model risk as market conditions drift from training data."
        )
        recs = [
            f"Retrain: {', '.join(str(m) for m in stale_models[:3])} — data drift likely after 30 days.",
            "Schedule nightly retraining in ml_retrain.py for all production models.",
            "Run walk-forward backtest before deploying retrained model to production.",
        ]
    elif accuracy is not None and float(accuracy) < 0.55:
        analysis = (
            f"Model accuracy ({float(accuracy):.1%}) is below the 55% signal threshold. "
            "Below-chance accuracy suggests model degradation or feature distribution shift."
        )
        recs = [
            "Retrain model with 6 months of fresh data before next production cycle.",
            "Re-run feature importance — drop features with near-zero SHAP contribution.",
            "Try reducing model complexity (fewer layers) to combat overfitting.",
        ]
    else:
        analysis = (
            f"ML models are healthy: {models_found} model(s) loaded, "
            f"accuracy {f'{float(accuracy):.1%}' if accuracy else 'not measured'}. "
            "Ensemble is ready for signal generation."
        )
        recs = [
            "Continue monitoring model calibration — accuracy alone can mask confidence miscalibration.",
            "Run debug_signal_quality.py monthly to track IC decay.",
        ]

    return {
        "agent": "ml_agent",
        "analysis": analysis,
        "recommendations": recs[:4],
        "source": "rule_engine",
    }


def _generic_handler(ctx: dict) -> dict:
    task_type = ctx.get("task_type", "unknown")
    status = ctx.get("status", "ok")
    message = ctx.get("message", "Task completed.")

    analysis = (
        f"Task '{task_type}' completed with status '{status}'. {message} "
        "No specialist heuristics available for this task type — review the result manually."
    )
    recs = [
        f"Review raw output for '{task_type}' in agent logs.",
        "Add domain-specific heuristics to app/llm/rule_engine.py for richer analysis.",
    ]

    return {
        "agent": "strategy_agent",
        "analysis": analysis,
        "recommendations": recs,
        "source": "rule_engine",
    }


_HANDLERS = {
    "evaluate_strategies": _evaluate_strategies,
    "risk_check": _risk_check,
    "alpha_mining": _alpha_mining,
    "slippage_analysis": _slippage_analysis,
    "fetch_ohlcv": _fetch_ohlcv,
    "evaluate_models": _evaluate_models,
}
