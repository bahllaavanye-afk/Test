"""Employee agents: each role reasons in its own domain through the shared gateway.

This is the "engineer-wise" division of labour. The platform already groups these
engineers into desks/teams (see app/tasks/org/desk_registry.py); here each one gets
a job-description system prompt and reasons about the tasks assigned to it.

Honesty rule: when no free LLM provider is configured the helper returns
``{"llm": "unavailable"}`` — it never fabricates analysis.
"""
from __future__ import annotations

import json

from app.llm.gateway import complete

# role -> (agent name for budgeting, job-description system prompt)
ROLE_PROMPTS: dict[str, tuple[str, str]] = {
    "strategy_agent": (
        "strategy_agent",
        "You are the Strategy Engineer at a quant trading firm. You review strategy "
        "performance and recommend which strategies to keep, tune, or disable. Be "
        "concrete and risk-aware. Never invent numbers — reason only from the data given.",
    ),
    "research_agent": (
        "research_agent",
        "You are the Quant Researcher. You propose and critique alpha ideas and explain "
        "why a factor might carry signal or be spurious. Be skeptical of overfitting.",
    ),
    "risk_agent": (
        "risk_agent",
        "You are the Risk Manager. You assess portfolio risk, regime, and circuit-breaker "
        "posture. Flag concentration, drawdown, and correlation risks. Be conservative.",
    ),
    "execution_agent": (
        "execution_agent",
        "You are the Execution Engineer. You analyse slippage and recommend execution-algo "
        "changes (TWAP/VWAP/limit-first) to reduce implementation shortfall.",
    ),
    "data_agent": (
        "data_agent",
        "You are the Data Engineer. You assess data freshness/coverage gaps and recommend "
        "what to backfill or which feeds to repair.",
    ),
    "ml_agent": (
        "ml_agent",
        "You are the ML Engineer. You evaluate model accuracy/calibration and recommend "
        "retraining, feature changes, or ensemble reweighting. Flag stale models.",
    ),
}

# task_type -> role responsible (the org chart for autonomous work)
TASK_ROUTING: dict[str, str] = {
    "evaluate_strategies": "strategy_agent",
    "alpha_mining": "research_agent",
    "risk_check": "risk_agent",
    "slippage_analysis": "execution_agent",
    "fetch_ohlcv": "data_agent",
    "evaluate_models": "ml_agent",
}


async def reason_about_task(task_type: str, context: dict, max_tokens: int = 400) -> dict:
    """Have the responsible employee agent reason about a task's computed result.

    ``context`` is the rule-based result already computed by the dispatcher.
    Returns {"agent", "analysis", "recommendations"} or {"llm": "unavailable"}.
    """
    role = TASK_ROUTING.get(task_type, "strategy_agent")
    agent_name, system_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS["strategy_agent"])

    user_prompt = (
        f"Task type: {task_type}\n"
        f"Computed result (ground truth — do not contradict):\n"
        f"{json.dumps(context, default=str)[:2000]}\n\n"
        "Respond with a JSON object only, shape:\n"
        '{"analysis": "<2-3 sentence read of the situation>", '
        '"recommendations": ["<action 1>", "<action 2>"]}'
    )

    text = await complete(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        agent=agent_name,
    )
    if text is None:
        return {"llm": "unavailable", "agent": agent_name}

    parsed = _extract_json(text)
    if parsed is None:
        # LLM answered but not as JSON — keep the prose, do not invent structure.
        return {"agent": agent_name, "analysis": text[:800], "recommendations": []}

    return {
        "agent": agent_name,
        "analysis": str(parsed.get("analysis", ""))[:800],
        "recommendations": [str(r)[:200] for r in (parsed.get("recommendations") or [])][:5],
    }


def _extract_json(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None
