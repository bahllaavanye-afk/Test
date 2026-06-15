"""
Model distribution router — the single place that decides WHICH model does WHAT.

Design (two orthogonal axes — do not conflate them):

  AXIS 1 — task complexity → model tier (this is the task-based mapping):
    T0 deterministic   no LLM        rule engine        heartbeats, status, health
    T1 fast/chatter    free, fastest  groq, cerebras     standup replies, desk notes, intros
    T2 analysis        free, balanced gemini, sambanova  strategy eval, slippage, scans
    T3 reasoning       free, strong   deepseek, gemini   alpha mining, validation, code review
    T4 high-stakes     Claude*        senior roles only  final review, risk sign-off, architecture

  AXIS 2 — account/key load-balancing (maximize free throughput):
    Each employee is pinned to their own numbered key (GROQ_API_KEY_3, ...) in
    slack_agent_team so 47 agents don't collide on one account's rate limit.
    That layer is unchanged — this router only decides the provider ORDER.

Token policy ("maximize free LLMs, save Claude tokens"):
  * Free providers are tried FIRST for every tier, including T4.
  * Claude is reached only when ALL of these hold:
      - the task is high-stakes (T4), AND
      - the role is senior (team lead / director / VP / C-suite), AND
      - a small daily Claude budget remains, AND
      - env ALLOW_CLAUDE_SENIOR is not "false".
    So Claude is a correctness backstop for senior, high-stakes work — never
    the default. ALLOW_PAID_APIS stays False: this flag narrowly enables only
    Claude for seniors, not any other paid API.

Shared brain: every provider call goes through llm_common, which injects the
shared company_brain.json context into the prompt — so all models reason over
the same memory regardless of vendor.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

_STATE_DIR = Path(os.environ.get("GITHUB_WORKSPACE", ".")) / ".github" / "state"
_BUDGET_FILE = _STATE_DIR / "claude_budget.json"

# ── Seniority registry ────────────────────────────────────────────────────────
# Team leads, directors, VPs, and C-suite may escalate to Claude on high-stakes
# tasks. Everyone else is free-LLM only.
SENIOR_ROLES: set[str] = {
    # C-suite
    "cto", "cro", "ceo", "cfo",
    # VPs
    "vp_eng", "vp_research", "vp_ml", "vp_frontend", "vp_devops",
    "vp_security", "vp_product", "vp_quant",
    # Directors / desk + squad leads
    "alpha_dir", "ml_lead", "backend_lead", "qa_dir", "devops_dir",
    "exec_lead", "frontend_lead", "data_lead", "security_lead",
    "product_lead", "ml_infra", "equity_lead", "senior_quant",
    "portfolio_manager", "research_ops",
}

# ── Task → tier ───────────────────────────────────────────────────────────────
TASK_TIERS: dict[str, str] = {
    # T0 — deterministic, no LLM (handled by rule engine upstream)
    "heartbeat": "deterministic",
    "status_check": "deterministic",
    "health_check": "deterministic",
    # T1 — fast chatter
    "standup": "fast",
    "standup_reply": "fast",
    "desk_note": "fast",
    "intro": "fast",
    "chat": "fast",
    "slack_reply": "fast",
    # T2 — analysis
    "evaluate_strategies": "analysis",
    "slippage_analysis": "analysis",
    "market_scan": "analysis",
    "fetch_ohlcv": "analysis",
    "risk_check": "analysis",
    "feature_note": "analysis",
    # T3 — reasoning
    "alpha_mining": "reasoning",
    "evaluate_models": "reasoning",
    "model_validation": "reasoning",
    "walk_forward": "reasoning",
    "code_review": "reasoning",
    "research": "reasoning",
    # T4 — high-stakes (Claude-eligible for seniors)
    "final_review": "highstakes",
    "risk_signoff": "highstakes",
    "architecture": "highstakes",
    "investor_synthesis": "highstakes",
    "merge_decision": "highstakes",
}

# ── Tier → ordered free-provider chain ────────────────────────────────────────
# Names must match llm_common._PROVIDERS.
TIER_PROVIDERS: dict[str, list[str]] = {
    "fast":       ["groq", "cerebras", "sambanova", "gemini", "together", "openrouter"],
    "analysis":   ["gemini", "sambanova", "groq", "deepseek", "cerebras", "together"],
    "reasoning":  ["deepseek", "gemini", "sambanova", "groq", "together", "openrouter"],
    "highstakes": ["deepseek", "gemini", "sambanova", "groq"],
}

CLAUDE_DAILY_BUDGET = int(os.environ.get("CLAUDE_DAILY_BUDGET", "40"))


def tier_for_task(task_type: str) -> str:
    return TASK_TIERS.get(task_type, "analysis")


def is_senior(role: str) -> bool:
    return (role or "").lower() in SENIOR_ROLES


def _claude_enabled() -> bool:
    # Fail-closed: paid Claude escalation requires explicit opt-in
    # (ALLOW_CLAUDE_SENIOR=true) AND a key. Any workflow that forgets the flag
    # gets free-LLM-only behaviour, never silent paid spend.
    return os.environ.get("ALLOW_CLAUDE_SENIOR", "false").lower() == "true" and bool(
        os.environ.get("ANTHROPIC_API_KEY", "").strip()
    )


def _budget_state() -> dict:
    today = time.strftime("%Y-%m-%d", time.gmtime())
    try:
        data = json.loads(_BUDGET_FILE.read_text())
    except Exception:
        data = {}
    if data.get("date") != today:
        data = {"date": today, "used": 0}
    return data


def claude_budget_remaining() -> int:
    return max(0, CLAUDE_DAILY_BUDGET - _budget_state().get("used", 0))


def _consume_claude_budget() -> None:
    data = _budget_state()
    data["used"] = data.get("used", 0) + 1
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        _BUDGET_FILE.write_text(json.dumps(data))
    except Exception:
        pass


def route(task_type: str, role: str) -> dict:
    """
    Return the routing plan for (task_type, role):
      {
        "tier": str,
        "providers": [free provider names, in order],
        "claude_eligible": bool,   # may escalate to Claude after free chain
        "deterministic": bool,     # caller should use the rule engine, not an LLM
      }
    """
    tier = tier_for_task(task_type)
    if tier == "deterministic":
        return {"tier": tier, "providers": [], "claude_eligible": False, "deterministic": True}

    providers = list(TIER_PROVIDERS.get(tier, TIER_PROVIDERS["analysis"]))
    claude_eligible = (
        tier == "highstakes"
        and is_senior(role)
        and _claude_enabled()
        and claude_budget_remaining() > 0
    )
    return {
        "tier": tier,
        "providers": providers,
        "claude_eligible": claude_eligible,
        "deterministic": False,
    }


def smart_llm(
    task_type: str,
    role: str,
    prompt: str,
    system: str = "You are an expert at QuantEdge, a quantitative trading firm.",
    max_tokens: int = 400,
    temperature: float = 0.7,
) -> tuple[str, str]:
    """
    Route a prompt to the right model for (task_type, role).
    Free providers first (always); Claude only as a senior+high-stakes backstop.
    Returns (text, provider_used).
    """
    plan = route(task_type, role)

    if plan["deterministic"]:
        return "", "rule_engine"  # caller should not have called an LLM for this

    from llm_common import llm, llm_with_provider

    # Try the free chain strictly in TIER order — fallback_to_cascade=False so a
    # miss advances to the next tier provider instead of being overridden by the
    # fixed gemini-first race (finding L2).
    for provider in plan["providers"]:
        text, used = llm_with_provider(
            prompt, provider, system=system, max_tokens=max_tokens,
            temperature=temperature, inject_company_context=True,
            fallback_to_cascade=False,
        )
        if text and not text.startswith("[LLM unavailable"):
            return text, used

    # Senior + high-stakes backstop: Claude, budget-gated.
    if plan["claude_eligible"]:
        try:
            from slack_agent_team import call_claude
            text = call_claude(system, prompt, max_tokens=max_tokens)
            if text:
                _consume_claude_budget()
                return text, "claude"
        except Exception:
            pass

    # Last resort: the full llm_common cascade (all 11 providers) so an agent
    # never goes silent while free capacity sits idle (finding L10).
    text = llm(prompt, system=system, max_tokens=max_tokens,
               temperature=temperature, use_cache=False)
    if text and not text.startswith("[LLM unavailable"):
        return text, "cascade"

    return "[LLM unavailable — all providers failed]", "none"


if __name__ == "__main__":
    # Quick self-check of the routing table (no network calls).
    for tt, rl in [
        ("standup", "data_engineer_2"),
        ("alpha_mining", "quant_researcher"),
        ("risk_signoff", "cro"),
        ("risk_signoff", "data_engineer_2"),
        ("architecture", "vp_eng"),
        ("heartbeat", "cto"),
    ]:
        p = route(tt, rl)
        print(f"{tt:20s} {rl:18s} → tier={p['tier']:12s} claude={p['claude_eligible']} "
              f"providers={p['providers'][:3]}")
    print(f"\nClaude budget remaining today: {claude_budget_remaining()}/{CLAUDE_DAILY_BUDGET}")
