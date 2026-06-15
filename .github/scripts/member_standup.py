"""
3-hourly per-member standup with threaded replies on every channel.

For each desk/squad channel:
  1. The team lead opens a standup thread (root message).
  2. Every member of that channel posts a REPLY in the thread with their own
     persona-specific update — so each channel shows a lead post + a thread of
     member replies, 8x/day (every 3 hours).

Model policy (via model_router): standup is the "fast" tier → free LLMs only
(groq/cerebras/gemini). Claude is never used here — it's reserved for senior,
high-stakes work. Every member therefore uses an LLM on every run, entirely on
free providers, which is exactly "maximize free LLMs / save Claude tokens".

Shared brain: each prompt is enriched with company_brain.json context by
llm_common, so all members reason over the same memory.

No Slack token → prints what it would post (dev/paper mode) and exits 0.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from llm_common import slack_post  # noqa: E402
from model_router import smart_llm  # noqa: E402

try:
    from slack_agent_team import _EMPLOYEE_PERSONAS
except Exception:
    _EMPLOYEE_PERSONAS = {}

_DEFAULT_PERSONA = (
    "You are a member of QuantEdge, an institutional quant trading firm. "
    "Give a concrete, specific standup update — cite real files, metrics, or "
    "tickers. No vague status. Under 60 words."
)

# channel → (lead_role, lead_display, [member_roles])
# Members are persona keys from slack_agent_team._EMPLOYEE_PERSONAS.
CHANNEL_TEAMS: dict[str, tuple[str, str, list[str]]] = {
    "desk-equities": ("alpha_dir", "Aleksandr Petrov · Alpha Dir",
                      ["equity_lead", "momentum_quant", "quant_researcher", "stat_arb_desk"]),
    "desk-crypto": ("ml_lead", "Kai Zhang · Crypto Lead",
                    ["crypto_quant", "crypto_defi_desk", "derivatives_desk"]),
    "desk-polymarket": ("poly_desk", "Lior Avraham · Poly Desk",
                        ["poly_desk", "kalshi_desk", "regime_analyst"]),
    "desk-fx-rates": ("vp_research", "Marcus Polk · VP Research",
                      ["macro_researcher", "fixed_income_desk"]),
    "desk-options": ("vol_trader", "Emma Schmidt · Vol Desk",
                     ["vol_trader", "derivatives_desk", "market_maker"]),
    "ml-experiments": ("ml_researcher", "Tomas Lindqvist · ML Research",
                       ["model_validator", "feature_engineer", "nlp_researcher",
                        "rl_trader", "graph_ml_researcher"]),
    "squad-backend": ("backend_lead", "Anna Hoffmann · Backend Lead",
                      ["backend_lead", "data_engineer_2", "infra_lead", "latency_engineer"]),
    "squad-frontend": ("frontend_lead", "Priya Iyer · Frontend Lead",
                       ["frontend", "frontend_lead"]),
    "squad-execution": ("exec_lead", "Ying Chen · Execution",
                        ["exec_eng", "arb_trader", "market_maker"]),
    "risk-alerts": ("cro", "Marina Volkov · CRO",
                    ["risk_eng", "portfolio_manager", "regime_analyst"]),
    "squad-qa": ("qa_dir", "Maria Garcia · QA Dir",
                 ["qa_dir", "backtest_engineer"]),
    "infra-alerts": ("devops_dir", "Liu Wei · DevOps Dir",
                     ["devops_dir", "infra_lead", "ml_infra", "security_lead"]),
}


def _persona(role: str) -> str:
    return _EMPLOYEE_PERSONAS.get(role, _DEFAULT_PERSONA)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%a %b %d, %H:%M UTC")


_LLM_KEY_VARS = [
    "GROQ_API_KEY", "GROQ_API_KEY_1", "GEMINI_API_KEY", "GEMINI_API_KEY_1",
    "DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_1", "SAMBANOVA_API_KEY", "SAMBANOVA_API_KEY_1",
    "CEREBRAS_API_KEY", "CEREBRAS_API_KEY_1", "TOGETHER_API_KEY", "OPENROUTER_API_KEY",
]


def _any_llm_key() -> bool:
    return any(os.environ.get(k, "").strip() for k in _LLM_KEY_VARS)


def run_member_standup() -> dict:
    if not _any_llm_key():
        # No free LLM key → don't attempt network calls (they would hang/retry).
        # The standup runs once keys are synced to the runtime.
        print("member_standup: no LLM keys available — skipping (set free provider keys to enable).")
        return {"channels": 0, "lead_posts": 0, "member_replies": 0, "skipped": "no_llm_key"}

    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    dry = not token
    summary = {"channels": 0, "lead_posts": 0, "member_replies": 0, "dry_run": dry}
    stamp = _now()

    for channel, (lead_role, lead_name, members) in CHANNEL_TEAMS.items():
        # 1) Lead opens the standup thread.
        lead_prompt = (
            f"Open the {stamp} standup for #{channel}. In 2 short sentences, set "
            f"today's focus for the desk/squad and ask the team for their updates. "
            f"Reference one concrete current priority (a file, strategy, metric, or ticker)."
        )
        lead_text, _ = smart_llm("standup", lead_role, lead_prompt,
                                 system=_persona(lead_role), max_tokens=160)
        root = f"*🗓️ Standup — {stamp}*\n{lead_text}\n\n_— {lead_name}_"

        if dry:
            print(f"\n#{channel} (lead {lead_role}):\n  {lead_text[:120]}")
            thread_ts = None
        else:
            resp = slack_post(channel, root)
            thread_ts = resp.get("ts") if resp.get("ok") else None
            if thread_ts:
                summary["lead_posts"] += 1
            else:
                print(f"  #{channel}: lead post failed ({resp.get('error')}) — skipping thread")
                continue
        summary["channels"] += 1

        # 2) Each member replies in the thread.
        for role in members:
            reply_prompt = (
                f"Post your standup reply in #{channel} for {stamp}. State, in <50 words: "
                f"(1) what you shipped/checked since the last standup, (2) today's focus, "
                f"(3) any blocker. Be specific and reference real files/metrics/tickers for your domain."
            )
            reply_text, provider = smart_llm("standup_reply", role, reply_prompt,
                                             system=_persona(role), max_tokens=140)
            body = f"*{role}*: {reply_text}"
            if dry:
                print(f"    ↳ {role} [{provider}]: {reply_text[:90]}")
                summary["member_replies"] += 1
            else:
                r = slack_post(channel, body, thread_ts=thread_ts)
                if r.get("ok"):
                    summary["member_replies"] += 1
                time.sleep(0.4)  # gentle on Slack rate limits

    print(f"\nMember standup complete: {summary}")
    return summary


if __name__ == "__main__":
    run_member_standup()
