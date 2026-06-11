"""
Agent Chat Handler — runs when a user sends a message to an agent.

Works entirely on free LLMs (Groq → DeepSeek → Gemini).
No Claude tokens needed — this is the fallback when Claude is unavailable.

Triggered by: agent-chatbot.yml workflow_dispatch
              or future webhook from the frontend /agents/chat endpoint
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).parent))
from llm_common import llm, slack_post as _lc_slack_post, memory_write

ALLOW_PAID_APIS = os.environ.get("ALLOW_PAID_APIS", "False")
if ALLOW_PAID_APIS.lower() == "true":
    sys.exit(1)

SLACK_TOKEN   = os.environ.get("SLACK_BOT_TOKEN", "")

REPO_ROOT   = Path(__file__).resolve().parents[2]
MEMORY_FILE = REPO_ROOT / ".github" / "state" / "agent_memory.json"
SKILL_FILE  = REPO_ROOT / ".github" / "state" / "skill_library.json"

AGENT_ROLES = {
    "continuous_improver":   "Improves Python code quality across backend + scripts. Expert in FastAPI, SQLAlchemy, and async Python.",
    "signal_runner":         "Generates trading signals every 5 min across all desks. Expert in market microstructure.",
    "quick_backtest":        "Runs lightweight backtests, ranks strategies by Sharpe. Expert in vectorbt and pandas.",
    "peer_reviewer":         "Reviews AI agent commits. Expert in code quality and security.",
    "frontend_design":       "Improves React/TypeScript UI. Expert in Tailwind, TanStack Query, Redux.",
    "token_monitor":         "Tracks API usage and costs. Expert in optimization.",
    "strategy_generator":    "Generates new trading strategy ideas. Expert in quantitative finance.",
    "free_agent_engineer":   "General-purpose engineer. Fixes bugs, adds features. Full-stack expert.",
    "desk_trader":           "Paper trades across crypto/equity/polymarket. Expert in execution and slippage.",
    "system_watchdog":       "Health checks and self-healing. Expert in DevOps and monitoring.",
    "ml_trainer":            "Trains ML models. Expert in PyTorch, LSTM, XGBoost.",
    "standup_agent":         "Posts daily standups to Slack. Expert in org communication.",
    "investor_pipeline":     "Tracks investor relations pipeline. Expert in fundraising.",
    "run_experiments":       "Runs strategy experiments. Expert in scientific methodology.",
    "algo_agent":            "UCB1 bandit for strategy exploration. Expert in reinforcement learning.",
    "self_improver":         "Autonomous code quality improvement. Expert in RLVR and self-play.",
    "research_scientist":    "Discovers new alpha from research papers. Expert in academic ML/finance.",
    "modeling_engineer":     "Monitors model drift and retraining. Expert in MLOps.",
}


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def post_slack(channel: str, text: str, username: str, icon: str = "robot_face") -> bool:
    if not SLACK_TOKEN:
        print(f"[#{channel}] {text[:200]}")
        return False
    try:
        r = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_TOKEN}", "Content-Type": "application/json"},
            json={"channel": channel, "text": text, "mrkdwn": True,
                  "username": username, "icon_emoji": f":{icon}:"},
            timeout=15,
        )
        ok = r.json().get("ok", False)
        if not ok:
            print(f"Slack error: {r.json().get('error')}")
        return ok
    except Exception as e:
        print(f"Slack: {e}")
        return False


def main():
    agent_name = os.environ.get("AGENT_NAME", "free_agent_engineer")
    user_message = os.environ.get("USER_MESSAGE", "What are you working on?")
    channel = os.environ.get("SLACK_CHANNEL", "general")
    now = datetime.now(timezone.utc)

    print(f"[{now.strftime('%H:%M UTC')}] Chat: user → {agent_name}: {user_message[:80]}")

    # Load shared context
    mem = _read_json(MEMORY_FILE)
    skills = _read_json(SKILL_FILE).get("skills", [])[-8:]
    failures = mem.get("failure_traces", [])[-3:]
    peer_learnings = mem.get("peer_learnings", [])[-5:]
    agent_stats = mem.get("improvement_stats", {}).get(agent_name, {})

    role = AGENT_ROLES.get(agent_name, "senior engineer on QuantEdge quantitative trading platform")

    system_parts = [
        f"You are the **{agent_name}** autonomous agent on QuantEdge, an institutional-grade",
        f"quantitative trading platform. Your role: {role}.",
        "",
        "Platform stack: FastAPI + SQLAlchemy async backend. React 18 + TypeScript frontend.",
        "ML: PyTorch (LSTM, TFT, XGBoost, Lorentzian KNN, SSM). Brokers: Alpaca, Binance, Polymarket.",
        "You operate 24/7 via GitHub Actions. Branch: main.",
        "",
        "Speak as this agent in first person. Be concise, technical, specific.",
        "Reference actual file paths and function names. No disclaimers.",
    ]

    if skills:
        system_parts += ["", "KNOWN PATTERNS:", *[f"  • {s}" for s in skills]]
    if failures:
        system_parts += ["", "RECENT FAILURES TO AVOID:", *[
            f"  • {f.get('what_failed','')}: {f.get('error','')}" for f in failures
        ]]
    if peer_learnings:
        system_parts += ["", "TEAM LEARNINGS:", *[f"  • {l}" for l in peer_learnings[-3:]]]

    if agent_stats:
        system_parts += [
            "",
            f"YOUR STATS: {agent_stats.get('runs', 0)} runs, "
            f"{agent_stats.get('successes', 0)} successes.",
        ]
        if agent_stats.get("last_summary"):
            system_parts += [f"Last task: {agent_stats['last_summary'][:120]}"]

    reply = llm(user_message, system="\n".join(system_parts), max_tokens=800, inject_company_context=False)
    print(f"Reply ({len(reply)} chars): {reply[:100]}…")

    # Post to Slack
    slack_text = (
        f"*{agent_name}* → responding to: _{user_message[:120]}_\n\n"
        f"{reply}\n\n"
        f"_Powered by free LLMs (Groq/DeepSeek/Gemini) · {now.strftime('%H:%M UTC')}_"
    )
    posted = post_slack(channel, slack_text, username=f"Agent: {agent_name}")

    # Also post to #engineering
    if channel != "engineering":
        post_slack("engineering", slack_text, username=f"Agent: {agent_name}")

    # Log to memory
    mem.setdefault("agent_conversations", [])
    mem["agent_conversations"].append({
        "agent": agent_name,
        "user_message": user_message,
        "reply": reply[:500],
        "channel": channel,
        "timestamp": now.isoformat(),
    })
    mem["agent_conversations"] = mem["agent_conversations"][-100:]

    # Update peer learnings
    mem.setdefault("peer_learnings", [])
    mem["peer_learnings"].append(
        f"[{agent_name} @ {now.strftime('%Y-%m-%dT%H:%M')}] Responded to: {user_message[:80]}"
    )
    mem["peer_learnings"] = mem["peer_learnings"][-200:]

    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    mem["last_updated"] = now.isoformat()
    MEMORY_FILE.write_text(json.dumps(mem, indent=2))

    print(f"✓ Posted to #{channel}: {posted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
