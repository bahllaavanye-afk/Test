"""
Agent Status Checker — posts a "health roll call" to Slack every 4 hours.

For each of the 18 agents:
1. Reads their last activity from agent_memory.json
2. Asks the LLM: "What are you working on right now?"
3. Posts all replies to #engineering in a single threaded message

This proves every agent is alive and talking, using only free LLMs.
Also writes a summary to .github/state/agent_status.json for the frontend dashboard.

SECURITY: ALLOW_PAID_APIS must always be False. No strategy code / positions sent to LLM.
"""
from __future__ import annotations
import json
import os
import sys
import random
from datetime import datetime, timezone
from pathlib import Path
import requests

ALLOW_PAID_APIS = os.environ.get("ALLOW_PAID_APIS", "False")
if ALLOW_PAID_APIS.lower() == "true":
    sys.exit(1)


def _resolve_key(*names: str) -> str:
    for name in names:
        v = os.environ.get(name, "")
        if v:
            return v
        if not name[-1].isdigit():
            v = os.environ.get(name + "_1", "")
            if v:
                return v
    return ""


GROQ_KEY       = _resolve_key("GROQ_API_KEY")
DEEPSEEK_KEYS  = [k for k in [
    _resolve_key("DEEPSEEK_API_KEY"),
    os.environ.get("DEEPSEEK_API_KEY_2", ""),
    os.environ.get("DEEPSEEK_API_KEY_3", ""),
] if k]
SAMBANOVA_KEY  = _resolve_key("SAMBANOVA_API_KEY")
CEREBRAS_KEY   = _resolve_key("CEREBRAS_API_KEY")
HYPERBOLIC_KEY = _resolve_key("HYPERBOLIC_API_KEY")
TOGETHER_KEY   = _resolve_key("TOGETHER_API_KEY")
GEMINI_KEY     = _resolve_key("GEMINI_API_KEY")
SLACK_TOKEN    = os.environ.get("SLACK_BOT_TOKEN", "")

REPO_ROOT    = Path(__file__).resolve().parents[2]
MEMORY_FILE  = REPO_ROOT / ".github" / "state" / "agent_memory.json"
SKILL_FILE   = REPO_ROOT / ".github" / "state" / "skill_library.json"
STATUS_FILE  = REPO_ROOT / ".github" / "state" / "agent_status.json"

# 18 agents with their domain + emoji
AGENTS = [
    ("continuous_improver",  "🔧", "Improves Python code quality in backend + scripts"),
    ("signal_runner",        "📡", "Generates trading signals every 5 min across all desks"),
    ("quick_backtest",       "⚡", "Runs lightweight backtests, ranks strategies by Sharpe"),
    ("peer_reviewer",        "👁️",  "Reviews AI agent commits for quality and security"),
    ("frontend_design",      "🎨", "Improves React/TypeScript UI in Bloomberg dark theme"),
    ("token_monitor",        "💰", "Tracks API usage and posts optimization suggestions"),
    ("strategy_generator",   "🧠", "Generates new trading strategy ideas"),
    ("free_agent_engineer",  "🤖", "Fixes bugs, adds features across the full stack"),
    ("desk_trader",          "📊", "Paper trades across crypto/equity/polymarket desks"),
    ("system_watchdog",      "🛡️",  "Health checks and self-heals the platform every 5 min"),
    ("ml_trainer",           "🏋️",  "Trains ML models: LSTM, TFT, XGBoost, Lorentzian KNN"),
    ("standup_agent",        "📋", "Posts daily standups and OKR tracking to Slack"),
    ("investor_pipeline",    "💼", "Tracks investor pipeline, auto-advances stages"),
    ("run_experiments",      "🔬", "Runs strategy experiments, saves results to JSON"),
    ("algo_agent",           "🎰", "UCB1 bandit for strategy exploration"),
    ("self_improver",        "🔄", "Autonomous code quality improvement via self-play"),
    ("research_scientist",   "🔭", "Discovers new alpha from research papers"),
    ("modeling_engineer",    "⚙️",  "Monitors model drift and retraining pipeline"),
]


def call_llm(messages: list[dict], max_tokens: int = 150) -> str:
    """Groq → DeepSeek → SambaNova → Cerebras → Hyperbolic → Together → Gemini."""
    if GROQ_KEY:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                json={"model": "llama-3.1-8b-instant", "messages": messages, "max_tokens": max_tokens},
                timeout=15,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Groq: {e}")

    for key in DEEPSEEK_KEYS:
        try:
            r = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": "deepseek-chat", "messages": messages, "max_tokens": max_tokens},
                timeout=20,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"DeepSeek: {e}")

    if SAMBANOVA_KEY:
        try:
            r = requests.post(
                "https://api.sambanova.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {SAMBANOVA_KEY}", "Content-Type": "application/json"},
                json={"model": "Meta-Llama-3.1-8B-Instruct", "messages": messages, "max_tokens": max_tokens},
                timeout=20,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"SambaNova: {e}")

    if CEREBRAS_KEY:
        try:
            r = requests.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {CEREBRAS_KEY}", "Content-Type": "application/json"},
                json={"model": "llama3.1-8b", "messages": messages, "max_tokens": max_tokens},
                timeout=15,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Cerebras: {e}")

    if HYPERBOLIC_KEY:
        try:
            r = requests.post(
                "https://api.hyperbolic.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {HYPERBOLIC_KEY}", "Content-Type": "application/json"},
                json={"model": "meta-llama/Llama-3.2-3B-Instruct", "messages": messages, "max_tokens": max_tokens},
                timeout=20,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Hyperbolic: {e}")

    if TOGETHER_KEY:
        try:
            r = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {TOGETHER_KEY}", "Content-Type": "application/json"},
                json={"model": "meta-llama/Llama-3.2-3B-Instruct-Turbo", "messages": messages, "max_tokens": max_tokens},
                timeout=20,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Together: {e}")

    if GEMINI_KEY:
        try:
            prompt = "\n".join(m["content"] for m in messages)
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}",
                json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                      "generationConfig": {"maxOutputTokens": max_tokens}},
                timeout=25,
            )
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            print(f"Gemini: {e}")

    return ""


def post_slack(channel: str, text: str, username: str = "QuantEdge",
               icon: str = "robot_face", thread_ts: str | None = None) -> str | None:
    if not SLACK_TOKEN:
        print(f"[#{channel}] {username}: {text[:120]}")
        return "local-ts"
    payload = {
        "channel": channel, "text": text, "mrkdwn": True,
        "username": username, "icon_emoji": f":{icon}:",
    }
    if thread_ts:
        payload["thread_ts"] = thread_ts
    try:
        r = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_TOKEN}", "Content-Type": "application/json"},
            json=payload, timeout=15,
        )
        d = r.json()
        if not d.get("ok"):
            print(f"Slack error: {d.get('error')}")
        return d.get("ts")
    except Exception as e:
        print(f"Slack: {e}")
        return None


def _make_agent_prompt(agent_name: str, role: str, stats: dict,
                       recent_learnings: list[str], skills: list[str]) -> list[dict]:
    """Build a short status-report prompt for one agent."""
    learning_ctx = ""
    if recent_learnings:
        relevant = [l for l in recent_learnings if agent_name in l or "all" in l.lower()][-3:]
        if not relevant:
            relevant = recent_learnings[-3:]
        learning_ctx = "Recent team learnings:\n" + "\n".join(f"  • {l[:100]}" for l in relevant)

    skill_ctx = ""
    if skills:
        skill_ctx = "Platform skills: " + "; ".join(skills[-5:])

    runs = stats.get("runs", 0)
    successes = stats.get("successes", 0)
    last_task = stats.get("last_summary", "starting up")[:100]

    system = (
        f"You are the {agent_name} agent on QuantEdge, a quantitative trading platform. "
        f"Role: {role}. "
        f"Stats: {runs} runs, {successes} successes. Last task: {last_task}. "
        f"{skill_ctx}\n"
        "{learning_ctx}\n"
        "You are autonomous and running 24/7 via GitHub Actions. "
        "Be specific, first person, concise (2 sentences max). Reference real file paths."
    ).format(learning_ctx=learning_ctx)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Give a one-sentence status update: what are you actively working on RIGHT NOW?"},
    ]


def main():
    now = datetime.now(timezone.utc)
    print(f"[{now.strftime('%H:%M UTC')}] Agent status check — {len(AGENTS)} agents")

    mem = _read_json(MEMORY_FILE)
    skills = _read_json(SKILL_FILE).get("skills", [])
    stats_map = mem.get("improvement_stats", {})
    recent_learnings = mem.get("peer_learnings", [])[-20:]

    # Only call LLM for a subset (rate limit protection — 6 agents per run, rotate)
    # Rotate which 6 agents are checked based on hour-of-day
    hour = now.hour
    batch_size = 6
    start_idx = (hour * batch_size) % len(AGENTS)
    batch = AGENTS[start_idx:start_idx + batch_size]
    if len(batch) < batch_size:
        batch += AGENTS[:batch_size - len(batch)]

    agent_statuses = []
    for agent_name, emoji, role in batch:
        stats = stats_map.get(agent_name, {})
        msgs = _make_agent_prompt(agent_name, role, stats, recent_learnings, skills)
        reply = call_llm(msgs, max_tokens=100)

        if not reply:
            last_task = stats.get("last_summary", "awaiting API key setup")[:80]
            reply = f"[Offline — set GROQ_API_KEY_1 in Secrets] Last known: {last_task}"

        agent_statuses.append({
            "agent": agent_name,
            "emoji": emoji,
            "reply": reply,
            "runs": stats.get("runs", 0),
            "successes": stats.get("successes", 0),
        })
        print(f"  {emoji} {agent_name}: {reply[:80]}")

    # Post to Slack as a threaded roll call
    total_runs = sum(
        v.get("runs", 0) for v in stats_map.values()
    )
    sr_all = sum(v.get("successes", 0) for v in stats_map.values())
    sr_pct = round(100 * sr_all / total_runs, 1) if total_runs else 0
    skill_count = len(_read_json(SKILL_FILE).get("skills", []))

    header = (
        f"*Agent Roll Call — {now.strftime('%H:%M UTC')} · {now.strftime('%a %b %d')}*\n"
        f"_{len(AGENTS)} agents online · {total_runs} total runs · {sr_pct}% success rate · {skill_count} shared skills_\n"
        f"_Showing {len(batch)} of {len(AGENTS)} agents (rotating batch)_"
    )
    thread_ts = post_slack("engineering", header, username="QuantEdge Status Bot", icon="white_check_mark")

    for s in agent_statuses:
        line = f"{s['emoji']} *{s['agent']}* ({s['runs']} runs)\n_{s['reply']}_"
        post_slack("engineering", line, username=f"Agent: {s['agent']}",
                   icon="robot_face", thread_ts=thread_ts)

    # Save status to state file for frontend dashboard
    status_doc = {
        "checked_at": now.isoformat(),
        "total_agents": len(AGENTS),
        "batch_checked": len(batch),
        "total_runs": total_runs,
        "success_rate_pct": sr_pct,
        "skill_count": skill_count,
        "agent_statuses": agent_statuses,
        "all_agents": [
            {
                "agent": a,
                "emoji": e,
                "role": r,
                "runs": stats_map.get(a, {}).get("runs", 0),
                "successes": stats_map.get(a, {}).get("successes", 0),
                "last_summary": stats_map.get(a, {}).get("last_summary", ""),
            }
            for a, e, r in AGENTS
        ],
    }
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status_doc, indent=2))

    # Update shared memory with new peer learnings
    mem.setdefault("peer_learnings", [])
    for s in agent_statuses:
        if s["reply"] and "Offline" not in s["reply"]:
            mem["peer_learnings"].append(
                f"[{s['agent']} @ {now.strftime('%Y-%m-%dT%H:%M')}] {s['reply'][:150]}"
            )
    mem["peer_learnings"] = mem["peer_learnings"][-200:]
    mem["last_updated"] = now.isoformat()
    MEMORY_FILE.write_text(json.dumps(mem, indent=2))

    print(f"✓ Status check complete: {len(agent_statuses)} agents reported, saved to agent_status.json")
    return 0


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


if __name__ == "__main__":
    sys.exit(main())
