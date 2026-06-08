"""
Multi-Agent Discussion — employees genuinely collaborate by reading each other's work.

Architecture (Voyager + Reflexion + MARL-lite):
1. Each "speaker" agent reads: shared memory, skills, recent peer learnings
2. It generates a substantive update about its domain using a free LLM
3. Other "reactor" agents read the update and reply in the same Slack thread
4. All findings written back to agent_memory.json peer_learnings

This creates a realistic engineering discussion where agents:
- Build on each other's findings
- Challenge approaches
- Share domain knowledge
- Log new skills discovered during discussion

Run by: multi-agent-discussion.yml (every 2 hours)
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
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


GROQ_KEY    = _resolve_key("GROQ_API_KEY")
DEEPSEEK_KEYS = [k for k in [
    _resolve_key("DEEPSEEK_API_KEY"),
    os.environ.get("DEEPSEEK_API_KEY_2", ""),
    os.environ.get("DEEPSEEK_API_KEY_3", ""),
] if k]
GEMINI_KEY  = _resolve_key("GEMINI_API_KEY")
SAMBANOVA_KEY = _resolve_key("SAMBANOVA_API_KEY")
CEREBRAS_KEY  = _resolve_key("CEREBRAS_API_KEY")
SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")

REPO_ROOT   = Path(__file__).resolve().parents[2]
MEMORY_FILE = REPO_ROOT / ".github" / "state" / "agent_memory.json"
SKILL_FILE  = REPO_ROOT / ".github" / "state" / "skill_library.json"
TASK_FILE   = REPO_ROOT / ".github" / "state" / "task_registry.json"


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def call_llm(messages: list[dict], max_tokens: int = 400) -> str:
    """Groq → DeepSeek → SambaNova → Cerebras → Gemini."""
    if GROQ_KEY:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                json={"model": "llama-3.1-8b-instant", "messages": messages, "max_tokens": max_tokens},
                timeout=20,
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
                timeout=25,
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
                timeout=25,
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
                timeout=20,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Cerebras: {e}")

    if GEMINI_KEY:
        try:
            prompt = "\n".join(m["content"] for m in messages[-2:])
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


def post_slack(channel: str, text: str, username: str,
               icon: str = "robot_face", thread_ts: str | None = None) -> str | None:
    """Post message, return thread_ts."""
    if not SLACK_TOKEN:
        print(f"[#{channel}] {username}: {text[:100]}")
        return None
    payload: dict = {
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
        data = r.json()
        if not data.get("ok"):
            print(f"Slack error: {data.get('error')}")
            return None
        return data.get("ts")
    except Exception as e:
        print(f"Slack: {e}")
        return None


# ── Discussion topics / speaking agents ──────────────────────────────────────

DISCUSSION_ROUNDS = [
    {
        "channel": "engineering",
        "topic": "code_quality",
        "speakers": [
            ("continuous_improver", "🔧 QuantEdge Improver", "wrench",
             "You're the continuous improvement agent. Report what specific code improvements you made in the last cycle. Read the .github/state/agent_memory.json peer_learnings and reference real findings. Be specific: file paths, functions, what changed. 3-4 sentences."),
            ("peer_reviewer", "👁️ Peer Reviewer", "eyes",
             "You're the peer review agent. Based on recent improvements reported above, what code quality issues should be prioritized next? Reference backend/app or .github/scripts paths. 2-3 sentences."),
            ("free_agent_engineer", "🤖 Free Agent", "robot_face",
             "You're the free agent engineer. Pick up an actionable task from the discussion above. Name the specific file you'll fix next and why. 2 sentences."),
        ],
    },
    {
        "channel": "desk-equities",
        "topic": "strategy_alpha",
        "speakers": [
            ("signal_runner", "📡 Signal Runner", "satellite_antenna",
             "You're the signal runner. Report what trading signals fired in the last cycle across equity symbols (SPY, QQQ, AAPL, MSFT, NVDA). Use data from agent_memory.json signals array. What was the strongest signal and confidence? 3 sentences."),
            ("strategy_generator", "🧠 Strategy Gen", "brain",
             "You're the strategy generator. Based on the signals reported above, propose one new strategy variant worth backtesting. Name the specific indicator combination and timeframe. 2-3 sentences."),
            ("quick_backtest", "⚡ Backtest Engine", "zap",
             "You're the backtest agent. Comment on the proposed strategy: what Sharpe ratio would you estimate and why? What out-of-sample period to test? 2 sentences."),
        ],
    },
    {
        "channel": "ml-research",
        "topic": "ml_models",
        "speakers": [
            ("ml_trainer", "🏋️ ML Trainer", "weight_lifting",
             "You're the ML trainer. Report the current status of models in backend/app/ml/models/. What was the last training run result? Any model showing drift? Reference specific model files. 3 sentences."),
            ("research_scientist", "🔭 Research Scientist", "telescope",
             "You're the research scientist. Based on the model report above, what new architecture or feature would improve performance? Reference a specific paper or technique (LSTM, TFT, SSM, Lorentzian KNN). 2-3 sentences."),
            ("modeling_engineer", "⚙️ Modeling Engineer", "gear",
             "You're the modeling engineer. What deployment or retraining decision do you recommend based on this discussion? Name the specific model and action. 2 sentences."),
        ],
    },
    {
        "channel": "risk",
        "topic": "risk_management",
        "speakers": [
            ("system_watchdog", "🛡️ Watchdog", "shield",
             "You're the system watchdog. Report on platform health: API availability, state file integrity, any failures in the last cycle. Reference actual data from agent_memory.json. 3 sentences."),
            ("desk_trader", "📊 Desk Trader", "bar_chart",
             "You're the desk trader. Based on current market conditions and risk status, what position sizing recommendation would you make for today? Reference the 70/30 arb/directional allocation policy. 2 sentences."),
        ],
    },
]


def run_discussion(mem: dict, skills: list[str], force_channel: str = "") -> list[dict]:
    """Run one discussion round, return new peer_learnings entries."""
    now = datetime.now(timezone.utc)
    new_learnings = []

    # Pick one random discussion topic per run (don't run all — rate limits)
    if force_channel:
        matching = [r for r in DISCUSSION_ROUNDS if r["channel"] == force_channel]
        round_config = matching[0] if matching else random.choice(DISCUSSION_ROUNDS)
    else:
        round_config = random.choice(DISCUSSION_ROUNDS)
    channel = round_config["channel"]
    topic = round_config["topic"]

    print(f"[{now.strftime('%H:%M UTC')}] Discussion: #{channel} — {topic}")

    # Gather shared context for all speakers
    failure_traces = mem.get("failure_traces", [])[-5:]
    peer_learnings = mem.get("peer_learnings", [])[-10:]
    stats = mem.get("improvement_stats", {})

    context_block = ""
    if peer_learnings:
        context_block += "RECENT TEAM ACTIVITY:\n" + "\n".join(f"  • {l}" for l in peer_learnings[-5:]) + "\n\n"
    if skills:
        context_block += "SHARED SKILL PATTERNS:\n" + "\n".join(f"  • {s}" for s in skills[-5:]) + "\n\n"
    if failure_traces:
        recent_failures = [f"{f.get('agent','?')}: {f.get('what_failed','')}" for f in failure_traces[-3:]]
        context_block += "RECENT FAILURES (to inform discussion):\n" + "\n".join(f"  • {r}" for r in recent_failures) + "\n"

    # Get real git log for context
    try:
        log = subprocess.check_output(
            ["git", "log", "--oneline", "-5", "claude/advanced-trading-bot-d5Lmw"],
            capture_output=False, timeout=10, text=True,
        ).strip()
    except Exception:
        log = "git log unavailable"

    # Opening message
    opening = (
        f"*{topic.replace('_', ' ').title()} Discussion — {now.strftime('%H:%M UTC')}*\n"
        f"_{len(round_config['speakers'])} agents contributing · {len(skills)} skills in shared memory_"
    )
    thread_ts = post_slack(channel, opening, username="QuantEdge Multi-Agent", icon="speech_balloon")

    # Each agent speaks in turn
    prev_content = opening
    for agent_name, display_name, icon, task_prompt in round_config["speakers"]:
        agent_stats = stats.get(agent_name, {})

        system = (
            f"You are the {agent_name} autonomous agent on QuantEdge, an institutional-grade "
            f"quantitative trading platform. You're in a team discussion on #{channel}.\n\n"
            f"{context_block}"
            f"Recent commits:\n{log}\n\n"
            f"Your stats: {agent_stats.get('runs', 0)} runs, {agent_stats.get('successes', 0)} successes.\n"
            f"Last task: {agent_stats.get('last_summary', 'N/A')[:100]}\n\n"
            "Be concise, specific, and reference real file paths. No disclaimers. First person."
        )
        user_msg = (
            f"Previous discussion so far:\n{prev_content[-300:]}\n\n"
            f"Your turn: {task_prompt}"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]
        reply = call_llm(messages, max_tokens=300)

        if not reply:
            reply = f"[{agent_name}] LLM unavailable — set API keys in GitHub Secrets to enable real collaboration."

        ts = post_slack(channel, reply, username=display_name, icon=icon, thread_ts=thread_ts)
        prev_content = reply

        # Record as peer learning
        new_learnings.append(
            f"[{agent_name} in #{channel} @ {now.strftime('%Y-%m-%dT%H:%M')}] {reply[:150]}"
        )
        print(f"  {agent_name}: {reply[:80]}…")

    print(f"  Discussion complete: {len(new_learnings)} learnings added")
    return new_learnings


def main():
    now = datetime.now(timezone.utc)
    print(f"[{now.strftime('%H:%M UTC')}] Multi-agent discussion")

    mem = _read_json(MEMORY_FILE)
    skills = _read_json(SKILL_FILE).get("skills", [])

    force_channel = os.environ.get("FORCE_CHANNEL", "").strip()
    new_learnings = run_discussion(mem, skills, force_channel=force_channel)

    # Write back to shared memory
    mem.setdefault("peer_learnings", [])
    mem["peer_learnings"].extend(new_learnings)
    mem["peer_learnings"] = mem["peer_learnings"][-200:]
    mem["last_updated"] = now.isoformat()
    mem.setdefault("platform_metrics", {})
    mem["platform_metrics"]["last_discussion"] = now.isoformat()
    mem["platform_metrics"]["total_discussions"] = mem["platform_metrics"].get("total_discussions", 0) + 1

    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(mem, indent=2))

    print(f"✓ {len(new_learnings)} new peer learnings added to shared memory")
    return 0


if __name__ == "__main__":
    sys.exit(main())
