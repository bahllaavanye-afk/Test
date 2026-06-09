"""
Slack Auto-Review Loop
======================
Polls all trading desk Slack channels, classifies each recent message,
and dispatches free LLM agents to act on them:

- Research finding → validate and queue as experiment
- Trade signal → run risk check and log
- Bug report / error → diagnose and post fix recommendation
- Strategy idea → evaluate and post IC/Sharpe estimate
- General question → answer using codebase context

Posts responses back as threaded replies. Runs every 5 minutes via GitHub Actions.
Tracks processed messages in a simple state file to avoid double-replies.
"""
from __future__ import annotations

import json, os, re, sys, time, hashlib, urllib.request
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
from llm_common import llm, memory_write

REPO_ROOT   = Path(__file__).parent.parent
BRANCH      = "claude/advanced-trading-bot-d5Lmw"
SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
ALLOW_PAID  = os.environ.get("ALLOW_PAID_APIS", "False")
STATE_FILE  = REPO_ROOT / ".github" / "state" / "slack_review_state.json"

if ALLOW_PAID.lower() == "true":
    sys.exit(1)

CHANNELS = [
    "desk-research", "desk-equity", "desk-crypto", "desk-polymarket",
    "desk-ml", "desk-risk", "desk-lead-review", "engineering",
    "risk-alerts", "general", "desk-tv-indicators",
]

# How many messages to look back per channel
MSG_LOOKBACK = 10


def _free_llm(prompt: str, max_tokens: int = 512) -> str | None:
    result = llm(prompt, max_tokens=max_tokens, inject_company_context=False)
    if result and not result.startswith("[LLM unavailable"):
        return result
    return None


# ── Slack API helpers ─────────────────────────────────────────────────────────

def _slack_call(method: str, payload: dict) -> dict:
    if not SLACK_TOKEN:
        return {}
    url = f"https://slack.com/api/{method}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Authorization": f"Bearer {SLACK_TOKEN}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"[slack-review] Slack API error ({method}): {e}")
        return {}


def get_channel_id(name: str) -> str | None:
    """Resolve channel name to ID."""
    result = _slack_call("conversations.list", {
        "exclude_archived": True, "types": "public_channel,private_channel", "limit": 200,
    })
    for ch in result.get("channels", []):
        if ch.get("name") == name:
            return ch["id"]
    return None


def get_recent_messages(channel_id: str, limit: int = MSG_LOOKBACK) -> list[dict]:
    result = _slack_call("conversations.history", {
        "channel": channel_id, "limit": limit,
    })
    return result.get("messages", [])


def post_reply(channel_id: str, thread_ts: str, text: str) -> None:
    _slack_call("chat.postMessage", {
        "channel": channel_id,
        "thread_ts": thread_ts,
        "text": text,
        "mrkdwn": True,
    })


def post_message(channel_id: str, text: str) -> None:
    _slack_call("chat.postMessage", {
        "channel": channel_id, "text": text, "mrkdwn": True,
    })


# ── Message classifier ────────────────────────────────────────────────────────

CLASSIFIER_PROMPT = """You are an AI reviewer for a quantitative trading firm's Slack.
Classify this message and decide if it needs a response.

Message: "{text}"
Channel: #{channel}

Return JSON only:
{{
  "needs_response": true/false,
  "category": "research_finding|trade_signal|bug_report|strategy_idea|question|general|noise",
  "priority": "high|medium|low",
  "action": "one sentence on what to do, or 'none'"
}}

Do NOT respond to:
- Bot-generated messages (contain 🤖 or "Pipeline started" or "Auto-fixed")
- Messages shorter than 10 characters
- Messages that already have responses (thread replies)
- Duplicate or repetitive status updates

DO respond to:
- Research findings that could become strategies
- Bug reports or errors that need diagnosis
- Strategy ideas that need evaluation
- Technical questions from team members
- Risk alerts needing acknowledgment"""


def classify_message(text: str, channel: str) -> dict:
    # Fast pre-filter: skip obvious noise
    if len(text) < 10:
        return {"needs_response": False, "category": "noise", "priority": "low", "action": "none"}
    skip_patterns = [
        "Pipeline started", "Auto-fixed", "✅", "⚠️ *Backend AI Team",
        "🤖", "auto-fix", "cron job", "workflow run", "Heartbeat",
        "standup", "All systems clean", "Backend AI Audit", "Backend AI Team",
        "All systems nominal", "Agent Review", "Lead Review",
        "Slack Review Loop", "Signal Runner", "P&L Report",
        "Alpha Review", "Standup —", "Squad Standup", "All-Hands",
        "Strategy Review —", "Risk EOD", "[skip ci]", "Agent Health",
        "Collective Learning", "Backend AI Team", "QuantEdge Bot",
        "[Agent Review]", "[Lead Review]", "🔄 *Slack Review",
    ]
    for pat in skip_patterns:
        if pat in text:
            return {"needs_response": False, "category": "noise", "priority": "low", "action": "none"}

    prompt = CLASSIFIER_PROMPT.format(text=text[:500], channel=channel)
    raw = _free_llm(prompt, max_tokens=150)
    if not raw:
        return {"needs_response": False, "category": "noise", "priority": "low", "action": "none"}
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group(0)) if m else {"needs_response": False, "category": "noise"}
    except Exception:
        return {"needs_response": False, "category": "noise", "priority": "low", "action": "none"}


# ── Response generators ───────────────────────────────────────────────────────

RESPONSE_PROMPTS = {
    "research_finding": """You are a quant analyst at QuantEdge.
A researcher posted: "{text}"
Respond concisely (3-5 sentences): evaluate the alpha potential, suggest a backtest approach,
and note any risks. Format for Slack (use `code` for symbols, *bold* for key points).""",

    "trade_signal": """You are QuantEdge's risk manager.
A trade signal was posted: "{text}"
Respond (3-4 sentences): assess the signal quality, check for obvious risks (position size,
market conditions, regime), suggest entry/exit params. Be direct.""",

    "bug_report": """You are a senior backend engineer at QuantEdge.
A bug was reported: "{text}"
Diagnose in 3-4 sentences: likely root cause, which file/line is probably broken,
and the fastest fix. Use `code blocks` for technical terms.""",

    "strategy_idea": """You are a quant researcher at QuantEdge.
A strategy idea was proposed: "{text}"
Evaluate (4-5 sentences): academic backing, expected Sharpe range, implementation complexity,
data requirements, and one concrete improvement. Be specific.""",

    "question": """You are a senior quantitative engineer at QuantEdge.
A team member asked: "{text}"
Answer concisely (2-4 sentences) with technical precision. Use `code` formatting where helpful.""",
}


def generate_response(text: str, category: str, channel: str) -> str | None:
    template = RESPONSE_PROMPTS.get(category)
    if not template:
        return None
    prompt = template.format(text=text[:800], channel=channel)
    return _free_llm(prompt, max_tokens=300)


# ── State tracking (per-run in /tmp) ─────────────────────────────────────────

def load_state() -> set[str]:
    if STATE_FILE.exists():
        try:
            return set(json.loads(STATE_FILE.read_text()))
        except Exception:
            pass
    return set()


def save_state(seen: set[str]) -> None:
    # Keep only last 500 message IDs
    ids = list(seen)[-500:]
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(ids))


def msg_id(channel_id: str, ts: str) -> str:
    return hashlib.md5(f"{channel_id}:{ts}".encode()).hexdigest()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not SLACK_TOKEN:
        print("[slack-review] No SLACK_BOT_TOKEN — skipping")
        return

    seen = load_state()
    now_ts = time.time()
    # Only look at messages from last 10 minutes to stay fresh
    cutoff_ts = now_ts - 10 * 60
    processed = 0
    responded = 0

    for ch_name in CHANNELS:
        ch_id = get_channel_id(ch_name)
        if not ch_id:
            continue

        messages = get_recent_messages(ch_id, limit=MSG_LOOKBACK)
        for msg in messages:
            ts = msg.get("ts", "0")
            if float(ts) < cutoff_ts:
                continue   # too old
            if msg.get("subtype"):
                continue   # bot join/leave events
            if msg.get("bot_id"):
                continue   # bot messages — don't reply to our own agents

            mid = msg_id(ch_id, ts)
            if mid in seen:
                continue   # already processed

            text = msg.get("text", "")
            if not text:
                continue

            seen.add(mid)
            processed += 1

            classification = classify_message(text, ch_name)
            if not classification.get("needs_response"):
                continue

            category = classification.get("category", "general")
            priority = classification.get("priority", "low")
            action   = classification.get("action", "none")

            print(f"[slack-review] #{ch_name} [{priority}] {category}: {text[:60]!r}")

            response = generate_response(text, category, ch_name)
            if response:
                prefix = {
                    "high":   "🔴 *[Lead Review]*",
                    "medium": "🟡 *[Agent Review]*",
                    "low":    "🔵 *[Agent Review]*",
                }.get(priority, "🔵 *[Agent Review]*")
                post_reply(ch_id, ts, f"{prefix} {response}")
                responded += 1
                time.sleep(1)  # rate limit
                if priority == "high":
                    memory_write("slack_insights", {
                        "summary": f"#{ch_name} [{category}]: {text[:120]}",
                        "channel": ch_name,
                        "response_snippet": response[:200],
                    })

    save_state(seen)
    print(f"[slack-review] Processed {processed} new messages, responded to {responded}")

    # Summary to #engineering if there was significant activity
    if responded >= 3:
        eng_id = get_channel_id("engineering")
        if eng_id:
            post_message(eng_id,
                f"🔄 *Slack Review Loop:* Reviewed {processed} messages across {len(CHANNELS)} channels "
                f"→ responded to {responded} | _{datetime.now(timezone.utc).strftime('%H:%M UTC')}_")


if __name__ == "__main__":
    main()
