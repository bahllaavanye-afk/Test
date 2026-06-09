"""
Claude ↔ Employee Conversations — Real Slack threads with Gemini-powered responses.

Claude posts an opening message to each channel.
The channel's resident employee responds using Gemini Flash (free tier).
Both sides are posted to Slack so you can see the real thread.

Usage (via GitHub Actions workflow_dispatch or direct):
    python claude_conversations.py [--channel CHANNEL] [--all]
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path


def _resolve_key(*names: str) -> str:
    for name in names:
        v = os.environ.get(name, "")
        if v: return v
        if not name[-1].isdigit():
            v = os.environ.get(name + "_1", "")
            if v: return v
    return ""


REPO_ROOT = Path(__file__).resolve().parents[2]

SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
GEMINI_API_KEY    = _resolve_key("GEMINI_API_KEY", "GEMINI_API_KEY_1")
GROQ_API_KEY      = _resolve_key("GROQ_API_KEY", "GROQ_API_KEY_1")
DEEPSEEK_API_KEY  = _resolve_key("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_1")
SAMBANOVA_API_KEY = _resolve_key("SAMBANOVA_API_KEY", "SAMBANOVA_API_KEY_1")
CEREBRAS_API_KEY  = _resolve_key("CEREBRAS_API_KEY", "CEREBRAS_API_KEY_1")
HYPERBOLIC_API_KEY = _resolve_key("HYPERBOLIC_API_KEY", "HYPERBOLIC_API_KEY_1")
TOGETHER_API_KEY  = _resolve_key("TOGETHER_API_KEY", "TOGETHER_API_KEY_1")

# ── Shared memory ──────────────────────────────────────────────────────────────

STATE_FILE = Path(__file__).resolve().parents[2] / ".github" / "state" / "agent_memory.json"

def load_memory() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"conversations": {}, "thread_state": {}, "employee_context": {}, "platform_metrics": {}}

def save_memory(memory: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    memory["last_updated"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(memory, indent=2))

# ── Channel → Employee mapping ─────────────────────────────────────────────────

CHANNEL_EMPLOYEES: dict[str, dict] = {
    "engineering": {
        "emp_key": "vp_eng",
        "name": "VP Engineering",
        "emoji": "⚙️",
        "slack_name": "VP-Eng · Alex Chen",
        "claude_opener": (
            "Hey Alex, I just reviewed the workflow runs. desk-trading is now green, "
            "free-agent-engineer exits 0 on partial fixes, and gemini-ml-training has "
            "continue-on-error on the training step. 461 tests passing. "
            "What's the biggest remaining reliability risk you see in the backend right now?"
        ),
    },
    "desk-crypto": {
        "emp_key": "ml_lead",
        "name": "ML Lead / Crypto Desk",
        "emoji": "₿",
        "slack_name": "ML-Lead · Kai Zhang",
        "claude_opener": (
            "Kai, the DEX-CEX arb strategy is live (dex_cex_arb.py), social sentiment "
            "features are wired into engineer.py for crypto symbols, and we have 13 crypto "
            "strategies registered. What's your read on the current crypto alpha — "
            "which desk strategy has the best signal right now?"
        ),
    },
    "desk-polymarket": {
        "emp_key": "poly_desk",
        "name": "Polymarket Desk",
        "emoji": "🎯",
        "slack_name": "Poly-Desk · Lior Avraham",
        "claude_opener": (
            "Lior, we have 5 Polymarket strategies: poly_binary_arb, poly_calibration_arb, "
            "poly_late_resolution, poly_market_maker, polymarket_sentiment_momentum. "
            "The YES+NO < $0.97 arb is the core position. "
            "What calibration gaps are you seeing vs Metaculus right now?"
        ),
    },
    "desk-equities": {
        "emp_key": "alpha_dir",
        "name": "Alpha Research Director",
        "emoji": "📈",
        "slack_name": "Alpha-Dir · Sofia Karlsson",
        "claude_opener": (
            "Sofia, equities desk has 43 strategies live including cross_sectional_momentum, "
            "opening_range_breakout, vwap_reversion, and the full ML-enhanced suite. "
            "Walk-forward validation is enforced on all backtests. "
            "Which equity strategy do you see as most likely to survive a regime change?"
        ),
    },
    "risk": {
        "emp_key": "cro",
        "name": "Chief Risk Officer",
        "emoji": "🛡️",
        "slack_name": "CRO · Marcus Olufemi",
        "claude_opener": (
            "Marcus, capital split is 70% arb / 30% directional as configured. "
            "Bot exit checker runs every 5 min and creates Trade records at TP/SL. "
            "Supabase keep-alive pings every 5 days. "
            "What's the single biggest firm-level risk you'd flag right now?"
        ),
    },
    "ml-research": {
        "emp_key": "ml_researcher",
        "name": "ML Researcher",
        "emoji": "🧠",
        "slack_name": "ML-Research · Tomas Lindqvist",
        "claude_opener": (
            "Tomas, we have 17 model architectures: LSTM, TFT, XGBoost, LightGBM, "
            "SSM (Mamba-style), PatchTST, iTransformer, GNN, A3C-LSTM, Ensemble. "
            "62 experiment configs, Gemini cloud training runs nightly. "
            "Which architecture is showing the best OOS Sharpe right now?"
        ),
    },
    "backend": {
        "emp_key": "backend_lead",
        "name": "Backend Lead",
        "emoji": "🔧",
        "slack_name": "Backend-Lead · Anna Hoffmann",
        "claude_opener": (
            "Anna, FastAPI backend is live on Render. 78 strategies registered (28 arb, 44 equity, 6 crypto). "
            "APScheduler running 10 jobs (snapshot, retrain, order_sync, bot_exit_checker, etc.). "
            "490 tests passing, TypeScript clean. check_bot_exits() creates Trade records every 5min at TP/SL. "
            "What's the highest-priority backend tech debt you'd tackle next?"
        ),
    },
    "incidents": {
        "emp_key": "devops_dir",
        "name": "DevOps Director",
        "emoji": "🚨",
        "slack_name": "DevOps-Dir · Kenji Watanabe",
        "claude_opener": (
            "Kenji, 60 GitHub Actions workflows running on main branch: signal-runner every 5min, "
            "system-watchdog every 5min, quick-backtest every 15min, continuous-improvement every 30min, "
            "slack-agent-team 4x/day. All 7 LLM providers configured. "
            "Which workflow is the highest single point of failure right now?"
        ),
    },
    "frontend": {
        "emp_key": "frontend_lead",
        "name": "Frontend Lead",
        "emoji": "🎨",
        "slack_name": "Frontend · Priya Iyer",
        "claude_opener": (
            "Priya, the Bloomberg dark theme is live on all pages. LWCharts equity curves, "
            "comparison chart, and drawdown monitor are all rendering. "
            "The frontend design agent commits improvements every 2h. "
            "What UX improvement would have the biggest investor impact?"
        ),
    },
    "data-engineering": {
        "emp_key": "data_lead",
        "name": "Data Engineering Lead",
        "emoji": "🗄️",
        "slack_name": "Data-Eng · Jiwoo Park",
        "claude_opener": (
            "Jiwoo, real-time feeds from Alpaca + Binance WebSocket are live. "
            "Redis price cache TTL is set. Historical OHLCV pipeline runs nightly. "
            "Where is the biggest data quality risk right now — stale cache or feed interruptions?"
        ),
    },
    "alpha-research": {
        "emp_key": "alpha_researcher",
        "name": "Alpha Researcher",
        "emoji": "⚗️",
        "slack_name": "Alpha · Aleksandr Petrov",
        "claude_opener": (
            "Aleksandr, 78 strategies live: 44 equity + 28 arb + 6 crypto. "
            "Walk-forward validation enforced on all. poly_binary_arb near risk-free, "
            "avellaneda_stoikov_mm live, HMM regime gating directional strategies in bear markets. "
            "Which factor has the most unexploited alpha right now?"
        ),
    },
    "execution": {
        "emp_key": "exec_lead",
        "name": "Execution Lead",
        "emoji": "⚡",
        "slack_name": "Execution · Ying Chen",
        "claude_opener": (
            "Ying, TWAP/VWAP/LimitFirst/Iceberg all implemented. Smart router selects algo by "
            "order size and urgency. PPO RL execution agent trained for minimizing implementation shortfall. "
            "Slippage tracker logs every fill vs signal price. "
            "What execution edge are we leaving on the table right now?"
        ),
    },
    "security": {
        "emp_key": "security_lead",
        "name": "Security Lead",
        "emoji": "🔐",
        "slack_name": "Security · Naoko Tanaka",
        "claude_opener": (
            "Naoko, JWT auth on all endpoints, AES-256 broker key encryption, rate limiting via slowapi. "
            "No raw SQL (ORM only), CSP headers on Vercel, secret scanning active. "
            "TRADING_MODE is paper-only, enforced server-side. "
            "What's the highest-priority security gap to close before Series A diligence?"
        ),
    },
    "product": {
        "emp_key": "product_lead",
        "name": "Product Manager",
        "emoji": "📋",
        "slack_name": "Product · Sarah Kim",
        "claude_opener": (
            "Sarah, OKR 1 (CEO): investor pipeline at 10 contacts, Series A target D90. "
            "OKR 1 (CTO): 50+ commits/day via continuous improvement bots. "
            "Tearsheet endpoint live for investor pitch. "
            "What's the single most investor-impressive feature we could ship this week?"
        ),
    },
    "devops": {
        "emp_key": "devops_dir",
        "name": "DevOps Director",
        "emoji": "🚀",
        "slack_name": "DevOps · Liu Wei",
        "claude_opener": (
            "Liu, 42 GitHub Actions workflows deployed, all running hourly. Render backend + Vercel frontend live. "
            "Agent heartbeat monitors every 30 min. P0 watchdog alerts every hour. "
            "UptimeRobot pings /health every 5 min. "
            "What's the weakest link in our deployment pipeline?"
        ),
    },
    "ml-infra": {
        "emp_key": "ml_infra",
        "name": "ML Infrastructure Lead",
        "emoji": "🏗️",
        "slack_name": "ML-Infra · Felix Andersen",
        "claude_opener": (
            "Felix, Gemini cloud training runs every 4h across all symbols. "
            "LSTM, TFT, XGBoost, LightGBM, SSM, Lorentzian KNN, Ensemble all in registry. "
            "62 experiment configs, walk-forward validation enforced. "
            "How should we prioritize model retraining frequency vs training cost?"
        ),
    },
}

# ── Gemini call ────────────────────────────────────────────────────────────────

def call_gemini(system_prompt: str, user_message: str, model: str = "gemini-2.0-flash") -> str | None:
    """Call Gemini Flash (free tier) and return text response."""
    if not GEMINI_API_KEY:
        return None
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {
            "maxOutputTokens": 400,
            "temperature": 0.7,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"[gemini] error: {e}")
        return None


def call_groq(system_prompt: str, user_message: str) -> str | None:
    """Fallback: Groq Llama (free)."""
    if not GROQ_API_KEY:
        return None
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 400,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[groq] error: {e}")
        return None


def call_deepseek(system_prompt: str, user_message: str) -> str | None:
    """DeepSeek V3 (free tier)."""
    key = DEEPSEEK_API_KEY
    if not key:
        return None
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 400,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[deepseek] error: {e}")
        return None


def call_sambanova(system_prompt: str, user_message: str) -> str | None:
    """SambaNova Cloud (free tier)."""
    key = SAMBANOVA_API_KEY
    if not key:
        return None
    payload = {
        "model": "Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 400,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        "https://api.sambanova.ai/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[sambanova] error: {e}")
        return None


def call_cerebras(system_prompt: str, user_message: str) -> str | None:
    """Cerebras Llama (free tier)."""
    key = CEREBRAS_API_KEY
    if not key:
        return None
    payload = {
        "model": "llama3.1-8b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 400,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        "https://api.cerebras.ai/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[cerebras] error: {e}")
        return None


def call_hyperbolic(system_prompt: str, user_message: str) -> str | None:
    """Hyperbolic (free tier)."""
    key = HYPERBOLIC_API_KEY
    if not key:
        return None
    payload = {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 400,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        "https://api.hyperbolic.xyz/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[hyperbolic] error: {e}")
        return None


def call_together(system_prompt: str, user_message: str) -> str | None:
    """Together AI (free tier)."""
    key = TOGETHER_API_KEY
    if not key:
        return None
    payload = {
        "model": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 400,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        "https://api.together.xyz/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[together] error: {e}")
        return None


def get_employee_response(emp_key: str, context: str) -> tuple[str, str]:
    """7-provider cascade for employee response. Returns (text, provider)."""
    sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))
    try:
        from slack_agent_team import _EMPLOYEE_PERSONAS
        persona = _EMPLOYEE_PERSONAS.get(emp_key, "You are a senior quant engineer.")
    except Exception:
        persona = "You are a senior quant engineer at QuantEdge, an algorithmic trading platform."

    user_msg = (
        f"Claude (platform AI) just posted this to your Slack channel:\n\n"
        f"\"{context}\"\n\n"
        f"Reply directly and concisely as yourself. Be specific — cite file names, metrics, "
        f"numbers. Max 120 words. Slack format (*bold* for emphasis). No headers."
    )

    # Full 7-provider cascade — first success wins
    for fn, name in [
        (call_gemini,    "Gemini 2.0 Flash"),
        (call_cerebras,  "Cerebras Llama-3.1"),
        (call_groq,      "Groq Llama-3.1"),
        (call_deepseek,  "DeepSeek V3"),
        (call_sambanova, "SambaNova Llama-3.1"),
        (call_hyperbolic,"Hyperbolic Llama-3.2"),
        (call_together,  "Together Llama-3.2"),
    ]:
        try:
            text = fn(persona, user_msg)
            if text:
                return text, name
        except Exception:
            continue

    return (
        "⚠️ All 7 free LLM providers unavailable — check secrets in GitHub Actions.",
        "none",
    )


# ── Slack helpers ──────────────────────────────────────────────────────────────

def slack_api(method: str, payload: dict) -> dict:
    url = f"https://slack.com/api/{method}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {SLACK_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_channel_id(channel_name: str) -> str | None:
    name = channel_name.lstrip("#")
    # Try up to 1000 channels (paginated)
    cursor = ""
    while True:
        payload: dict = {"limit": 200, "types": "public_channel,private_channel"}
        if cursor:
            payload["cursor"] = cursor
        resp = slack_api("conversations.list", payload)
        if resp.get("ok"):
            for ch in resp.get("channels", []):
                if ch.get("name") == name:
                    return ch["id"]
            cursor = resp.get("response_metadata", {}).get("next_cursor", "")
            if not cursor:
                break
        else:
            break
    return None


def get_or_create_channel(channel_name: str) -> str | None:
    """Return channel ID, auto-creating the public channel if it doesn't exist yet."""
    ch_id = get_channel_id(channel_name)
    if ch_id:
        return ch_id
    name = channel_name.lstrip("#")
    resp = slack_api("conversations.create", {"name": name, "is_private": False})
    if resp.get("ok"):
        ch_id = resp.get("channel", {}).get("id")
        print(f"  ✅ Created missing channel #{name} → {ch_id}")
        return ch_id
    print(f"  ⚠️  Could not create #{name}: {resp.get('error')} (need channels:manage scope)")
    return None


def post_message(channel: str, text: str, thread_ts: str | None = None,
                 username: str | None = None, icon_emoji: str | None = None) -> dict:
    payload: dict = {"channel": channel, "text": text, "mrkdwn": True}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    if username:
        payload["username"] = username
    if icon_emoji:
        payload["icon_emoji"] = icon_emoji
    return slack_api("chat.postMessage", payload)


def ensure_in_channel(channel_id: str) -> None:
    slack_api("conversations.join", {"channel": channel_id})


# ── Thread follow-up helpers ───────────────────────────────────────────────────

def get_channel_history(channel_id: str, limit: int = 10) -> list[dict]:
    """Read recent messages from a Slack channel."""
    if not SLACK_TOKEN:
        return []
    try:
        req = urllib.request.Request(
            f"https://slack.com/api/conversations.history?channel={channel_id}&limit={limit}",
            headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data.get("messages", []) if data.get("ok") else []
    except Exception:
        return []

def get_thread_replies(channel_id: str, thread_ts: str) -> list[dict]:
    """Read replies in a specific thread."""
    if not SLACK_TOKEN:
        return []
    try:
        url = f"https://slack.com/api/conversations.replies?channel={channel_id}&ts={thread_ts}&limit=20"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {SLACK_TOKEN}"}, method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data.get("messages", [])[1:] if data.get("ok") else []  # skip root message
    except Exception:
        return []

def follow_up_on_threads(memory: dict):
    """Read recent threads in all channels and have employees respond to new messages."""
    if not SLACK_TOKEN:
        return

    thread_state = memory.setdefault("thread_state", {})

    for channel_name, config in CHANNEL_EMPLOYEES.items():
        ch_id = get_channel_id(channel_name)
        if not ch_id:
            continue

        # Get recent channel messages
        messages = get_channel_history(ch_id, limit=5)

        for msg in messages:
            ts = msg.get("ts", "")
            reply_count = msg.get("reply_count", 0)

            if not ts or reply_count == 0:
                continue

            state_key = f"{channel_name}:{ts}"
            last_seen_reply = thread_state.get(state_key, {}).get("last_reply_count", 0)

            # Only respond if there are new replies we haven't seen
            if reply_count <= last_seen_reply:
                continue

            replies = get_thread_replies(ch_id, ts)
            if not replies:
                continue

            # Get the latest reply text
            latest_reply = replies[-1].get("text", "")
            if not latest_reply:
                continue

            # Have the employee respond to the latest message in the thread
            emp_key = config["emp_key"]
            context = f"In your Slack channel, someone just said: \"{latest_reply}\"\n\nRespond briefly as yourself. Max 80 words."
            emp_text, provider = get_employee_response(emp_key, context)

            if emp_text and "unavailable" not in emp_text.lower():
                reply_payload = {
                    "channel": ch_id,
                    "text": emp_text,
                    "thread_ts": ts,
                    "username": config["slack_name"],
                }
                slack_api("chat.postMessage", reply_payload)
                print(f"  ↩ Follow-up in #{channel_name} thread by {config['slack_name']}")

            # Update state
            thread_state[state_key] = {"last_reply_count": reply_count}

    memory["thread_state"] = thread_state


# ── Main conversation loop ─────────────────────────────────────────────────────

def run_conversation(channel_name: str) -> dict:
    """Run a single Claude ↔ Employee conversation. Returns result summary."""
    config = CHANNEL_EMPLOYEES.get(channel_name)
    if not config:
        return {"ok": False, "error": f"No employee mapped for #{channel_name}"}

    print(f"\n{'='*60}")
    print(f"Channel: #{channel_name} → {config['name']}")
    print(f"{'='*60}")

    if not SLACK_TOKEN:
        print("⚠️  No SLACK_BOT_TOKEN — conversation will be logged only, not posted to Slack.")

    # Step 1: Post Claude's opening message
    claude_text = (
        f"*Claude (Platform AI) → {config['name']}*\n\n"
        f"{config['claude_opener']}"
    )
    print(f"\n[CLAUDE POSTS]:\n{claude_text}\n")

    thread_ts = None
    ch_id = None

    if SLACK_TOKEN:
        ch_id = get_or_create_channel(channel_name)
        if not ch_id:
            print(f"⚠️  Channel #{channel_name} not found/created — skipping Slack post")
        else:
            ensure_in_channel(ch_id)
            result = post_message(
                ch_id,
                claude_text,
                username="Claude · Platform AI",
                icon_emoji=":robot_face:",
            )
            if result.get("ok"):
                thread_ts = result.get("ts")
                print(f"✅ Claude message posted (ts={thread_ts})")
            else:
                print(f"❌ Failed to post: {result.get('error')}")

    # Step 2: Get employee response via Gemini
    print(f"\n[CALLING GEMINI for {config['name']}] ...")
    emp_text, provider = get_employee_response(config["emp_key"], config["claude_opener"])
    print(f"[{config['name']} via {provider}]:\n{emp_text}\n")

    # Step 3: Post employee response as reply in thread
    if SLACK_TOKEN and ch_id:
        formatted = (
            f"*{config['slack_name']}* {config['emoji']}\n"
            f"_{provider}_\n\n"
            f"{emp_text}"
        )
        reply = post_message(
            ch_id,
            formatted,
            thread_ts=thread_ts,
            username=config["slack_name"],
            icon_emoji=config["emoji"],
        )
        if reply.get("ok"):
            print(f"✅ Employee reply posted in thread")
        else:
            print(f"❌ Employee reply failed: {reply.get('error')}")

    return {
        "channel": channel_name,
        "employee": config["name"],
        "provider": provider,
        "claude_message": config["claude_opener"],
        "employee_response": emp_text,
        "posted_to_slack": bool(SLACK_TOKEN and ch_id and thread_ts),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="", help="Single channel to chat with")
    parser.add_argument("--all", action="store_true", help="Chat with all channels")
    args = parser.parse_args()

    if not GEMINI_API_KEY and not GROQ_API_KEY:
        print("WARNING: No GEMINI_API_KEY or GROQ_API_KEY — employees will use fallback responses")
        print("To enable real Gemini responses: add GEMINI_API_KEY to GitHub Secrets → Settings → Secrets → Actions")

    # Load shared memory
    memory = load_memory()
    print(f"[memory] loaded — last_updated: {memory.get('last_updated', 'never')}")

    # Follow up on existing threads before posting new conversations
    print("\n[thread follow-ups] checking for new replies in all channels...")
    follow_up_on_threads(memory)

    channels = list(CHANNEL_EMPLOYEES.keys()) if args.all else (
        [args.channel] if args.channel else list(CHANNEL_EMPLOYEES.keys())
    )

    results = []
    for ch in channels:
        result = run_conversation(ch)
        results.append(result)
        if len(channels) > 1:
            time.sleep(2)  # rate limit

    # Update memory with conversation results — log each exchange as timestamped entries
    conversations = memory.setdefault("conversations", {})
    for r in results:
        ch = r.get("channel", "")
        if not ch:
            continue

        # Claude's opening message entry
        ts_claude = datetime.now(timezone.utc).isoformat()
        conversations[ts_claude] = {
            "channel": ch,
            "speaker": "claude",
            "message": r.get("claude_message", "")[:500],
            "timestamp": ts_claude,
            "provider": "claude",
        }

        # Employee response entry (only if we got a real response)
        emp_text = r.get("employee_response", "")
        if emp_text and "unavailable" not in emp_text.lower():
            ts_emp = datetime.now(timezone.utc).isoformat()
            conversations[ts_emp] = {
                "channel": ch,
                "speaker": r.get("employee", ch),
                "message": emp_text[:500],
                "timestamp": ts_emp,
                "provider": r.get("provider", "none"),
                "posted_to_slack": r.get("posted_to_slack", False),
            }

    memory["conversations"] = conversations

    # Save updated memory
    save_memory(memory)
    print(f"[memory] saved to {STATE_FILE}")

    print(f"\n{'='*60}")
    print(f"CONVERSATION SUMMARY — {len(results)} channels")
    print(f"{'='*60}")
    for r in results:
        status = "✅ Slack" if r.get("posted_to_slack") else "📋 Log-only"
        print(f"{status}  #{r['channel']:20} {r['employee']:25} [{r['provider']}]")

    # Write summary JSON for GitHub step summary
    summary_path = Path("/tmp/conversations_summary.json")
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results: {summary_path}")


if __name__ == "__main__":
    main()
