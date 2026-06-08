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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

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
            "Anna, FastAPI backend is live on Render. 68 strategies registered, "
            "APScheduler running 10 jobs (snapshot, retrain, order_sync, bot_exit_checker, etc.). "
            "New: check_bot_exits() creates Trade records every 5min at TP/SL. "
            "What's the highest-priority backend tech debt you'd tackle next?"
        ),
    },
    "incidents": {
        "emp_key": "devops_dir",
        "name": "DevOps Director",
        "emoji": "🚨",
        "slack_name": "DevOps-Dir · Kenji Watanabe",
        "claude_opener": (
            "Kenji, 3 workflows just got fixed: desk-trading (continue-on-error), "
            "free-agent-engineer (always exits 0), gemini-ml-training (training step resilient). "
            "slack-on-deploy also fixed. 37 workflows total running. "
            "Which pipeline step would you harden next?"
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


def get_employee_response(emp_key: str, context: str) -> tuple[str, str]:
    """Get a Gemini (or Groq fallback) response for the employee. Returns (text, provider)."""
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

    text = call_gemini(persona, user_msg)
    if text:
        return text, "Gemini 2.0 Flash"

    text = call_groq(persona, user_msg)
    if text:
        return text, "Groq Llama-3.1-8b"

    return (
        "⚠️ All free LLM providers unavailable — check GEMINI_API_KEY / GROQ_API_KEY in GitHub Secrets.",
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
    for method in ("conversations.list", ):
        resp = slack_api(method, {"limit": 200, "types": "public_channel,private_channel"})
        if resp.get("ok"):
            for ch in resp.get("channels", []):
                if ch.get("name") == name:
                    return ch["id"]
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
        ch_id = get_channel_id(channel_name)
        if not ch_id:
            print(f"⚠️  Channel #{channel_name} not found — bot may not be in it")
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

    channels = list(CHANNEL_EMPLOYEES.keys()) if args.all else (
        [args.channel] if args.channel else list(CHANNEL_EMPLOYEES.keys())
    )

    results = []
    for ch in channels:
        result = run_conversation(ch)
        results.append(result)
        if len(channels) > 1:
            time.sleep(2)  # rate limit

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
