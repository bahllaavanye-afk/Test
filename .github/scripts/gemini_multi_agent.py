"""
Multi-Agent Gemini Manager — handles quota rotation, downtime prevention,
and context sharing so the platform keeps running 24/7 even when Claude is unavailable.

Key features:
- Rotates through multiple Gemini API keys (GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3)
- Falls back to Groq (Llama-3.1, Mixtral) when all Gemini keys are exhausted
- Posts Slack alert when quota is hit so owner can add a new key
- Shares platform context across all agents so they stay in sync
- Exports context snapshot to /tmp/agent_context.json for agent hand-off

Usage:
    from gemini_multi_agent import MultiAgentLLM
    llm = MultiAgentLLM()
    response = llm.call(prompt, role="vp_eng")
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import requests

# ── Key pool — add GEMINI_API_KEY_2, _3 in GitHub Secrets to scale ───────────

GEMINI_KEYS: list[str] = [
    k for k in [
        os.environ.get("GEMINI_API_KEY", ""),
        os.environ.get("GEMINI_API_KEY_2", ""),
        os.environ.get("GEMINI_API_KEY_3", ""),
    ] if k
]

GROQ_KEYS: list[str] = [
    k for k in [
        os.environ.get("GROQ_API_KEY", ""),
        os.environ.get("GROQ_API_KEY_2", ""),
    ] if k
]

SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
GH_REPO = os.environ.get("GH_REPO", "bahllaavanye-afk/test")

# Groq model rotation for variety
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]


class MultiAgentLLM:
    """
    LLM abstraction that rotates through Gemini keys then falls back to Groq.
    Tracks quota exhaustion and alerts via Slack.
    Zero-downtime: always returns a response even if all Gemini keys are exhausted.
    """

    def __init__(self):
        self._gemini_exhausted: set[str] = set()
        self._groq_key_idx = 0
        self._groq_model_idx = 0
        self._call_count = 0
        self._quota_alert_sent = False

    def call(
        self,
        prompt: str,
        role: str = "agent",
        max_tokens: int = 800,
        temperature: float = 0.7,
    ) -> str:
        self._call_count += 1
        # Try Gemini keys in order
        for key in GEMINI_KEYS:
            if key in self._gemini_exhausted:
                continue
            result = self._call_gemini(key, prompt, max_tokens, temperature)
            if result:
                return result

        # All Gemini keys exhausted — alert once
        if GEMINI_KEYS and not self._quota_alert_sent:
            self._quota_alert_sent = True
            self._alert_quota_exhausted()

        # Fall back to Groq
        return self._call_groq(prompt, max_tokens) or "[no LLM response — check API keys]"

    def _call_gemini(self, key: str, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
                },
                timeout=35,
            )
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            if resp.status_code == 429:
                print(f"Gemini key ...{key[-8:]} quota exhausted → marking as done for today")
                self._gemini_exhausted.add(key)
            elif resp.status_code == 403:
                print(f"Gemini key ...{key[-8:]} invalid/unauthorized")
                self._gemini_exhausted.add(key)
        except requests.Timeout:
            print(f"Gemini timeout on key ...{key[-8:]}")
        except Exception as e:
            print(f"Gemini error: {e}")
        return ""

    def _call_groq(self, prompt: str, max_tokens: int) -> str:
        if not GROQ_KEYS:
            return ""
        key = GROQ_KEYS[self._groq_key_idx % len(GROQ_KEYS)]
        model = GROQ_MODELS[self._groq_model_idx % len(GROQ_MODELS)]
        self._groq_model_idx += 1  # Rotate models for variety
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": min(max_tokens, 2048),
                },
                timeout=25,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            if resp.status_code == 429:
                # Try next Groq key
                self._groq_key_idx += 1
        except Exception as e:
            print(f"Groq error ({model}): {e}")
        return ""

    def _alert_quota_exhausted(self):
        """Post Slack alert so owner knows to add a new Gemini key."""
        msg = (
            "⚠️ *All Gemini API keys exhausted for today*\n"
            "Platform continues on Groq fallback — zero downtime.\n\n"
            "*To restore full Gemini capacity:*\n"
            "1. Go to https://aistudio.google.com → get a new free API key\n"
            "2. Add it to GitHub Secrets as `GEMINI_API_KEY_2` (or `_3`)\n"
            "3. Gemini resets at midnight UTC — capacity auto-restores tomorrow\n\n"
            "_Current key pool: " + str(len(GEMINI_KEYS)) + " Gemini key(s), " + str(len(GROQ_KEYS)) + " Groq key(s)_"
        )
        if not SLACK_TOKEN:
            print(msg)
            return
        for ch in ["engineering", "incidents"]:
            try:
                requests.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={"Authorization": f"Bearer {SLACK_TOKEN}", "Content-Type": "application/json"},
                    json={"channel": ch, "text": msg, "mrkdwn": True},
                    timeout=10,
                )
            except Exception:
                pass
        print("✓ Quota alert posted to Slack")

    def status(self) -> dict:
        return {
            "gemini_keys_available": len(GEMINI_KEYS) - len(self._gemini_exhausted),
            "gemini_keys_exhausted": len(self._gemini_exhausted),
            "groq_keys_available": len(GROQ_KEYS),
            "call_count_this_session": self._call_count,
            "quota_alert_sent": self._quota_alert_sent,
        }


# ── Shared platform context ───────────────────────────────────────────────────

def build_platform_context(gh_token: str = "") -> dict:
    """
    Builds a shared context snapshot that all agents can load.
    Lets Gemini agents understand the current platform state without Claude.
    Saved to /tmp/agent_context.json for inter-agent hand-off.
    """
    import glob

    ctx = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": "QuantEdge — institutional quant trading platform",
        "branch": "claude/advanced-trading-bot-d5Lmw",
        "rules": {
            "allow_paid_apis": False,
            "trading_mode": "paper",
            "never_ask_for_keys": True,
            "free_apis_only": ["Gemini free tier", "Groq free tier", "Binance public REST", "yfinance", "GitHub API"],
        },
    }

    # Count strategies
    manual = glob.glob("backend/app/strategies/manual/*.py")
    ml = glob.glob("backend/app/strategies/ml_enhanced/*.py")
    ctx["strategy_count"] = {
        "manual": len([f for f in manual if "__init__" not in f]),
        "ml_enhanced": len([f for f in ml if "__init__" not in f]),
    }

    # Count workflows
    workflows = glob.glob(".github/workflows/*.yml")
    ctx["workflow_count"] = len(workflows)

    # Count experiments
    experiments = glob.glob("backend/experiments/configs/*.yaml") + glob.glob("experiments/configs/*.yaml")
    ctx["experiment_configs"] = len(experiments)

    # GitHub state
    if gh_token:
        headers = {"Authorization": f"token {gh_token}"}
        try:
            r = requests.get(
                f"https://api.github.com/repos/{GH_REPO}/issues?state=open&labels=agent-fix-needed&per_page=10",
                headers=headers, timeout=10
            )
            if r.status_code == 200:
                ctx["agent_fix_queue"] = [{"number": i["number"], "title": i["title"][:60]} for i in r.json()]
        except Exception:
            pass
        try:
            r = requests.get(
                f"https://api.github.com/repos/{GH_REPO}/commits?per_page=5&sha=claude/advanced-trading-bot-d5Lmw",
                headers=headers, timeout=10
            )
            if r.status_code == 200:
                ctx["recent_commits"] = [c["commit"]["message"][:80] for c in r.json()]
        except Exception:
            pass

    # Save for hand-off
    with open("/tmp/agent_context.json", "w") as f:
        json.dump(ctx, f, indent=2)

    return ctx


def load_platform_context() -> dict:
    """Load the shared context if available."""
    try:
        with open("/tmp/agent_context.json") as f:
            return json.load(f)
    except Exception:
        return {}


# ── Singleton for convenience ─────────────────────────────────────────────────

_llm: MultiAgentLLM | None = None

def get_llm() -> MultiAgentLLM:
    global _llm
    if _llm is None:
        _llm = MultiAgentLLM()
    return _llm


if __name__ == "__main__":
    # Self-test
    print(f"Gemini keys configured: {len(GEMINI_KEYS)}")
    print(f"Groq keys configured: {len(GROQ_KEYS)}")
    llm = MultiAgentLLM()
    result = llm.call("Say 'QuantEdge multi-agent system online' in exactly those words.", max_tokens=20)
    print(f"LLM test: {result}")
    print(f"Status: {llm.status()}")
    ctx = build_platform_context(os.environ.get("GH_TOKEN", ""))
    print(f"Platform context: {json.dumps(ctx, indent=2)}")
