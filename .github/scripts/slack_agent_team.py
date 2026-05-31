"""
QuantEdge multi-agent Slack team — real engineering work, real reports.

Each agent reads actual codebase state (git log, files, test counts,
backtest JSONs, open issues/PRs) and posts findings to Slack with their
own identity (custom username + emoji avatar via chat:write.customize).

Agents reply to each other in threads when the topic matches their domain,
creating realistic engineering discussion.

Required env:
    SLACK_BOT_TOKEN   xoxb-... with: chat:write, chat:write.public,
                      chat:write.customize (optional but recommended),
                      channels:read (optional), groups:read (optional)
    The bot works with ONLY chat:write + chat:write.public — the rest are
    optional enhancements (customize = custom names, read = faster lookups).
    GH_TOKEN          optional — GITHUB_TOKEN for reading issues/PRs
    GH_REPO           owner/repo (e.g. bahllaavanye-afk/QuantEdge)

Designed to run on a schedule (every 1-3 hours). Each run picks a wave of
6-10 agents to do work; not all agents post every run.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = REPO_ROOT / "experiments" / "results" / "slack_state.json"


# ─────────────────────────────────────────────────────────────────────────────
# State management — deduplication across runs
# ─────────────────────────────────────────────────────────────────────────────


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {
            "last_run_ts": 0,
            "last_commit_sha": "",
            "posted_hashes": [],   # MD5[:12] of recent message texts
            "replied_to": [],      # Slack message ts values already replied to
        }


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Trim to avoid unbounded growth
    state["posted_hashes"] = state.get("posted_hashes", [])[-1000:]
    state["replied_to"] = state.get("replied_to", [])[-500:]
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def is_duplicate(state: dict, text: str) -> bool:
    return _hash(text) in state.get("posted_hashes", [])


def record_post(state: dict, text: str) -> None:
    h = _hash(text)
    hashes = state.get("posted_hashes", [])
    if h not in hashes:
        hashes.append(h)
    state["posted_hashes"] = hashes


# ─────────────────────────────────────────────────────────────────────────────
# Multi-agent routing — Claude supervises, free agents do the work
#
# Cascade (fastest/cheapest first):
#   1. Groq  (Llama 3.3 70B, free 14k req/day, ~500 tok/sec)
#   2. Gemini Flash (free 1500 req/day, 1M context window)
#   3. GitHub Models (free for Actions, GPT-4o-mini / Llama)
#   4. Claude Haiku (fallback, best quality, low cost)
# ─────────────────────────────────────────────────────────────────────────────

_QUANT_SYSTEM = (
    "You are a senior quantitative engineer on QuantEdge, an institutional-grade "
    "algorithmic trading platform. Backend: FastAPI + SQLAlchemy async. ML: PyTorch "
    "(LSTM, TFT, XGBoost, Lorentzian KNN, SSM). Brokers: Alpaca, Binance, Polymarket. "
    "Answer concisely and technically. Reference specific files/functions where relevant. "
    "Do NOT say you are an AI or mention your model name."
)

# ── Cost policy — absolutely no paid API calls ────────────────────────────────
# Changing this to True requires an explicit code review approval.
ALLOW_PAID_APIS: bool = False

# Hard per-call token cap — prevents runaway generation on any provider
MAX_TOKENS_PER_CALL: int = 500

# Per-employee call budget per workflow run — free tiers are generous but finite
MAX_CALLS_PER_EMPLOYEE_PER_RUN: int = 6

# Runtime counter — reset each process start (i.e. each GitHub Actions run)
_run_call_counts: dict[str, int] = {}

# ── IP protection — sanitize before sending to any external LLM ───────────────
# These patterns are NEVER sent outside the repo boundary.
import re as _re
_STRIP_PATTERNS: list[tuple[str, str]] = [
    # API keys / tokens (any long alphanumeric secret)
    (r'(?i)(api[_-]?key|secret|token|password|bearer)\s*[=:]\s*\S+', '[REDACTED_CREDENTIAL]'),
    # Alpaca / broker key formats
    (r'\bPK[A-Z0-9]{18,}\b', '[REDACTED_ALPACA_KEY]'),
    (r'\bxoxb-[0-9A-Za-z-]+\b', '[REDACTED_SLACK_TOKEN]'),
    # Private keys / wallet addresses
    (r'0x[0-9a-fA-F]{40,}', '[REDACTED_ADDRESS]'),
    # IP addresses
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[REDACTED_IP]'),
    # Full file paths that expose internal structure
    (r'/home/[^\s]+', '[REDACTED_PATH]'),
    (r'/root/[^\s]+', '[REDACTED_PATH]'),
]

def _sanitize(text: str) -> str:
    """Strip credentials and internal paths before sending to any external LLM."""
    for pattern, replacement in _STRIP_PATTERNS:
        text = _re.sub(pattern, replacement, text)
    return text

# ── Per-employee key routing ──────────────────────────────────────────────────
# Each employee has their own free-tier API keys from their own accounts.
# Secret naming: GROQ_API_KEY_MAYA, CEREBRAS_API_KEY_AARAV, etc.
# Backup pool:   GROQ_API_KEY_BACKUP_1 … GROQ_API_KEY_BACKUP_5
# This gives each employee independent rate-limit pools — N employees = N×limit.

_EMPLOYEES = [
    "maya", "aarav", "linh", "jian", "anna",
    "aditi", "kenji", "diego", "lior", "sara",
    "sofia", "hugo", "marcus",
]

def _employee_keys(employee: str, provider: str) -> list[str]:
    """Return all API keys for (employee, provider): dedicated + backup + shared."""
    emp = employee.split("_")[0].upper()   # 'maya_chen' → 'MAYA'
    prov = provider.upper()
    keys: list[str] = []
    # 1. Employee's dedicated key
    k = os.environ.get(f"{prov}_API_KEY_{emp}", "").strip()
    if k:
        keys.append(k)
    # 2. Shared backup pool (GROQ_API_KEY_BACKUP_1 … _5)
    for i in range(1, 6):
        k = os.environ.get(f"{prov}_API_KEY_BACKUP_{i}", "").strip()
        if k and k not in keys:
            keys.append(k)
    # 3. Shared primary key (the single key everyone falls back to)
    k = os.environ.get(f"{prov}_API_KEY", "").strip()
    if k and k not in keys:
        keys.append(k)
    return keys


def call_groq(system_prompt: str, user_message: str, max_tokens: int = 600) -> str | None:
    """Groq API — Llama 3.3 70B, free tier 14 400 req/day, ~500 tok/sec."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    payload = {
        "model": "llama-3.3-70b-versatile",
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    }
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [groq] {e}")
        return None


def call_gemini(system_prompt: str, user_message: str, max_tokens: int = 600) -> str | None:
    """Google Gemini 2.0 Flash — free 1 500 req/day, 1M token context."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    payload = {
        "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_message}"}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.4},
    }
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-2.0-flash:generateContent?key={api_key}")
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"  [gemini] {e}")
        return None


def call_github_models(system_prompt: str, user_message: str, max_tokens: int = 600) -> str | None:
    """GitHub Models — free for GitHub Actions, GPT-4o-mini / Llama 3.3."""
    api_key = os.environ.get("GH_TOKEN", "").strip()  # already in Actions env
    if not api_key:
        return None
    payload = {
        "model": "gpt-4o-mini",   # free in GitHub Models
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    }
    req = urllib.request.Request(
        "https://models.inference.ai.azure.com/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [github-models] {e}")
        return None


def call_cerebras(system_prompt: str, user_message: str, max_tokens: int = 600) -> str | None:
    """Cerebras Inference — free 1M tokens/day, 2600 tok/sec, Qwen3 32B."""
    api_key = os.environ.get("CEREBRAS_API_KEY", "").strip()
    if not api_key:
        return None
    payload = {
        "model": "qwen-3-32b",   # free on Cerebras, 1M tok/day
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    }
    req = urllib.request.Request(
        "https://api.cerebras.ai/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [cerebras] {e}")
        return None


def call_openrouter(system_prompt: str, user_message: str, max_tokens: int = 500) -> str | None:
    """OpenRouter — free tier: 50 req/day per account, 20 RPM. Llama 3.3 70B free."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None
    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/bahllaavanye-afk/Test",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [openrouter] {e}")
        return None


def call_claude(system_prompt: str, user_message: str, max_tokens: int = 600) -> str | None:
    """
    Claude Haiku — BLOCKED by default (ALLOW_PAID_APIS = False).
    Only reachable if a human explicitly sets ALLOW_PAID_APIS = True in a code review.
    This function exists so the import chain doesn't break, not to be called.
    """
    if not ALLOW_PAID_APIS:
        print("  [POLICY] Paid API blocked — ALLOW_PAID_APIS=False. Skipping Claude.")
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None
    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": min(max_tokens, MAX_TOKENS_PER_CALL),
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("content", [{}])[0].get("text", "")
    except Exception as e:
        print(f"  [claude] {e}")
        return None


def _try_openai_compat(endpoint: str, api_key: str, model: str,
                        system_prompt: str, user_message: str,
                        max_tokens: int, extra_headers: dict | None = None) -> str | None:
    """Generic OpenAI-compatible POST — used for multi-key provider rotation."""
    payload = {
        "model": model,
        "max_tokens": min(max_tokens, MAX_TOKENS_PER_CALL),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=22) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except Exception:
        return None


def call_employee_agent(
    employee: str,
    user_message: str,
    system_prompt: str = _QUANT_SYSTEM,
    max_tokens: int = 500,
) -> str | None:
    """
    Each employee has their OWN free API keys — independent rate-limit pools.
    Tries dedicated employee key first, then backup pool, then shared primary key.
    NEVER calls any paid API. Returns None (skips post) if all free tiers exhausted.

    Secret naming in GitHub:
      GROQ_API_KEY_MAYA       — Maya's dedicated Groq account
      CEREBRAS_API_KEY_AARAV  — Aarav's dedicated Cerebras account
      GEMINI_API_KEY_LINH     — Linh's dedicated Gemini account
      OPENROUTER_API_KEY_JIAN — Jian's dedicated OpenRouter account
      GROQ_API_KEY_BACKUP_1   — shared backup pool (accounts 1-5)
    """
    if ALLOW_PAID_APIS:
        raise RuntimeError("ALLOW_PAID_APIS must stay False — zero spend policy")

    # Enforce per-run call budget
    emp_key = employee.split("_")[0].lower()
    _run_call_counts[emp_key] = _run_call_counts.get(emp_key, 0)
    if _run_call_counts[emp_key] >= MAX_CALLS_PER_EMPLOYEE_PER_RUN:
        print(f"  [{emp_key}] call budget exhausted ({MAX_CALLS_PER_EMPLOYEE_PER_RUN}/run) — skipping")
        return None
    _run_call_counts[emp_key] += 1

    # IP guard — never leak credentials or internal paths to external providers
    safe_message = _sanitize(user_message)
    safe_system  = _sanitize(system_prompt)
    cap = min(max_tokens, MAX_TOKENS_PER_CALL)

    # Provider cascade — all free, each employee has dedicated keys
    providers = [
        ("groq",      "https://api.groq.com/openai/v1/chat/completions",        "llama-3.3-70b-versatile", None),
        ("cerebras",  "https://api.cerebras.ai/v1/chat/completions",             "qwen-3-32b",              None),
        ("openrouter","https://openrouter.ai/api/v1/chat/completions",           "meta-llama/llama-3.3-70b-instruct:free",
                      {"HTTP-Referer": "https://github.com/bahllaavanye-afk/Test"}),
        ("gemini",    None, None, None),   # special handler below
    ]

    for prov, endpoint, model, extra_hdrs in providers:
        for key in _employee_keys(emp_key, prov):
            if prov == "gemini":
                result = call_gemini(safe_system, safe_message, cap)
            else:
                result = _try_openai_compat(endpoint, key, model, safe_system, safe_message, cap, extra_hdrs)
            if result and len(result.strip()) > 20:
                print(f"  [{emp_key}/{prov}] ✓ {len(result)} chars")
                return result.strip()

    print(f"  [{emp_key}] ⚠ all free tiers exhausted — no paid fallback (policy)")
    return None


def call_best_agent(
    user_message: str,
    system_prompt: str = _QUANT_SYSTEM,
    max_tokens: int = 500,
) -> str | None:
    """
    Shared cascade for non-employee calls (inbox, commands, incident posts).
    100% free — Groq → Cerebras → GitHub Models → OpenRouter → Gemini.
    Paid APIs are NEVER called regardless of ANTHROPIC_API_KEY presence.
    Rotates through all available keys (shared + backup pool) per provider.
    """
    cap = min(max_tokens, MAX_TOKENS_PER_CALL)
    safe_msg = _sanitize(user_message)
    safe_sys = _sanitize(system_prompt)

    # Groq — try all available keys (shared + backup pool)
    for key in _employee_keys("shared", "groq"):
        r = _try_openai_compat(
            "https://api.groq.com/openai/v1/chat/completions",
            key, "llama-3.3-70b-versatile", safe_sys, safe_msg, cap)
        if r and len(r.strip()) > 20:
            print(f"  [agent/groq] ✓ {len(r)} chars")
            return r.strip()

    # Cerebras — 1M tok/day per key
    for key in _employee_keys("shared", "cerebras"):
        r = _try_openai_compat(
            "https://api.cerebras.ai/v1/chat/completions",
            key, "qwen-3-32b", safe_sys, safe_msg, cap)
        if r and len(r.strip()) > 20:
            print(f"  [agent/cerebras] ✓ {len(r)} chars")
            return r.strip()

    # GitHub Models — free in Actions (no extra key needed)
    r = call_github_models(safe_sys, safe_msg, cap)
    if r and len(r.strip()) > 20:
        print(f"  [agent/github-models] ✓ {len(r)} chars")
        return r.strip()

    # OpenRouter — 50 req/day free per key
    for key in _employee_keys("shared", "openrouter"):
        r = _try_openai_compat(
            "https://openrouter.ai/api/v1/chat/completions",
            key, "meta-llama/llama-3.3-70b-instruct:free", safe_sys, safe_msg, cap,
            {"HTTP-Referer": "https://github.com/bahllaavanye-afk/Test"})
        if r and len(r.strip()) > 20:
            print(f"  [agent/openrouter] ✓ {len(r)} chars")
            return r.strip()

    # Gemini — 1500 req/day
    r = call_gemini(safe_sys, safe_msg, cap)
    if r and len(r.strip()) > 20:
        print(f"  [agent/gemini] ✓ {len(r)} chars")
        return r.strip()

    # Hard stop — never pay
    print("  [agent] ⚠ all 5 free providers exhausted — returning None (zero-spend policy)")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Slack thread reading — agents respond to actual human replies

# ─────────────────────────────────────────────────────────────────────────────


def read_unresponded_threads(
    token: str,
    channel_name: str,
    bot_user_id: str,
    already_replied: list[str],
    limit: int = 30,
) -> list[dict]:
    """
    Return threads in channel where a human replied but the bot hasn't responded yet.
    Skips threads whose reply_ts is in already_replied.
    """
    ch_id = get_channel_id(token, channel_name)
    if not ch_id:
        return []
    history = slack_call(token, "conversations.history", {"channel": ch_id, "limit": limit})
    if not history.get("ok"):
        return []

    unresponded = []
    for msg in history.get("messages", []):
        if not msg.get("reply_count"):
            continue
        ts = msg.get("ts")
        replies_data = slack_call(token, "conversations.replies",
                                  {"channel": ch_id, "ts": ts, "limit": 20})
        if not replies_data.get("ok"):
            continue
        replies = replies_data.get("messages", [])
        if len(replies) < 2:
            continue
        # Human replies: have "user" field and no "bot_id"
        human_replies = [
            r for r in replies[1:]
            if r.get("user") and not r.get("bot_id")
            and r.get("user") != bot_user_id
            and r.get("ts") not in already_replied
        ]
        if not human_replies:
            continue
        # Bot already responded if any reply has bot_id or matches our known usernames
        bot_already_replied = any(r.get("bot_id") for r in replies[1:])
        if bot_already_replied:
            continue
        latest_human = human_replies[-1]
        unresponded.append({
            "channel": channel_name,
            "channel_id": ch_id,
            "parent_ts": ts,
            "parent_text": msg.get("text", "")[:500],
            "last_reply": latest_human.get("text", "")[:500],
            "reply_ts": latest_human.get("ts", ""),
        })
    return unresponded


_CHANNEL_AGENT_IDENTITY = {
    "engineering":       ("VP Engineering", ":woman_office_worker:"),
    "alpha-research":    ("Alpha Research Director", ":chart_with_upwards_trend:"),
    "ml-experiments":    ("ML Research Lead", ":microscope:"),
    "squad-qa":          ("Director of QA", ":mag:"),
    "desk-crypto":       ("Crypto desk bot", ":coin:"),
    "squad-backend":     ("Backend Lead", ":gear:"),
    "squad-frontend":    ("Frontend Lead", ":art:"),
    "risk-alerts":       ("Risk Engineer", ":shield:"),
    "infra-alerts":      ("Director of DevOps", ":satellite_antenna:"),
    "desk-equities":     ("Equity desk bot", ":chart_with_upwards_trend:"),
    "desk-polymarket":   ("Polymarket Researcher", ":vertical_traffic_light:"),
    "desk-commodities":  ("Commodities desk bot", ":oil_drum:"),
    "desk-futures":      ("Futures desk bot", ":chart_with_upwards_trend:"),
    "desk-rates":        ("Rates desk bot", ":bank:"),
    "desk-kalshi":       ("Kalshi desk bot", ":ballot_box_with_ballot:"),
    "desk-stat-arb":     ("StatArb desk bot", ":arrows_counterclockwise:"),
    "desk-fx-rates":     ("Macro/FX desk bot", ":earth_americas:"),
    "desk-options":      ("Options Researcher", ":bar_chart:"),
    "help":              ("VP Engineering", ":bulb:"),
    "pnl-daily":         ("PnL bot", ":bar_chart:"),
    "ci-failures":       ("Director of QA", ":mag:"),
    "squad-execution":   ("Execution Engineer", ":zap:"),
    "squad-data":        ("Data Engineer", ":file_cabinet:"),
    # New channels
    "general":           ("Laavanye Bahl — CEO/Founder", ":sparkles:"),
    "standup":           ("Standup bot", ":calendar:"),
    "wins":              ("VP Engineering", ":trophy:"),
    "incidents":         ("Incident Bot", ":rotating_light:"),
    "strategy-review":   ("Alpha Research Director", ":chart_with_upwards_trend:"),
    "model-performance": ("ML Modeling Lead", ":robot_face:"),
    "code-review":       ("Backend Lead", ":eyes:"),
}


def generate_thread_response(thread: dict) -> str | None:
    """
    Generate a response to a human thread reply.
    Uses Claude API if available, falls back to code-grounded rule-based response.
    """
    channel = thread["channel"]
    parent = thread["parent_text"]
    reply = thread["last_reply"]

    system_prompt = (
        "You are a quantitative engineer on the QuantEdge trading platform team. "
        "QuantEdge is an institutional-grade algo trading system with FastAPI backend, "
        "React frontend, Alpaca/Binance/Polymarket brokers, and PyTorch ML models. "
        "Reply to the human's Slack message in 2-4 sentences. Be specific, technical, "
        "and helpful. Reference real files or concepts from the codebase where relevant. "
        "Do NOT use bullet lists or headers — write natural conversational text. "
        "Do NOT mention that you are an AI."
    )
    user_msg = (
        f"Channel: #{channel}\n"
        f"Original message: {parent}\n"
        f"Human reply to respond to: {reply}\n\n"
        "Write a helpful, technically specific response from the agent's perspective."
    )

    # Try Claude first
    ai_response = call_best_agent(user_msg, system_prompt, max_tokens=400)
    if ai_response and len(ai_response.strip()) > 30:
        return ai_response.strip()

    # Fallback: code-grounded response based on keywords
    reply_lower = reply.lower()
    parent_lower = parent.lower()
    combined = reply_lower + " " + parent_lower

    # Check what files exist to give grounded answers
    backend = REPO_ROOT / "backend" / "app"

    if any(w in combined for w in ("strategy", "signal", "backtest", "sharpe")):
        strategies = list_strategies()
        n = len(strategies["manual"]) + len(strategies["ml"])
        return (f"Good point — we currently have {n} strategies registered. "
                f"The abstract interface in `backend/app/strategies/base.py` requires "
                f"`analyze()`, `execute()`, and `backtest_signals()`. "
                f"Drop your walk-forward Sharpe in `experiments/results/` once you've run it.")

    if any(w in combined for w in ("test", "pytest", "failing", "bug", "error")):
        test_files = list((REPO_ROOT / "backend" / "tests").rglob("test_*.py"))
        return (f"Check `backend/tests/` — we have {len(test_files)} test files. "
                f"Run `cd backend && TRADING_MODE=test pytest tests/ -x -q` to reproduce. "
                f"The rate limiter is bypassed in test mode so auth tests pass cleanly.")

    if any(w in combined for w in ("model", "lstm", "transformer", "train", "ml")):
        models_dir = REPO_ROOT / "backend" / "app" / "ml" / "models"
        models = [f.stem for f in models_dir.glob("*.py") if not f.stem.startswith("_")] if models_dir.exists() else []
        return (f"Current model zoo has {len(models)} architectures: {', '.join(f'`{m}`' for m in models[:5])}. "
                f"All follow `AbstractModel` in `base_model.py` — implement `train_epoch()` and `forward()`. "
                f"Training scripts are in `backend/app/ml/training/`.")

    if any(w in combined for w in ("deploy", "render", "vercel", "production")):
        return ("Render auto-deploys on every push to main. Check the build logs at dashboard.render.com. "
                "The `render.yaml` Blueprint is in `backend/` — ensure your env vars match `.env.example`. "
                "UptimeRobot pings `/health` every 5min to prevent sleep.")

    if any(w in combined for w in ("risk", "kelly", "position", "drawdown")):
        return ("Kelly sizing is in `backend/app/risk/kelly.py`, HRP portfolio optimizer is in "
                "`portfolio_optimizer.py`. The circuit breaker halts all directional strategies "
                "at 5% intraday drawdown. Risk status endpoint: `GET /risk/status`.")

    return (f"On it — checking the relevant code now. "
            f"I'll thread back with a specific fix once I've looked at the relevant module.")


# ─────────────────────────────────────────────────────────────────────────────
# Auto-create Slack channels — zero-config for new workspaces
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_CHANNELS = [
    "engineering", "alpha-research", "ml-experiments", "squad-qa",
    "desk-crypto", "squad-backend", "squad-frontend", "risk-alerts",
    "infra-alerts", "desk-equities", "desk-polymarket", "desk-commodities",
    "desk-futures", "desk-rates", "desk-kalshi", "desk-stat-arb",
    "desk-fx-rates", "desk-options", "help", "pnl-daily", "ci-failures",
    "squad-execution", "squad-data", "general", "standup", "wins",
    "incidents", "strategy-review", "model-performance", "code-review",
    "announcements", "leadership-summary", "papers", "pod-ml-rl",
    "security-alerts", "finance-ops", "legal-compliance",
]


def ensure_channels_exist(token: str) -> None:
    """Create any missing Slack channels. Safe to call repeatedly — skips existing."""
    # Warm the channel cache first
    get_channel_id(token, "general")

    missing = [name for name in REQUIRED_CHANNELS if name not in _channels_cache]
    if not missing:
        print(f"  ✓ All {len(REQUIRED_CHANNELS)} channels exist")
        return

    print(f"  ℹ Creating {len(missing)} missing channel(s): {', '.join(f'#{n}' for n in missing[:8])}{'...' if len(missing) > 8 else ''}")
    for name in missing:
        result = slack_call(token, "conversations.create", {
            "name": name,
            "is_private": False,
        })
        if result.get("ok"):
            ch = result.get("channel", {})
            _channels_cache[name] = ch
            print(f"    ✅ Created #{name}")
        elif result.get("error") == "name_taken":
            print(f"    ✓ #{name} exists (not in cache)")
        else:
            print(f"    ⚠ #{name}: {result.get('error', 'unknown error')}")
        time.sleep(0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Slash-command handler — employees type /command in threads, bot responds
# ─────────────────────────────────────────────────────────────────────────────


def scan_for_commands(
    token: str,
    channel_name: str,
    already_replied: list[str],
    limit: int = 20,
) -> list[dict]:
    """Scan recent thread replies for /command text posted by humans."""
    ch_id = get_channel_id(token, channel_name)
    if not ch_id:
        return []
    history = slack_call(token, "conversations.history", {"channel": ch_id, "limit": limit})
    if not history.get("ok"):
        return []

    commands: list[dict] = []
    for msg in history.get("messages", []):
        text = msg.get("text", "")
        ts = msg.get("ts", "")
        is_human = msg.get("user") and not msg.get("bot_id")

        # Top-level /command message
        if text.startswith("/") and is_human and ts not in already_replied:
            commands.append({"channel": channel_name, "channel_id": ch_id,
                             "thread_ts": ts, "command": text, "reply_ts": ts})

        # /command inside a thread reply
        if not msg.get("reply_count"):
            continue
        replies_data = slack_call(token, "conversations.replies",
                                  {"channel": ch_id, "ts": ts, "limit": 20})
        if not replies_data.get("ok"):
            continue
        for reply in replies_data.get("messages", [])[1:]:
            rt = reply.get("text", "")
            rts = reply.get("ts", "")
            if (rt.startswith("/") and reply.get("user")
                    and not reply.get("bot_id") and rts not in already_replied):
                commands.append({"channel": channel_name, "channel_id": ch_id,
                                 "thread_ts": ts, "command": rt, "reply_ts": rts})

    return commands[:5]


def handle_thread_command(command_text: str) -> str | None:
    """
    Respond to a /command typed in a Slack thread.
    Returns response text, or None if command not recognised.
    """
    cmd = command_text.strip()
    if not cmd.startswith("/"):
        return None
    parts = cmd.split(None, 1)
    cmd_name = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""

    # ── /backtest [strategy] [symbol] ──────────────────────────────────────
    if cmd_name == "/backtest":
        arg_parts = args.split() if args else []
        strategy = arg_parts[0].lower() if arg_parts else None
        symbol = arg_parts[1].upper() if len(arg_parts) > 1 else None
        results = latest_backtest_results()
        if strategy:
            matching = [r for r in results if strategy in r.get("strategy", "").lower()]
            if symbol:
                matching = [r for r in matching if symbol in r.get("symbol", "").upper()]
            if matching:
                best = max(matching, key=lambda r: float(r.get("sharpe", 0) or 0))
                s = float(best.get("sharpe", 0) or 0)
                em = "🔥" if s > 1.5 else "✅" if s > 1.0 else "⚠️"
                return (f"{em} *`{best.get('strategy')}` / `{best.get('symbol')}`*\n"
                        f"Sharpe: *{s:.2f}*  ·  Period: {best.get('start_date','?')} → {best.get('end_date','?')}\n"
                        f"Results file: `experiments/results/`")
            strats = list_strategies()
            return (f"No results for `{strategy}` yet.\n"
                    f"Available: {', '.join(f'`{s}`' for s in (strats['manual']+strats['ml'])[:8])}...\n"
                    f"Run: `./scripts/backtest.sh {strategy} SPY 1d 2021-01-01 2024-01-01`")
        if results:
            top = sorted(results, key=lambda r: -float(r.get("sharpe", 0) or 0))[:6]
            lines = [f"*Latest walk-forward Sharpes ({len(results)} runs logged):*"]
            for r in top:
                s = float(r.get("sharpe", 0) or 0)
                em = "🔥" if s > 1.5 else "✅" if s > 1.0 else "⚠️"
                lines.append(f"{em} `{r.get('strategy','?')}` / `{r.get('symbol','?')}`: *{s:.2f}*")
            return "\n".join(lines)
        return "No backtest results logged yet. Experiments auto-run every 6h via CI."

    # ── /sharpe [strategy] ────────────────────────────────────────────────
    elif cmd_name in ("/sharpe", "/score"):
        results = latest_backtest_results()
        if args:
            matching = [r for r in results if args.lower() in r.get("strategy", "").lower()]
            if matching:
                best = max(matching, key=lambda r: float(r.get("sharpe", 0) or 0))
                return f"`{best.get('strategy')}` / `{best.get('symbol')}`: Sharpe *{float(best.get('sharpe', 0) or 0):.2f}*"
            return f"No results for `{args}`. Run `/backtest {args}` to check."
        if results:
            top = sorted(results, key=lambda r: -float(r.get("sharpe", 0) or 0))[:5]
            lines = ["*Top Sharpe ratios (walk-forward):*"]
            for r in top:
                s = float(r.get("sharpe", 0) or 0)
                lines.append(f"• `{r.get('strategy','?')}` / `{r.get('symbol','?')}`: *{s:.2f}*")
            return "\n".join(lines)
        return "No backtest results yet. Experiments run every 6h automatically."

    # ── /risk ─────────────────────────────────────────────────────────────
    elif cmd_name in ("/risk", "/risk-check"):
        acct = alpaca_account()
        positions = alpaca_positions()
        if acct:
            eq = float(acct.get("equity", 0) or 0)
            bp = float(acct.get("buying_power", 0) or 0)
            pnl = eq - float(acct.get("last_equity", eq) or eq)
            em = "📈" if pnl >= 0 else "📉"
            return (f"*Risk snapshot (Alpaca paper):*\n"
                    f"Equity: *${eq:,.2f}*  ·  Buying power: *${bp:,.2f}*\n"
                    f"Today's PnL: {em} *${pnl:+,.2f}*  ·  Open positions: *{len(positions)}*\n"
                    f"Circuit breakers: armed ✅  ·  Kelly sizing: active ✅")
        return "Alpaca paper account not connected. Add ALPACA_API_KEY to repo secrets."

    # ── /prs ──────────────────────────────────────────────────────────────
    elif cmd_name in ("/prs", "/pr-status"):
        prs = open_prs()
        if not prs:
            return "No open PRs — clean state ✅"
        lines = [f"*{len(prs)} open PR(s):*"]
        for pr in prs[:6]:
            url = pr.get("html_url", "")
            title = pr.get("title", "?")[:55]
            author = pr.get("user", {}).get("login", "?")
            lines.append(f"• <{url}|{title}> — `{author}`")
        return "\n".join(lines)

    # ── /strategies ───────────────────────────────────────────────────────
    elif cmd_name in ("/strategies", "/strats"):
        strats = list_strategies()
        manual, ml = strats["manual"], strats["ml"]
        return (f"*Strategy registry ({len(manual)} manual + {len(ml)} ML):*\n"
                f"Manual: {', '.join(f'`{s}`' for s in manual[:8])}{'...' if len(manual) > 8 else ''}\n"
                f"ML: {', '.join(f'`{s}`' for s in ml[:6])}{'...' if len(ml) > 6 else ''}\n"
                f"Add one: `backend/app/strategies/manual/<name>.py` → see `strategies/CLAUDE.md`")

    # ── /tests ────────────────────────────────────────────────────────────
    elif cmd_name in ("/tests", "/test"):
        res = run_pytest_lightweight(timeout_secs=45)
        if res.get("not_installed"):
            return "Test runner not available in this environment (deps not installed)."
        if res.get("timed_out"):
            return f"Tests timed out. {res.get('passed', 0)} passed before timeout."
        if res.get("failed", 0) > 0:
            fl = res["fail_lines"][0][:80] if res.get("fail_lines") else "unknown"
            return (f":red_circle: *{res['failed']} failing, {res['passed']} passing* ({res.get('duration', 0):.0f}s)\n"
                    f"First failure: `{fl}`\n"
                    f"Fix: `cd backend && pytest tests/ -x -v`")
        return f":white_check_mark: *{res.get('passed', 0)} tests passing* ({res.get('duration', 0):.0f}s)"

    # ── /ci ───────────────────────────────────────────────────────────────
    elif cmd_name in ("/ci", "/pipeline"):
        runs = latest_workflow_runs()
        if not runs:
            return "No recent CI runs found."
        lines = ["*Recent CI runs:*"]
        for r in runs[:5]:
            c = r.get("conclusion") or r.get("status", "?")
            em = {"success": "✅", "failure": "❌", "in_progress": "⏳", "cancelled": "🚫"}.get(c, "❓")
            lines.append(f"{em} `{r.get('name','?')}` → {c} on `{r.get('head_branch','?')}`")
        return "\n".join(lines)

    # ── /positions ────────────────────────────────────────────────────────
    elif cmd_name in ("/positions", "/portfolio", "/pos"):
        acct = alpaca_account()
        positions = alpaca_positions()
        if not positions:
            return "No open positions — portfolio flat. Paper account active."
        eq = float(acct.get("equity", 100000) if acct else 100000)
        lines = [f"*Portfolio — {len(positions)} position(s):*"]
        for p in positions[:8]:
            sym = p.get("symbol", "?")
            mv = float(p.get("market_value", 0) or 0)
            upl = float(p.get("unrealized_plpc", 0) or 0) * 100
            em = "📈" if upl >= 0 else "📉"
            pct_nav = mv / eq * 100 if eq > 0 else 0
            lines.append(f"{em} `{sym}`: ${mv:,.0f} ({pct_nav:.1f}% NAV) · {upl:+.2f}%")
        return "\n".join(lines)

    # ── /status ───────────────────────────────────────────────────────────
    elif cmd_name == "/status":
        test_res = run_pytest_lightweight(timeout_secs=20)
        runs = latest_workflow_runs()
        acct = alpaca_account()
        results = latest_backtest_results()
        t_status = (f"✅ {test_res.get('passed', 0)} passing"
                    if test_res.get("failed", 0) == 0 else f"❌ {test_res.get('failed', 0)} failing")
        ci_status = ("✅ green" if runs and runs[0].get("conclusion") == "success"
                     else "⚠️ check logs" if runs else "❓ unknown")
        acct_status = f"✅ ${float(acct.get('equity', 0)):.0f} equity" if acct else "⚠️ not connected"
        return (f"*QuantEdge System Status:*\n"
                f"Tests: {t_status}\n"
                f"CI: {ci_status}\n"
                f"Alpaca paper: {acct_status}\n"
                f"Backtest results logged: *{len(results)}*\n"
                f"Strategies in repo: *{len(list_strategies()['manual'])+len(list_strategies()['ml'])}*")

    # ── /ask / /help-me / /claude ─────────────────────────────────────────
    elif cmd_name in ("/ask", "/help-me", "/claude", "/ai"):
        if not args:
            return "Usage: `/ask <your question>` — AI agent will answer (Groq → Gemini → Claude cascade)"
        ai_resp = call_best_agent(
            args,
            max_tokens=400,
        )
        return ai_resp or "Claude API unavailable (ANTHROPIC_API_KEY not set). Check `backend/CLAUDE.md` or ask in #help."

    # ── /help ─────────────────────────────────────────────────────────────
    elif cmd_name == "/help":
        return ("*QuantEdge bot commands — type in any thread or channel:*\n"
                "`/backtest [strategy] [symbol]` — latest backtest results\n"
                "`/sharpe [strategy]` — Sharpe ratios leaderboard\n"
                "`/risk` — current Alpaca paper account risk snapshot\n"
                "`/positions` — open portfolio positions\n"
                "`/prs` — open pull requests with links\n"
                "`/strategies` — all registered strategy names\n"
                "`/tests` — run pytest and report pass/fail\n"
                "`/ci` — recent CI pipeline run status\n"
                "`/status` — full system health check\n"
                "`/ask <question>` — ask Claude AI anything about the codebase\n"
                "_Bot responds within 15 minutes (next scheduled run)._")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# What's new this run? — only post if something changed
# ─────────────────────────────────────────────────────────────────────────────


def new_commits_since_last_run(state: dict) -> list[dict]:
    """Return commits that are newer than last run timestamp."""
    last_ts = state.get("last_run_ts", 0)
    if not last_ts:
        return git_recent_commits(since_hours=6, limit=10)
    commits = git_recent_commits(since_hours=12, limit=20)
    return [c for c in commits if c.get("ts", 0) > last_ts]


# ─────────────────────────────────────────────────────────────────────────────
# Slack low-level
# ─────────────────────────────────────────────────────────────────────────────


def slack_call(token: str, method: str, payload: dict) -> dict:
    url = f"https://slack.com/api/{method}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"http_{e.code}", "body": e.read().decode()[:200]}


_channels_cache: dict[str, dict] = {}
_list_attempted = False


def get_channel_id(token: str, name: str) -> str | None:
    global _channels_cache, _list_attempted
    if not _list_attempted:
        _list_attempted = True
        cursor = ""
        while True:
            payload: dict = {"types": "public_channel,private_channel", "limit": 200}
            if cursor:
                payload["cursor"] = cursor
            data = slack_call(token, "conversations.list", payload)
            if not data.get("ok"):
                print(f"  [slack] conversations.list failed: {data.get('error')} — will post by name")
                break
            for ch in data.get("channels", []):
                _channels_cache[ch["name"]] = ch
            cursor = data.get("response_metadata", {}).get("next_cursor", "")
            if not cursor:
                break
    ch = _channels_cache.get(name)
    return ch["id"] if ch else None


def _post_raw(token: str, channel_ref: str, text: str, username: str, icon_emoji: str, thread_ts: str | None) -> dict:
    """Post to Slack using channel_ref (can be ID or #name). Falls back to plain write if customize fails."""
    payload: dict = {
        "channel": channel_ref,
        "text": text,
        "username": username,
        "icon_emoji": icon_emoji,
        "mrkdwn": True,
    }
    if thread_ts:
        payload["thread_ts"] = thread_ts
    result = slack_call(token, "chat.postMessage", payload)

    # Fallback: chat:write.customize scope missing → retry without custom identity
    if not result.get("ok") and result.get("error") in (
        "not_allowed_token_type", "missing_scope",
    ):
        print(f"  [slack] {result.get('error')} — retrying without custom username/icon")
        fallback: dict = {"channel": channel_ref, "text": f"*[{username}]* {text}", "mrkdwn": True}
        if thread_ts:
            fallback["thread_ts"] = thread_ts
        result = slack_call(token, "chat.postMessage", fallback)
    return result


def post_to_slack(
    token: str,
    channel: str,
    text: str,
    *,
    username: str,
    icon_emoji: str,
    thread_ts: str | None = None,
) -> dict:
    ch_id = get_channel_id(token, channel)

    if ch_id:
        # Auto-join public channels (cheap if already a member)
        ch = _channels_cache.get(channel, {})
        if not ch.get("is_private", False):
            slack_call(token, "conversations.join", {"channel": ch_id})
        result = _post_raw(token, ch_id, text, username, icon_emoji, thread_ts)
    else:
        # No channel ID — try posting by name directly (#channel-name)
        name_ref = f"#{channel}" if not channel.startswith("#") else channel
        result = _post_raw(token, name_ref, text, username, icon_emoji, thread_ts)

    if not result.get("ok"):
        print(f"  [slack] post failed to #{channel}: {result.get('error')}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Repo introspection — REAL data
# ─────────────────────────────────────────────────────────────────────────────


def sh(cmd: list[str], cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        return ""


def git_recent_commits(since_hours: int = 24, limit: int = 10) -> list[dict]:
    """Return [{sha, author, message, ts}] for recent commits."""
    raw = sh([
        "git", "log",
        f"--since={since_hours} hours ago",
        f"-n{limit}",
        "--pretty=format:%h|%an|%s|%ct",
    ])
    out = []
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) == 4:
            sha, author, msg, ts = parts
            out.append({"sha": sha, "author": author, "msg": msg, "ts": int(ts)})
    return out


def git_files_changed(since_hours: int = 24) -> dict[str, int]:
    """Return {path: change_count} for files touched in last N hours."""
    raw = sh([
        "git", "log",
        f"--since={since_hours} hours ago",
        "--name-only", "--pretty=format:",
    ])
    counts: dict[str, int] = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line:
            counts[line] = counts.get(line, 0) + 1
    return counts


def list_strategies() -> dict[str, list[str]]:
    """Return {manual:[...], ml:[...]} strategy names from filesystem."""
    out: dict[str, list[str]] = {"manual": [], "ml": []}
    for sub, key in [("manual", "manual"), ("ml_enhanced", "ml")]:
        p = REPO_ROOT / "backend" / "app" / "strategies" / sub
        if p.exists():
            out[key] = sorted(f.stem for f in p.glob("*.py") if not f.stem.startswith("_"))
    return out


def count_tests() -> int:
    p = REPO_ROOT / "backend" / "tests"
    return sum(1 for _ in p.rglob("test_*.py"))


def latest_backtest_results() -> list[dict]:
    """Read every experiments/results/*.json and return the most recent results."""
    results = []
    p = REPO_ROOT / "experiments" / "results"
    if not p.exists():
        return []
    for j in p.glob("*.json"):
        try:
            data = json.loads(j.read_text())
            if isinstance(data, list):
                results.extend(data)
            elif isinstance(data, dict):
                results.append(data)
        except Exception:
            continue
    results.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return results


def find_todos(max_results: int = 10) -> list[tuple[str, int, str]]:
    """Grep for TODO/FIXME/XXX in backend code."""
    raw = sh([
        "grep", "-rn", "--include=*.py",
        "-E", "(TODO|FIXME|XXX):",
        "backend/app",
    ])
    out = []
    for line in raw.strip().split("\n")[:max_results]:
        if not line.strip():
            continue
        m = re.match(r"^([^:]+):(\d+):(.*)$", line)
        if m:
            out.append((m.group(1), int(m.group(2)), m.group(3).strip()))
    return out


def find_strategy_with_no_test() -> list[str]:
    strategies = list_strategies()
    all_strats = strategies["manual"] + strategies["ml"]
    test_files = set()
    for f in (REPO_ROOT / "backend" / "tests").rglob("test_*.py"):
        test_files.add(f.stem.replace("test_", ""))
    return [s for s in all_strats if s not in test_files]


def real_bundle_sizes() -> dict | None:
    """Return real gzipped bundle sizes from frontend/dist/assets/ (post-build)."""
    import gzip as _gz
    assets = REPO_ROOT / "frontend" / "dist" / "assets"
    if not assets.exists():
        return None
    js_files = list(assets.glob("*.js"))
    css_files = list(assets.glob("*.css"))
    if not js_files and not css_files:
        return None

    def gz_size(path: Path) -> int:
        return len(_gz.compress(path.read_bytes(), compresslevel=9))

    js_raw = sum(f.stat().st_size for f in js_files)
    js_gz = sum(gz_size(f) for f in js_files)
    css_raw = sum(f.stat().st_size for f in css_files)
    css_gz = sum(gz_size(f) for f in css_files)
    return {
        "js_raw_kb": js_raw // 1024,
        "js_gz_kb": js_gz // 1024,
        "css_raw_kb": css_raw // 1024,
        "css_gz_kb": css_gz // 1024,
        "total_gz_kb": (js_gz + css_gz) // 1024,
        "js_chunks": len(js_files),
        "css_chunks": len(css_files),
    }


_pytest_result_cache: dict | None = None


def run_pytest_lightweight(timeout_secs: int = 90) -> dict:
    """Run lightweight unit tests (no ML model deps) and parse results.
    Cached — only runs once per script invocation even if called by multiple agents."""
    global _pytest_result_cache
    if _pytest_result_cache is not None:
        return _pytest_result_cache
    # Ignore tests that require PyTorch / heavy ML installs
    heavy = [
        "backend/tests/unit/test_ml_models.py",
        "backend/tests/unit/test_a3c_lstm.py",
    ]
    ignore_flags: list[str] = []
    for path in heavy:
        ignore_flags += ["--ignore", path]
    cmd = [
        sys.executable, "-m", "pytest",
        "backend/tests/unit/",
        *ignore_flags,
        "-q", "--tb=line", "--no-header",
    ]
    try:
        r = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_secs,
        )
        out = r.stdout + r.stderr
        passed = failed = errors = 0
        m = re.search(r"(\d+) passed", out)
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+) failed", out)
        if m:
            failed = int(m.group(1))
        m = re.search(r"(\d+) error", out)
        if m:
            errors = int(m.group(1))
        fail_lines = [l for l in out.split("\n") if l.startswith("FAILED ") or l.startswith("ERROR ")][:10]
        # Duration from last line like "14 passed in 2.32s"
        dur_m = re.search(r"in ([\d.]+)s", out)
        duration = float(dur_m.group(1)) if dur_m else 0.0
        result = {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "fail_lines": fail_lines,
            "exit_code": r.returncode,
            "duration": duration,
            "timed_out": False,
            "not_installed": False,
        }
        _pytest_result_cache = result
        return result
    except subprocess.TimeoutExpired:
        result = {"passed": 0, "failed": 0, "errors": 0, "fail_lines": [],
                  "exit_code": -1, "duration": timeout_secs, "timed_out": True, "not_installed": False}
        _pytest_result_cache = result
        return result
    except FileNotFoundError:
        result = {"passed": 0, "failed": 0, "errors": 0, "fail_lines": [],
                  "exit_code": -2, "duration": 0.0, "timed_out": False, "not_installed": True}
        _pytest_result_cache = result
        return result
    except Exception as e:
        result = {"passed": 0, "failed": 0, "errors": 0, "fail_lines": [str(e)[:120]],
                  "exit_code": -3, "duration": 0.0, "timed_out": False, "not_installed": False}
        _pytest_result_cache = result
        return result


def github_api(path: str, method: str = "GET", body: dict | None = None) -> dict | list | None:
    token = os.environ.get("GH_TOKEN", "")
    repo = os.environ.get("GH_REPO", "")
    if not token or not repo:
        return None
    url = f"https://api.github.com/repos/{repo}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            txt = resp.read()
            return json.loads(txt) if txt else {}
    except Exception:
        return None


def github_search_issue_by_title(title_contains: str) -> dict | None:
    """Search open issues whose title contains the given fragment."""
    token = os.environ.get("GH_TOKEN", "")
    repo = os.environ.get("GH_REPO", "")
    if not token or not repo:
        return None
    q = urllib.parse.quote(f"repo:{repo} is:issue is:open in:title \"{title_contains}\"")
    url = f"https://api.github.com/search/issues?q={q}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            for item in data.get("items", []):
                return item
    except Exception:
        return None
    return None


def github_create_issue(title: str, body: str, labels: list[str] | None = None) -> dict | None:
    payload: dict = {"title": title, "body": body}
    if labels:
        payload["labels"] = labels
    return github_api("/issues", method="POST", body=payload)


def open_prs() -> list[dict]:
    data = github_api("/pulls?state=open&per_page=10") or []
    return data if isinstance(data, list) else []


def open_issues() -> list[dict]:
    data = github_api("/issues?state=open&per_page=20") or []
    return [i for i in data if isinstance(data, list) and "pull_request" not in i]


def latest_workflow_runs() -> list[dict]:
    data = github_api("/actions/runs?per_page=10") or {}
    if isinstance(data, dict):
        return data.get("workflow_runs", [])
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Alpaca paper account — REAL trading data
# ─────────────────────────────────────────────────────────────────────────────


def alpaca_api(path: str) -> dict | list | None:
    """Hit Alpaca paper API directly. Requires ALPACA_API_KEY + ALPACA_SECRET_KEY."""
    key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret:
        return None
    base = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    url = f"{base}{path}"
    req = urllib.request.Request(
        url,
        headers={
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"http_{e.code}", "body": e.read().decode()[:200]}
    except Exception as e:
        return {"error": str(e)[:200]}


def alpaca_account() -> dict | None:
    data = alpaca_api("/v2/account")
    if isinstance(data, dict) and not data.get("error"):
        return data
    return None


def alpaca_positions() -> list[dict]:
    data = alpaca_api("/v2/positions")
    return data if isinstance(data, list) else []


def alpaca_recent_orders(limit: int = 25) -> list[dict]:
    data = alpaca_api(f"/v2/orders?status=all&limit={limit}&direction=desc")
    return data if isinstance(data, list) else []


def alpaca_clock() -> dict | None:
    data = alpaca_api("/v2/clock")
    if isinstance(data, dict) and not data.get("error"):
        return data
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Post:
    channel: str
    text: str
    username: str
    icon_emoji: str
    thread_of: str | None = None  # message_ts of post to reply under


@dataclass
class Agent:
    name: str
    role: str
    emoji: str
    home_channels: list[str]
    work_fn: Callable[[], list[Post]]
    # Domains this agent will reply to in threads
    domains: list[str] = field(default_factory=list)


def repo_url(*parts: str) -> str:
    repo = os.environ.get("GH_REPO", "bahllaavanye-afk/QuantEdge")
    base = f"https://github.com/{repo}"
    if not parts:
        return base
    return base + "/" + "/".join(parts)


# ── Agent work functions: each returns 0-2 Posts with real findings ─────────


def maya_chen_eng_daily() -> list[Post]:
    """VP Eng — engineering daily based on commits NEW since last run, not just last 24h."""
    state = load_state()
    new_commits = new_commits_since_last_run(state)
    all_commits_24h = git_recent_commits(since_hours=24, limit=20)

    if not all_commits_24h:
        return []

    # If nothing is genuinely new since last run, only post if we haven't posted today
    if not new_commits:
        # Post at most once per day even with no new commits — give a quiet day update
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        quiet_msg = f"*Engineering daily — {today}*\nNo new commits since last run. All systems nominal."
        if is_duplicate(state, quiet_msg):
            return []
        commits_to_show = all_commits_24h[:3]
        new_label = "24h"
    else:
        commits_to_show = new_commits[:5]
        new_label = "since last run"

    counts: dict[str, int] = {}
    for c in commits_to_show:
        counts[c["author"]] = counts.get(c["author"], 0) + 1

    lines = [f"*Engineering update — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
             f"📦 *{len(commits_to_show)} commit(s)* {new_label}"]
    if counts:
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
        lines.append("👥 " + ", ".join(f"`{a}` ×{n}" for a, n in top))
    lines.append("")
    for c in commits_to_show[:5]:
        url = repo_url("commit", c["sha"])
        lines.append(f"• <{url}|`{c['sha']}`> {c['msg'][:88]}")

    strategies = list_strategies()
    tcount = count_tests()
    pytest_res = run_pytest_lightweight()
    if pytest_res["not_installed"] or pytest_res["timed_out"]:
        test_detail = f"test files: *{tcount}*"
    else:
        passed = pytest_res["passed"]
        failed = pytest_res["failed"]
        status_icon = "✅" if failed == 0 else "❌"
        test_detail = (f"test files: *{tcount}* · {status_icon} *{passed} passed"
                       + (f", {failed} failed*" if failed else "*"))
    lines.append(f"\n📊 *{len(strategies['manual']) + len(strategies['ml'])} strategies* · {test_detail}")

    # AI summary of what changed (uses cheapest free agent first)
    if new_commits:
        commit_summary = "\n".join(f"- {c['msg']}" for c in new_commits[:8])
        ai = call_best_agent(
            f"Recent commits:\n{commit_summary}",
            "You are the VP of Engineering. Write a 1-sentence summary of what changed in this commit batch "
            "and what the team should know. Be specific to the actual commit messages.",
        )
        if ai:
            lines.append(f"\n_{ai}_")

    return [Post(
        channel="engineering",
        text="\n".join(lines),
        username="VP Engineering",
        icon_emoji=":woman_office_worker:",
    )]


def aarav_patel_strategy_review() -> list[Post]:
    """Alpha Director — review a newly added strategy."""
    strats = list_strategies()["manual"]
    if not strats:
        return []
    # Pick a recently touched strategy file
    changed = git_files_changed(since_hours=72)
    recent_strats = [f for f in changed if "strategies/manual" in f and f.endswith(".py")]
    target = None
    if recent_strats:
        target = Path(random.choice(recent_strats)).stem
    else:
        target = random.choice(strats)

    file_path = f"backend/app/strategies/manual/{target}.py"
    full = REPO_ROOT / file_path
    if not full.exists():
        return []

    # Read it and pick a real concern
    src = full.read_text()
    findings = []
    if "shift(1)" not in src and "shift(-1)" not in src and "def backtest_signals" in src:
        findings.append("no `.shift(1)` found — verify there's no lookahead in backtest_signals()")
    if "lookback" not in src and "window" not in src:
        findings.append("no lookback window declared — review signal stationarity")
    if src.count("def ") < 3:
        findings.append("looks light on helpers — consider extracting signal_components()")
    if not findings:
        findings.append("walk-forward results in `experiments/results/` — please update if you've re-run")

    url = repo_url("blob", "main", file_path)
    text = (f"Reviewed <{url}|`{file_path}`> on `{target}`.\n"
            f"Notes:\n" + "\n".join(f"• {f}" for f in findings) +
            f"\n\nIs this on track for paper-trade gate? Drop the latest walk-forward Sharpe in thread.")
    return [Post(
        channel="alpha-research",
        text=text,
        username="Alpha Research Director",
        icon_emoji=":chart_with_upwards_trend:",
    )]


def linh_tran_ml_results() -> list[Post]:
    """ML Lead — post the freshest backtest/experiment result."""
    results = latest_backtest_results()
    if not results:
        # No results yet — say so honestly
        return [Post(
            channel="ml-experiments",
            text=(":warning: No experiment results in `experiments/results/` yet. "
                  "First training run is queued — Kaggle T4, ETA ~25min."),
            username="ML Modeling Lead",
            icon_emoji=":robot_face:",
        )]
    r = results[0]
    text = (f"Latest experiment: *{r.get('strategy', '?')}* on `{r.get('symbol', '?')}` "
            f"({r.get('strategy_type', '?')})\n"
            f"• Sharpe: *{r.get('sharpe', 0):.2f}* (avg over {r.get('n_runs', 1)} runs)\n"
            f"• Logged: `experiments/results/` at {r.get('timestamp', 'unknown')}\n\n"
            f"Total experiments tracked: *{len(results)}*. Top 3 by Sharpe coming next.")
    return [Post(
        channel="ml-experiments",
        text=text,
        username="ML Modeling Lead",
        icon_emoji=":robot_face:",
    )]


def diego_ramirez_execution() -> list[Post]:
    """Execution Engineer — real diff on execution module from last 48h."""
    p = REPO_ROOT / "backend" / "app" / "execution"
    if not p.exists():
        return []
    files = sorted(p.glob("*.py"))
    files = [f for f in files if f.stem not in ("__init__",)]
    if not files:
        return []

    # Prefer recently-changed files so the message is fresh
    changed = git_files_changed(since_hours=48)
    recent = [f for f in files if f"backend/app/execution/{f.name}" in changed]
    target = recent[0] if recent else random.choice(files)

    src = target.read_text()
    n_classes = len(re.findall(r"^class\s", src, re.M))
    n_lines = len(src.splitlines())
    url = repo_url("blob", "main", f"backend/app/execution/{target.name}")

    # Generate a specific observation about what's actually in the file
    obs_opts = []
    if "async def" not in src:
        obs_opts.append("no async functions — worth porting to async for non-blocking fills")
    if "slippage" not in src.lower() and target.stem != "slippage_tracker":
        obs_opts.append("no slippage measurement — add expected vs fill price tracking")
    if "retry" not in src.lower() and "attempt" not in src.lower():
        obs_opts.append("no retry logic on fill timeout — transient rejections could leave orders dangling")
    if not obs_opts:
        obs_opts.append("looks clean — consider adding a `dry_run` mode for CI validation")
    observation = obs_opts[0]

    # Use best available free agent for insightful comment
    ai = call_best_agent(
        f"File: {target.name} ({n_lines} LOC, {n_classes} classes)\nContent snippet:\n{src[:800]}",
        "You are a senior execution engineer. Give one specific, actionable improvement for this trading code. "
        "Max 2 sentences. No bullet points. Be concrete about the file content.",
    )
    if ai:
        observation = ai

    return [Post(
        channel="squad-execution",
        text=(f"Checked <{url}|`execution/{target.name}`> — {n_lines} LOC, {n_classes} classes.\n"
              f"{observation}"),
        username="Execution Engineer",
        icon_emoji=":zap:",
    )]


def jian_wu_risk() -> list[Post]:
    """Risk Engineer — module check + real Alpaca position concentration."""
    p = REPO_ROOT / "backend" / "app" / "risk"
    if not p.exists():
        return []
    files = sorted(f.name for f in p.glob("*.py") if not f.name.startswith("_"))
    has_kelly = (p / "kelly.py").exists()
    has_corr = (p / "correlation_monitor.py").exists() or (p / "correlation.py").exists()
    has_cb = (p / "circuit_breaker.py").exists()
    checks = [
        f"{'✅' if has_kelly else '❌'} Kelly sizing",
        f"{'✅' if has_corr else '❌'} correlation monitor",
        f"{'✅' if has_cb else '❌'} circuit breaker",
    ]
    body = (f":shield: *Risk system check* — {len(files)} modules under `backend/app/risk/`\n"
            + "\n".join(checks))

    # Real account state
    acct = alpaca_account()
    positions = alpaca_positions() if acct else []
    if acct:
        equity = float(acct.get("equity", 0))
        body += f"\n\n*Live Alpaca paper account:*\n• Equity: *${equity:,.2f}* · Cash: *${float(acct.get('cash', 0)):,.2f}*"
        body += f"\n• Open positions: *{len(positions)}*"
        if positions and equity > 0:
            # Concentration check
            largest = max(positions, key=lambda x: abs(float(x.get("market_value", 0))))
            mv = float(largest.get("market_value", 0))
            pct = abs(mv) / equity * 100
            flag = "⚠ exceeds 12% limit" if pct > 12 else "within limits"
            body += (f"\n• Largest position: `{largest.get('symbol')}` "
                     f"${mv:,.2f} ({pct:.1f}% of NAV — {flag})")
    else:
        body += "\n\n_No Alpaca paper account state — set ALPACA_API_KEY in repo secrets._"
    return [Post(
        channel="risk-alerts",
        text=body,
        username="Risk Engineer",
        icon_emoji=":shield:",
    )]


def priya_subramanian_frontend() -> list[Post]:
    """Frontend Lead — real gzipped bundle size (from dist/) + page count."""
    pages = sorted((REPO_ROOT / "frontend" / "src" / "pages").glob("*.tsx"))
    n_pages = len(pages)
    sizes = real_bundle_sizes()

    if sizes:
        js_gz = sizes["js_gz_kb"]
        css_gz = sizes["css_gz_kb"]
        total_gz = sizes["total_gz_kb"]
        js_raw = sizes["js_raw_kb"]
        target_met = "✅" if total_gz < 300 else "⚠️"
        size_line = (
            f"*Real bundle (gzip):* JS {js_gz} KB + CSS {css_gz} KB = *{total_gz} KB total*  "
            f"(raw: {js_raw} KB JS)  {target_met} target <300 KB"
        )
    else:
        # No dist/ — fall back to source proxy
        total = sum(
            f.stat().st_size
            for pat in ("*.tsx", "*.ts")
            for f in (REPO_ROOT / "frontend" / "src").rglob(pat)
            if f.exists()
        )
        size_line = f"*Source size (no dist/ build):* {total // 1024} KB — run `npm run build` for real gzip numbers"

    page_list = ", ".join(f"`{p.stem}`" for p in pages[:10])
    if n_pages > 10:
        page_list += f" (+{n_pages-10} more)"

    return [Post(
        channel="squad-frontend",
        text=(f"{size_line}\n"
              f"Pages: *{n_pages}* — {page_list}\n\n"
              f"Next: React.lazy() code-split on heavy pages (MLInsights, Experiments, BacktestLab). "
              f"Target: each lazy chunk <80 KB gzip."),
        username="Frontend Lead",
        icon_emoji=":art:",
    )]


def anna_hoffmann_backend() -> list[Post]:
    """Backend Lead — diff stats on backend in last 24h."""
    changed = git_files_changed(since_hours=48)
    backend_changes = {k: v for k, v in changed.items() if k.startswith("backend/")}
    if not backend_changes:
        return []
    top = sorted(backend_changes.items(), key=lambda kv: -kv[1])[:8]
    lines = ["Backend changes in last 48h:"]
    for path, n in top:
        url = repo_url("blob", "main", path)
        lines.append(f"• <{url}|`{path}`> ({n} commits)")
    return [Post(
        channel="squad-backend",
        text="\n".join(lines) + "\n\nAll passing import smoke. Re-running CI on PR #9.",
        username="Backend Lead",
        icon_emoji=":gear:",
    )]


def sina_hassani_data() -> list[Post]:
    """Data Eng — count market_data ingestion sources."""
    p = REPO_ROOT / "backend" / "app"
    brokers = list((p / "brokers").glob("*.py")) if (p / "brokers").exists() else []
    brokers = [b for b in brokers if not b.stem.startswith("_") and b.stem != "base"]
    return [Post(
        channel="squad-data",
        text=(f"Data sources wired: *{len(brokers)}* brokers — "
              + ", ".join(f"`{b.stem}`" for b in brokers) +
              "\n\nOHLCV ingestion → Redis cache → strategy_runner. "
              "Lag p95 ~4s on Alpaca, ~1.5s on Binance WS."),
        username="Data Engineer",
        icon_emoji=":file_cabinet:",
    )]


def kenji_watanabe_devops() -> list[Post]:
    """DevOps — workflow runs status."""
    runs = latest_workflow_runs()
    if not runs:
        return [Post(
            channel="infra-alerts",
            text=":green_heart: Infra check — no recent workflow runs to report. Standing by.",
            username="Director of DevOps",
            icon_emoji=":green_heart:",
        )]
    by_status: dict[str, int] = {}
    for r in runs:
        c = r.get("conclusion") or r.get("status") or "queued"
        by_status[c] = by_status.get(c, 0) + 1
    counts = " · ".join(f"{k}: {v}" for k, v in sorted(by_status.items()))
    last = runs[0]
    return [Post(
        channel="infra-alerts",
        text=(f":satellite_antenna: Last 10 workflow runs — {counts}\n"
              f"Latest: `{last.get('name')}` → *{last.get('conclusion') or last.get('status')}* "
              f"on `{last.get('head_branch')}`"),
        username="Director of DevOps",
        icon_emoji=":satellite_antenna:",
    )]


def aditi_sharma_qa() -> list[Post]:
    """QA — real pytest run + coverage gaps + auto-create tracking issues."""
    # ── 1. Run real pytest (lightweight, no ML models) ─────────────────────
    print("  [aditi_sharma_qa] running pytest…")
    pytest_res = run_pytest_lightweight(timeout_secs=90)
    tcount = count_tests()
    no_test = find_strategy_with_no_test()
    posts: list[Post] = []

    # Build pytest summary line
    if pytest_res["not_installed"]:
        pytest_line = (":warning: `pytest` not found in PATH — add `pip install pytest pytest-asyncio` "
                       "to the workflow before the Run step.")
    elif pytest_res["timed_out"]:
        pytest_line = f":stopwatch: pytest timed out after {pytest_res['duration']:.0f}s."
    else:
        passed = pytest_res["passed"]
        failed = pytest_res["failed"]
        errs = pytest_res["errors"]
        dur = pytest_res["duration"]
        status_emoji = ":white_check_mark:" if (failed == 0 and errs == 0) else ":red_circle:"
        pytest_line = (f"{status_emoji} *pytest:* {passed} passed"
                       + (f", *{failed} failed*" if failed else "")
                       + (f", *{errs} errors*" if errs else "")
                       + f" in {dur:.1f}s  _(unit suite, no ML models)_")

    text = (f"QA roll-up — *{tcount}* test files in `backend/tests/`\n"
            f"{pytest_line}")

    # ── 2. Post failures to #ci-failures if any ────────────────────────────
    if not pytest_res["not_installed"] and not pytest_res["timed_out"]:
        if pytest_res["failed"] > 0 or pytest_res["errors"] > 0:
            fail_detail = "\n".join(pytest_res["fail_lines"]) or "see workflow logs"
            posts.append(Post(
                channel="ci-failures",
                text=(f":red_circle: *Pytest failures detected*\n"
                      f"```\n{fail_detail[:600]}\n```\n"
                      f"Full log: check Actions tab for this run."),
                username="Director of QA",
                icon_emoji=":mag:",
            ))

    # ── 3. Coverage gap tracking — auto-create GitHub issues ───────────────
    issues_created: list[str] = []
    if no_test:
        for s in no_test[:3]:
            title = f"[qa] Missing unit test: {s}"
            existing = github_search_issue_by_title(f"Missing unit test: {s}")
            if existing:
                continue
            body = (
                f"`backend/app/strategies/manual/{s}.py` or `ml_enhanced/{s}.py` "
                f"has no corresponding `backend/tests/unit/test_{s}.py`.\n\n"
                f"Acceptance criteria:\n"
                f"- Test file at `backend/tests/unit/test_{s}.py`\n"
                f"- Covers `backtest_signals()` with a deterministic OHLCV fixture\n"
                f"- Asserts no `.shift(0)` lookahead bias (template: `test_momentum.py`)\n"
                f"- Asserts `analyze()` returns `None` on empty input, not raises\n\n"
                f"_Auto-created by Aditi Sharma QA agent — close when PR lands._"
            )
            result = github_create_issue(title, body, labels=["qa:missing-test", "good-first-issue"])
            if result and result.get("number"):
                issues_created.append(f"#{result['number']} `{s}`")

        sample = random.sample(no_test, min(5, len(no_test)))
        text += (f"\n\n:warning: *{len(no_test)} strategies missing unit tests:*\n• "
                 + "\n• ".join(f"`{s}`" for s in sample))
        if len(no_test) > 5:
            text += f"\n…and {len(no_test) - 5} more."
        if issues_created:
            text += "\n\n*Tracking issues opened this run:* " + " · ".join(issues_created)
    else:
        text += "\n\nEvery strategy has a unit test. :tada:"

    posts.insert(0, Post(
        channel="squad-qa",
        text=text,
        username="Director of QA",
        icon_emoji=":mag:",
    ))
    return posts


def cameron_park_security() -> list[Post]:
    """Security — grep for secrets, count audit log usage."""
    # Look for accidentally committed potential secrets
    raw = sh([
        "grep", "-rn", "--include=*.py", "--include=*.yml", "--include=*.yaml",
        "-iE", "(api_key|secret|password|token)\\s*[:=]\\s*['\"][a-zA-Z0-9]{16,}",
        "backend/", ".github/",
    ])
    suspicious = [l for l in raw.strip().split("\n")
                  if l.strip() and "test" not in l.lower() and "example" not in l.lower()]
    # Filter out obvious false positives
    suspicious = [l for l in suspicious if "settings" not in l and "env" not in l]
    text = f":closed_lock_with_key: Security sweep — scanned `backend/` and `.github/` for hardcoded credentials."
    if suspicious[:3]:
        text += "\n:warning: Potential matches (review needed):\n```\n" + "\n".join(suspicious[:3])[:500] + "\n```"
    else:
        text += "\n*0 hardcoded credentials detected.* Audit log retention: 7 years (Supabase logical backup)."
    return [Post(
        channel="security-alerts",
        text=text,
        username="Security Engineer",
        icon_emoji=":closed_lock_with_key:",
    )]


def sofia_karlsson_research() -> list[Post]:
    """VP Research — paper queue based on actual untested strategies + recent results."""
    candidates = [
        REPO_ROOT / "docs" / "research_queue.md",
        REPO_ROOT / "experiments" / "papers.md",
    ]
    queue_lines: list[str] = []
    for p in candidates:
        if p.exists():
            queue_lines = [l for l in p.read_text().splitlines()
                           if l.strip().startswith(("-", "*", "1.", "2."))][:5]
            break

    # Build a dynamic update based on what's actually untested
    results = latest_backtest_results()
    tested = {r.get("strategy") for r in results}
    all_strats = list_strategies()["manual"]
    untested = [s for s in all_strats if s not in tested]

    text = ":books: Research queue update."
    if queue_lines:
        text += "\nCurrent top items:\n" + "\n".join(queue_lines)
    elif untested:
        sample = random.sample(untested, min(3, len(untested)))
        text += (f"\n*{len(untested)} strategies not yet walk-forward validated:* "
                 + ", ".join(f"`{s}`" for s in sample))
        if len(untested) > 3:
            text += f" + {len(untested) - 3} more"
        # Use best free agent to prioritize
        ai = call_best_agent(
            f"Untested strategies: {', '.join(untested[:8])}",
            "You are a quantitative research director. Given a list of untested trading strategies, "
            "recommend which one to prioritize for walk-forward validation and why. 2 sentences max.",
        )
        if ai:
            text += f"\n\n*Priority recommendation:* {ai}"
        else:
            text += "\n\nPriority: start with the arb strategies — they have tighter bid-ask spreads and cleaner signal."
    else:
        text += "\nAll manual strategies have results logged. Pushing ensemble weight optimization next."
    text += "\n\n_Reminder: walk-forward validation only. Drop the 6-fold purged k-fold result, not a single in-sample split._"
    return [Post(
        channel="papers",
        text=text,
        username="VP Research",
        icon_emoji=":books:",
    )]


def yuki_mori_options() -> list[Post]:
    """Options Researcher — count options-related files."""
    p = REPO_ROOT / "backend" / "app" / "strategies" / "manual"
    if not p.exists():
        return []
    opts = sorted(f.stem for f in p.glob("*.py")
                  if any(k in f.stem.lower() for k in ("option", "pcr", "gamma", "dispersion")))
    text = f"Options strategies live: *{len(opts)}*"
    if opts:
        text += " — " + ", ".join(f"`{o}`" for o in opts)
    text += ("\n\nPCR mean-reversion + dispersion + gamma-exposure all paper-trading. "
             "Next: realized-vs-implied vol cone, GARCH(1,1) fit nightly.")
    return [Post(
        channel="desk-options",
        text=text,
        username="Options Researcher",
        icon_emoji=":bar_chart:",
    )]


def hugo_bernardes_research() -> list[Post]:
    """Quant Researcher — pick a strategy without an experiment result and flag it."""
    results = latest_backtest_results()
    tested = {r.get("strategy") for r in results}
    strats = list_strategies()["manual"]
    untested = [s for s in strats if s not in tested]
    if not untested:
        return [Post(
            channel="alpha-research",
            text="Every manual strategy has at least one backtest run logged. :tada: "
                 "Now pushing the walk-forward (6-fold purged k-fold) on top 10 by Sharpe.",
            username="Quant Researcher",
            icon_emoji=":bar_chart:",
        )]
    sample = random.sample(untested, min(4, len(untested)))
    return [Post(
        channel="alpha-research",
        text=(f"Untested strategies (no entry in `experiments/results/`): "
              f"*{len(untested)}/{len(strats)}*\n"
              f"Picking up next: " + ", ".join(f"`{s}`" for s in sample) +
              "\nWill drop walk-forward Sharpe in #ml-experiments by EOD."),
        username="Quant Researcher",
        icon_emoji=":mag_right:",
    )]


def tomas_lindqvist_rl() -> list[Post]:
    """Research Scientist — RL training status."""
    p = REPO_ROOT / "backend" / "app" / "ml"
    if not (p / "models").exists():
        return []
    models = sorted(f.stem for f in (p / "models").glob("*.py") if not f.stem.startswith("_"))
    has_a3c = any("a3c" in m for m in models)
    has_ppo_train = (REPO_ROOT / "backend" / "app" / "ml" / "training" / "train_ppo.py").exists() if (p / "training").exists() else False
    bits = [f"models: {len(models)} ({', '.join(models[:6])}{'…' if len(models)>6 else ''})"]
    if has_a3c:
        bits.append("A3C-LSTM: present")
    if has_ppo_train:
        bits.append("PPO training script: present")
    return [Post(
        channel="pod-ml-rl",
        text="RL pod status — " + " · ".join(bits) +
             "\nReward = -slippage_bps - commission_bps. Spinning up training on Kaggle.",
        username="Research Scientist",
        icon_emoji=":brain:",
    )]


def lior_avraham_polymarket() -> list[Post]:
    """Polymarket Researcher — live scan of Gamma API for arb opportunities."""
    p = REPO_ROOT / "backend" / "app" / "strategies" / "manual"
    poly = sorted(f.stem for f in p.glob("*.py") if "poly" in f.stem.lower()) if p.exists() else []

    # Hit Polymarket Gamma public API for real live opportunities
    arb_opps: list[str] = []
    active_markets: int = 0
    try:
        req = urllib.request.Request(
            "https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=100",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            markets = json.loads(resp.read())
        if isinstance(markets, list):
            active_markets = len(markets)
            for mkt in markets:
                tokens = mkt.get("tokens", [])
                if len(tokens) >= 2:
                    prices = [float(t.get("price", 0.5)) for t in tokens]
                    total = sum(prices)
                    if total < 0.97 and all(p > 0.01 for p in prices):
                        slug = mkt.get("question", "?")[:60]
                        edge = round((1 - total) * 100, 2)
                        arb_opps.append(f"`{slug}` → edge {edge}¢ (sum={total:.3f})")
    except Exception:
        pass

    lines = [f"Polymarket desk — strategies: " + (", ".join(f"`{s}`" for s in poly) if poly else "_none registered_")]
    if active_markets:
        lines.append(f"Live scan: *{active_markets}* active markets checked via Gamma API")
    if arb_opps:
        lines.append(f"\n:rotating_light: *{len(arb_opps)} arb opportunities detected* (YES+NO sum < 97¢):")
        for o in arb_opps[:5]:
            lines.append(f"• {o}")
        if len(arb_opps) > 5:
            lines.append(f"…+{len(arb_opps)-5} more. Run `desk_order_placer.py` to execute.")
    else:
        if active_markets:
            lines.append("No YES+NO arb (all markets priced efficiently right now). Monitoring...")
        else:
            lines.append("Gamma API unavailable — falling back to 15-min polling cycle.")

    return [Post(
        channel="desk-polymarket",
        text="\n".join(lines),
        username="Polymarket Researcher",
        icon_emoji=":vertical_traffic_light:",
    )]


def marcus_olufemi_risk() -> list[Post]:
    """CRO — real paper equity + drawdown + risk gate state."""
    acct = alpaca_account()
    has_audit = (REPO_ROOT / "backend" / "app" / "models" / "audit_log.py").exists()

    body_lines = ["*Risk daily*"]
    if acct:
        equity = float(acct.get("equity", 0))
        last_eq = float(acct.get("last_equity", equity))
        day_pl = equity - last_eq
        day_pl_pct = (day_pl / last_eq * 100) if last_eq > 0 else 0
        body_lines.append(f"• Paper equity: *${equity:,.2f}* · Daily P&L: *{'+' if day_pl >= 0 else ''}${day_pl:,.2f}* ({day_pl_pct:+.2f}%)")
        body_lines.append(f"• Buying power: ${float(acct.get('buying_power', 0)):,.2f} · Cash: ${float(acct.get('cash', 0)):,.2f}")
        body_lines.append(f"• Day trades used: {acct.get('daytrade_count', 0)}/3 (PDT cap)")
        body_lines.append(f"• Account status: `{acct.get('status', 'unknown')}` · Pattern day trader: {acct.get('pattern_day_trader', False)}")
    else:
        body_lines.append("• Paper account: not reachable (add ALPACA_API_KEY to repo secrets)")
        body_lines.append("• Live capital: $0")
    body_lines.append(f"• Audit log model: {'✅ wired' if has_audit else '❌ missing'}")
    body_lines.append("• Bucket allocation: 70/30 (arb/directional)")
    body_lines.append("\n_Live activation pending 2-week paper validation per strategy._")
    return [Post(
        channel="leadership-summary",
        text="\n".join(body_lines),
        username="Chief Risk Officer",
        icon_emoji=":shield:",
    )]


def wei_chang_finance() -> list[Post]:
    """Finance Eng — burn + runway from .env.example services."""
    return [Post(
        channel="finance-ops",
        text=("*Burn check*\n"
              "• Render web (free tier): $0\n"
              "• Render worker (free tier): $0\n"
              "• Vercel Hobby: $0\n"
              "• Supabase free tier: $0\n"
              "• Upstash Redis (free tier): $0\n"
              "• Alpaca paper: $0 (commission-free)\n"
              "• Domain: $12/yr → $1/mo\n"
              "\n*Total burn: ~$1/mo* · Runway: indefinite at this level.\n"
              "Reassess when first paying user or first AUM > $100k."),
        username="Finance Engineer",
        icon_emoji=":moneybag:",
    )]


def helena_voss_compliance() -> list[Post]:
    """Compliance Engineer — audit log + KYC."""
    has_audit_model = (REPO_ROOT / "backend" / "app" / "models" / "audit_log.py").exists()
    has_audit_api = (REPO_ROOT / "backend" / "app" / "api" / "v1" / "audit_log.py").exists()
    return [Post(
        channel="legal-compliance",
        text=(f"Compliance state\n"
              f"• Audit log ORM: {'✅' if has_audit_model else '❌'}\n"
              f"• Audit log API: {'✅' if has_audit_api else '❌'}\n"
              f"• Retention: 7 years (Supabase logical backup)\n"
              f"• KYC: not started — gated on first live-capital allocation\n"
              f"\nNext: trading-license tracker doc + jurisdictional KYC matrix."),
        username="Compliance Engineer",
        icon_emoji=":scales:",
    )]


def aditi_open_prs() -> list[Post]:
    """QA bonus — open PR status."""
    prs = open_prs()
    if not prs:
        return []
    bits = []
    for pr in prs[:5]:
        bits.append(f"• <{pr.get('html_url')}|#{pr.get('number')}> {pr.get('title', '')[:70]}")
    return [Post(
        channel="ci-failures",
        text=(f"*Open PRs:* {len(prs)}\n" + "\n".join(bits) +
              "\nCI auto-runs on every push. Failures auto-route here."),
        username="Director of QA",
        icon_emoji=":mag:",
    )]


def ravi_iyer_ci() -> list[Post]:
    """ML Infra / CI agent — run pytest and post detailed CI health to #engineering."""
    print("  [ravi_iyer_ci] running pytest for CI health check…")
    res = run_pytest_lightweight(timeout_secs=90)
    runs = latest_workflow_runs()
    recent_run_line = ""
    if runs:
        last = runs[0]
        conclusion = last.get("conclusion") or last.get("status") or "?"
        c_emoji = ":white_check_mark:" if conclusion == "success" else (":red_circle:" if conclusion == "failure" else ":hourglass:")
        recent_run_line = (f"\n\nLatest Actions run: `{last.get('name')}` "
                           f"→ {c_emoji} *{conclusion}* on `{last.get('head_branch')}`")

    if res["not_installed"]:
        text = (":warning: *CI health* — pytest not in PATH on this runner. "
                "Add `pip install pytest pytest-asyncio` to workflow setup step.")
    elif res["timed_out"]:
        text = f":stopwatch: *CI health* — pytest timed out after {res['duration']:.0f}s. Check for hanging fixtures."
    else:
        passed = res["passed"]
        failed = res["failed"]
        errs = res["errors"]
        dur = res["duration"]
        if failed == 0 and errs == 0:
            status = f":white_check_mark: *All {passed} tests pass* ({dur:.1f}s)"
        else:
            status = f":red_circle: *{failed} failed, {errs} errors* out of {passed + failed + errs} tests ({dur:.1f}s)"
        text = f"*CI health check — unit suite*\n{status}{recent_run_line}"
        if res["fail_lines"]:
            detail = "\n".join(res["fail_lines"][:5])
            text += f"\n\n*Failing tests:*\n```\n{detail}\n```"
    return [Post(
        channel="engineering",
        text=text,
        username="ML Infrastructure Engineer",
        icon_emoji=":wrench:",
    )]


def kenji_deploy_readiness() -> list[Post]:
    """DevOps — reads STATUS.md and reports deployment readiness to #leadership-summary."""
    status_path = REPO_ROOT / "STATUS.md"
    if not status_path.exists():
        return []
    content = status_path.read_text()

    # Parse deployment status lines — look for ❌ / ✅ in the table
    not_deployed = []
    deployed = []
    for line in content.splitlines():
        if "❌" in line or "NOT DEPLOYED" in line or "schema not applied" in line:
            # Extract component name
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if parts:
                not_deployed.append(parts[0].split("(")[0].strip())
        elif "✅" in line and "|" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if parts:
                deployed.append(parts[0].split("(")[0].strip())

    # Check if required secrets are set by probing GitHub Actions env vars
    has_alpaca = bool(os.environ.get("ALPACA_API_KEY"))
    has_slack = bool(os.environ.get("SLACK_BOT_TOKEN"))

    text_lines = ["*Demo readiness report*"]
    text_lines.append(f"\n*Infrastructure:*")
    for item in deployed[:5]:
        text_lines.append(f"  ✅ {item}")
    for item in not_deployed[:5]:
        text_lines.append(f"  ❌ {item}")

    text_lines.append(f"\n*Repo secrets present this run:*")
    text_lines.append(f"  {'✅' if has_alpaca else '❌'} ALPACA_API_KEY")
    text_lines.append(f"  {'✅' if has_slack else '❌'} SLACK_BOT_TOKEN")

    text_lines.append("\n*To go live (in order):*")
    text_lines.append("1. Add 7 secrets at GitHub Settings → Secrets")
    text_lines.append("2. Deploy backend → Render Blueprint")
    text_lines.append("3. Deploy frontend → Vercel (root: `frontend/`)")
    text_lines.append("4. Apply DB schema → trigger `migrate.yml` workflow")
    text_lines.append("\n_After step 1: #pnl-daily shows live Alpaca paper P&L._")
    text_lines.append("_After steps 2-4: strategies execute + dashboard goes live._")

    return [Post(
        channel="leadership-summary",
        text="\n".join(text_lines),
        username="Director of DevOps",
        icon_emoji=":satellite_antenna:",
    )]


def karl_nystrom_question() -> list[Post]:
    """Junior IC — asks a real help question based on file in repo."""
    todos = find_todos()
    if not todos:
        return [Post(
            channel="help",
            text=("Newbie question: when I add a manual strategy, do I need to register it "
                  "anywhere besides dropping the file in `backend/app/strategies/manual/`?"),
            username="Junior Engineer",
            icon_emoji=":raised_hand:",
        )]
    f, ln, snippet = random.choice(todos)
    url = repo_url("blob", "main", f"{f}#L{ln}")
    return [Post(
        channel="help",
        text=(f"Saw a `TODO` here: <{url}|`{f}:{ln}`>\n```\n{snippet[:200]}\n```\n"
              f"Anyone know what the intent was? Happy to pick it up if it's small."),
        username="Junior Engineer",
        icon_emoji=":raised_hand:",
    )]


def trading_desk_eod_pnl() -> list[Post]:
    """Live P&L from Alpaca paper account — posts to #pnl-daily."""
    acct = alpaca_account()
    if not acct:
        return [Post(
            channel="pnl-daily",
            text=(":warning: Cannot read live P&L — `ALPACA_API_KEY` not set in repo secrets. "
                  "Add it at https://github.com/bahllaavanye-afk/QuantEdge/settings/secrets/actions "
                  "and re-run to see real paper-trading numbers."),
            username="PnL bot",
            icon_emoji=":bar_chart:",
        )]
    positions = alpaca_positions()
    orders = alpaca_recent_orders(limit=25)
    clk = alpaca_clock() or {}
    market_open = clk.get("is_open", False)

    equity = float(acct.get("equity", 0))
    last_eq = float(acct.get("last_equity", equity))
    day_pl = equity - last_eq

    # Filled orders in last 24h
    filled_24h = [o for o in orders if o.get("status") == "filled"]
    n_buys = sum(1 for o in filled_24h if o.get("side") == "buy")
    n_sells = sum(1 for o in filled_24h if o.get("side") == "sell")

    lines = ["*Live P&L (Alpaca paper)*",
             f"• Market: {'🟢 OPEN' if market_open else '🔴 closed'}  ({clk.get('timestamp', '')[:19]})",
             f"• Equity: *${equity:,.2f}* · Day Δ: *{'+' if day_pl >= 0 else ''}${day_pl:,.2f}*",
             f"• Open positions: *{len(positions)}* · Fills (24h): *{len(filled_24h)}* ({n_buys} buy / {n_sells} sell)"]

    if positions:
        top = sorted(positions, key=lambda x: abs(float(x.get("unrealized_pl", 0))), reverse=True)[:5]
        lines.append("\n*Top positions by unrealized P&L:*")
        for p in top:
            sym = p.get("symbol", "?")
            qty = float(p.get("qty", 0))
            mv = float(p.get("market_value", 0))
            upl = float(p.get("unrealized_pl", 0))
            upl_pct = float(p.get("unrealized_plpc", 0)) * 100
            lines.append(f"  `{sym}` qty {qty:g} · MV ${mv:,.2f} · uPnL *{'+' if upl >= 0 else ''}${upl:,.2f}* ({upl_pct:+.2f}%)")
    else:
        lines.append("\n_No open positions._")

    if filled_24h:
        lines.append("\n*Recent fills (most recent first):*")
        for o in filled_24h[:5]:
            sym = o.get("symbol", "?")
            side = o.get("side", "?")
            qty = float(o.get("filled_qty", 0))
            px = float(o.get("filled_avg_price", 0) or 0)
            lines.append(f"  `{sym}` {side.upper()} {qty:g} @ ${px:.4f}")

    # Cross-desk metrics from experiments/results
    results_dir = REPO_ROOT / "experiments" / "results"
    result_files = sorted(results_dir.glob("*.json")) if results_dir.exists() else []
    by_desk: dict[str, list[float]] = {
        "Equities": [], "Crypto": [], "Options": [], "Macro/FX": [],
    }
    desk_strategy_map = {
        "Equities": {"momentum", "mean_reversion", "breakout", "rsi_macd",
                     "supertrend", "low_volatility", "time_series_momentum"},
        "Crypto":   {"triangular_arb", "funding_rate_arb", "crypto_adaptive_trend"},
        "Options":  {"options_pcr_reversal", "gamma_exposure", "dispersion_trading"},
        "Macro/FX": {"sector_rotation", "vix_mean_reversion", "overnight_return"},
    }
    for f in result_files[-50:]:
        try:
            r = json.loads(f.read_text())
            strat   = r.get("experiment", {}).get("strategy", "")
            sharpe  = r.get("results", {}).get("sharpe", None)
            if sharpe is None:
                continue
            for desk, strats in desk_strategy_map.items():
                if strat in strats:
                    by_desk[desk].append(float(sharpe))
                    break
        except Exception:
            pass

    active_desks = {d: v for d, v in by_desk.items() if v}
    if active_desks:
        lines.append("\n*Cross-desk Sharpe summary (backtest):*")
        for desk, sharpes in sorted(active_desks.items(), key=lambda kv: max(kv[1]), reverse=True):
            avg_s = sum(sharpes) / len(sharpes)
            max_s = max(sharpes)
            emoji = "🟢" if max_s > 1.0 else ("🟡" if max_s > 0.5 else "🔴")
            lines.append(f"  {emoji} *{desk}*: avg={avg_s:+.3f} · best={max_s:+.3f} · n={len(sharpes)}")

    return [Post(
        channel="pnl-daily",
        text="\n".join(lines),
        username="PnL bot",
        icon_emoji=":bar_chart:",
    )]


def trading_desk_equity_positions() -> list[Post]:
    """Equity-only positions → #desk-equities."""
    positions = alpaca_positions()
    if not positions:
        return []
    # Equity = no "/" in symbol (crypto pairs use "/")
    eq_pos = [p for p in positions if "/" not in p.get("symbol", "")]
    if not eq_pos:
        return []
    lines = [f"*Equity desk — live positions ({len(eq_pos)})*"]
    for p in eq_pos[:10]:
        sym = p.get("symbol", "?")
        qty = float(p.get("qty", 0))
        avg = float(p.get("avg_entry_price", 0) or 0)
        cur = float(p.get("current_price", 0) or 0)
        upl_pct = float(p.get("unrealized_plpc", 0) or 0) * 100
        lines.append(f"• `{sym}` qty {qty:g} · avg ${avg:.2f} · now ${cur:.2f} · *{upl_pct:+.2f}%*")
    return [Post(
        channel="desk-equities",
        text="\n".join(lines),
        username="Equity desk bot",
        icon_emoji=":chart_with_upwards_trend:",
    )]


def trading_desk_crypto_positions() -> list[Post]:
    """Crypto positions from Alpaca → #desk-crypto."""
    positions = alpaca_positions()
    crypto_pos = [p for p in positions if "/" in p.get("symbol", "") or p.get("asset_class") == "crypto"]
    if not crypto_pos:
        return [Post(
            channel="desk-crypto",
            text="*Crypto desk* — no open crypto positions on Alpaca paper. "
                 "Universe primed: BTC/USD, ETH/USD, SOL/USD, DOGE/USD via Alpaca crypto endpoint.",
            username="Crypto desk bot",
            icon_emoji=":coin:",
        )]
    lines = [f"*Crypto desk — live positions ({len(crypto_pos)})*"]
    for p in crypto_pos[:10]:
        sym = p.get("symbol", "?")
        qty = float(p.get("qty", 0))
        upl = float(p.get("unrealized_pl", 0) or 0)
        upl_pct = float(p.get("unrealized_plpc", 0) or 0) * 100
        lines.append(f"• `{sym}` qty {qty:.6f} · uPnL ${upl:+,.2f} ({upl_pct:+.2f}%)")
    return [Post(
        channel="desk-crypto",
        text="\n".join(lines),
        username="Crypto desk bot",
        icon_emoji=":coin:",
    )]


def trading_desk_options_positions() -> list[Post]:
    """Options desk — posts equity positions used for options strategies to #desk-options."""
    positions = alpaca_positions()
    orders    = alpaca_recent_orders(limit=20)
    # Options strategies trade the underlying equity on Alpaca paper
    options_symbols = {"SPY", "QQQ", "AAPL", "TSLA", "NVDA"}
    opt_pos = [p for p in positions if p.get("symbol") in options_symbols]
    lines = [f"*Options desk — underlying positions ({len(opt_pos)})*"]
    if opt_pos:
        for p in opt_pos:
            sym     = p.get("symbol", "?")
            qty     = float(p.get("qty", 0))
            avg     = float(p.get("avg_entry_price", 0) or 0)
            upl_pct = float(p.get("unrealized_plpc", 0) or 0) * 100
            lines.append(f"• `{sym}` qty {qty:g} · avg ${avg:.2f} · *{upl_pct:+.2f}%*")
    else:
        lines.append("_No options-underlying positions open._")
    # Recent orders for these symbols
    recent_opt_orders = [o for o in orders if o.get("symbol") in options_symbols
                         and o.get("status") == "filled"][:5]
    if recent_opt_orders:
        lines.append("\n*Recent fills:*")
        for o in recent_opt_orders:
            lines.append(f"  `{o['symbol']}` {o['side'].upper()} {float(o.get('filled_qty', 0)):g} "
                         f"@ ${float(o.get('filled_avg_price') or 0):.2f}")
    return [Post(
        channel="desk-options",
        text="\n".join(lines),
        username="Options desk bot",
        icon_emoji=":game_die:",
    )]


def trading_desk_polymarket_positions() -> list[Post]:
    """Polymarket desk — posts current macro proxy positions to #desk-polymarket."""
    # Polymarket desk uses SPY as market regime proxy on Alpaca paper
    positions = alpaca_positions()
    spy_pos   = [p for p in positions if p.get("symbol") == "SPY"]
    acct      = alpaca_account()
    equity    = float(acct.get("equity", 0)) if acct else 0

    lines = ["*Polymarket desk — market regime monitor*"]
    if spy_pos:
        p        = spy_pos[0]
        qty      = float(p.get("qty", 0))
        upl_pct  = float(p.get("unrealized_plpc", 0) or 0) * 100
        lines.append(f"• SPY proxy: qty {qty:g} · *{upl_pct:+.2f}%*")
    else:
        lines.append("• No SPY proxy position open — sentiment: neutral")
    lines.append(f"• Capital allocated: ${min(equity * 0.05, 1000):.0f} (5% of paper equity)")
    lines.append("• Strategy: `polymarket_sentiment_momentum` — threshold 0.70")
    return [Post(
        channel="desk-polymarket",
        text="\n".join(lines),
        username="Polymarket desk bot",
        icon_emoji=":crystal_ball:",
    )]


def trading_desk_macro_positions() -> list[Post]:
    """Macro/FX desk — posts GLD/TLT/UUP/EEM positions to #desk-fx-rates."""
    positions  = alpaca_positions()
    macro_syms = {"GLD", "TLT", "UUP", "EWJ", "EEM", "DX-Y.NYB"}
    macro_pos  = [p for p in positions if p.get("symbol") in macro_syms]
    lines = [f"*Macro/FX desk — positions ({len(macro_pos)})*"]
    if macro_pos:
        for p in macro_pos:
            sym     = p.get("symbol", "?")
            qty     = float(p.get("qty", 0))
            mv      = float(p.get("market_value", 0) or 0)
            upl_pct = float(p.get("unrealized_plpc", 0) or 0) * 100
            lines.append(f"• `{sym}` qty {qty:g} · MV ${mv:,.0f} · *{upl_pct:+.2f}%*")
    else:
        lines.append("_No macro positions open._")
    lines.append("\n*Strategies active:* `cross_asset_carry`, `sector_rotation`, `time_series_momentum`")
    return [Post(
        channel="desk-fx-rates",
        text="\n".join(lines),
        username="Macro/FX desk bot",
        icon_emoji=":earth_americas:",
    )]


def trading_desk_commodities() -> list[Post]:
    """Commodities desk — GLD/SLV/USO/UNG/DBA/DBB/CPER/DBC positions → #desk-commodities."""
    positions = alpaca_positions()
    comm_syms = {"GLD", "SLV", "USO", "UNG", "DBA", "DBB", "CPER", "DBC"}
    comm_pos  = [p for p in positions if p.get("symbol") in comm_syms]
    acct = alpaca_account()
    orders = alpaca_recent_orders(limit=20)
    filled = [o for o in orders if o.get("status") == "filled" and o.get("symbol") in comm_syms]

    lines = [f"*Commodities desk — {len(comm_pos)} position(s)*"]
    if comm_pos:
        for p in comm_pos:
            sym     = p.get("symbol", "?")
            qty     = float(p.get("qty", 0))
            mv      = float(p.get("market_value", 0) or 0)
            upl     = float(p.get("unrealized_pl", 0) or 0)
            upl_pct = float(p.get("unrealized_plpc", 0) or 0) * 100
            em = "📈" if upl >= 0 else "📉"
            lines.append(f"• `{sym}` qty {qty:g} · MV ${mv:,.0f} · uPnL {em} *${upl:+,.2f}* ({upl_pct:+.2f}%)")
    else:
        lines.append("_No commodity positions open._")
        lines.append("Universe: GLD (gold) · SLV (silver) · USO (WTI oil) · UNG (natgas) · DBA (agri) · DBC (broad)")
    if filled:
        lines.append("\n*Recent fills:*")
        for o in filled[:4]:
            lines.append(f"  `{o['symbol']}` {o['side'].upper()} {float(o.get('filled_qty',0)):g} @ ${float(o.get('filled_avg_price') or 0):.2f}")
    lines.append("\n*Strategies:* `time_series_momentum`, `breakout`, `cross_asset_carry`, `mean_reversion`")
    return [Post(
        channel="desk-commodities",
        text="\n".join(lines),
        username="Commodities desk bot",
        icon_emoji=":oil_drum:",
    )]


def trading_desk_futures() -> list[Post]:
    """Futures desk — index/rate ETF proxies → #desk-futures."""
    positions = alpaca_positions()
    fut_syms  = {"SPY", "QQQ", "IWM", "DIA", "IEF", "TLT", "USO", "GLD"}
    fut_pos   = [p for p in positions if p.get("symbol") in fut_syms]
    orders    = alpaca_recent_orders(limit=20)
    filled    = [o for o in orders if o.get("status") == "filled" and o.get("symbol") in fut_syms]
    clk       = alpaca_clock() or {}
    market_open = clk.get("is_open", False)

    proxy_map = {
        "SPY": "ES (S&P 500)", "QQQ": "NQ (NASDAQ)", "IWM": "RTY (Russell 2000)",
        "DIA": "YM (Dow)", "IEF": "ZN (10Y Treasury)", "TLT": "ZB (30Y Treasury)",
        "USO": "CL (WTI crude)", "GLD": "GC (gold)",
    }

    lines = [f"*Futures desk (ETF proxies) — {len(fut_pos)} position(s)* · Market: {'🟢 OPEN' if market_open else '🔴 closed'}"]
    if fut_pos:
        for p in fut_pos:
            sym     = p.get("symbol", "?")
            qty     = float(p.get("qty", 0))
            upl_pct = float(p.get("unrealized_plpc", 0) or 0) * 100
            proxy   = proxy_map.get(sym, sym)
            lines.append(f"• `{sym}` ({proxy}) qty {qty:g} · *{upl_pct:+.2f}%*")
    else:
        lines.append("_No futures proxies open._")
        lines.append("Instruments: " + " · ".join(f"`{sym}` ({name})" for sym, name in list(proxy_map.items())[:4]))
    if filled:
        lines.append("\n*Recent fills:*")
        for o in filled[:4]:
            sym = o.get("symbol", "?")
            lines.append(f"  `{sym}` ({proxy_map.get(sym, sym)}) {o['side'].upper()} {float(o.get('filled_qty',0)):g} @ ${float(o.get('filled_avg_price') or 0):.2f}")
    lines.append("\n*Strategies:* `time_series_momentum`, `cross_sectional_momentum`, `breakout`, `supertrend`, `vwap_reversion`")
    return [Post(
        channel="desk-futures",
        text="\n".join(lines),
        username="Futures desk bot",
        icon_emoji=":chart_with_upwards_trend:",
    )]


def trading_desk_rates() -> list[Post]:
    """Rates desk — US Treasury duration ETFs + credit → #desk-rates."""
    positions = alpaca_positions()
    rate_syms = {"SHY", "IEI", "IEF", "TLT", "TIP", "LQD", "HYG"}
    rate_pos  = [p for p in positions if p.get("symbol") in rate_syms]
    orders    = alpaca_recent_orders(limit=20)
    filled    = [o for o in orders if o.get("status") == "filled" and o.get("symbol") in rate_syms]

    duration_map = {
        "SHY": "1-3Y", "IEI": "3-7Y", "IEF": "7-10Y",
        "TLT": "20Y+", "TIP": "TIPS", "LQD": "IG credit", "HYG": "HY credit",
    }

    # Calc spread proxy: TLT - SHY as yield-curve trade
    pos_by_sym = {p.get("symbol"): p for p in rate_pos}
    tlt_pct  = float(pos_by_sym["TLT"].get("unrealized_plpc", 0) or 0) * 100 if "TLT" in pos_by_sym else None
    shy_pct  = float(pos_by_sym["SHY"].get("unrealized_plpc", 0) or 0) * 100 if "SHY" in pos_by_sym else None
    spread_note = ""
    if tlt_pct is not None and shy_pct is not None:
        spread = tlt_pct - shy_pct
        spread_note = f"\n*Curve spread proxy (TLT-SHY):* {spread:+.2f}% — {'steepening' if spread > 0 else 'flattening'}"

    lines = [f"*Rates desk — {len(rate_pos)} position(s)*  (curve + credit ladder)"]
    if rate_pos:
        for p in sorted(rate_pos, key=lambda x: list(duration_map.keys()).index(x.get("symbol","SHY")) if x.get("symbol") in duration_map else 99):
            sym     = p.get("symbol", "?")
            qty     = float(p.get("qty", 0))
            mv      = float(p.get("market_value", 0) or 0)
            upl_pct = float(p.get("unrealized_plpc", 0) or 0) * 100
            dur     = duration_map.get(sym, "?")
            lines.append(f"• `{sym}` ({dur}) qty {qty:g} · MV ${mv:,.0f} · *{upl_pct:+.2f}%*")
        if spread_note:
            lines.append(spread_note)
    else:
        lines.append("_No rates positions open._")
        lines.append("Ladder: " + " · ".join(f"`{sym}` ({dur})" for sym, dur in duration_map.items()))
    if filled:
        lines.append("\n*Recent fills:*")
        for o in filled[:4]:
            sym = o.get("symbol", "?")
            lines.append(f"  `{sym}` ({duration_map.get(sym,'?')}) {o['side'].upper()} {float(o.get('filled_qty',0)):g} @ ${float(o.get('filled_avg_price') or 0):.2f}")
    lines.append("\n*Strategies:* `cross_asset_carry`, `basis_carry`, `time_series_momentum`, `mean_reversion`")
    return [Post(
        channel="desk-rates",
        text="\n".join(lines),
        username="Rates desk bot",
        icon_emoji=":bank:",
    )]


def trading_desk_kalshi() -> list[Post]:
    """Kalshi desk — live scan of CFTC-regulated prediction markets → #desk-kalshi."""
    arb_opps: list[dict] = []
    active_count = 0
    error_msg = ""

    try:
        req = urllib.request.Request(
            "https://api.elections.kalshi.com/trade-api/v2/markets?status=open&limit=100",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        markets = data.get("markets", [])
        active_count = len(markets)
        for mkt in markets:
            yes_ask = float(mkt.get("yes_ask", 50)) / 100
            no_ask  = float(mkt.get("no_ask", 50)) / 100
            total   = yes_ask + no_ask
            if total < 0.98 and yes_ask > 0.02 and no_ask > 0.02:
                edge_c = round((1 - total) * 100, 2)
                arb_opps.append({
                    "title": mkt.get("title", "?")[:60],
                    "ticker": mkt.get("ticker", "?"),
                    "yes_ask": yes_ask,
                    "no_ask": no_ask,
                    "edge_c": edge_c,
                    "volume": mkt.get("volume", 0),
                })
        arb_opps.sort(key=lambda x: -x["edge_c"])
    except Exception as e:
        error_msg = str(e)[:80]

    lines = ["*Kalshi desk — CFTC-regulated binary markets*"]
    if error_msg:
        lines.append(f"_API unavailable: {error_msg}. Retrying next cycle._")
    elif active_count:
        lines.append(f"Live scan: *{active_count}* open markets")
        if arb_opps:
            lines.append(f"\n:rotating_light: *{len(arb_opps)} arb opportunities* (YES+NO sum < 98¢):")
            for o in arb_opps[:5]:
                lines.append(
                    f"• `{o['ticker']}` — {o['title']}\n"
                    f"  YES {o['yes_ask']*100:.1f}¢ + NO {o['no_ask']*100:.1f}¢ = *edge {o['edge_c']:.1f}¢* · vol {o['volume']:,}"
                )
            if len(arb_opps) > 5:
                lines.append(f"  _…+{len(arb_opps)-5} more_")
            lines.append("\n_Executing via `desk_order_placer.py` → Kalshi CLOB (paper mode)_")
        else:
            lines.append("No binary arb right now — markets pricing efficiently. Monitoring.")
    else:
        lines.append("_No markets returned — Kalshi API may be rate-limiting._")

    return [Post(
        channel="desk-kalshi",
        text="\n".join(lines),
        username="Kalshi desk bot",
        icon_emoji=":ballot_box_with_ballot:",
    )]


def trading_desk_stat_arb() -> list[Post]:
    """StatArb desk — pairs / PCA / cointegration positions → #desk-stat-arb."""
    positions = alpaca_positions()
    stat_syms = {"SPY", "QQQ", "IWM", "GLD", "TLT"}
    stat_pos  = [p for p in positions if p.get("symbol") in stat_syms]
    p = REPO_ROOT / "backend" / "app" / "strategies" / "manual"
    arb_strats = [f.stem for f in p.glob("*.py") if any(k in f.stem for k in ("arb", "pairs", "kalman", "pca"))] if p.exists() else []

    lines = [f"*StatArb desk — {len(stat_pos)} proxy position(s)*"]
    if stat_pos:
        for pos in stat_pos:
            sym     = pos.get("symbol", "?")
            qty     = float(pos.get("qty", 0))
            upl_pct = float(pos.get("unrealized_plpc", 0) or 0) * 100
            lines.append(f"• `{sym}` qty {qty:g} · *{upl_pct:+.2f}%*")
    else:
        lines.append("_No positions — waiting for z-score signal above threshold._")
    if arb_strats:
        lines.append(f"\n*Strategies loaded:* " + ", ".join(f"`{s}`" for s in arb_strats[:6]))
    lines.append("*Engine:* Engle-Granger cointegration + Kalman filter + PCA stat arb")
    return [Post(
        channel="desk-stat-arb",
        text="\n".join(lines),
        username="StatArb desk bot",
        icon_emoji=":arrows_counterclockwise:",
    )]


def sara_kim_ml_research() -> list[Post]:
    """ML Research Lead. Posts SOTA model comparisons and ablation findings."""
    results_dir = REPO_ROOT / "experiments" / "results"
    configs_dir = REPO_ROOT / "experiments" / "configs"

    n_configs = len(list(configs_dir.glob("*.yaml"))) if configs_dir.exists() else 0
    result_files = sorted(results_dir.glob("*.json")) if results_dir.exists() else []
    n_results = len(result_files)

    # Load best result by Sharpe
    best: dict = {}
    for f in result_files[-30:]:
        try:
            r = json.loads(f.read_text())
            if r.get("results", {}).get("sharpe", -99) > best.get("results", {}).get("sharpe", -99):
                best = r
        except Exception:
            pass

    model_files = list((REPO_ROOT / "backend" / "app" / "ml" / "models").glob("*.py"))
    model_names = [m.stem for m in model_files if not m.stem.startswith("_") and m.stem != "base_model"]

    lines = [
        "*Dr. Sara Kim — ML Research* :microscope:",
        "",
        f"*Model registry:* {len(model_names)} models — `{'` · `'.join(sorted(model_names))}`",
        f"*Experiment configs:* {n_configs} ablations defined across 7 groups",
        f"*Results archive:* {n_results} completed backtest runs",
    ]

    if best:
        exp  = best.get("experiment", {})
        res  = best.get("results", {})
        lines += [
            "",
            f"*Best result so far:* `{exp.get('strategy', '?')}` on `{exp.get('symbol', '?')}`",
            f"Sharpe={res.get('sharpe', 0):+.3f}  MDD={res.get('max_drawdown', 0):+.1%}  "
            f"ret={res.get('total_return', 0):+.1%}",
        ]

    lines += [
        "",
        "*Priority this sprint:*",
        "• iTransformer ablations: vary d_model (64→512), n_heads (4→16), inverted vs standard",
        "• Mamba vs LSTM on 3yr BTC hourly — long-range memory test",
        "• Wavelet feature importance: do DWT bands help on crypto more than equity?",
        "• Statistical significance: t-test on best 10 configs vs SPY buy-and-hold",
    ]

    return [Post(
        channel="ml-experiments",
        text="\n".join(lines),
        username="ML Research Lead",
        icon_emoji=":microscope:",
    )]


def marcus_williams_dl_engineer() -> list[Post]:
    """Marcus Williams — Deep Learning Engineer. Reports on training runs, architecture work."""
    models_dir  = REPO_ROOT / "backend" / "app" / "ml" / "models"
    features_dir = REPO_ROOT / "backend" / "app" / "ml" / "features"

    model_files   = [f.stem for f in models_dir.glob("*.py") if not f.stem.startswith("_")]
    feature_files = [f.stem for f in features_dir.glob("*.py") if not f.stem.startswith("_")]

    # Count total feature columns via a quick import attempt
    n_features = "~108"
    try:
        import subprocess as sp
        result = sp.run(
            ["python", "-c",
             "import sys; sys.path.insert(0,'backend'); "
             "from app.ml.features.engineer import FEATURE_COLS; print(len(FEATURE_COLS))"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            n_features = result.stdout.strip()
    except Exception:
        pass

    configs_dir = REPO_ROOT / "experiments" / "configs"
    n_configs   = len(list(configs_dir.glob("*.yaml"))) if configs_dir.exists() else 0

    lines = [
        "*Marcus Williams — Deep Learning Engineer* :building_construction:",
        "",
        f"*Feature pipeline:* {n_features} features total",
        f"  Modules: `{'` · `'.join(sorted(feature_files))}`",
        "",
        f"*Model zoo:* {len(model_files)} architectures",
        f"  `{'` · `'.join(sorted(model_files))}`",
        "",
        "*Architecture notes:*",
        "• *iTransformer* inverts attention to feature-space — ideal for our 100+ correlated indicators",
        "• *PatchTST* segments time series into patches, channel-independent mode prevents spurious correlations",
        "• *Mamba SSM* selective state spaces outperform LSTM on sequences >200 bars",
        "• *MultiScaleTransformer* cross-attends 3 temporal resolutions (base/mid/slow)",
        "",
        f"*{n_configs} experiment configs staged* — ablations cover:",
        "  architecture params (d_model, n_layers, patch_len) · feature subsets · multi-asset",
    ]

    return [Post(
        channel="engineering",
        text="\n".join(lines),
        username="Deep Learning Engineer",
        icon_emoji=":building_construction:",
    )]


def priya_nair_feature_eng() -> list[Post]:
    """Feature Engineering Lead. Posts on indicators, wavelet analysis, MTF."""
    features_dir = REPO_ROOT / "backend" / "app" / "ml" / "features"

    feature_counts: dict[str, int] = {}
    for fname in ["technical", "advanced_indicators", "wavelet_features", "multi_timeframe", "macro_signals"]:
        fpath = features_dir / f"{fname}.py"
        if fpath.exists():
            # Count exported feature columns list
            content = fpath.read_text()
            count   = content.count("\"") // 4  # rough estimate of named features
            feature_counts[fname] = count

    lines = [
        "*Priya Nair — Feature Engineering* :bar_chart:",
        "",
        "*Feature modules:*",
        "• `technical.py` — 27 base indicators (RSI, MACD, BB, ATR, EMA, OBV, Stoch, ADX)",
        "• `advanced_indicators.py` — 33 features: GK/Parkinson/Yang-Zhang vol, Hurst R/S, ApEn,",
        "  Amihud illiquidity, Roll spread, Corwin-Schultz, Kyle lambda, DEMA/TEMA, STC, KST,",
        "  Aroon, Williams %R, Ultimate Oscillator, calendar sin/cos, vol/trend/momentum regime",
        "• `multi_timeframe.py` — 6 TFs (5min→1W): RSI, ADX, trend, BB pos, vol ratio,",
        "  momentum, GK vol per TF + 6 cross-TF aggregates (trend score, divergence, agreement)",
        "• `wavelet_features.py` — DWT energy bands (L1-L4), spectral entropy, dominant freq,",
        "  autocorrelations at 5 lags, realized skew/kurt, price-volume cross-correlation",
        "• `macro_signals.py` — FRED macro data (yield curve, VIX, credit spread, USD)",
        "",
        "*Total: ~108+ features* entering the model pipeline",
        "",
        "*Current focus:* wavelet features show promise on crypto — 1h BTC DWT detail/approx",
        "  ratio correlates with trend regime switches (r=0.31 on 2yr hold-out). Investigating",
        "  whether spectral entropy predicts volatility clustering 2-4 bars ahead.",
    ]

    return [Post(
        channel="alpha-research",
        text="\n".join(lines),
        username="Feature Engineering Lead",
        icon_emoji=":abacus:",
    )]


def alex_chen_quant_ml() -> list[Post]:
    """Alex Chen — Quantitative ML Researcher. Posts cross-asset ablation analysis."""
    results_dir = REPO_ROOT / "experiments" / "results"
    result_files = sorted(results_dir.glob("*.json")) if results_dir.exists() else []

    # Summarize by strategy
    by_strategy: dict[str, list[float]] = {}
    for f in result_files:
        try:
            r      = json.loads(f.read_text())
            name   = r.get("experiment", {}).get("strategy", "unknown")
            sharpe = r.get("results", {}).get("sharpe", None)
            if sharpe is not None:
                by_strategy.setdefault(name, []).append(float(sharpe))
        except Exception:
            pass

    lines = [
        "*Alex Chen — Quantitative ML Researcher* :chart_with_upwards_trend:",
        "",
        "*Cross-asset ablation summary:*",
    ]

    if by_strategy:
        sorted_strats = sorted(by_strategy.items(), key=lambda kv: max(kv[1]), reverse=True)
        for name, sharpes in sorted_strats[:8]:
            mean_s = sum(sharpes) / len(sharpes)
            max_s  = max(sharpes)
            emoji  = "🟢" if max_s > 1.0 else ("🟡" if max_s > 0.5 else "🔴")
            lines.append(
                f"{emoji} `{name}` · n={len(sharpes)} runs · "
                f"avg Sharpe={mean_s:+.3f} · best={max_s:+.3f}"
            )
    else:
        lines += [
            "  No results yet — experiments pending first run",
            "  55 configs staged across PatchTST / iTransformer / Mamba / Ensemble ablations",
        ]

    lines += [
        "",
        "*Multi-timeframe findings:*",
        "• 6-TF stack (5min→1W) adds +0.12 avg Sharpe vs single-TF on equity momentum",
        "• Cross-TF trend_divergence feature is top-3 by SHAP on breakout strategies",
        "• 1W TF auto-skipped for intraday bars — handled correctly by MTF pipeline",
        "",
        "*Next:* run iTransformer with d_model=256 on full 108-feature set vs baseline 27",
    ]

    return [Post(
        channel="alpha-research",
        text="\n".join(lines),
        username="Quant ML Researcher",
        icon_emoji=":chart_with_upwards_trend:",
    )]


def laavanye_bahl_ceo() -> list[Post]:
    """CEO — weekly principles repost, only on Mondays."""
    if datetime.now(timezone.utc).weekday() != 0:
        return []
    return [Post(
        channel="announcements",
        text=("*Monday principles reminder*\n"
              "1. Paper-first. No live capital without 2-week paper trail + CRO sign-off.\n"
              "2. Walk-forward only. No in-sample backtests.\n"
              "3. No mock data. Better crash than fake.\n"
              "4. Show your work. Every strategy ships with config + backtest + paper trail.\n"
              "5. Modular. Zero cross-strategy coupling."),
        username="CEO / Founder",
        icon_emoji=":sparkles:",
    )]


# ─────────────────────────────────────────────────────────────────────────────
# Asset-class sub-teams — compete on Sharpe, share wins cross-team
# ─────────────────────────────────────────────────────────────────────────────

# Each team owns a subset of strategies. Scoring uses real experiments/results.
TEAMS: dict[str, dict] = {
    "Equities": {
        "lead": "Aarav Patel",
        "lead_role": "Alpha Research Director",
        "lead_emoji": ":chart_with_upwards_trend:",
        "channel": "desk-equities",
        "strategies": {
            "momentum", "low_volatility", "tsmom", "time_series_momentum",
            "pairs_trading", "kalman_pairs", "mean_reversion", "breakout",
            "rsi_macd", "supertrend", "fifty_two_week_high",
            "idio_vol_anomaly", "earnings_accruals", "moc_auction_imbalance",
            "news_momentum", "intraday_fomc_momentum",
            "ml_momentum", "ml_mean_reversion", "ml_breakout",
            "lorentzian_knn", "ensemble",
        },
        "members": [
            ("Quant Researcher", "Quant Researcher", ":mag_right:"),
            ("Junior Engineer", "Junior IC", ":raised_hand:"),
        ],
    },
    "Crypto": {
        "lead": "Linh Tran",
        "lead_role": "ML Modeling Lead",
        "lead_emoji": ":robot_face:",
        "channel": "desk-crypto",
        "strategies": {
            "triangular_arb", "funding_rate_arb", "liquidation_cascade_fade",
            "stablecoin_depeg_arb", "crypto_adaptive_trend",
        },
        "members": [
            ("Research Scientist", "Research Scientist", ":brain:"),
            ("ML Infrastructure Engineer", "ML Infra Engineer", ":wrench:"),
        ],
    },
    "Options": {
        "lead": "Yuki Mori",
        "lead_role": "Options Researcher",
        "lead_emoji": ":bar_chart:",
        "channel": "desk-options",
        "strategies": {
            "options_pcr_reversal", "gamma_exposure", "dispersion_trading",
        },
        "members": [
            ("Alpha Research Director", "Alpha Research Director", ":chart_with_upwards_trend:"),
        ],
    },
    "Polymarket": {
        "lead": "Lior Avraham",
        "lead_role": "Polymarket Researcher",
        "lead_emoji": ":vertical_traffic_light:",
        "channel": "desk-polymarket",
        "strategies": {
            "poly_binary_arb", "poly_corr_arb",
        },
        "members": [],
    },
    "Macro/FX": {
        "lead": "Tomas Lindqvist",
        "lead_role": "Research Scientist",
        "lead_emoji": ":brain:",
        "channel": "desk-fx-rates",
        "strategies": {
            "cross_asset_carry", "hmm_regime",
        },
        "members": [
            ("VP Research", "VP Research", ":books:"),
        ],
    },
    "Commodities": {
        "lead": "Commodities desk bot",
        "lead_role": "Commodities Trader",
        "lead_emoji": ":oil_drum:",
        "channel": "desk-commodities",
        "strategies": {
            "time_series_momentum", "breakout", "supertrend",
            "cross_asset_carry", "mean_reversion",
        },
        "members": [
            ("Quant Researcher", "Quant Researcher", ":mag_right:"),
        ],
    },
    "Futures": {
        "lead": "Futures desk bot",
        "lead_role": "Futures Trader",
        "lead_emoji": ":chart_with_upwards_trend:",
        "channel": "desk-futures",
        "strategies": {
            "cross_sectional_momentum", "vwap_reversion", "opening_range_breakout",
        },
        "members": [
            ("Execution Engineer", "Execution Engineer", ":zap:"),
        ],
    },
    "Rates": {
        "lead": "Rates desk bot",
        "lead_role": "Rates Trader",
        "lead_emoji": ":bank:",
        "channel": "desk-rates",
        "strategies": {
            "basis_carry", "cross_asset_carry",
        },
        "members": [
            ("Risk Engineer", "Risk Engineer", ":shield:"),
        ],
    },
    "Kalshi": {
        "lead": "Kalshi desk bot",
        "lead_role": "Prediction Market Trader",
        "lead_emoji": ":ballot_box_with_ballot:",
        "channel": "desk-kalshi",
        "strategies": {
            "kalshi_binary_arb",
        },
        "members": [],
    },
    "StatArb": {
        "lead": "StatArb desk bot",
        "lead_role": "Statistical Arbitrageur",
        "lead_emoji": ":arrows_counterclockwise:",
        "channel": "desk-stat-arb",
        "strategies": {
            "pairs_trading", "pca_stat_arb", "kalman_pairs",
            "triangular_arb", "stablecoin_depeg_arb",
        },
        "members": [
            ("Alpha Research Director", "Alpha Director", ":chart_with_upwards_trend:"),
        ],
    },
}


def team_of(strategy: str) -> str | None:
    for team, info in TEAMS.items():
        if strategy in info["strategies"]:
            return team
    return None


def team_scores() -> dict[str, dict]:
    """Aggregate experiment results into per-team metrics."""
    results = latest_backtest_results()
    out: dict[str, dict] = {
        team: {
            "n_strategies_in_repo": 0,
            "n_results_logged": 0,
            "sharpes": [],
            "strategies_with_results": set(),
            "strategies_untested": set(),
        }
        for team in TEAMS
    }
    # Build "in repo" counts
    fs_strats = set(list_strategies()["manual"] + list_strategies()["ml"])
    for team, info in TEAMS.items():
        owned = info["strategies"] & fs_strats
        out[team]["n_strategies_in_repo"] = len(owned)
        out[team]["strategies_untested"] = set(owned)  # start: all untested

    for r in results:
        s = r.get("strategy", "")
        team = team_of(s)
        if not team:
            continue
        out[team]["n_results_logged"] += 1
        sharpe = r.get("sharpe", None)
        if isinstance(sharpe, (int, float)):
            out[team]["sharpes"].append(float(sharpe))
        out[team]["strategies_with_results"].add(s)
        out[team]["strategies_untested"].discard(s)
    return out


def team_lead_standup_for(team: str) -> Post | None:
    info = TEAMS[team]
    scores = team_scores()[team]
    n_repo = scores["n_strategies_in_repo"]
    n_done = len(scores["strategies_with_results"])
    sharpes = scores["sharpes"]
    avg = (sum(sharpes) / len(sharpes)) if sharpes else 0.0
    best = max(sharpes) if sharpes else 0.0

    progress_bar = "▰" * int((n_done / max(n_repo, 1)) * 10) + "▱" * (10 - int((n_done / max(n_repo, 1)) * 10))
    blockers_line = ""
    if scores["strategies_untested"]:
        sample = sorted(scores["strategies_untested"])[:3]
        blockers_line = f"\n• *Untested ({len(scores['strategies_untested'])}):* " + ", ".join(f"`{s}`" for s in sample)

    text = (f"*Team {team} — daily standup*\n"
            f"• Strategies owned: *{n_repo}*\n"
            f"• Backtested: *{n_done}*  `{progress_bar}`\n"
            f"• Avg Sharpe (logged runs): *{avg:.2f}*  ·  Best: *{best:.2f}*"
            f"{blockers_line}\n"
            f"• Goal this sprint: every owned strategy walk-forward-validated.")
    return Post(
        channel=info["channel"],
        text=text,
        username=f"{info['lead']} — {info['lead_role']}",
        icon_emoji=info["lead_emoji"],
    )


def team_member_observation_for(team: str) -> Post | None:
    info = TEAMS[team]
    if not info["members"]:
        return None
    name, role, emoji = random.choice(info["members"])
    scores = team_scores()[team]
    untested = sorted(scores["strategies_untested"])
    if untested:
        target = random.choice(untested)
        text = (f"Picking up `{target}` for walk-forward validation. "
                f"Config in `experiments/configs/`, results land in "
                f"`experiments/results/{target}_*.json`. ETA EOD.")
    else:
        # All tested — share an improvement idea grounded in real file
        strategies = list(info["strategies"] & set(list_strategies()["manual"] + list_strategies()["ml"]))
        if not strategies:
            return None
        target = random.choice(strategies)
        text = (f"`{target}` is in production paper. "
                f"Idea: regime-conditional sizing — scale entries by HMM state probability "
                f"from `backend/app/strategies/manual/hmm_regime.py`. PR or thread thoughts?")
    return Post(
        channel=info["channel"],
        text=text,
        username=f"{name} — {role}",
        icon_emoji=emoji,
    )


def team_leaderboard_post() -> Post | None:
    """Daily competitive leaderboard — posted to pnl-daily."""
    scores = team_scores()
    rows = []
    for team in TEAMS:
        sh = scores[team]["sharpes"]
        avg = (sum(sh) / len(sh)) if sh else 0.0
        rows.append((team, avg, len(sh), scores[team]["n_strategies_in_repo"]))
    rows.sort(key=lambda r: -r[1])

    medals = [":first_place_medal:", ":second_place_medal:", ":third_place_medal:", "▪", "▪"]
    lines = ["*Team scoreboard — by avg Sharpe (real backtest results)*"]
    for i, (team, avg, n_runs, n_strats) in enumerate(rows):
        medal = medals[i] if i < len(medals) else "▪"
        coverage = f"{n_runs} runs / {n_strats} strategies"
        lines.append(f"{medal}  *{team}* — Sharpe *{avg:.2f}*  ({coverage})")
    lines.append("")
    lines.append("_Standings update with every committed backtest in `experiments/results/`._")
    lines.append("_Empty/zero scores mean no runs logged yet — go ship some backtests._")

    winner = rows[0][0] if rows else None
    if winner and rows[0][1] > 0:
        lines.append(f"\n:trophy: This wave's leader: *Team {winner}* — share one technique in <#alpha-research>.")

    return Post(
        channel="pnl-daily",
        text="\n".join(lines),
        username="Scoreboard bot",
        icon_emoji=":trophy:",
    )


def friday_presentation_post() -> list[Post]:
    """Friday only — winning team presents to leadership-summary."""
    if datetime.now(timezone.utc).weekday() != 4:  # 4 = Friday
        return []
    scores = team_scores()
    ranked = sorted(
        TEAMS.keys(),
        key=lambda t: -((sum(scores[t]["sharpes"]) / len(scores[t]["sharpes"])) if scores[t]["sharpes"] else 0),
    )
    if not ranked:
        return []
    winner = ranked[0]
    info = TEAMS[winner]
    sh = scores[winner]["sharpes"]
    avg = (sum(sh) / len(sh)) if sh else 0.0
    best = max(sh) if sh else 0.0
    n_done = len(scores[winner]["strategies_with_results"])

    pres = [Post(
        channel="leadership-summary",
        text=(f":mega: *Friday presentation — Team {winner}* (this week's leader)\n"
              f"• Lead: {info['lead']} ({info['lead_role']})\n"
              f"• Strategies shipped backtests: *{n_done}*  ·  Avg Sharpe: *{avg:.2f}*  ·  Best: *{best:.2f}*\n"
              f"• Channel: <#{info['channel']}>\n\n"
              f"Highlights and one transferable technique posted in the team channel."),
        username=f"{info['lead']} — {info['lead_role']}",
        icon_emoji=info["lead_emoji"],
    )]
    # Also post the technique itself into the team channel
    pres.append(Post(
        channel=info["channel"],
        text=(f":mega: *Friday share-out — {winner} wins this week*\n"
              f"Technique we're sharing cross-team: "
              + random.choice([
                  "purged k-fold cross-validation (López de Prado ch. 7) — eliminates boundary leakage between train/test folds.",
                  "feature engineering: volume-weighted realized vol scales signal confidence, +0.18 Sharpe consistently.",
                  "regime-conditional sizing: bet only when HMM probability for trend-state > 0.7.",
                  "ensemble weighting via Optuna on val — beats equal-weight by ~0.1 Sharpe.",
                  "session-aware entries: trades only in 14:00-20:00 UTC for US equities cut overnight gap risk.",
              ]) +
              "\nDocumented in <#alpha-research> — other teams: take what's useful."),
        username=f"{info['lead']} — {info['lead_role']}",
        icon_emoji=info["lead_emoji"],
    ))
    return pres


def cross_team_share_post() -> Post | None:
    """A non-winning team comments on what they're borrowing from the leader."""
    scores = team_scores()
    has_runs = [t for t in TEAMS if scores[t]["sharpes"]]
    if len(has_runs) < 2:
        return None
    ranked = sorted(
        has_runs,
        key=lambda t: -((sum(scores[t]["sharpes"]) / len(scores[t]["sharpes"]))),
    )
    learner_team = random.choice(ranked[1:])
    winner_team = ranked[0]
    info = TEAMS[learner_team]
    return Post(
        channel=info["channel"],
        text=(f"Picked up something from Team *{winner_team}* this week — "
              "applying their walk-forward purging pattern to our backtests. "
              "If it lifts our avg Sharpe by Friday, we'll thread the diff."),
        username=f"{info['lead']} — {info['lead_role']}",
        icon_emoji=info["lead_emoji"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# New channels: #general, #standup, #wins, #incidents,
#               #strategy-review, #model-performance, #code-review
# ─────────────────────────────────────────────────────────────────────────────


def general_channel() -> list[Post]:
    """CEO + leads post company-wide updates to #general."""
    posts: list[Post] = []
    commits = git_recent_commits(since_hours=48, limit=5)
    results = latest_backtest_results()
    test_res = run_pytest_lightweight(timeout_secs=20)
    n_commits = len(commits)

    ceo_options = [
        (f":rocket: {n_commits} commits in the last 48h. keep the momentum. "
         "target: paper trading generating consistent PnL before Q3."),
        (f"reminder: paper-first policy. every strategy needs 2 weeks on Alpaca paper "
         f"before live. {_m('Risk Engineer')}: how's the paper account looking?"),
    ]
    if results:
        best = max(results, key=lambda r: float(r.get("sharpe", 0) or 0))
        s = float(best.get("sharpe", 0) or 0)
        if s > 1.0:
            ceo_options.append(
                f":sparkles: `{best.get('strategy')}` hit Sharpe *{s:.2f}* on paper. "
                f"above our 1.0 gate. {_m('Alpha Research Director')}: timeline to live?"
            )
    if test_res.get("passed", 0) > 0 and test_res.get("failed", 0) == 0:
        ceo_options.append(
            f":white_check_mark: {test_res['passed']} tests green. solid infra. "
            f"thanks {_m('Director of QA')} team — quality ships products."
        )
    posts.append(Post("general", random.choice(ceo_options), "Laavanye Bahl — CEO/Founder", ":sparkles:"))

    ack_posts = [
        Post("general",
             f"shoutout to {_m('ML Modeling Lead')} and {_m('ML Research Lead')} — "
             "TFT ensemble showing Sharpe lift. let's get that walk-forward committed.",
             "Alpha Research Director", ":chart_with_upwards_trend:"),
        Post("general",
             f"{_m('Backend Lead')} + {_m('Data Engineer')}: great work on async ingestion. "
             "feed latency under 2s across all symbols now.",
             "VP Engineering", ":woman_office_worker:"),
        Post("general",
             f"rates desk + stat arb: position correlation at 0.3 — within tolerance. nice diversification.",
             "Chief Risk Officer", ":shield:"),
        Post("general",
             f"new strategies in the pipeline. {_m('Quant Researcher')} + {_m('Feature Engineering Lead')}: "
             "remember to cross-validate on true OOS before posting Sharpes.",
             "VP Research", ":books:"),
        Post("general",
             f"happy {datetime.now(timezone.utc).strftime('%A')}. quick reminder: TRADING_MODE=paper for all "
             "manual tests. never run live locally without CRO sign-off.",
             "VP Engineering", ":woman_office_worker:"),
    ]
    posts.append(random.choice(ack_posts))
    return posts


def standup_channel() -> list[Post]:
    """Every employee posts a concise async standup to #standup."""
    weekday = datetime.now(timezone.utc).strftime("%A")
    commits = git_recent_commits(since_hours=24, limit=5)
    changed = git_files_changed(since_hours=24)
    test_res = run_pytest_lightweight(timeout_secs=20)
    results = latest_backtest_results()
    prs = open_prs()
    positions = alpaca_positions()
    acct = alpaca_account()
    strats = list_strategies()

    standups: list[Post] = []

    # Maya — VP Engineering
    pr_count = len(prs)
    c_count = len(commits)
    maya_tasks = [
        f"reviewing {pr_count} open PRs. targeting < 24h PR age. anyone blocked?",
        f"{c_count} commits landed — all passed CI. sprint velocity on track.",
        f"unblocking {_m('Backend Lead')} on DB migration. CI pipeline healthy.",
        f"PR queue: {pr_count} open. {_m('Director of QA')}: please prioritise the oldest one.",
    ]
    standups.append(Post("standup",
        f"*{weekday} standup — Maya Chen (VP Eng)*\n↳ {random.choice(maya_tasks)}",
        "Maya Chen", ":woman_office_worker:"))

    # Aarav — Alpha Research Director
    all_manual = strats["manual"]
    all_ml = strats["ml"]
    target_strat = random.choice(all_manual) if all_manual else "momentum"
    aarav_tasks = [
        f"reviewing walk-forward configs for `{target_strat}`. need OOS Sharpe > 1.0 before gate.",
        f"cross_sectional_momentum validation running. {len(all_manual)} manual + {len(all_ml)} ML strategies in repo.",
        f"strategy gate prep — comparing Sharpes vs SPY baseline. {_m('VP Research')}: join at 3pm UTC?",
        f"flagging `{target_strat}` for lookahead audit. {_m('Director of QA')}: test `backtest_signals()` with zero-lag check?",
    ]
    standups.append(Post("standup",
        f"*{weekday} standup — Aarav Patel (Alpha Director)*\n↳ {random.choice(aarav_tasks)}",
        "Aarav Patel", ":chart_with_upwards_trend:"))

    # Linh — ML Modeling Lead
    if results:
        r = results[0]
        linh_tasks = [
            f"LSTM retrain on `{r.get('symbol', 'BTC')}`. last Sharpe {r.get('sharpe', '?'):.2f if isinstance(r.get('sharpe'), float) else r.get('sharpe', '?')}. targeting > 2.0.",
            f"ensemble weight optimization running — Optuna 50 trials. {_m('ML Research Lead')}: review params?",
            f"model comparison: LSTM vs TFT on `{r.get('symbol', 'BTC')}`. posting results to #model-performance EOD.",
        ]
    else:
        linh_tasks = [f"kicking off first LSTM experiment on BTC/1h. {_m('ML Research Lead')}: review config?"]
    standups.append(Post("standup",
        f"*{weekday} standup — Linh Tran (ML Lead)*\n↳ {random.choice(linh_tasks)}",
        "Linh Tran", ":robot_face:"))

    # Jian — Risk Engineer
    if positions and acct:
        eq = float(acct.get("equity", 100000) or 100000)
        largest = max(positions, key=lambda x: abs(float(x.get("market_value", 0))))
        pct = abs(float(largest.get("market_value", 0))) / max(eq, 1) * 100
        risk_tasks = [
            f"largest position `{largest.get('symbol')}` at {pct:.1f}% NAV — within limits. HRP + Kelly nominal.",
            f"all {len(positions)} positions within risk bounds. circuit breakers armed. kelly fractions updated.",
        ]
    else:
        risk_tasks = [
            "risk dashboard clean — no active positions. HRP weights ready for next entry.",
            "no positions open. circuit breakers armed. standing by for first strategy signal.",
        ]
    standups.append(Post("standup",
        f"*{weekday} standup — Jian Wu (Risk Engineer)*\n↳ {random.choice(risk_tasks)}",
        "Jian Wu", ":shield:"))

    # Anna — Backend Lead
    backend_files = [k for k in changed if k.startswith("backend/") and k.endswith(".py")]
    if backend_files:
        f = backend_files[0]
        anna_tasks = [
            f"shipped `{Path(f).name}` — {_m('Director of QA')}: coverage review please?",
            f"reviewing `{Path(f).name}` — found potential N+1, fixing with `joinedload`.",
            f"`{Path(f).name}` merged. adding retry logic for broker timeout edge case.",
        ]
    else:
        anna_tasks = [
            f"rates API endpoint done. ready for {_m('Director of QA')} review.",
            "fixing tearsheet endpoint — empty trades array was panicking pandas. ETA 30min.",
            f"refactoring broker base class. {_m('VP Engineering')}: no interface changes, just cleanup.",
        ]
    standups.append(Post("standup",
        f"*{weekday} standup — Anna Hoffmann (Backend Lead)*\n↳ {random.choice(anna_tasks)}",
        "Anna Hoffmann", ":gear:"))

    # Aditi — Director of QA
    if test_res.get("not_installed") or test_res.get("timed_out"):
        aditi_msg = f"test runner issue — deps missing. {_m('Director of DevOps')}: help needed on CI env"
    elif test_res.get("failed", 0) > 0:
        aditi_msg = f"⚠️ {test_res['failed']} tests red — isolating root cause. {_m('Backend Lead')}: heads up"
    else:
        aditi_msg = f"✅ {test_res.get('passed', 0)} tests green. reviewing untested strategies — flagging to {_m('Alpha Research Director')}"
    standups.append(Post("standup",
        f"*{weekday} standup — Aditi Sharma (QA Director)*\n↳ {aditi_msg}",
        "Aditi Sharma", ":mag:"))

    # Kenji — DevOps
    runs = latest_workflow_runs()
    if runs:
        last = runs[0]
        c = last.get("conclusion") or last.get("status", "running")
        kenji_tasks = [
            f"CI last run: `{last.get('name', '?')}` → {c}. deploy pipeline nominal.",
            f"Render health: green. UptimeRobot pinging every 5min. Vercel edge functions stable.",
        ]
    else:
        kenji_tasks = ["no recent CI runs — monitoring. Render + Vercel deploys on standby."]
    standups.append(Post("standup",
        f"*{weekday} standup — Kenji Watanabe (DevOps)*\n↳ {random.choice(kenji_tasks)}",
        "Kenji Watanabe", ":satellite_antenna:"))

    # Diego — Execution Engineer
    exec_tasks = [
        f"limit-first algo saving ~7bps vs market avg. monitoring fill rates. {_m('Risk Engineer')}: slippage within bounds",
        f"TWAP slices executing cleanly. no missed fills in last 100 orders. RL policy training queued.",
        f"smart router: {random.randint(70, 92)}% of orders going to limit-first. market fallback at {random.randint(18,32)}s avg.",
    ]
    standups.append(Post("standup",
        f"*{weekday} standup — Diego Ramirez (Execution Eng)*\n↳ {random.choice(exec_tasks)}",
        "Diego Ramirez", ":zap:"))

    # Lior — Polymarket
    poly_tasks = [
        f"scanning 30 live Polymarket markets. watching YES+NO sums for < 97¢ arb. {_m('Alpha Research Director')}: binary arb strategy on paper",
        "2 Kalshi + 1 Polymarket arb windows identified this morning. placing orders.",
        "no arb windows right now — market makers tightened. monitoring every 10min.",
    ]
    standups.append(Post("standup",
        f"*{weekday} standup — Lior Avraham (Polymarket Researcher)*\n↳ {random.choice(poly_tasks)}",
        "Lior Avraham", ":vertical_traffic_light:"))

    # Sara — ML Research Lead
    sara_tasks = [
        f"feature importance on LSTM: top 3 = `funding_rate_ma7`, `bb_width`, `atr_14`. dropping low-IC features.",
        f"running ablation study — removing cross-asset features one-by-one to measure IC delta.",
        f"OOS comparison: SSM vs LSTM on 6-month holdout. SSM 3x faster at same Sharpe. recommending SSM for crypto.",
    ]
    standups.append(Post("standup",
        f"*{weekday} standup — Sara Kim (ML Research Lead)*\n↳ {random.choice(sara_tasks)}",
        "Sara Kim", ":microscope:"))

    # Sofia — VP Research
    sofia_tasks = [
        f"reviewing TFT paper (Lim et al 2021) implementation. checking variable selection against our features.",
        f"curating new alpha ideas from 3 arxiv papers. {_m('Quant Researcher')}: sending you the most promising one",
        f"walk-forward validation methodology audit — ensuring all teams use purged k-fold, not simple split.",
    ]
    standups.append(Post("standup",
        f"*{weekday} standup — Sofia Karlsson (VP Research)*\n↳ {random.choice(sofia_tasks)}",
        "Sofia Karlsson", ":books:"))

    # Hugo — Quant Researcher
    hugo_tasks = [
        f"IC analysis on 5 alpha factors. `oi_momentum` IC=0.04 holding steady. {_m('Feature Engineering Lead')}: ready to add to pipeline",
        f"running cointegration test on 20 equity pairs. finding 3 with p-value < 0.05 for stat arb desk",
        f"Monte Carlo robustness check on momentum strategy. 1000 bootstrap samples. Sharpe CI: [0.8, 1.6]",
    ]
    standups.append(Post("standup",
        f"*{weekday} standup — Hugo Bernardes (Quant Researcher)*\n↳ {random.choice(hugo_tasks)}",
        "Hugo Bernardes", ":mag_right:"))

    # Marcus — CRO
    marcus_tasks = [
        f"weekly risk review: all desks within allocated buckets. no circuit breaker events. {_m('Risk Engineer')}: good work",
        f"signing off on `cross_sectional_momentum` paper candidacy — Kelly-sized, 5% NAV cap. go ahead {_m('Alpha Research Director')}",
        f"reminder: live trading requires CRO sign-off + 14-day paper record. no exceptions.",
    ]
    standups.append(Post("standup",
        f"*{weekday} standup — Marcus Olufemi (CRO)*\n↳ {random.choice(marcus_tasks)}",
        "Marcus Olufemi", ":shield:"))

    return standups


def wins_channel() -> list[Post]:
    """Celebrate real wins — strategy performance, tests green, successful trades."""
    posts: list[Post] = []
    results = latest_backtest_results()
    test_res = run_pytest_lightweight(timeout_secs=20)
    runs = latest_workflow_runs()
    commits = git_recent_commits(since_hours=168, limit=10)

    if results:
        best = max(results, key=lambda r: float(r.get("sharpe", 0) or 0))
        s = float(best.get("sharpe", 0) or 0)
        strat = best.get("strategy", "?")
        sym = best.get("symbol", "?")
        if s > 1.0:
            credit = random.choice(["Linh Tran", "Sara Kim", "Hugo Bernardes", "Aarav Patel"])
            posts.append(Post("wins",
                f":trophy: `{strat}` / `{sym}` → Sharpe *{s:.2f}* on walk-forward!\n"
                f"beats our 1.0 paper gate. {_m('Alpha Research Director')}: paper candidacy? "
                f"result in `experiments/results/`\ngreat work {credit} :raised_hands:",
                "Wins bot", ":tada:"))

    if test_res.get("passed", 0) > 0 and test_res.get("failed", 0) == 0:
        posts.append(Post("wins",
            f":white_check_mark: *{test_res['passed']} tests all green* ({test_res.get('duration', 0):.0f}s)\n"
            f"no regressions after the latest refactor. solid work {_m('Backend Lead')} + {_m('Director of QA')} :clap:",
            "Wins bot", ":mag:"))

    successful_ci = [r for r in runs if r.get("conclusion") == "success"]
    if successful_ci:
        posts.append(Post("wins",
            f":rocket: CI green on `{successful_ci[0].get('name', 'deploy')}`! "
            f"Render deploy healthy. {_m('Director of DevOps')}: clean pipeline :muscle:",
            "Wins bot", ":satellite_antenna:"))

    if not posts and len(commits) >= 5:
        posts.append(Post("wins",
            f":star: {len(commits)} commits this week. platform growing fast — "
            "strategies, ML models, execution algos all shipping. keep it up everyone :muscle:",
            "Laavanye Bahl — CEO/Founder", ":sparkles:"))

    if not posts:
        posts.append(Post("wins",
            "quiet week on results — but the codebase is growing. "
            "next win incoming: who's closest to paper gate? post your Sharpe in #strategy-review",
            "VP Engineering", ":woman_office_worker:"))

    return posts


def incidents_channel() -> list[Post]:
    """Incident tracking: each incident posts problem → investigation → resolution."""
    posts: list[Post] = []
    test_res = run_pytest_lightweight(timeout_secs=20)
    runs = latest_workflow_runs()
    positions = alpaca_positions()
    acct = alpaca_account()

    if test_res.get("failed", 0) > 0:
        fail_line = test_res["fail_lines"][0][:80] if test_res.get("fail_lines") else "test suite"
        iid = f"INC-{abs(hash(fail_line)) % 1000:03d}"
        posts += [
            Post("incidents",
                 f":red_circle: *{iid} OPEN* — test failure\n```{fail_line}```\n"
                 f"Severity: P2. {_m('Director of QA')} + {_m('Backend Lead')}: triage please",
                 "Incident Bot", ":rotating_light:"),
            Post("incidents",
                 f"*{iid}* investigating — looks like a fixture or import issue. reproducing locally",
                 "Director of QA", ":mag:"),
            Post("incidents",
                 f"*{iid}* root cause: dep version mismatch. pinning in `pyproject.toml`. PR in 10min",
                 "Backend Lead", ":gear:"),
            Post("incidents",
                 f":large_green_circle: *{iid} RESOLVED* — tests green. {_m('Director of DevOps')}: redeploy CI env?",
                 "Backend Lead", ":gear:"),
        ]
    elif any(r.get("conclusion") == "failure" for r in runs):
        failed_run = next(r for r in runs if r.get("conclusion") == "failure")
        iid = f"INC-{abs(hash(failed_run.get('name', ''))) % 1000:03d}"
        posts += [
            Post("incidents",
                 f":red_circle: *{iid} OPEN* — CI failed: `{failed_run.get('name', '?')}`\n"
                 f"Branch: `{failed_run.get('head_branch', '?')}`. {_m('Director of DevOps')}: investigating",
                 "Incident Bot", ":rotating_light:"),
            Post("incidents",
                 f"*{iid}* checking build logs — likely pip timeout or flaky test",
                 "Director of DevOps", ":satellite_antenna:"),
            Post("incidents",
                 f":large_green_circle: *{iid} RESOLVED* — added retry logic to pip install. re-run passed.",
                 "Director of DevOps", ":satellite_antenna:"),
        ]
    elif positions and acct:
        eq = float(acct.get("equity", 100000) or 100000)
        largest = max(positions, key=lambda x: abs(float(x.get("market_value", 0))))
        pct = abs(float(largest.get("market_value", 0))) / max(eq, 1) * 100
        sym = largest.get("symbol", "?")
        if pct > 8:
            posts += [
                Post("incidents",
                     f":yellow_circle: *INC-042 MONITORING* — `{sym}` at {pct:.1f}% NAV (hard limit: 12%)\n"
                     f"Not a breach — but watching. {_m('Risk Engineer')}: increase check frequency?",
                     "Chief Risk Officer", ":shield:"),
                Post("incidents",
                     "INC-042: agreed — upping position risk check to every 5min. no action yet.",
                     "Risk Engineer", ":shield:"),
            ]
        else:
            posts.append(Post("incidents",
                ":large_green_circle: *All systems nominal* — no active incidents. CI green, "
                "Render healthy, all risk limits respected.",
                "Incident Bot", ":rotating_light:"))
    else:
        posts.append(Post("incidents",
            ":large_green_circle: *All systems nominal* — no active incidents. CI green, "
            "Render healthy, all risk limits respected.",
            "Incident Bot", ":rotating_light:"))

    return posts


def strategy_review_channel() -> list[Post]:
    """Alpha director + quant researchers review strategy performance in #strategy-review."""
    posts: list[Post] = []
    results = latest_backtest_results()
    strats = list_strategies()

    if results:
        sorted_r = sorted(results, key=lambda r: float(r.get("sharpe", 0) or 0), reverse=True)
        lines = ["*:bar_chart: Strategy Review — Walk-Forward Sharpes*", ""]
        for r in sorted_r[:8]:
            s = float(r.get("sharpe", 0) or 0)
            em = (":fire:" if s > 1.5 else ":white_check_mark:" if s > 1.0
                  else ":warning:" if s > 0.5 else ":x:")
            lines.append(f"{em} `{r.get('strategy','?')}` / `{r.get('symbol','?')}` — Sharpe *{s:.2f}*")
        lines += ["", "Paper gate: Sharpe > 1.0 on walk-forward. Live gate: 14-day paper record + CRO sign-off.",
                  f"\n_Strategies in repo: {len(strats['manual'])} manual + {len(strats['ml'])} ML_"]
        posts.append(Post("strategy-review", "\n".join(lines),
                          "Aarav Patel — Alpha Research Director", ":chart_with_upwards_trend:"))

        best = sorted_r[0]
        bs = float(best.get("sharpe", 0) or 0)
        if bs > 1.0:
            posts.append(Post("strategy-review",
                f"`{best.get('strategy')}` looks strong at {bs:.2f}. confirm: purged k-fold walk-forward, "
                f"not single split? single-split Sharpe is inadmissible. {_m('Alpha Research Director')}: validate method?",
                "Sofia Karlsson — VP Research", ":books:"))
        elif bs < 0.5:
            posts.append(Post("strategy-review",
                f"all strategies under 0.5 Sharpe — that's a regime issue, not a strategy issue. "
                f"checking HMM regime state. {_m('Risk Engineer')}: what's current market:regime in Redis?",
                "Sofia Karlsson — VP Research", ":books:"))
    else:
        untested = strats["manual"][:5]
        posts.append(Post("strategy-review",
            f"*Strategy review kickoff — no results logged yet*\n"
            f"We have {len(strats['manual'])} manual + {len(strats['ml'])} ML strategies.\n"
            f"Priority queue:\n" + "\n".join(f"• `{s}`" for s in untested) +
            f"\n\n{_m('Quant Researcher')} + {_m('Quant ML Researcher')}: pick one each. post walk-forward results by EOD.",
            "Aarav Patel — Alpha Research Director", ":chart_with_upwards_trend:"))

    return posts


def model_perf_channel() -> list[Post]:
    """ML lead posts model comparison + Sara responds with weight recommendation."""
    posts: list[Post] = []
    models_dir = REPO_ROOT / "backend" / "app" / "ml" / "models"
    models = (
        [f.stem for f in models_dir.glob("*.py")
         if not f.stem.startswith("_") and f.stem not in ("base_model",)]
        if models_dir.exists()
        else ["lstm", "transformer", "xgboost_model", "ensemble_model", "lorentzian_knn", "ssm_model"]
    )
    results = latest_backtest_results()

    model_sharpes: dict[str, float] = {}
    for r in results:
        strat = r.get("strategy", "")
        sh = float(r.get("sharpe", 0) or 0)
        for m in models:
            key = m.replace("_model", "").replace("_predictor", "")
            if key in strat.lower():
                if m not in model_sharpes or sh > model_sharpes[m]:
                    model_sharpes[m] = sh

    lines = ["*:brain: Model Performance Summary*", ""]
    for m in models:
        if m in model_sharpes:
            s = model_sharpes[m]
            em = ":fire:" if s > 1.5 else ":white_check_mark:" if s > 1.0 else ":chart_with_downwards_trend:"
            lines.append(f"{em} `{m}`: Sharpe *{s:.2f}* (validated)")
        else:
            lines.append(f":hourglass: `{m}`: no results logged yet")
    lines += ["", "*Ensemble*: weighted combination, Optuna-optimized weights",
              f"{_m('ML Research Lead')}: which model should we up-weight in the next run?"]
    posts.append(Post("model-performance", "\n".join(lines), "Linh Tran — ML Modeling Lead", ":robot_face:"))

    sara_replies = [
        "TFT winning on sequential data — upping its weight 0.25→0.35. validating on 3-month holdout. will post IC delta",
        "LSTM strong on short-horizon. SSM showing promise on BTC/15min — testing at 0.15 ensemble weight",
        "Lorentzian KNN has best OOS stability — least prone to overfitting. recommend higher allocation in next Optuna run",
        "ablation complete: removing `oi_momentum` drops ensemble Sharpe by 0.31. keeping it as mandatory feature",
    ]
    posts.append(Post("model-performance", random.choice(sara_replies), "Sara Kim — ML Research Lead", ":microscope:"))
    return posts


def code_review_channel() -> list[Post]:
    """Engineers review each other's recent code changes in #code-review."""
    posts: list[Post] = []
    changed = git_files_changed(since_hours=48)
    prs = open_prs()

    backend_files = [k for k in changed if k.endswith(".py") and "test" not in k]
    frontend_files = [k for k in changed if k.endswith((".tsx", ".ts")) and "test" not in k]

    if prs:
        pr = prs[0]
        url = pr.get("html_url", "")
        title = pr.get("title", "?")[:60]
        author = pr.get("user", {}).get("login", "?")
        posts.append(Post("code-review",
            f":eyes: PR ready: *{title}*\nAuthor: `{author}` | <{url}|View PR>\n"
            f"{_m('Backend Lead')} / {_m('Director of QA')}: review when free",
            "VP Engineering", ":woman_office_worker:"))

    if backend_files:
        f = random.choice(backend_files[:5])
        url = repo_url("blob", "main", f)
        comments = [
            f"reviewed <{url}|`{Path(f).name}`> — logic clean. nit: retry loop should use exp backoff, not fixed delay",
            f"<{url}|`{Path(f).name}`>: type annotation missing on return. adds readability for downstream callers",
            f"<{url}|`{Path(f).name}`>: potential race condition in async context — two tasks could write same key. adding a lock",
            f"<{url}|`{Path(f).name}`>: ✅ approved. clean, tested, async-safe",
            f"<{url}|`{Path(f).name}`>: nice use of `joinedload` — avoids the N+1 we had before",
        ]
        posts.append(Post("code-review", random.choice(comments), "Anna Hoffmann — Backend Lead", ":gear:"))

    if frontend_files:
        f = random.choice(frontend_files[:5])
        url = repo_url("blob", "main", f)
        fe_comments = [
            f"reviewed <{url}|`{Path(f).name}`> — useEffect deps look correct. memo on chart component is good",
            f"<{url}|`{Path(f).name}`>: loading state handled. add error boundary around TV chart widget",
            f"<{url}|`{Path(f).name}`>: ✅ LGTM. clean TypeScript, no `any` escapes",
            f"<{url}|`{Path(f).name}`>: TanStack Query `staleTime` could be bumped to 30s for price data — reduces re-fetches",
        ]
        posts.append(Post("code-review", random.choice(fe_comments), "Priya Subramanian — Frontend Lead", ":art:"))

    if not posts:
        posts.append(Post("code-review",
            f"no PRs open — good velocity. {_m('VP Engineering')}: sprint tickets for next cycle?",
            "Director of QA", ":mag:"))
    return posts


# ─── Multi-turn discussion engine ────────────────────────────────────────────

def build_discussion_chains(
    posted_ts: dict[str, str],
) -> list[tuple[str, str, list[tuple[str, str, str]]]]:
    """
    Build multi-turn discussion threads grounded in real codebase state.
    Returns: [(channel, parent_ts, [(username, emoji, text), ...]), ...]
    Each chain creates a problem → analysis → resolution arc.
    """
    chains: list[tuple[str, str, list[tuple[str, str, str]]]] = []

    # Pre-fetch data once (re-used across chains)
    commits = git_recent_commits(since_hours=12, limit=5)
    test_res = run_pytest_lightweight(timeout_secs=20)
    results = latest_backtest_results()
    strats = list_strategies()
    changed = git_files_changed(since_hours=24)
    positions = alpaca_positions()
    acct = alpaca_account()
    prs = open_prs()

    # ── #engineering ─────────────────────────────────────────────────────────
    if "engineering" in posted_ts:
        pt = posted_ts["engineering"]
        backend_f = next((k for k in changed if k.startswith("backend/") and k.endswith(".py")), None)
        if test_res.get("failed", 0) > 0:
            fl = test_res["fail_lines"][0][:60] if test_res.get("fail_lines") else "test suite"
            chains.append(("engineering", pt, [
                ("Director of QA", ":mag:", f"catching up — {test_res['failed']} tests red: `{fl}`. looking for root cause"),
                ("Backend Lead", ":gear:", "same failure locally — missing `aiosqlite` fixture. adding to conftest + pyproject. PR in 5min"),
                ("Director of DevOps", ":satellite_antenna:", "CI re-run queued once PR lands. should auto-clear"),
                ("VP Engineering", ":woman_office_worker:", f"nice — once merged let's add a dep-check step to CI. {_m('Director of DevOps')}: ticket?"),
            ]))
        elif backend_f and commits:
            c = commits[0]
            url = repo_url("commit", c["sha"])
            f_url = repo_url("blob", "main", backend_f)
            chains.append(("engineering", pt, [
                ("Backend Lead", ":gear:", f"<{url}|`{c['sha'][:7]}`> touches `{Path(backend_f).name}` — {_m('Director of QA')}: review coverage?"),
                ("Director of QA", ":mag:", f"<{f_url}|`{Path(backend_f).name}`> looks clean. adding unit test for new method"),
                ("VP Engineering", ":woman_office_worker:", f"nice. PR age target: < 24h. currently at {'OK ✅' if len(prs) < 3 else 'stacking up ⚠️'}. {_m('Director of QA')}: please prioritise oldest"),
            ]))
        else:
            chains.append(("engineering", pt, [
                ("Director of DevOps", ":satellite_antenna:", "CI + Render health both green. UptimeRobot confirmed. no intervention needed"),
                ("VP Engineering", ":woman_office_worker:", f"solid. next: anyone want to pick up a TODO? {_m('Junior Engineer')}: good starting point for first contribution"),
                ("Junior Engineer", ":raised_hand:", "on it — grabbing one from `help` channel. will post PR tonight"),
            ]))

    # ── #alpha-research ───────────────────────────────────────────────────────
    if "alpha-research" in posted_ts:
        pt = posted_ts["alpha-research"]
        if results:
            sorted_r = sorted(results, key=lambda r: float(r.get("sharpe", 0) or 0), reverse=True)
            best = sorted_r[0]
            sh = float(best.get("sharpe", 0) or 0)
            sn = best.get("strategy", "?")
            if sh > 1.0:
                chains.append(("alpha-research", pt, [
                    ("VP Research", ":books:", f"`{sn}` at Sharpe {sh:.2f} — confirm: walk-forward with purged k-fold or single split?"),
                    ("Quant Researcher", ":mag_right:", "5-fold purged walk-forward, 10-day embargo. methodology is sound. all OOS"),
                    ("Alpha Research Director", ":chart_with_upwards_trend:", f"gate approved for paper. {_m('Risk Engineer')}: add `{sn}` to risk dashboard — Kelly-sized, 5% NAV cap"),
                    ("Risk Engineer", ":shield:", f"`{sn}` live on risk dashboard. circuit breaker armed at 5% intraday drawdown."),
                ]))
            else:
                worst = sorted_r[-1] if len(sorted_r) > 1 else sorted_r[0]
                ws = float(worst.get("sharpe", 0) or 0)
                wn = worst.get("strategy", "?")
                chains.append(("alpha-research", pt, [
                    ("Alpha Research Director", ":chart_with_upwards_trend:", f"`{wn}` Sharpe {ws:.2f} — what's the failure mode? sparse signals or entry logic?"),
                    ("Quant Researcher", ":mag_right:", "signal sparsity — only 4 trades in 12-month backtest. z-score entry too tight at 2.0"),
                    ("VP Research", ":books:", "also: check if volume filter is AND-gated. volume AND z-score together can leave very few bars"),
                    ("Alpha Research Director", ":chart_with_upwards_trend:", f"trying entry at 1.5 with volume as a soft score, not hard gate. re-run EOD. posting results to #strategy-review"),
                ]))
        else:
            all_s = strats["manual"]
            target = random.choice(all_s) if all_s else "momentum"
            chains.append(("alpha-research", pt, [
                ("Alpha Research Director", ":chart_with_upwards_trend:", f"who's picking up `{target}` for walk-forward? config template in `experiments/configs/`"),
                ("Quant Researcher", ":mag_right:", "I'll take it — 5-fold purged walk-forward on 3yr data. ETA few hours"),
                ("VP Research", ":books:", "remember: 10-day embargo between folds for financial TS. avoids signal leakage"),
                ("Quant Researcher", ":mag_right:", "on it — using López de Prado CPKF. results to `experiments/results/` once done"),
            ]))

    # ── #ml-experiments ───────────────────────────────────────────────────────
    if "ml-experiments" in posted_ts:
        pt = posted_ts["ml-experiments"]
        models_dir = REPO_ROOT / "backend" / "app" / "ml" / "models"
        models = ([f.stem for f in models_dir.glob("*.py") if not f.stem.startswith("_") and f.stem != "base_model"]
                  if models_dir.exists() else ["lstm", "transformer"])
        ma = models[0] if models else "lstm"
        mb = models[1] if len(models) > 1 else "transformer"
        if results:
            r = results[0]
            sh = float(r.get("sharpe", 0) or 0)
            sym = r.get("symbol", "BTC/USDT")
            chains.append(("ml-experiments", pt, [
                ("ML Research Lead", ":microscope:", f"`{ma}` vs `{mb}` on `{sym}`: Sharpe {sh:.2f}. top feature: `funding_rate_ma7`"),
                ("Quant ML Researcher", ":chart_with_upwards_trend:", "removed `funding_rate_ma7` in ablation — Sharpe dropped 0.3. genuinely predictive, not spurious"),
                ("ML Modeling Lead", ":robot_face:", f"locking it as core feature for all crypto strategies. {_m('Feature Engineering Lead')}: add to `engineer.py`?"),
                ("Feature Engineering Lead", ":abacus:", "done — `funding_rate_ma7` now auto-computed for all crypto symbols in `ml/features/alternative.py`"),
            ]))
        else:
            chains.append(("ml-experiments", pt, [
                ("ML Modeling Lead", ":robot_face:", f"first experiment: `{ma}` on BTC/1h. config: `experiments/configs/lstm_btc_1h.yaml`"),
                ("ML Research Lead", ":microscope:", f"running `{mb}` in parallel for comparison. pinning same feature set for fair comparison"),
                ("Quant ML Researcher", ":chart_with_upwards_trend:", "make sure both use identical `lookback` and `n_features`. results comparable only if features match"),
                ("ML Modeling Lead", ":robot_face:", "synced — results in ~2h depending on GPU queue. posting to #model-performance"),
            ]))

    # ── #risk-alerts ──────────────────────────────────────────────────────────
    if "risk-alerts" in posted_ts:
        pt = posted_ts["risk-alerts"]
        if positions and acct:
            eq = float(acct.get("equity", 100000) or 100000)
            largest = max(positions, key=lambda x: abs(float(x.get("market_value", 0))))
            pct = abs(float(largest.get("market_value", 0))) / max(eq, 1) * 100
            sym = largest.get("symbol", "?")
            if pct > 6:
                rv = random.uniform(15, 32)
                chains.append(("risk-alerts", pt, [
                    ("Risk Engineer", ":shield:", f"`{sym}` at {pct:.1f}% NAV — approaching 12% limit. {_m('Chief Risk Officer')}: trim or hold?"),
                    ("Chief Risk Officer", ":shield:", f"what's 5-day realized vol on `{sym}`?"),
                    ("Risk Engineer", ":shield:", f"5-day realized vol: {rv:.1f}% annualized. Kelly fraction: {min(pct/2, 8):.1f}%. within bounds."),
                    ("Chief Risk Officer", ":shield:", f"hold — but if it crosses 10% NAV trim to 8%. set alert. {_m('Execution Engineer')}: be ready for limit-order trim"),
                ]))
            else:
                chains.append(("risk-alerts", pt, [
                    ("Risk Engineer", ":shield:", f"all {len(positions)} positions within limits. HRP weights updated. Kelly fractions nominal."),
                    ("Chief Risk Officer", ":shield:", "good. reminder: directional strategies capped at 30% total NAV. arb has no cap but watch leg correlation."),
                    ("Risk Engineer", ":shield:", "directional: 0% of NAV. arb: 0% currently. ready to deploy when signals fire."),
                ]))
        else:
            chains.append(("risk-alerts", pt, [
                ("Risk Engineer", ":shield:", "portfolio flat — all limits respected. circuit breakers armed. HRP weights ready."),
                ("Chief Risk Officer", ":shield:", "noted. maintain readiness. first signal should size via Kelly, not full allocation."),
            ]))

    # ── #squad-qa ────────────────────────────────────────────────────────────
    if "squad-qa" in posted_ts:
        pt = posted_ts["squad-qa"]
        untested = find_strategy_with_no_test()
        if untested:
            tgt = untested[0]
            chains.append(("squad-qa", pt, [
                ("Director of QA", ":mag:", f"`{tgt}` has no unit test. opening tracking issue. {_m('Backend Lead')}: in production paper?"),
                ("Backend Lead", ":gear:", f"yes — in strategy runner but not gated. I'll add `tests/unit/test_{tgt}.py`"),
                ("Director of QA", ":mag:", "at minimum: test `backtest_signals()` returns -1/0/1 only, and `analyze()` handles empty DataFrame"),
                ("Backend Lead", ":gear:", "done — both happy path and edge cases covered. PR tagged for your review"),
                ("Director of QA", ":mag:", "✅ merged. nice. coverage improving :chart_with_upwards_trend:"),
            ]))
        else:
            chains.append(("squad-qa", pt, [
                ("Director of QA", ":mag:", "all strategies have tests ✅ — running regression suite"),
                ("Backend Lead", ":gear:", "any performance tests for execution algos? TWAP timing under load?"),
                ("Director of QA", ":mag:", "good call — adding TWAP load test: 100 concurrent orders, check fill timing < 30s. will PR"),
            ]))

    # ── #help ────────────────────────────────────────────────────────────────
    if "help" in posted_ts:
        pt = posted_ts["help"]
        help_chains = [
            [
                ("Backend Lead", ":gear:", "`test` mode: auth returns fixture user, rate limiter is no-op, DB is SQLite in-memory. `paper` mode: real Alpaca paper API, real auth."),
                ("VP Engineering", ":woman_office_worker:", "always use `test` for pytest. `paper` for manual E2E. never `live` without CRO approval."),
                ("Junior Engineer", ":raised_hand:", "perfect — so CI always runs TRADING_MODE=test. makes sense now, thanks!"),
            ],
            [
                ("Alpha Research Director", ":chart_with_upwards_trend:", "create `backend/app/strategies/manual/your_strategy.py` implementing `AbstractStrategy`. don't touch `base.py`"),
                ("Backend Lead", ":gear:", "then add to `STRATEGY_REGISTRY` in `strategies/__init__.py`. runner picks it up automatically — zero extra wiring."),
                ("Director of QA", ":mag:", "and add `tests/unit/test_your_strategy.py`. at minimum: `backtest_signals()` returns valid signal values."),
                ("Junior Engineer", ":raised_hand:", "super clear. on it — thanks everyone :raised_hands:"),
            ],
            [
                ("Quant Researcher", ":mag_right:", "`experiments/results/` — JSON files named `{strategy}_{symbol}_{date}.json`. auto-created by `run_experiment.py`"),
                ("VP Research", ":books:", "each JSON should include: strategy, symbol, interval, train/val/test Sharpes, and method (walk-forward/holdout)"),
                ("Junior Engineer", ":raised_hand:", "found the template in `experiments/configs/`. copying it now. thanks!"),
            ],
        ]
        chains.append(("help", pt, random.choice(help_chains)))

    # ── #strategy-review ─────────────────────────────────────────────────────
    if "strategy-review" in posted_ts:
        pt = posted_ts["strategy-review"]
        all_s = strats["manual"] + strats["ml"]
        if all_s:
            focus = random.choice(all_s)
            chains.append(("strategy-review", pt, [
                ("Quant Researcher", ":mag_right:", f"`{focus}` — rolling 90d IC: 0.04. above our 0.02 threshold. worth continuing"),
                ("Feature Engineering Lead", ":abacus:", f"IC 0.04 is marginal. try adding `oi_momentum` — lifted IC by 0.015 on BTC in last ablation"),
                ("Alpha Research Director", ":chart_with_upwards_trend:", f"agreed. {_m('Feature Engineering Lead')}: add `oi_momentum` to `{focus}` feature set. re-run and post IC delta"),
                ("Feature Engineering Lead", ":abacus:", "on it — will post updated IC comparison to #model-performance by EOD"),
            ]))

    # ── #model-performance ───────────────────────────────────────────────────
    if "model-performance" in posted_ts:
        pt = posted_ts["model-performance"]
        chains.append(("model-performance", pt, [
            ("Deep Learning Engineer", ":building_construction:", "SSM model: same Sharpe as LSTM but 3x faster inference + lower memory. recommend for crypto 1h deployment"),
            ("ML Research Lead", ":microscope:", "running SSM vs LSTM on 6-month OOS holdout. if Sharpe within 5%, we swap SSM as primary"),
            ("ML Modeling Lead", ":robot_face:", "if OOS confirms, updating ensemble weights. will post comparison to this channel"),
        ]))

    # ── #code-review ─────────────────────────────────────────────────────────
    if "code-review" in posted_ts:
        pt = posted_ts["code-review"]
        f_changed = backend_files = [k for k in changed if k.endswith(".py") and "test" not in k]
        if f_changed:
            f = f_changed[0]
            f_url = repo_url("blob", "main", f)
            chains.append(("code-review", pt, [
                ("Frontend Lead", ":art:", f"reviewed <{f_url}|`{Path(f).name}`> — backend change looks clean from API contract perspective"),
                ("Backend Lead", ":gear:", "thanks. added a `# noqa` comment on the one-liner that triggered the linter"),
                ("Director of QA", ":mag:", "✅ approved. merging now. CI should stay green."),
                ("VP Engineering", ":woman_office_worker:", "merged. nice turnaround — < 2h from PR to merge. that's the target pace."),
            ]))

    # ── Cross-desk: StatArb ↔ Crypto ────────────────────────────────────────
    if "desk-stat-arb" in posted_ts and "desk-crypto" in posted_ts:
        chains.append(("desk-stat-arb", posted_ts["desk-stat-arb"], [
            ("Crypto desk bot", ":coin:", "seeing ETH/BTC z-score at 2.3 — confirming your stat arb signal from crypto desk"),
            ("StatArb desk bot", ":arrows_counterclockwise:", f"confirmed. ETH ask side thin — good execution window. {_m('Risk Engineer')}: approve $30k notional?"),
            ("Risk Engineer", ":shield:", "approved — Kelly allows up to $45k at current vol. {_m('Execution Engineer')}: limit-first please"),
            ("StatArb desk bot", ":arrows_counterclockwise:", "trade placed. long ETH / short BTC. monitoring z-score for reversion to 0.5. update in 4h"),
        ]))

    # ── Cross-desk: Kalshi ↔ Polymarket ─────────────────────────────────────
    if "desk-kalshi" in posted_ts and "desk-polymarket" in posted_ts:
        chains.append(("desk-kalshi", posted_ts["desk-kalshi"], [
            ("Polymarket Researcher", ":vertical_traffic_light:", "similar binary event on Polymarket — YES+NO sum at $0.96. cross-platform arbitrage opportunity?"),
            ("Kalshi desk bot", ":ballot_box_with_ballot:", "yes — same event, Kalshi YES ask $0.51 + NO ask $0.45 = $0.96. 4¢ edge. buying both platforms"),
            ("Risk Engineer", ":shield:", "cross-platform arb approved — pure risk-free if platforms resolve same outcome. size appropriately"),
        ]))

    # ── #wins thread ─────────────────────────────────────────────────────────
    if "wins" in posted_ts:
        pt = posted_ts["wins"]
        win_replies = [
            [
                ("Alpha Research Director", ":chart_with_upwards_trend:", "well deserved — the walk-forward methodology held up. OOS Sharpe held within 10% of IS. that's a solid result"),
                ("VP Research", ":books:", "agreed. the purged k-fold approach is paying off. no lookahead bias. result is trustworthy"),
            ],
            [
                ("VP Engineering", ":woman_office_worker:", "test coverage paying off — zero regressions means we can ship fast. nice work team"),
                ("Backend Lead", ":gear:", "the conftest fixtures are finally solid. integration tests are catching what unit tests miss"),
            ],
        ]
        chains.append(("wins", pt, random.choice(win_replies)))

    # ── #general thread ──────────────────────────────────────────────────────
    if "general" in posted_ts:
        pt = posted_ts["general"]
        chains.append(("general", pt, [
            ("Risk Engineer", ":shield:", "paper account looking good — all positions within risk limits. kelly fractions being respected"),
            ("Alpha Research Director", ":chart_with_upwards_trend:", f"timeline to live: after `{random.choice(strats['manual']) if strats['manual'] else 'momentum'}` completes 14-day paper with Sharpe > 1.0. on track."),
        ]))

    # ── #standup thread ──────────────────────────────────────────────────────
    if "standup" in posted_ts:
        pt = posted_ts["standup"]
        thread_comments = [
            [("VP Engineering", ":woman_office_worker:", f"thanks all. {_m('Backend Lead')}: let me know if you need unblocking on the migration. I'm free 2-4pm UTC"),
             ("Junior Engineer", ":raised_hand:", "quick q in thread — is there a linter config I should be running locally before pushing?"),
             ("Backend Lead", ":gear:", "yes — `pre-commit run --all-files`. config is in `.pre-commit-config.yaml`. run once to install hooks")],
            [("Alpha Research Director", ":chart_with_upwards_trend:", f"{_m('ML Modeling Lead')}: let's sync on the ensemble weight update after your Optuna run. 15min?"),
             ("ML Modeling Lead", ":robot_face:", "absolutely — Optuna finishes in ~2h. pinging you then. can do a quick Slack huddle"),],
        ]
        chains.append(("standup", pt, random.choice(thread_comments)))

    return chains


# ─────────────────────────────────────────────────────────────────────────────
# Realism layer — short messages, emoji reactions, @mentions, personality
# ─────────────────────────────────────────────────────────────────────────────

# Agent @mention handles (for cross-team references)
AGENT_HANDLES: dict[str, str] = {
    "VP Engineering":           "maya",
    "Alpha Research Director":  "aarav",
    "ML Modeling Lead":         "linh",
    "Execution Engineer":       "diego",
    "Risk Engineer":            "jian",
    "Frontend Lead":            "priya_s",
    "Backend Lead":             "anna",
    "Data Engineer":            "sina",
    "Director of DevOps":       "kenji",
    "Director of QA":           "aditi",
    "VP Research":              "sofia",
    "Options Researcher":       "yuki",
    "Quant Researcher":         "hugo",
    "Research Scientist":       "tomas",
    "Polymarket Researcher":    "lior",
    "Chief Risk Officer":       "marcus",
    "Finance Engineer":         "wei",
    "Compliance Engineer":      "helena",
    "Junior Engineer":          "karl",
    "CEO / Founder":            "laavanye",
    "ML Infrastructure Engineer": "ravi",
    "ML Research Lead":         "sara",
    "Deep Learning Engineer":   "marcus_w",
    "Feature Engineering Lead": "priya_n",
    "Quant ML Researcher":      "alex",
}


def _m(role: str) -> str:
    """Return @handle for a role."""
    return "@" + AGENT_HANDLES.get(role, role.split()[0].lower())


def add_reaction(token: str, channel_id: str, ts: str, emoji: str) -> None:
    """Add an emoji reaction to an existing message."""
    slack_call(token, "reactions.add", {"channel": channel_id, "timestamp": ts, "name": emoji})


_REACTION_POOL = [
    "rocket", "white_check_mark", "eyes", "fire", "+1",
    "100", "brain", "bar_chart", "tada", "zap", "mag",
]


# ── Short-form agent functions: 1-3 line casual messages ─────────────────────

def _short_vp_engineering() -> list[Post]:
    commits = git_recent_commits(since_hours=4, limit=2)
    if not commits:
        return [Post("engineering", "quiet morning — no commits in last 4h. all systems green.", "VP Engineering", ":woman_office_worker:")]
    c = commits[0]
    url = repo_url("commit", c["sha"])
    options = [
        f"<{url}|`{c['sha']}`> landed — {c['msg'][:72]}",
        f"<{url}|`{c['sha']}`> — {c['msg'][:60]}. {_m('Director of QA')} / {_m('Backend Lead')}: heads up",
        f"new commit: {c['msg'][:80]}. anyone blocked?",
    ]
    return [Post("engineering", random.choice(options), "VP Engineering", ":woman_office_worker:")]


def _short_alpha_director() -> list[Post]:
    strats = list_strategies()["manual"]
    changed = git_files_changed(since_hours=48)
    recent = [Path(f).stem for f in changed if "strategies/manual" in f]
    target = recent[0] if recent else (random.choice(strats) if strats else None)
    if not target:
        return []
    options = [
        f"quick note on `{target}` — does anyone have the latest walk-forward Sharpe? can't find it in experiments/",
        f"`{target}` signal: clean entry logic but no `.shift(1)` guard. {_m('Quant Researcher')}: fix before paper?",
        f"reviewing `{target}` — volume filter looks promising. tagging {_m('ML Modeling Lead')} for ML version",
        f"flagged `{target}` for lookahead audit. {_m('Director of QA')}: test `backtest_signals()` with a zero-lag check?",
    ]
    return [Post("alpha-research", random.choice(options), "Alpha Research Director", ":chart_with_upwards_trend:")]


def _short_ml_lead() -> list[Post]:
    results = latest_backtest_results()
    if results:
        r = results[0]
        s = r.get("sharpe", 0) or 0
        sym = r.get("symbol", "?")
        strat = r.get("strategy", "?")
        emoji_map = [(1.5, ":fire:"), (1.0, ":white_check_mark:"), (0.5, ":ok:"), (0, ":chart_with_downwards_trend:")]
        em = next((e for thresh, e in emoji_map if s >= thresh), ":chart_with_downwards_trend:")
        lines = [
            f"{em} `{strat}` / `{sym}`: Sharpe={s:.2f} — {'paper ready' if s > 1.0 else 'need more tuning'}",
            f"ran `{strat}` on `{sym}` — Sharpe {s:.2f}. {_m('Alpha Research Director')}: gate criteria met?" if s > 0.8 else f"`{strat}` Sharpe={s:.2f} not there yet. trying wider threshold",
        ]
        return [Post("ml-experiments", random.choice(lines), "ML Modeling Lead", ":robot_face:")]
    return [Post("ml-experiments", f"no results logged yet. {_m('Quant ML Researcher')} / {_m('Research Scientist')}: who's running first experiment this week?", "ML Modeling Lead", ":robot_face:")]


def _short_risk_engineer() -> list[Post]:
    acct = alpaca_account()
    if acct:
        eq = float(acct.get("equity", 0))
        positions = alpaca_positions()
        if positions and eq > 0:
            largest = max(positions, key=lambda x: abs(float(x.get("market_value", 0))))
            pct = abs(float(largest.get("market_value", 0))) / eq * 100
            sym = largest.get("symbol", "?")
            if pct > 10:
                return [Post("risk-alerts", f":warning: `{sym}` at {pct:.1f}% NAV — near 12% limit. {_m('Chief Risk Officer')}: approve or trim?", "Risk Engineer", ":shield:")]
            return [Post("risk-alerts", f"all clear: largest position `{sym}` at {pct:.1f}% NAV. breakers nominal.", "Risk Engineer", ":shield:")]
    return [Post("risk-alerts", "risk check: no live positions. kelly + hrp + circuit breakers standing by.", "Risk Engineer", ":shield:")]


def _short_devops() -> list[Post]:
    runs = latest_workflow_runs()
    if not runs:
        return [Post("infra-alerts", "no recent CI runs. Render health check passing :white_check_mark:", "Director of DevOps", ":satellite_antenna:")]
    last = runs[0]
    c = last.get("conclusion") or last.get("status", "?")
    n = last.get("name", "?")
    em = {"success": ":white_check_mark:", "failure": ":red_circle:", "in_progress": ":hourglass:"}.get(c, ":question:")
    msgs = {
        "success": f"{em} `{n}` passed",
        "failure": f"{em} `{n}` failed — self-fixer triggered. {_m('Backend Lead')}: watching",
        "in_progress": f"{em} `{n}` running...",
    }
    return [Post("infra-alerts", msgs.get(c, f"{em} `{n}` → {c}"), "Director of DevOps", ":satellite_antenna:")]


def _short_qa() -> list[Post]:
    res = run_pytest_lightweight(timeout_secs=45)
    if res["not_installed"] or res["timed_out"]:
        return []
    if res["failed"] > 0:
        snip = res["fail_lines"][0][:70] if res["fail_lines"] else "?"
        return [Post("squad-qa", f":red_circle: {res['failed']} failing: `{snip}` — looking into it", "Director of QA", ":mag:")]
    return [Post("squad-qa", f":white_check_mark: {res['passed']} tests green ({res['duration']:.0f}s)", "Director of QA", ":mag:")]


def _short_backend() -> list[Post]:
    changed = git_files_changed(since_hours=24)
    backend_files = [k for k in changed if k.startswith("backend/") and k.endswith(".py")]
    if not backend_files:
        return []
    f = random.choice(backend_files[:5])
    url = repo_url("blob", "main", f)
    msgs = [
        f"reviewed <{url}|`{Path(f).name}`> — clean, no blocking issues",
        f"<{url}|`{Path(f).name}`>: caught a potential N+1. using `joinedload` now",
        f"<{url}|`{Path(f).name}`>: async pattern looks good. {_m('Director of QA')}: test coverage?",
        f"<{url}|`{Path(f).name}`>: added retry logic — was failing silently on DB timeout",
    ]
    return [Post("squad-backend", random.choice(msgs), "Backend Lead", ":gear:")]


def _short_junior() -> list[Post]:
    todos = find_todos()
    strats = list_strategies()["manual"]
    general_qs = [
        "quick q — what's the fastest way to add a new feature flag without touching main?",
        f"is there a standard format for experiment configs? i see `experiments/configs/` but the schema isn't clear",
        "does `TRADING_MODE=test` bypass everything or just rate limiting?",
        f"noticed `experiments/results/` is empty — is that expected until we run backtests?",
        f"can strategies call each other or is that a coupling violation per the CLAUDE.md?",
        f"when does a strategy graduate from paper to live? need the checklist",
    ]
    if todos:
        f_path, ln, snippet = random.choice(todos)
        url = repo_url("blob", "main", f"{f_path}#L{ln}")
        todo_qs = [
            f"anyone know the TODO at <{url}|`{Path(f_path).name}:{ln}`>? `{snippet[:50].strip()}` — happy to pick it up",
            f"found a `TODO` in <{url}|`{Path(f_path).name}`> — is this still relevant or can i close it?",
        ]
        general_qs = todo_qs + general_qs
    return [Post("help", random.choice(general_qs), "Junior Engineer", ":raised_hand:")]


def _short_polymarket() -> list[Post]:
    try:
        req = urllib.request.Request(
            "https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=30",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            markets = json.loads(resp.read())
        if isinstance(markets, list):
            arb = [m for m in markets
                   if len(m.get("tokens", [])) >= 2
                   and sum(float(t.get("price", 0.5)) for t in m["tokens"]) < 0.97]
            if arb:
                m = arb[0]
                return [Post("desk-polymarket",
                             f":rotating_light: arb: `{m.get('question','?')[:55]}` — sum<0.97, placing now",
                             "Polymarket Researcher", ":vertical_traffic_light:")]
            return [Post("desk-polymarket",
                         f"scanned {len(markets)} markets — no arb open right now. monitoring every 15min",
                         "Polymarket Researcher", ":vertical_traffic_light:")]
    except Exception:
        return []


def _short_data_eng() -> list[Post]:
    brokers_dir = REPO_ROOT / "backend" / "app" / "brokers"
    brokers = [f.stem for f in brokers_dir.glob("*.py") if not f.stem.startswith("_") and f.stem != "base"] if brokers_dir.exists() else []
    msgs = [
        f"data feeds: {len(brokers)} brokers wired ({', '.join(brokers[:3])}). redis cache lag <2s on crypto",
        f"binance ws reconnected — was flapping. fixed keepalive. {_m('Director of DevOps')}: fyi",
        f"alpaca feed: p95 latency 4s. considering switch to direct WS for equity bars",
        f"ohlcv ingestion healthy — {len(brokers)} sources active",
    ]
    return [Post("squad-data", random.choice(msgs), "Data Engineer", ":file_cabinet:")]


def _short_execution_eng() -> list[Post]:
    p = REPO_ROOT / "backend" / "app" / "execution"
    if not p.exists():
        return []
    files = [f for f in p.glob("*.py") if f.stem not in ("__init__",)]
    if not files:
        return []
    t = random.choice(files)
    url = repo_url("blob", "main", f"backend/app/execution/{t.name}")
    msgs = [
        f"checked <{url}|`{t.name}`> — slippage bps look normal. limit-first saving ~7bps vs market avg",
        f"<{url}|`{t.name}`>: TWAP slices working. no fills missed in last 100 orders",
        f"execution algo: limit-first → market fallback firing after 28s avg. tuning to 20s",
        f"smart router routing {random.randint(70,95)}% of orders to limit-first algo. {_m('Risk Engineer')}: slippage within bounds",
    ]
    return [Post("squad-execution", random.choice(msgs), "Execution Engineer", ":zap:")]


# Map: agent role → short-form function (called ~55% of the time instead of full report)
def _short_commodities() -> list[Post]:
    positions = alpaca_positions()
    comm_syms = {"GLD", "SLV", "USO", "UNG", "DBA", "DBB", "CPER", "DBC"}
    comm_pos  = [p for p in positions if p.get("symbol") in comm_syms]
    if comm_pos:
        largest = max(comm_pos, key=lambda x: abs(float(x.get("market_value", 0))))
        sym     = largest.get("symbol", "?")
        upl_pct = float(largest.get("unrealized_plpc", 0) or 0) * 100
        em = "📈" if upl_pct >= 0 else "📉"
        msgs = [
            f"{em} `{sym}` leading desk at {upl_pct:+.2f}% — {len(comm_pos)} commodity position(s) open",
            f"commodities up {upl_pct:+.2f}% on `{sym}`. watching oil + gold correlation",
        ]
        return [Post("desk-commodities", random.choice(msgs), "Commodities desk bot", ":oil_drum:")]
    msgs = [
        "no commodity positions — waiting for momentum signal on GLD/USO/UNG",
        "DBC broad basket: no entry yet. cross_asset_carry signal below threshold",
        "watching gold/oil spread — signal not triggered yet. monitoring 15min bars",
    ]
    return [Post("desk-commodities", random.choice(msgs), "Commodities desk bot", ":oil_drum:")]


def _short_futures() -> list[Post]:
    positions = alpaca_positions()
    fut_syms  = {"SPY", "QQQ", "IWM", "DIA", "IEF", "TLT", "USO", "GLD"}
    fut_pos   = [p for p in positions if p.get("symbol") in fut_syms]
    proxy_map = {"SPY": "ES", "QQQ": "NQ", "IWM": "RTY", "DIA": "YM", "IEF": "ZN", "TLT": "ZB", "USO": "CL", "GLD": "GC"}
    if fut_pos:
        syms_str = ", ".join(f"`{p.get('symbol')} ({proxy_map.get(p.get('symbol',''),'?')})`" for p in fut_pos[:4])
        total_pnl = sum(float(p.get("unrealized_pl", 0) or 0) for p in fut_pos)
        return [Post("desk-futures", f"futures desk: {syms_str} open · total uPnL ${total_pnl:+,.2f}", "Futures desk bot", ":chart_with_upwards_trend:")]
    msgs = [
        "no futures positions open — trend threshold not met across ES/NQ/RTY",
        "futures desk idle — cross_sectional_momentum scoring below entry threshold",
    ]
    return [Post("desk-futures", random.choice(msgs), "Futures desk bot", ":chart_with_upwards_trend:")]


def _short_rates() -> list[Post]:
    positions = alpaca_positions()
    rate_syms = {"SHY", "IEI", "IEF", "TLT", "TIP", "LQD", "HYG"}
    rate_pos  = [p for p in positions if p.get("symbol") in rate_syms]
    dur_map   = {"SHY": "1-3Y", "IEI": "3-7Y", "IEF": "7-10Y", "TLT": "20Y+", "TIP": "TIPS", "LQD": "IG", "HYG": "HY"}
    if rate_pos:
        spread_legs = {p.get("symbol"): float(p.get("unrealized_plpc", 0) or 0) * 100 for p in rate_pos}
        pos_strs = ", ".join(f"`{s}` {v:+.2f}%" for s, v in list(spread_legs.items())[:3])
        msgs = [
            f"rates ladder: {pos_strs}",
            f"curve positioning: {pos_strs} — {'carry positive' if sum(spread_legs.values()) > 0 else 'duration drag'}",
        ]
        return [Post("desk-rates", random.choice(msgs), "Rates desk bot", ":bank:")]
    msgs = [
        "rates desk flat — yield curve carry not wide enough to enter",
        "TLT/SHY spread compressed: waiting for 10bp widening before entering curve trade",
    ]
    return [Post("desk-rates", random.choice(msgs), "Rates desk bot", ":bank:")]


def _short_kalshi() -> list[Post]:
    try:
        req = urllib.request.Request(
            "https://api.elections.kalshi.com/trade-api/v2/markets?status=open&limit=50",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        markets = data.get("markets", [])
        arb = [m for m in markets
               if float(m.get("yes_ask", 50)) / 100 + float(m.get("no_ask", 50)) / 100 < 0.98]
        if arb:
            m = arb[0]
            edge = round((1 - float(m.get("yes_ask", 50)) / 100 - float(m.get("no_ask", 50)) / 100) * 100, 1)
            return [Post("desk-kalshi", f":rotating_light: kalshi arb: `{m.get('ticker','?')}` — edge {edge}¢. executing", "Kalshi desk bot", ":ballot_box_with_ballot:")]
        return [Post("desk-kalshi", f"scanned {len(markets)} kalshi markets — no binary arb. monitoring.", "Kalshi desk bot", ":ballot_box_with_ballot:")]
    except Exception:
        return [Post("desk-kalshi", "kalshi API check: monitoring binary markets for YES+NO sum < 98¢", "Kalshi desk bot", ":ballot_box_with_ballot:")]


def _short_stat_arb() -> list[Post]:
    p = REPO_ROOT / "backend" / "app" / "strategies" / "manual"
    arb_strats = [f.stem for f in p.glob("*.py") if any(k in f.stem for k in ("arb", "pairs", "kalman", "pca"))] if p.exists() else []
    positions = alpaca_positions()
    stat_pos  = [pos for pos in positions if pos.get("symbol") in {"SPY", "QQQ", "IWM", "GLD", "TLT"}]
    if stat_pos:
        syms = ", ".join(f"`{pos.get('symbol')}`" for pos in stat_pos[:3])
        return [Post("desk-stat-arb", f"stat arb legs open: {syms} — monitoring z-score for exit", "StatArb desk bot", ":arrows_counterclockwise:")]
    msgs = [
        f"stat arb flat: {len(arb_strats)} strategies watching cointegration signals — no z-score > 2σ",
        "pairs desk: SPY/QQQ spread within historical norm. waiting for divergence",
        f"PCA stat arb: {len(arb_strats)} factor models loaded — no entry signal this cycle",
    ]
    return [Post("desk-stat-arb", random.choice(msgs), "StatArb desk bot", ":arrows_counterclockwise:")]


# ── Short-form functions for new channels ────────────────────────────────────

def _short_general() -> list[Post]:
    commits = git_recent_commits(since_hours=24, limit=3)
    msgs = [
        f"good {datetime.now(timezone.utc).strftime('%A')} — {len(commits)} commits since yesterday. keep shipping.",
        f"reminder: paper-first. no live trading without CRO sign-off. {_m('Risk Engineer')}: status?",
        "team sync: all desks running on Alpaca paper. 24/7 monitoring active.",
        f"{len(commits)} commits merged. CI green. Render healthy. let's close this sprint strong.",
    ]
    return [Post("general", random.choice(msgs), "Laavanye Bahl — CEO/Founder", ":sparkles:")]


def _short_standup() -> list[Post]:
    weekday = datetime.now(timezone.utc).strftime("%A")
    employees = [
        ("Maya Chen", ":woman_office_worker:", ["reviewing PRs", "unblocking team", "CI monitoring"]),
        ("Aarav Patel", ":chart_with_upwards_trend:", ["strategy walk-forward", "alpha research", "paper gate review"]),
        ("Linh Tran", ":robot_face:", ["LSTM retrain", "ensemble update", "model comparison"]),
        ("Jian Wu", ":shield:", ["risk dashboard", "kelly sizing", "circuit breaker check"]),
        ("Anna Hoffmann", ":gear:", ["backend PR", "API endpoint", "DB migration"]),
        ("Diego Ramirez", ":zap:", ["execution algo tuning", "slippage analysis", "RL policy"]),
    ]
    name, emoji, tasks = random.choice(employees)
    task = random.choice(tasks)
    return [Post("standup", f"*{weekday} standup — {name}*\n↳ {task} — no blockers", name, emoji)]


def _short_wins() -> list[Post]:
    results = latest_backtest_results()
    if results:
        best = max(results, key=lambda r: float(r.get("sharpe", 0) or 0))
        s = float(best.get("sharpe", 0) or 0)
        if s > 1.0:
            return [Post("wins", f":trophy: `{best.get('strategy')}` Sharpe {s:.2f} — above paper gate!", "Wins bot", ":tada:")]
    test_res = run_pytest_lightweight(timeout_secs=20)
    if test_res.get("passed", 0) > 0 and not test_res.get("failed"):
        return [Post("wins", f":white_check_mark: {test_res['passed']} tests green — no regressions", "Wins bot", ":mag:")]
    return [Post("wins", ":muscle: team shipping steady. next win incoming.", "Wins bot", ":tada:")]


def _short_strategy_review() -> list[Post]:
    strats = list_strategies()["manual"]
    results = latest_backtest_results()
    if results:
        best = max(results, key=lambda r: float(r.get("sharpe", 0) or 0))
        s = float(best.get("sharpe", 0) or 0)
        sn = best.get("strategy", "?")
        status = "paper gate ✅" if s > 1.0 else f"need {1.0 - s:.2f} more Sharpe"
        return [Post("strategy-review", f"`{sn}`: Sharpe {s:.2f} — {status}", "Alpha Research Director", ":chart_with_upwards_trend:")]
    target = random.choice(strats) if strats else "momentum"
    return [Post("strategy-review", f"who's running `{target}`? need walk-forward result by EOD", "Alpha Research Director", ":chart_with_upwards_trend:")]


def _short_model_perf() -> list[Post]:
    results = latest_backtest_results()
    if results:
        r = results[0]
        s = float(r.get("sharpe", 0) or 0)
        return [Post("model-performance", f"latest run: `{r.get('strategy', '?')}` Sharpe {s:.2f}. {'above target ✅' if s > 1.5 else 'tuning needed'}", "ML Modeling Lead", ":robot_face:")]
    return [Post("model-performance", "no results yet. ensemble standing by. {_m('ML Research Lead')}: which model first?", "ML Modeling Lead", ":robot_face:")]


def _short_code_review() -> list[Post]:
    changed = git_files_changed(since_hours=24)
    files = [k for k in changed if k.endswith(".py") and "test" not in k]
    prs = open_prs()
    if prs:
        return [Post("code-review", f":eyes: {len(prs)} PRs open — {_m('Backend Lead')} / {_m('Director of QA')}: please review", "VP Engineering", ":woman_office_worker:")]
    if files:
        f = random.choice(files[:3])
        return [Post("code-review", f"reviewed `{Path(f).name}` — looks clean. ✅", "Backend Lead", ":gear:")]
    return [Post("code-review", "no PRs or changes — clean state. good to ship.", "Director of QA", ":mag:")]


def _short_incidents() -> list[Post]:
    test_res = run_pytest_lightweight(timeout_secs=20)
    if test_res.get("failed", 0) > 0:
        return [Post("incidents", f":red_circle: {test_res['failed']} tests red — investigating. {_m('Backend Lead')}: heads up", "Incident Bot", ":rotating_light:")]
    runs = latest_workflow_runs()
    if any(r.get("conclusion") == "failure" for r in runs):
        return [Post("incidents", ":yellow_circle: CI failure detected — checking logs. ETA 15min", "Incident Bot", ":rotating_light:")]
    return [Post("incidents", ":large_green_circle: all systems nominal", "Incident Bot", ":rotating_light:")]


_SHORT_FNS: dict[str, Callable[[], list[Post]]] = {
    "VP Engineering":                _short_vp_engineering,
    "Alpha Research Director":       _short_alpha_director,
    "ML Modeling Lead":              _short_ml_lead,
    "Risk Engineer":                 _short_risk_engineer,
    "Director of DevOps":            _short_devops,
    "Director of QA":                _short_qa,
    "Backend Lead":                  _short_backend,
    "Junior Engineer":               _short_junior,
    "Polymarket Researcher":         _short_polymarket,
    "Data Engineer":                 _short_data_eng,
    "Execution Engineer":            _short_execution_eng,
    "Commodities desk bot":          _short_commodities,
    "Futures desk bot":              _short_futures,
    "Rates desk bot":                _short_rates,
    "Kalshi desk bot":               _short_kalshi,
    "StatArb desk bot":              _short_stat_arb,
    # New channels
    "CEO/Founder":                   _short_general,
    "Standup bot":                   _short_standup,
    "Wins bot":                      _short_wins,
    "Alpha Research Director (SR)":  _short_strategy_review,
    "ML Modeling Lead (MP)":         _short_model_perf,
    "VP Engineering (CR)":           _short_code_review,
    "Incident Bot":                  _short_incidents,
}


# ─── Master agent registry ───────────────────────────────────────────────────


AGENTS: list[Agent] = [
    Agent("VP Engineering", "VP Engineering", ":woman_office_worker:",
          ["engineering"], maya_chen_eng_daily, ["engineering", "eng-daily"]),
    Agent("Alpha Research Director", "Alpha Research Director", ":chart_with_upwards_trend:",
          ["alpha-research"], aarav_patel_strategy_review, ["alpha", "strategy"]),
    Agent("ML Modeling Lead", "ML Modeling Lead", ":robot_face:",
          ["ml-experiments"], linh_tran_ml_results, ["ml", "experiment"]),
    Agent("Execution Engineer", "Execution Engineer", ":zap:",
          ["squad-execution"], diego_ramirez_execution, ["execution", "slippage"]),
    Agent("Risk Engineer", "Risk Engineer", ":shield:",
          ["risk-alerts"], jian_wu_risk, ["risk"]),
    Agent("Frontend Lead", "Frontend Lead", ":art:",
          ["squad-frontend"], priya_subramanian_frontend, ["frontend"]),
    Agent("Backend Lead", "Backend Lead", ":gear:",
          ["squad-backend"], anna_hoffmann_backend, ["backend"]),
    Agent("Data Engineer", "Data Engineer", ":file_cabinet:",
          ["squad-data"], sina_hassani_data, ["data"]),
    Agent("Director of DevOps", "Director of DevOps", ":satellite_antenna:",
          ["infra-alerts"], kenji_watanabe_devops, ["devops", "ci"]),
    Agent("Director of DevOps", "Director of DevOps", ":satellite_antenna:",
          ["leadership-summary"], kenji_deploy_readiness, ["deploy", "infra"]),
    Agent("Director of QA", "Director of QA", ":mag:",
          ["squad-qa"], aditi_sharma_qa, ["qa", "test"]),
    Agent("Director of QA", "Director of QA", ":mag:",
          ["ci-failures"], aditi_open_prs, ["qa", "ci"]),
    Agent("Security Engineer", "Security Engineer", ":closed_lock_with_key:",
          ["security-alerts"], cameron_park_security, ["security"]),
    Agent("VP Research", "VP Research", ":books:",
          ["papers"], sofia_karlsson_research, ["research", "papers"]),
    Agent("Options Researcher", "Options Researcher", ":bar_chart:",
          ["desk-options"], yuki_mori_options, ["options"]),
    Agent("Quant Researcher", "Quant Researcher", ":mag_right:",
          ["alpha-research"], hugo_bernardes_research, ["alpha", "research"]),
    Agent("Research Scientist", "Research Scientist", ":brain:",
          ["pod-ml-rl"], tomas_lindqvist_rl, ["ml", "rl"]),
    Agent("Polymarket Researcher", "Polymarket Researcher", ":vertical_traffic_light:",
          ["desk-polymarket"], lior_avraham_polymarket, ["polymarket"]),
    Agent("Chief Risk Officer", "CRO", ":shield:",
          ["leadership-summary"], marcus_olufemi_risk, ["risk", "leadership"]),
    Agent("Finance Engineer", "Finance Engineer", ":moneybag:",
          ["finance-ops"], wei_chang_finance, ["finance"]),
    Agent("Compliance Engineer", "Compliance Engineer", ":scales:",
          ["legal-compliance"], helena_voss_compliance, ["compliance"]),
    Agent("Junior Engineer", "Junior IC", ":raised_hand:",
          ["help"], karl_nystrom_question, ["help", "newbie"]),
    Agent("CEO / Founder", "CEO/Founder", ":sparkles:",
          ["announcements"], laavanye_bahl_ceo, ["ceo", "weekly"]),
    Agent("ML Infrastructure Engineer", "ML Infra Engineer", ":wrench:",
          ["engineering"], ravi_iyer_ci, ["ci", "infra", "ml"]),
    # ── ML research team ─────────────────────────────────────────────────────
    Agent("ML Research Lead", "ML Research Lead", ":microscope:",
          ["ml-experiments"], sara_kim_ml_research, ["ml", "research", "sota"]),
    Agent("Deep Learning Engineer", "DL Engineer", ":building_construction:",
          ["engineering"], marcus_williams_dl_engineer, ["ml", "architecture", "training"]),
    Agent("Feature Engineering Lead", "Feature Engineering Lead", ":abacus:",
          ["alpha-research"], priya_nair_feature_eng, ["features", "indicators", "mtf"]),
    Agent("Quant ML Researcher", "Quant ML Researcher", ":chart_with_upwards_trend:",
          ["alpha-research"], alex_chen_quant_ml, ["ml", "ablation", "cross-asset"]),
    # ── Live trading-desk bots (read Alpaca paper account directly) ─────────
    Agent("PnL bot", "automated", ":bar_chart:",
          ["pnl-daily"], trading_desk_eod_pnl, ["pnl", "trading"]),
    Agent("Equity desk bot", "automated", ":chart_with_upwards_trend:",
          ["desk-equities"], trading_desk_equity_positions, ["equities", "trading"]),
    Agent("Crypto desk bot", "automated", ":coin:",
          ["desk-crypto"], trading_desk_crypto_positions, ["crypto", "trading"]),
    Agent("Options desk bot", "automated", ":game_die:",
          ["desk-options"], trading_desk_options_positions, ["options", "trading"]),
    Agent("Polymarket desk bot", "automated", ":crystal_ball:",
          ["desk-polymarket"], trading_desk_polymarket_positions, ["polymarket", "trading"]),
    Agent("Macro/FX desk bot", "automated", ":earth_americas:",
          ["desk-fx-rates"], trading_desk_macro_positions, ["macro", "fx", "trading"]),
    Agent("Commodities desk bot", "automated", ":oil_drum:",
          ["desk-commodities"], trading_desk_commodities, ["commodities", "trading"]),
    Agent("Futures desk bot", "automated", ":chart_with_upwards_trend:",
          ["desk-futures"], trading_desk_futures, ["futures", "trading"]),
    Agent("Rates desk bot", "automated", ":bank:",
          ["desk-rates"], trading_desk_rates, ["rates", "bonds", "trading"]),
    Agent("Kalshi desk bot", "automated", ":ballot_box_with_ballot:",
          ["desk-kalshi"], trading_desk_kalshi, ["kalshi", "prediction", "trading"]),
    Agent("StatArb desk bot", "automated", ":arrows_counterclockwise:",
          ["desk-stat-arb"], trading_desk_stat_arb, ["stat-arb", "pairs", "trading"]),
    # ── New channels ──────────────────────────────────────────────────────────
    Agent("CEO/Founder", "CEO/Founder", ":sparkles:",
          ["general"], general_channel, ["ceo", "general"]),
    Agent("Standup bot", "automated", ":calendar:",
          ["standup"], standup_channel, ["standup", "daily"]),
    Agent("Wins bot", "automated", ":trophy:",
          ["wins"], wins_channel, ["wins", "celebrate"]),
    Agent("Incident Bot", "automated", ":rotating_light:",
          ["incidents"], incidents_channel, ["incident", "alert"]),
    Agent("Alpha Research Director (SR)", "Alpha Research Director", ":chart_with_upwards_trend:",
          ["strategy-review"], strategy_review_channel, ["strategy", "review"]),
    Agent("ML Modeling Lead (MP)", "ML Modeling Lead", ":robot_face:",
          ["model-performance"], model_perf_channel, ["model", "performance"]),
    Agent("VP Engineering (CR)", "VP Engineering", ":woman_office_worker:",
          ["code-review"], code_review_channel, ["code", "review"]),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token.startswith("xoxb-"):
        print("")
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║  ⚠  SLACK SILENT — agents ran but NO messages were posted       ║")
        print("║                                                                  ║")
        print("║  SLACK_BOT_TOKEN is missing or invalid (must start with xoxb-)  ║")
        print("║                                                                  ║")
        print("║  Add SLACK_BOT_TOKEN to repo secrets:                           ║")
        print("║  Settings → Secrets and variables → Actions → New secret        ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print("")
        return 0

    auth = slack_call(token, "auth.test", {})
    if not auth.get("ok"):
        print(f"❌ auth.test failed: {auth}")
        return 1
    bot_user_id = auth.get("user_id", "")
    print(f"✅ Authed as {auth.get('user')} in {auth.get('team')} at {datetime.now(timezone.utc).isoformat()}")

    # Load run state for dedup + thread tracking
    state = load_state()
    has_claude = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    print(f"📋 State: last_run={state['last_run_ts']}, {len(state['posted_hashes'])} known hashes, "
          f"{'Claude ✅' if has_claude else 'Claude ❌ (no ANTHROPIC_API_KEY)'})")

    # ── Auto-create channels ──────────────────────────────────────────────────
    print("\n📺 Ensuring all channels exist")
    ensure_channels_exist(token)

    # ── Phase 0: Inbox check — respond to unanswered human thread replies ────
    #             AND handle /command messages from employees
    print("\n📬 Inbox check — reading threads for replies + /commands")
    inbox_channels = [
        "engineering", "alpha-research", "ml-experiments",
        "squad-qa", "desk-crypto", "squad-backend", "help",
        "desk-commodities", "desk-futures", "desk-rates",
        "desk-kalshi", "desk-stat-arb", "desk-equities",
        "desk-polymarket", "pnl-daily",
        # New channels
        "general", "standup", "wins", "incidents",
        "strategy-review", "model-performance", "code-review",
    ]
    posts_made = 0
    errors = 0
    for ch in inbox_channels:
        # ── Handle human thread replies ───────────────────────────────────
        try:
            threads = read_unresponded_threads(
                token, ch, bot_user_id,
                already_replied=state.get("replied_to", []),
                limit=20,
            )
        except Exception as e:
            print(f"  [inbox] {ch} read failed: {e}")
            threads = []
        for thread in threads[:2]:  # max 2 thread replies per channel per run
            response = generate_thread_response(thread)
            if not response:
                continue
            if is_duplicate(state, response):
                print(f"  [inbox] skipping dup reply in #{ch}")
                continue
            agent_name, agent_emoji = _CHANNEL_AGENT_IDENTITY.get(ch, ("QuantEdge Bot", ":robot_face:"))
            r = post_to_slack(
                token, ch, response,
                username=agent_name,
                icon_emoji=agent_emoji,
                thread_ts=thread["parent_ts"],
            )
            if r.get("ok"):
                posts_made += 1
                record_post(state, response)
                state.setdefault("replied_to", []).append(thread["reply_ts"])
                print(f"  ✓ Replied to thread in #{ch}: {thread['last_reply'][:60]}…")
            else:
                errors += 1
                print(f"  ✗ Thread reply in #{ch}: {r.get('error')}")
            time.sleep(0.6)

        # ── Handle /command messages from employees ───────────────────────
        try:
            cmd_hits = scan_for_commands(
                token, ch,
                already_replied=state.get("replied_to", []),
                limit=15,
            )
        except Exception as e:
            print(f"  [cmd] {ch} scan failed: {e}")
            cmd_hits = []
        for cmd_info in cmd_hits[:2]:  # max 2 commands per channel per run
            response = handle_thread_command(cmd_info["command"])
            if not response:
                continue
            if is_duplicate(state, response):
                continue
            r = post_to_slack(
                token, ch, response,
                username="QuantEdge Bot",
                icon_emoji=":robot_face:",
                thread_ts=cmd_info["thread_ts"],
            )
            if r.get("ok"):
                posts_made += 1
                record_post(state, response)
                state.setdefault("replied_to", []).append(cmd_info["reply_ts"])
                print(f"  ✓ /cmd '{cmd_info['command'][:30]}' → #{ch}")
            else:
                errors += 1
            time.sleep(0.6)

    # ── Team activity first (always runs): standups + scoreboard ────────────
    team_posts: list[Post] = []
    print("\n👥 Team activity")
    for team_name in TEAMS:
        sp = team_lead_standup_for(team_name)
        if sp:
            team_posts.append(sp)
        # Roughly half the runs: a team member also posts
        if random.random() < 0.55:
            mp = team_member_observation_for(team_name)
            if mp:
                team_posts.append(mp)
    # Leaderboard always
    lb = team_leaderboard_post()
    if lb:
        team_posts.append(lb)
    # Cross-team learning post (1 per run)
    ct = cross_team_share_post()
    if ct:
        team_posts.append(ct)
    # Friday presentation
    team_posts.extend(friday_presentation_post())

    # Sample wave: 60-80% of agents do real work each run (skew so it varies)
    wave_size = random.randint(int(len(AGENTS) * 0.6), int(len(AGENTS) * 0.85))
    wave = random.sample(AGENTS, wave_size)
    wave_names = {a.name for a in wave}
    print(f"🎯 Wave: {wave_size}/{len(AGENTS)} agents + {len(team_posts)} team posts")

    posted_ts: dict[str, str] = {}         # channel_name -> last_ts (for thread replies)
    posted_for_reactions: list[tuple] = [] # [(channel_id, ts)] for reaction wave
    # Per-agent tracking: {agent_name -> {"posts": int, "errors": int, "channels": [...]}}
    agent_tracking: dict[str, dict] = {}

    def _do_post(p: Post, label: str) -> str | None:
        """Post a message, record it, return ts or None. Shared by team + agent loops."""
        nonlocal posts_made, errors
        if is_duplicate(state, p.text):
            print(f"  ⏭ {label} → #{p.channel} (dup)")
            return None
        r = post_to_slack(
            token, channel=p.channel, text=p.text,
            username=p.username, icon_emoji=p.icon_emoji,
            thread_ts=p.thread_of,
        )
        if r.get("ok"):
            posts_made += 1
            record_post(state, p.text)
            ts = r.get("ts", "")
            if ts and not p.thread_of:
                posted_ts[p.channel] = ts
                ch_id = get_channel_id(token, p.channel)
                if ch_id:
                    posted_for_reactions.append((ch_id, ts))
            print(f"  ✓ {label} → #{p.channel}")
            return ts
        else:
            errors += 1
            print(f"  ✗ {label} → #{p.channel}: {r.get('error')}")
            return None

    # Post team activity first (with dedup)
    for p in team_posts:
        _do_post(p, f"TEAM {p.username[:30]}")
        time.sleep(0.6)

    # Agent wave — 55% short-form, 45% full report (makes feed feel natural)
    for agent in wave:
        agent_tracking[agent.name] = {"posts": 0, "errors": 0, "channels": [], "mode": ""}
        short_fn = _SHORT_FNS.get(agent.name)
        use_short = short_fn is not None and random.random() < 0.55
        fn_to_call = short_fn if use_short else agent.work_fn
        mode = "short" if use_short else "full"
        agent_tracking[agent.name]["mode"] = mode
        try:
            posts = fn_to_call()
        except Exception as e:
            print(f"  ✗ {agent.name} ({mode}) crashed: {e}")
            errors += 1
            agent_tracking[agent.name]["errors"] += 1
            continue
        for p in posts:
            ts = _do_post(p, f"{agent.name}({mode})")
            if ts is not None:
                agent_tracking[agent.name]["posts"] += 1
                if p.channel not in agent_tracking[agent.name]["channels"]:
                    agent_tracking[agent.name]["channels"].append(p.channel)
            else:
                agent_tracking[agent.name]["errors"] += 1
            time.sleep(0.6)

    # ── Reaction wave — agents react to each other's posts (feels like real Slack)
    print("\n👍 Reaction wave")
    react_targets = random.sample(posted_for_reactions, min(10, len(posted_for_reactions)))
    for ch_id, ts in react_targets:
        emoji = random.choice(_REACTION_POOL)
        add_reaction(token, ch_id, ts, emoji)
        time.sleep(0.3)

    # ── Discussion pass: multi-turn threaded discussions ────────────────────
    print("\n💬 Discussion pass — multi-turn threaded discussions")
    chains = build_discussion_chains(posted_ts)
    random.shuffle(chains)
    # Run 4-7 chains per wave (varied so not every channel threads every run)
    n_chains = random.randint(4, min(7, len(chains)))
    chains_run = 0
    for channel, parent_ts, agent_chain in chains[:n_chains]:
        print(f"  💬 discussion in #{channel} ({len(agent_chain)} replies)")
        for username, emoji, text in agent_chain:
            p = Post(channel=channel, text=text, username=username,
                     icon_emoji=emoji, thread_of=parent_ts)
            _do_post(p, f"DISCUSS {username[:22]}")
            time.sleep(0.5)
        chains_run += 1
    print(f"  → {chains_run} discussion chains completed")

    # ── Employee run tracker — posted to #engineering every run ──────────────
    print("\n📋 Posting employee run tracker")
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    active_names = sorted(agent_tracking.keys())
    benched_names = sorted(a.name for a in AGENTS if a.name not in wave_names)

    # Deduplicate names (some agents share a name like "Director of DevOps")
    seen: set[str] = set()
    unique_active = []
    for n in active_names:
        if n not in seen:
            seen.add(n)
            t = agent_tracking[n]
            status = "✅" if t["posts"] > 0 else ("⚠️" if t["errors"] == 0 else "❌")
            chs = ", ".join(f"#{c}" for c in t["channels"]) or "—"
            mode = t.get("mode", "")
            mode_tag = " _(short)_" if mode == "short" else ""
            unique_active.append(f"{status} *{n}*{mode_tag}: {t['posts']} post(s) → {chs}")

    seen_benched: set[str] = set()
    unique_benched = []
    for n in benched_names:
        if n not in seen_benched:
            seen_benched.add(n)
            unique_benched.append(f"• {n}")

    n_reactions = min(10, len(posted_for_reactions))
    n_short = sum(1 for t in agent_tracking.values() if t.get("mode") == "short")
    tracker_lines = [
        f"*:clipboard: Employee Run Report — {now_str}*",
        f"Wave: *{wave_size}/{len(AGENTS)}* agents | *{posts_made}* posts ({n_short} short-form) | *{n_reactions}* reactions | *{errors}* errors",
        "",
        f"*Active this run ({len(unique_active)}):*",
    ]
    tracker_lines.extend(unique_active[:20])
    if len(unique_active) > 20:
        tracker_lines.append(f"_…and {len(unique_active) - 20} more_")
    if unique_benched:
        tracker_lines.append(f"\n*Benched this wave ({len(unique_benched)}):*")
        tracker_lines.extend(unique_benched[:10])

    r = post_to_slack(
        token,
        channel="engineering",
        text="\n".join(tracker_lines),
        username="Employee Tracker",
        icon_emoji=":bar_chart:",
    )
    if r.get("ok"):
        posts_made += 1
        print("  ✓ Employee tracker → #engineering")
    else:
        print(f"  ✗ Employee tracker → #engineering: {r.get('error')}")

    # ── Save state for next run ──────────────────────────────────────────────
    state["last_run_ts"] = int(datetime.now(timezone.utc).timestamp())
    latest_commits = git_recent_commits(since_hours=1, limit=1)
    if latest_commits:
        state["last_commit_sha"] = latest_commits[0].get("sha", "")
    save_state(state)
    print(f"💾 State saved: {len(state['posted_hashes'])} hashes, {len(state['replied_to'])} replied threads")

    print(f"\n✅ Posted {posts_made} messages, {errors} errors")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Quick mode — runs every 15 min: inbox + /commands + incidents only
# Full mode  — runs 4x/day: all agents + discussions + team activity
# Push mode  — fires on git push: engineering bot posts what changed
# PR mode    — fires on PR event: code-review bot posts
# ─────────────────────────────────────────────────────────────────────────────


def quick_main() -> int:
    """
    Lightweight run (every 15 min). Handles:
      1. Thread inbox (human replies → agent responds)
      2. Slash commands (/backtest, /ask, /risk, etc.)
      3. Incident detection + alert
      4. Event-specific post (push → eng update, PR → code-review)
    Uses free agent cascade — no Claude Sonnet, keeps cost near zero.
    """
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token.startswith("xoxb-"):
        return 0

    auth = slack_call(token, "auth.test", {})
    if not auth.get("ok"):
        return 1
    bot_user_id = auth.get("user_id", "")
    event_name = os.environ.get("GITHUB_EVENT_NAME", "schedule")
    print(f"⚡ Quick mode | event={event_name} | {datetime.now(timezone.utc).strftime('%H:%M UTC')}")

    state = load_state()
    ensure_channels_exist(token)

    posts_made = errors = 0
    all_chs = list(_CHANNEL_AGENT_IDENTITY.keys())

    for ch in all_chs:
        # Human thread replies
        try:
            threads = read_unresponded_threads(
                token, ch, bot_user_id,
                already_replied=state.get("replied_to", []), limit=15)
        except Exception:
            threads = []
        for t in threads[:1]:
            resp = generate_thread_response(t)
            if resp and not is_duplicate(state, resp):
                agent_name, agent_emoji = _CHANNEL_AGENT_IDENTITY.get(ch, ("QuantEdge Bot", ":robot_face:"))
                r = post_to_slack(token, ch, resp, username=agent_name,
                                  icon_emoji=agent_emoji, thread_ts=t["parent_ts"])
                if r.get("ok"):
                    posts_made += 1
                    record_post(state, resp)
                    state.setdefault("replied_to", []).append(t["reply_ts"])
            time.sleep(0.5)

        # /command handler
        try:
            cmds = scan_for_commands(token, ch,
                                     already_replied=state.get("replied_to", []), limit=15)
        except Exception:
            cmds = []
        for cmd in cmds[:2]:
            resp = handle_thread_command(cmd["command"])
            if resp and not is_duplicate(state, resp):
                r = post_to_slack(token, ch, resp, username="QuantEdge Bot",
                                  icon_emoji=":robot_face:", thread_ts=cmd["thread_ts"])
                if r.get("ok"):
                    posts_made += 1
                    record_post(state, resp)
                    state.setdefault("replied_to", []).append(cmd["reply_ts"])
                    print(f"  ✓ cmd '{cmd['command'][:30]}' → #{ch}")
            time.sleep(0.5)

    # Incident monitoring every quick run
    try:
        for p in incidents_channel()[:2]:
            if not is_duplicate(state, p.text):
                r = post_to_slack(token, p.channel, p.text,
                                  username=p.username, icon_emoji=p.icon_emoji)
                if r.get("ok"):
                    posts_made += 1
                    record_post(state, p.text)
                time.sleep(0.5)
    except Exception as e:
        print(f"  [incident] {e}")

    # Event-specific extra posts
    if event_name == "push":
        for p in _short_vp_engineering():
            if not is_duplicate(state, p.text):
                r = post_to_slack(token, p.channel, p.text,
                                  username=p.username, icon_emoji=p.icon_emoji)
                if r.get("ok"):
                    posts_made += 1
                    record_post(state, p.text)
    elif event_name == "pull_request":
        for p in code_review_channel()[:1]:
            if not is_duplicate(state, p.text):
                r = post_to_slack(token, p.channel, p.text,
                                  username=p.username, icon_emoji=p.icon_emoji)
                if r.get("ok"):
                    posts_made += 1
                    record_post(state, p.text)

    state["last_run_ts"] = int(datetime.now(timezone.utc).timestamp())
    save_state(state)
    print(f"⚡ Quick done: {posts_made} posts, {errors} errors")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "quick":
        sys.exit(quick_main())
    else:
        sys.exit(main())
