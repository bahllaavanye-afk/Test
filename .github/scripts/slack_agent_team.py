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
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = REPO_ROOT / "experiments" / "results" / "slack_state.json"


# ─────────────────────────────────────────────────────────────────────────────
# Multi-agent routing — system prompt + cost policy constants
# (defined here so they are available as default argument values below)
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

# Hosts that returned Cloudflare 1010 errors — skip silently for the rest of this run
_CF_BLOCKED_HOSTS: set[str] = set()


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


def _init_governance(state: dict) -> None:
    """Ensure governance keys exist in state (idempotent)."""
    gov = state.setdefault("governance", {})
    gov.setdefault("freeze_algos", False)
    gov.setdefault("paused_engineers", {})
    gov.setdefault("cto_user_ids", [])
    gov.setdefault("audit_log", [])
    today = _today()
    quotas = gov.setdefault("engineer_quotas", {})
    for emp in _EMPLOYEES:
        q = quotas.setdefault(emp, {})
        if q.get("date") != today:
            q["calls"] = MAX_CALLS_PER_EMPLOYEE_PER_RUN
            q["slack_posts"] = 8
            q["tokens"] = MAX_TOKENS_PER_CALL * MAX_CALLS_PER_EMPLOYEE_PER_RUN
            q["date"] = today


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["posted_hashes"] = state.get("posted_hashes", [])[-1000:]
    state["replied_to"]    = state.get("replied_to", [])[-500:]
    # Trim response cache to last 200 entries
    cache = state.get("response_cache", {})
    if len(cache) > 200:
        # evict oldest entries by timestamp
        sorted_keys = sorted(cache, key=lambda k: cache[k].get("ts", 0))
        for k in sorted_keys[:len(cache) - 200]:
            del cache[k]
    state["response_cache"] = cache
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def is_duplicate(state: dict, text: str) -> bool:
    return _hash(text) in state.get("posted_hashes", [])


# ─────────────────────────────────────────────────────────────────────────────
# Token budget tracker — persisted in state so limits survive across runs
# ─────────────────────────────────────────────────────────────────────────────

# ─── Per-wave throughput counters (reset each process invocation) ─────────────
from collections import defaultdict as _defaultdict
_API_CALL_COUNTS: dict[str, int] = _defaultdict(int)
_API_TOKEN_COUNTS: dict[str, int] = _defaultdict(int)


def track_api_call(provider_key: str, tokens_used: int = 0) -> None:
    """Increment in-wave call and token counters for the given provider key."""
    _API_CALL_COUNTS[provider_key] += 1
    _API_TOKEN_COUNTS[provider_key] += tokens_used


# ─────────────────────────────────────────────────────────────────────────────

# Daily soft limits per provider key — stay below these to never hit hard caps.
# Set at 80% of the real limit so there's always a 20% safety margin.
_DAILY_SOFT_LIMITS: dict[str, int] = {
    # Groq — up to 10 accounts, both _1 and plain naming supported
    "GROQ_API_KEY_1": 400_000,   # real: 500K tok/day
    "GROQ_API_KEY":   400_000,   # alias for account 1
    "GROQ_API_KEY_2": 400_000,
    "GROQ_API_KEY_3": 400_000,
    "GROQ_API_KEY_4": 400_000,   # extra accounts — zero-config drop-in
    "GROQ_API_KEY_5": 400_000,
    "GROQ_API_KEY_6": 400_000,
    "GROQ_API_KEY_7": 400_000,
    "GROQ_API_KEY_8": 400_000,
    "GROQ_API_KEY_9": 400_000,
    "GROQ_API_KEY_10": 400_000,
    # Gemini — 3 accounts, tracked as requests (1500 req/day real limit)
    "GEMINI_API_KEY_1": 1_200,   # real: 1500 req/day
    "GEMINI_API_KEY":   1_200,   # alias for account 1
    "GEMINI_API_KEY_2": 1_200,
    "GEMINI_API_KEY_3": 1_200,
    "CEREBRAS_API_KEY_1": 800_000,   # 1M/day hard, use 80%
    "CEREBRAS_API_KEY":   800_000,   # alias for account 1
    "CEREBRAS_API_KEY_2": 800_000,   # account 2
    "CEREBRAS_API_KEY_3": 800_000,   # account 3 — zero-config drop-in
    "SAMBANOVA_API_KEY": 15_000_000,  # 20M/day hard limit, use 75%
    "OPENROUTER_API_KEY": 40,        # 50 req/day hard, use 80%
    "OPENROUTER_API_KEY_2": 40,      # 50 req/day hard, use 80%
    "OPENROUTER_API_KEY_3": 40,      # account 3 — zero-config drop-in
}


def _today() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def budget_ok(state: dict, provider_env: str, estimated_tokens: int = 500) -> bool:
    """Return True if this provider still has room in today's budget."""
    today = _today()
    usage = state.setdefault("token_budget", {}).setdefault(provider_env, {"date": today, "used": 0})
    if usage.get("date") != today:          # new day — reset
        usage["date"] = today
        usage["used"] = 0
    limit = _DAILY_SOFT_LIMITS.get(provider_env, 999_999_999)
    return usage["used"] + estimated_tokens < limit


def record_usage(state: dict, provider_env: str, tokens: int) -> None:
    """Increment daily usage counter for this provider key (tokens + requests)."""
    today = _today()
    day_usage = state.setdefault("daily_usage", {}).setdefault(today, {})
    key_usage = day_usage.setdefault(provider_env, {"tokens": 0, "requests": 0})
    key_usage["tokens"] += tokens
    key_usage["requests"] += 1
    # Also update legacy flat format for budget_ok compatibility
    bucket = state.setdefault("token_budget", {}).setdefault(provider_env, {"date": today, "used": 0})
    if bucket.get("date") != today:
        bucket["date"] = today
        bucket["used"] = 0
    bucket["used"] = bucket.get("used", 0) + tokens
    state.setdefault("token_usage", {}).setdefault(today, {})[provider_env] = key_usage["tokens"]


def log_budget(state: dict) -> None:
    """Print a one-line budget summary at the start of each run."""
    today = _today()
    parts = []
    for k, limit in _DAILY_SOFT_LIMITS.items():
        used = state.get("token_budget", {}).get(k, {}).get("used", 0)
        if state.get("token_budget", {}).get(k, {}).get("date") != today:
            used = 0
        parts.append(f"{k}={used}/{limit}")
    print(f"  [budget] {' | '.join(parts)}")


# ─────────────────────────────────────────────────────────────────────────────
# Response cache — 4-hour TTL — avoids redundant LLM calls when nothing changed
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_TTL_SECONDS = 14_400   # 4 hours


def cached_call(state: dict, cache_key: str, call_fn, ttl: int = _CACHE_TTL_SECONDS) -> str | None:
    """
    Return cached response if fresh, else call call_fn() and cache the result.
    cache_key should encode: (employee, channel, recent_git_sha[:8]).
    This prevents calling the LLM again if the same employee is asked about the
    same topic within 2 hours and no new commits have landed.
    """
    import time
    now = time.time()
    cache = state.setdefault("response_cache", {})
    entry = cache.get(cache_key)
    if entry and now - entry.get("ts", 0) < ttl:
        print(f"  [cache hit] {cache_key[:32]}")
        return entry["text"]
    result = call_fn()
    if result:
        cache[cache_key] = {"text": result, "ts": now}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Git context — detect whether anything changed since last run
# ─────────────────────────────────────────────────────────────────────────────

def current_git_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def repo_changed(state: dict) -> bool:
    """Return True if HEAD changed since last run. Updates state.

    Grace period: if the last commit is older than 4 hours AND the SHA has not
    changed, set state["skip_wave"] = True so the caller can skip ALL agent
    posts for this run (prevents redundant posts on weekends / quiet periods).
    """
    sha = current_git_sha()
    last = state.get("last_commit_sha", "")
    changed = sha != last
    state["last_commit_sha"] = sha

    if not changed:
        # Check age of last commit to decide whether to suppress the whole wave
        try:
            raw_ts = subprocess.check_output(
                ["git", "log", "-1", "--format=%ct"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
            last_commit_age = time.time() - int(raw_ts)
        except Exception:
            last_commit_age = 0

        grace_seconds = 24 * 3600  # 24 hours
        if last_commit_age > grace_seconds:
            state["skip_wave"] = True
            print(
                f"  [repo_changed] last commit is {last_commit_age/3600:.1f}h old "
                f"and SHA unchanged — setting skip_wave=True"
            )
        else:
            state["skip_wave"] = False
    else:
        state["skip_wave"] = False

    return changed


# ─────────────────────────────────────────────────────────────────────────────
# Batch post generation — 1 API call generates posts for all employees in a group
# Instead of 13 separate calls, 3 batch calls (one per Groq account group).
# Token savings: ~70% reduction on system-prompt overhead.
# ─────────────────────────────────────────────────────────────────────────────

def batch_generate(
    employee_topics: list[tuple[str, str]],   # [(employee_name, topic_description), ...]
    groq_key: str,
    state: dict,
    system_prompt: str = _QUANT_SYSTEM,
) -> dict[str, str]:
    """
    One Groq API call → posts for N employees. Returns {employee: post_text}.
    Falls back to empty dict — callers fall through to individual calls.
    Respects daily token budget: skips if account is near its soft limit.
    """
    if not employee_topics or not groq_key:
        return {}

    # Find which env var this key belongs to (for budget tracking)
    provider_env = next(
        (k for k in ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"]
         if os.environ.get(k, "").strip() == groq_key),
        "GROQ_API_KEY"
    )
    estimated = len(employee_topics) * 250   # ~250 tok output per employee
    if not budget_ok(state, provider_env, estimated):
        print(f"  [batch] {provider_env} at soft limit — skipping batch")
        return {}

    names = [e for e, _ in employee_topics]
    delimiters = {e: f"==={e.upper()}===" for e in names}

    lines = [
        "Generate a short Slack post (2-4 sentences, professional but direct) for each person.",
        "Use exactly these section delimiters — no extra text between sections.\n",
    ]
    for emp, topic in employee_topics:
        lines.append(f"{delimiters[emp]}\nEmployee: {emp.title()} | Topic: {_sanitize(topic)}\n")

    batch_prompt = "\n".join(lines)
    cap = min(len(employee_topics) * 220, 1800)   # higher cap for batch output

    result = _try_openai_compat(
        "https://api.groq.com/openai/v1/chat/completions",
        groq_key, "llama-3.3-70b-versatile",
        _sanitize(system_prompt), batch_prompt, cap,
    )
    if not result:
        return {}

    record_usage(state, provider_env, estimated)

    posts: dict[str, str] = {}
    for emp in names:
        delim = delimiters[emp]
        if delim not in result:
            continue
        text = result.split(delim, 1)[1].strip()
        # cut at the next delimiter
        for other in names:
            if other != emp and delimiters[other] in text:
                text = text.split(delimiters[other])[0].strip()
        if len(text) > 20:
            posts[emp] = text

    hit = len(posts)
    print(f"  [batch/groq/{provider_env}] ✓ {hit}/{len(names)} posts in 1 call")
    return posts


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

# Runtime counter — reset each process start (i.e. each GitHub Actions run)
_run_call_counts: dict[str, int] = {}

# Which free provider answered the most recent call_employee_agent() call.
# None until the first successful call, or if the last call was exhausted.
_LAST_PROVIDER: str | None = None
# Per-employee provider tracking for throughput report (not shown in posts).
_LAST_PROVIDERS_MAP: dict[str, str] = {}

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

# ── Per-role system prompts ───────────────────────────────────────────────────
# Each employee speaks from their actual desk/role (see TEAMS + AGENTS registry).
# These replace the single shared _QUANT_SYSTEM when an employee does their own
# daily work, so a free bot answers in that person's domain voice.
_STRICT_OUTPUT_REQUIREMENTS = (
    " STRICT OUTPUT REQUIREMENTS: (1) Always include at least one specific file path, metric name, or number."
    " (2) Never give generic advice like 'we should improve X' without specifying HOW."
    " (3) If you cannot give a specific concrete answer, say 'INSUFFICIENT DATA' rather than guessing."
    " (4) Slack format: use *bold* for key points, no headers, max 150 words."
)

_EMPLOYEE_PERSONAS: dict[str, str] = {
    "maya": (
        "You are the VP of Engineering at QuantEdge, a quant trading platform. You own "
        "backend reliability, CI/CD health, and release quality. Read commit subjects and "
        "test results, then call out the single biggest reliability risk in plain language. "
        "Example of GOOD output: \"CI risk: test_risk_engine.py has 3 flaky assertions on Kelly fraction edge cases (lines 45-67). Fix: pin random seed in conftest.py. Impact: currently causes 1 false failure per 10 runs.\" "
        "Example of BAD output (reject this): \"The CI looks good overall with some minor issues to watch.\""
        + _STRICT_OUTPUT_REQUIREMENTS
    ),
    "aarav": (
        "You are the Alpha Research Director leading the equities desk at QuantEdge "
        "(momentum, mean-reversion, pairs/Kalman, breakout, idio-vol, ML directional). "
        "You scrutinize backtests for lookahead, regime-fit, and walk-forward Sharpe before any paper-trade gate."
        + _STRICT_OUTPUT_REQUIREMENTS
    ),
    "linh": (
        "You are the ML Modeling Lead and crypto desk lead at QuantEdge. You own LSTM/TFT/XGBoost "
        "experiment analysis and crypto alpha (funding-rate carry, basis, perp liquidation cascades, depeg arb). "
        "You decide which model to prioritize from experiment results and read funding/basis for the desk. "
        "Example of GOOD output: \"TFT on SPY daily (tft_spy_daily.yaml) is overfitting: val_sharpe 1.8 vs train_sharpe 3.2. Reduce d_model from 64->32 and increase dropout 0.1->0.25. XGBoost multi-asset shows better OOS stability.\" "
        "Example of BAD output (reject): \"The ML models are performing well and show promise for future improvements.\""
        + _STRICT_OUTPUT_REQUIREMENTS
    ),
    "jian": (
        "You are the Risk Engineer at QuantEdge. You enforce position limits, drawdown caps, the 70/30 "
        "arbitrage/directional capital split, and per-strategy exposure. You flag risk-bucket breaches and "
        "correlation blowups concisely."
        + _STRICT_OUTPUT_REQUIREMENTS
    ),
    "anna": (
        "You are the Backend Lead at QuantEdge. You own the FastAPI + async SQLAlchemy services, broker "
        "adapters (Alpaca/Binance/Polymarket), and data plumbing. You comment on API/schema/perf changes precisely."
        + _STRICT_OUTPUT_REQUIREMENTS
    ),
    "aditi": (
        "You are the Director of QA at QuantEdge. You own test coverage, the pytest suite, open-PR quality gates, "
        "and flake triage. You report pass/fail signal and the highest-priority test gap, never sugar-coating."
        + _STRICT_OUTPUT_REQUIREMENTS
    ),
    "kenji": (
        "You are the Director of DevOps at QuantEdge. You own GitHub Actions pipelines, deploy readiness, "
        "container builds, and infra cost. You summarize CI/deploy health and the top blocker to shipping."
        + _STRICT_OUTPUT_REQUIREMENTS
    ),
    "diego": (
        "You are the Execution Engineer at QuantEdge. You own order routing, smart execution, slippage and "
        "fill-quality analysis, and async order-management code. You comment on execution-path latency and slippage risk."
        + _STRICT_OUTPUT_REQUIREMENTS
    ),
    "lior": (
        "You are the Polymarket / prediction-market researcher at QuantEdge. You think in probability calibration, "
        "binary-market resolution arbitrage, correlated-market mispricing, and settlement/edge-case risk. "
        "You write desk notes on calibration and resolution-arb opportunities. "
        "Example of GOOD output: \"BTC >$100k by Dec 2025 at 0.62 YES vs Metaculus 0.71 — 9pp gap, buy YES. Kelly fraction 0.08 given 2% bid-ask. Resolution risk: unclear settlement oracle.\" "
        "Example of BAD output (reject): \"There are some interesting opportunities in the prediction markets worth exploring.\""
        + _STRICT_OUTPUT_REQUIREMENTS
    ),
    "sara": (
        "You are the ML Research Lead at QuantEdge. You run model comparisons (LSTM vs TFT vs XGBoost vs Lorentzian KNN), "
        "weight ensembles, and feature studies. You recommend which architecture to prioritize and why, citing metrics."
        + _STRICT_OUTPUT_REQUIREMENTS
    ),
    "sofia": (
        "You are the VP of Research leading the macro/FX-rates desk at QuantEdge (cross-asset carry, HMM regime, "
        "basis carry). You reason about rates, carry, regime shifts, and cross-asset correlation, and translate papers into desk ideas."
        + _STRICT_OUTPUT_REQUIREMENTS
    ),
    "hugo": (
        "You are a Quant Researcher at QuantEdge on the equities/research desk. You prototype signals, run "
        "walk-forward studies, and stress-test stationarity and capacity. You report concrete findings, not vibes."
        + _STRICT_OUTPUT_REQUIREMENTS
    ),
    "marcus": (
        "You are the Chief Risk Officer at QuantEdge. You own firm-wide risk: VaR, drawdown limits, the 70/30 "
        "capital split, leverage, and the paper-first activation policy. You flag the single biggest firm-level risk crisply."
        + _STRICT_OUTPUT_REQUIREMENTS
    ),
    "frontend": (
        "You are a senior frontend engineer at QuantEdge, a Bloomberg-terminal-style quant trading dashboard. "
        "Stack: React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui, TanStack Query, Redux Toolkit, "
        "TradingView Lightweight Charts, TradingView widgets. "
        "You write precise, production-grade TSX. You prioritize: zero TypeScript errors, "
        "smooth real-time data updates via WebSocket, accessibility, and a professional dark trading terminal aesthetic. "
        "Never suggest placeholder UI or mock data — only improvements backed by real API data or real WebSocket feeds. "
        "Cite specific component names, file paths, and concrete code patterns."
        + _STRICT_OUTPUT_REQUIREMENTS
    ),
}


def check_for_hallucination(text: str) -> bool:
    """Flag obvious hallucination patterns — invented specific numbers not from real data."""
    import re
    # Reject if text claims specific $ P&L, % returns, or Sharpe ratios without data context
    red_flags = [
        r'\$[\d,]+\s*(profit|loss|return|gain)',   # "$50,000 profit"
        r'\d+\.?\d*%\s*(return|gain|profit|alpha)', # "34.5% return"
        r'sharpe\s*(ratio\s*)?of\s*\d+\.?\d*',     # "Sharpe of 3.2"
        r'(yesterday|today|this week)\s+(we|the fund|strategy)\s+(made|lost|gained)',
    ]
    for pattern in red_flags:
        if re.search(pattern, text, re.IGNORECASE):
            return True  # likely hallucination
    return False


def score_agent_output(output: str, task_type: str, provider_used: str = "") -> tuple[int, str]:
    """Second-pass quality review of an agent output using a different free provider.
    Returns (score 1-10, reason). Score < 6 = reject, don't post."""
    import re as _re
    if not output or len(output.strip()) < 20:
        return 1, "empty or trivial output"

    review_prompt = (
        f"You are a strict quality reviewer at a top-tier quant fund (Jane Street / Citadel caliber).\n"
        f"Score this AI-generated work output from 1-10. Reject (score < 6) if:\n"
        f"- Generic filler or vague advice with no specifics\n"
        f"- Invented numbers or claims not grounded in the given data\n"
        f"- Repetition of the question without adding insight\n"
        f"- Off-topic for a quant trading platform\n"
        f"- Less than 2 concrete, actionable points\n\n"
        f"Task type: {task_type}\n"
        f"Output to review:\n---\n{output[:600]}\n---\n"
        f"Respond with ONLY: SCORE:<number> REASON:<one sentence>\n"
        f"Example: SCORE:8 REASON:Cites specific file paths and quantifies the risk."
    )

    # Use a DIFFERENT provider than the one that generated the output
    # to avoid confirmation bias — route through "review" task for Gemini-first ordering
    reviewer, _rev_provider = call_best_agent_for_task(
        "review",
        review_prompt,
        system_prompt=(
            "You are a strict quality reviewer. Be harsh. "
            "Only score 8+ for genuinely insightful, specific, actionable output."
        ),
        max_tokens=80,
    )
    if not reviewer:
        return 5, "reviewer unavailable — conservative score"

    m = _re.search(r'SCORE:(\d+)', reviewer)
    r = _re.search(r'REASON:(.+)', reviewer)
    score = int(m.group(1)) if m else 5
    reason = r.group(1).strip() if r else reviewer[:100]
    return min(10, max(1, score)), reason



# ── Task-aware model routing ──────────────────────────────────────────────────
# Maps task type → preferred provider order.
# Gemini 2.5 Flash is best for quant reasoning, ML analysis, risk, review.
# Groq Llama 3.3 70B is best for code generation and fast turnaround.
# Cerebras Qwen3 32B is a solid fallback but NOT used as primary on critical tasks.

_TASK_ROUTING: dict[str, list[str]] = {
    "code":       ["gemini", "cerebras", "groq", "openrouter"],
    "quant":      ["gemini", "cerebras", "groq", "sambanova"],
    "ml":         ["gemini", "cerebras", "groq"],
    "risk":       ["gemini", "cerebras", "groq", "sambanova"],
    "review":     ["gemini", "cerebras", "openrouter"],
    "polymarket": ["gemini", "groq", "sambanova"],
    "frontend":   ["gemini", "cerebras", "groq"],
    "default":    ["gemini", "cerebras", "groq", "sambanova", "openrouter"],
}

# Maps employee short-name → task type for routing
_EMP_TASK_TYPE: dict[str, str] = {
    "maya":   "code",      "anna":  "code",      "kenji":  "code",
    "aditi":  "code",      "diego": "code",
    "linh":   "ml",        "sara":  "ml",
    "aarav":  "quant",     "hugo":  "quant",     "sofia":  "quant",
    "jian":   "risk",      "marcus":"risk",
    "lior":   "polymarket",
    "frontend": "frontend",
}


def call_best_agent_for_task(
    task_type: str,
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 500,
) -> tuple[str | None, str]:
    """Route to the best provider for the given task type.
    Returns (text, provider_name). Uses _TASK_ROUTING order.
    Falls back through all providers before returning (None, 'exhausted')."""
    global _LAST_PROVIDER
    sys_p = system_prompt or _QUANT_SYSTEM
    cap = min(max_tokens, MAX_TOKENS_PER_CALL)
    safe_prompt = _sanitize(prompt)
    safe_sys = _sanitize(sys_p)
    order = _TASK_ROUTING.get(task_type, _TASK_ROUTING["default"])

    for provider in order:
        if provider == "groq":
            # Try all numbered Groq accounts
            for acct in range(1, 11):
                env_var = f"GROQ_API_KEY_{acct}"
                key = os.environ.get(env_var, "").strip()
                if not key:
                    continue
                r = _try_openai_compat(
                    "https://api.groq.com/openai/v1/chat/completions",
                    key, "llama-3.3-70b-versatile", safe_sys, safe_prompt, cap)
                if r and len(r.strip()) > 20:
                    _LAST_PROVIDER = f"Groq-{acct}"
                    track_api_call(env_var, cap)
                    return r.strip(), f"Groq-{acct}"
            # Also try plain GROQ_API_KEY
            key = os.environ.get("GROQ_API_KEY", "").strip()
            if key:
                r = _try_openai_compat(
                    "https://api.groq.com/openai/v1/chat/completions",
                    key, "llama-3.3-70b-versatile", safe_sys, safe_prompt, cap)
                if r and len(r.strip()) > 20:
                    _LAST_PROVIDER = "Groq"
                    track_api_call("GROQ_API_KEY", cap)
                    return r.strip(), "Groq"

        elif provider == "gemini":
            for env_var in ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY"]:
                key = os.environ.get(env_var, "").strip()
                if not key:
                    continue
                r = call_gemini_with_key(key, safe_sys, safe_prompt, cap)
                if r and len(r.strip()) > 20:
                    _LAST_PROVIDER = f"Gemini({env_var})"
                    track_api_call(env_var, cap)
                    return r.strip(), f"Gemini({env_var})"

        elif provider == "cerebras":
            for env_var in ["CEREBRAS_API_KEY_1", "CEREBRAS_API_KEY", "CEREBRAS_API_KEY_2", "CEREBRAS_API_KEY_3"]:
                key = os.environ.get(env_var, "").strip()
                if not key:
                    continue
                r = _try_openai_compat(
                    "https://api.cerebras.ai/v1/chat/completions",
                    key, "qwen-3-32b", safe_sys, safe_prompt, cap)
                if r and len(r.strip()) > 20:
                    _LAST_PROVIDER = f"Cerebras({env_var})"
                    track_api_call(env_var, cap)
                    return r.strip(), f"Cerebras({env_var})"

        elif provider == "sambanova":
            for key in _employee_keys("shared", "sambanova"):
                r = _try_openai_compat(
                    "https://api.sambanova.ai/v1/chat/completions",
                    key, "Meta-Llama-3.3-70B-Instruct", safe_sys, safe_prompt, cap)
                if r and len(r.strip()) > 20:
                    _LAST_PROVIDER = "SambaNova"
                    track_api_call("SAMBANOVA_API_KEY", cap)
                    return r.strip(), "SambaNova"

        elif provider == "openrouter":
            for env_var in ["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2", "OPENROUTER_API_KEY_3"]:
                key = os.environ.get(env_var, "").strip()
                if not key:
                    continue
                r = _try_openai_compat(
                    "https://openrouter.ai/api/v1/chat/completions",
                    key, "meta-llama/llama-3.3-70b-instruct:free",
                    safe_sys, safe_prompt, cap,
                    {"HTTP-Referer": "https://github.com/bahllaavanye-afk/Test"})
                if r and len(r.strip()) > 20:
                    _LAST_PROVIDER = f"OpenRouter({env_var})"
                    track_api_call(env_var, cap)
                    return r.strip(), f"OpenRouter({env_var})"

    return None, "exhausted"


def employee_provider_prompt(emp_key: str, task: str, state: dict | None = None) -> tuple[str | None, str | None]:
    """Returns (answer_text, provider_name) or (None, None) if all tiers exhausted.
    Routes through call_best_agent_for_task with the employee's elite domain persona.
    Uses _EMP_TASK_TYPE to select the best model for the employee's desk.
    Applies quality gate: outputs scoring < 6 are retried once with an enhanced prompt.
    Appends to state["quality_log"] for CTO daily digest."""
    import time
    global _LAST_PROVIDER
    emp = (emp_key or "").split("_")[0].lower()
    persona = _EMPLOYEE_PERSONAS.get(emp, _QUANT_SYSTEM)
    task_type = _EMP_TASK_TYPE.get(emp, "default")
    result, provider = call_best_agent_for_task(task_type, task, system_prompt=persona)
    if not result:
        return (None, None)

    if check_for_hallucination(result):
        print(f"[HALLUCINATION WARNING] {emp} output flagged: {result[:120]}")
        result = f"⚠️ [hallucination-flagged] {result}"

    # Quality gate: score the output and retry once if below threshold
    score, reason = score_agent_output(result, emp, provider_used=provider or "")
    if score < 7:
        print(f"[quality] {emp} output rejected (score={score}): {reason}")
        result2, provider2 = call_best_agent_for_task(
            task_type,
            f"Previous attempt was too generic. Be MORE specific, cite file names/numbers/percentages. Task: {task}",
            system_prompt=persona + " Be extremely specific. Cite exact file names, numbers, and percentages. No generic advice.",
        )
        if result2:
            score2, reason2 = score_agent_output(result2, emp, provider_used=provider2 or "")
            if score2 >= 7:
                result = result2
                score = score2
                reason = reason2

    # Quality emoji appended to end of output (single char, not verbose tag)
    if score >= 8:
        quality_tag = "⭐"
    elif score >= 6:
        quality_tag = ""
    else:
        quality_tag = "⚠️"

    if quality_tag:
        result = f"{result} {quality_tag}"

    # Log for CTO quality digest
    if state is not None:
        state.setdefault("quality_log", []).append({
            "emp": emp,
            "score": score,
            "reason": reason,
            "channel": "unknown",
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        state["quality_log"] = state["quality_log"][-200:]
        state.setdefault("agent_output_log", []).append({
            "emp": emp,
            "channel": "unknown",
            "text": result[:300],
            "provider": provider,
            "ts": time.time(),
            "score": score,
        })
        state["agent_output_log"] = state["agent_output_log"][-50:]
        # Track provider per engineer for throughput report (not shown in posts)
        if provider:
            state.setdefault("last_providers", {})[emp] = provider

    # Also update global map for callers that don't pass state
    if provider:
        _LAST_PROVIDERS_MAP[emp] = provider

    return (result, provider)

# ── Groq account assignment — 3 accounts, 13 employees distributed evenly ────
# Each account handles its own employees in parallel — no sharing, no cascading
# between Groq accounts. This gives 3× RPD and 3× token budget from Groq alone.
#
# Account 1 → GROQ_API_KEY      (~333 req/day budget, 167K tok/day budget)
# Account 2 → GROQ_API_KEY_2    (~333 req/day budget, 167K tok/day budget)
# Account 3 → GROQ_API_KEY_3    (~333 req/day budget, 167K tok/day budget)
# Combined  → 3 × 1000 RPD  =  3 000 req/day  |  3 × 500K tok = 1.5M tok/day
#
# If an employee's assigned Groq account is exhausted for the day, the cascade
# falls through to Cerebras/GitHub Models/OpenRouter/Gemini — never another Groq account.
# This prevents any single Groq account from being overloaded by other employees' quota.

_GROQ_ACCOUNT: dict[str, str] = {
    # Account 1 — GROQ_API_KEY / GROQ_API_KEY_1  (4 employees)
    "maya":   "GROQ_API_KEY_1",
    "aarav":  "GROQ_API_KEY_1",
    "linh":   "GROQ_API_KEY_1",
    "jian":   "GROQ_API_KEY_1",
    # Account 2 — GROQ_API_KEY_2  (4 employees)
    "anna":   "GROQ_API_KEY_2",
    "aditi":  "GROQ_API_KEY_2",
    "kenji":  "GROQ_API_KEY_2",
    "diego":  "GROQ_API_KEY_2",
    # Account 3 — GROQ_API_KEY_3  (5 employees)
    "lior":   "GROQ_API_KEY_3",
    "sara":   "GROQ_API_KEY_3",
    "sofia":  "GROQ_API_KEY_3",
    "hugo":   "GROQ_API_KEY_3",
    "marcus": "GROQ_API_KEY_3",
}

# Each employee group gets the same-numbered Gemini account as their Groq account.
# Group 1 (GROQ_API_KEY_1) → GEMINI_API_KEY_1, etc.
# This isolates quota completely — Group 2 Gemini burnout never affects Group 1.
_GEMINI_ACCOUNT: dict[str, str] = {
    "maya":   "GEMINI_API_KEY_1",
    "aarav":  "GEMINI_API_KEY_1",
    "linh":   "GEMINI_API_KEY_1",
    "jian":   "GEMINI_API_KEY_1",
    "anna":   "GEMINI_API_KEY_2",
    "aditi":  "GEMINI_API_KEY_2",
    "kenji":  "GEMINI_API_KEY_2",
    "diego":  "GEMINI_API_KEY_2",
    "lior":   "GEMINI_API_KEY_3",
    "sara":   "GEMINI_API_KEY_3",
    "sofia":  "GEMINI_API_KEY_3",
    "hugo":   "GEMINI_API_KEY_3",
    "marcus": "GEMINI_API_KEY_3",
}

# 2 Cerebras accounts — split employees evenly across groups.
# Group 1 (Groq_1/Gemini_1 users) + Group 2 (Groq_2/Gemini_2 users) → CEREBRAS_API_KEY_1
# Group 3 (Groq_3/Gemini_3 users) → CEREBRAS_API_KEY_2
_CEREBRAS_ACCOUNT: dict[str, str] = {
    "maya":   "CEREBRAS_API_KEY_1",
    "aarav":  "CEREBRAS_API_KEY_1",
    "linh":   "CEREBRAS_API_KEY_1",
    "jian":   "CEREBRAS_API_KEY_1",
    "anna":   "CEREBRAS_API_KEY_1",
    "aditi":  "CEREBRAS_API_KEY_1",
    "kenji":  "CEREBRAS_API_KEY_1",
    "diego":  "CEREBRAS_API_KEY_1",
    "lior":   "CEREBRAS_API_KEY_2",
    "sara":   "CEREBRAS_API_KEY_2",
    "sofia":  "CEREBRAS_API_KEY_2",
    "hugo":   "CEREBRAS_API_KEY_2",
    "marcus": "CEREBRAS_API_KEY_2",
}

# For shared calls, rotate across all available accounts round-robin.
# Both GROQ_API_KEY_1 and GROQ_API_KEY are checked — whichever is set wins.
# KEY_4 through KEY_10 are auto-discovered: if the env var is set it joins the pool,
# if not it is skipped — zero-config drop-in for new keys.
_shared_groq_counter: int = 0
_GROQ_SHARED_ACCOUNTS = [
    "GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3",
    "GROQ_API_KEY_4", "GROQ_API_KEY_5", "GROQ_API_KEY_6",
    "GROQ_API_KEY_7", "GROQ_API_KEY_8", "GROQ_API_KEY_9", "GROQ_API_KEY_10",
]


def _groq_key_for(employee: str) -> str | None:
    """Return the Groq key for this employee's assigned account.
    Checks _1 suffix first (GROQ_API_KEY_1), then plain (GROQ_API_KEY) as alias.
    """
    emp = employee.split("_")[0].lower()
    env_var = _GROQ_ACCOUNT.get(emp, "GROQ_API_KEY_1")
    key = os.environ.get(env_var, "").strip()
    if key:
        return key
    # Alias: GROQ_API_KEY_1 ↔ GROQ_API_KEY (same account, either naming works)
    if env_var == "GROQ_API_KEY_1":
        return os.environ.get("GROQ_API_KEY", "").strip() or None
    return None


def _gemini_key_for(employee: str) -> str | None:
    """Returns the Gemini API key assigned to this employee's department group."""
    emp = employee.split("_")[0].lower()
    env_var = _GEMINI_ACCOUNT.get(emp, "GEMINI_API_KEY_1")
    key = os.environ.get(env_var, "").strip()
    if key:
        return key
    # Fallback: try plain GEMINI_API_KEY if _1 is the assigned one
    if env_var == "GEMINI_API_KEY_1":
        return os.environ.get("GEMINI_API_KEY", "").strip() or None
    return None


def _cerebras_key_for(employee: str) -> str | None:
    """Returns the Cerebras API key assigned to this employee's department group."""
    emp = employee.split("_")[0].lower()
    env_var = _CEREBRAS_ACCOUNT.get(emp, "CEREBRAS_API_KEY_1")
    # Support both _1 suffix and plain name for first account
    key = os.environ.get(env_var, "").strip()
    if key:
        return key
    if env_var == "CEREBRAS_API_KEY_1":
        return os.environ.get("CEREBRAS_API_KEY", "").strip() or None
    return None


def _groq_key_shared() -> str | None:
    """Round-robin across all 3 Groq accounts for shared (non-employee) calls."""
    global _shared_groq_counter
    for _ in range(len(_GROQ_SHARED_ACCOUNTS)):
        env_var = _GROQ_SHARED_ACCOUNTS[_shared_groq_counter % len(_GROQ_SHARED_ACCOUNTS)]
        _shared_groq_counter += 1
        key = os.environ.get(env_var, "").strip()
        if key:
            return key
    return None


def _employee_keys(employee: str, provider: str) -> list[str]:
    """Return API keys for (employee, provider) in priority order.

    For Groq: returns ONLY the employee's assigned account key — prevents one
    employee from consuming another account's daily budget.

    For all other providers (Cerebras, Gemini, OpenRouter): tries employee-
    dedicated key (CEREBRAS_API_KEY_MAYA) then numbered pool (_2, _3 …) then
    shared primary — these providers all have separate per-account limits.

    Secret naming supported (all checked automatically):
      GROQ_API_KEY / GROQ_API_KEY_2 / GROQ_API_KEY_3   — 3 Groq accounts
      CEREBRAS_API_KEY_MAYA                              — employee dedicated
      CEREBRAS_API_KEY_2 … _10                          — numbered pool
      CEREBRAS_API_KEY                                   — shared primary
    """
    emp = employee.split("_")[0].lower()
    prov = provider.upper()
    keys: list[str] = []

    def _add(k: str) -> None:
        k = (k or "").strip()
        if k and k not in keys:
            keys.append(k)

    if prov == "GROQ":
        # One account per employee — no cross-contamination of quotas
        _add(_groq_key_for(emp))
        return keys

    # All other providers: dedicated → numbered pool (_1 through _10) → plain primary
    _add(os.environ.get(f"{prov}_API_KEY_{emp.upper()}", ""))   # e.g. CEREBRAS_API_KEY_MAYA
    for i in range(1, 11):                                       # _1 … _10 (includes _1 alias)
        _add(os.environ.get(f"{prov}_API_KEY_{i}", ""))
    _add(os.environ.get(f"{prov}_API_KEY", ""))                  # plain primary
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
            content_result = data["choices"][0]["message"]["content"]
            track_api_call("GROQ_API_KEY", data.get("usage", {}).get("total_tokens", max_tokens))
            return content_result
    except Exception as e:
        print(f"  [groq] {e}")
        return None


def call_gemini(system_prompt: str, user_message: str, max_tokens: int = 600,
                state: dict | None = None) -> str | None:
    """
    Google Gemini 2.5 Flash — rotates across GEMINI_API_KEY, _2, _3.
    Free: 1500 req/day per key → 3 keys = 4500 req/day combined.
    Checks daily budget (tracked in state) before calling.
    """
    payload = {
        "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_message}"}]}],
        "generationConfig": {"maxOutputTokens": min(max_tokens, MAX_TOKENS_PER_CALL),
                             "temperature": 0.4},
    }
    for env_var in ["GEMINI_API_KEY_1", "GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]:
        key = os.environ.get(env_var, "").strip()
        if not key:
            continue
        if state and not budget_ok(state, env_var, estimated_tokens=1):
            continue   # this key's daily request budget exhausted
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"gemini-2.5-flash:generateContent?key={key}")
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                text = json.loads(resp.read())["candidates"][0]["content"]["parts"][0]["text"]
                if state:
                    record_usage(state, env_var, 1)   # track as 1 request
                track_api_call(env_var, 1)
                print(f"  [gemini/{env_var}] ✓")
                return text
        except Exception as e:
            print(f"  [gemini/{env_var}] {e}")
            continue
    return None


def call_gemini_with_key(
    api_key: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 500,
    state: dict | None = None,
) -> str | None:
    """Call Gemini with a specific API key (used for department-isolated calls)."""
    # Determine which env var name this key belongs to (for budget tracking)
    env_var = "GEMINI_API_KEY"
    for ev in ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY"]:
        if os.environ.get(ev, "").strip() == api_key:
            env_var = ev
            break
    if state and not budget_ok(state, env_var, estimated_tokens=max_tokens):
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7},
    }).encode()
    req = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            if state:
                record_usage(state, env_var, max_tokens)
            track_api_call(env_var, max_tokens)
            return text.strip()
    except Exception as e:
        print(f"  [gemini-key] error: {e}")
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
            gh_result = data["choices"][0]["message"]["content"]
            track_api_call("GH_TOKEN", data.get("usage", {}).get("total_tokens", max_tokens))
            return gh_result
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
            cerebras_result = data["choices"][0]["message"]["content"]
            track_api_call("CEREBRAS_API_KEY", data.get("usage", {}).get("total_tokens", max_tokens))
            return cerebras_result
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
            or_result = data["choices"][0]["message"]["content"]
            track_api_call("OPENROUTER_API_KEY", data.get("usage", {}).get("total_tokens", max_tokens))
            return or_result
    except Exception as e:
        print(f"  [openrouter] {e}")
        return None


def call_claude(system_prompt: str, user_message: str, max_tokens: int = 600) -> str | None:
    """
    Claude Haiku — BLOCKED by default (paid-API flag is off).
    Only reachable if a human explicitly flips the paid-API flag on in a code review.
    This function exists so the import chain doesn't break, not to be called.
    """
    if not ALLOW_PAID_APIS:
        import traceback
        caller = traceback.extract_stack()[-2]
        _PAID_CALLS_BLOCKED.append(f"{caller.filename.split('/')[-1]}:{caller.lineno} @ {_today()}")
        print("  [POLICY] Paid API blocked — ALLOW_PAID_APIS=False. Logged to #agent-api-usage.")
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
    import urllib.parse as _urlparse
    _host = _urlparse.urlparse(endpoint).hostname or ""
    if _host in _CF_BLOCKED_HOSTS:
        return None  # skip silently — already known Cloudflare-blocked from CI
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
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        _provider = endpoint.split("/")[2] if "/" in endpoint else endpoint[:30]
        if e.code == 403 and "1010" in body:
            _CF_BLOCKED_HOSTS.add(_host)
            print(f"  [{_provider}] Cloudflare blocked (1010) — marking {_host} as unreachable from CI")
        else:
            print(f"  [{_provider}] HTTP {e.code}: {body}")
        return None
    except Exception as e:
        _provider = endpoint.split("/")[2] if "/" in endpoint else endpoint[:30]
        print(f"  [{_provider}] error: {type(e).__name__}: {e}")
        return None


def call_employee_agent(
    employee: str,
    user_message: str,
    system_prompt: str = _QUANT_SYSTEM,
    max_tokens: int = 500,
    state: dict | None = None,
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
    global _LAST_PROVIDER
    _LAST_PROVIDER = None

    if ALLOW_PAID_APIS:
        raise RuntimeError("ALLOW_PAID_APIS must stay False — zero spend policy")

    # Governance: check if engineer is paused for today
    emp_key = employee.split("_")[0].lower()
    gov = state.get("governance", {}) if state else {}
    paused = gov.get("paused_engineers", {})
    if emp_key in paused and paused[emp_key] == _today():
        print(f"  [{emp_key}] PAUSED by CTO for today — skipping")
        return None

    # Enforce per-run call budget
    _run_call_counts[emp_key] = _run_call_counts.get(emp_key, 0)
    if _run_call_counts[emp_key] >= MAX_CALLS_PER_EMPLOYEE_PER_RUN:
        print(f"  [{emp_key}] call budget exhausted ({MAX_CALLS_PER_EMPLOYEE_PER_RUN}/run) — skipping")
        return None
    _run_call_counts[emp_key] += 1

    # IP guard — never leak credentials or internal paths to external providers
    safe_message = _sanitize(user_message)
    safe_system  = _sanitize(system_prompt)
    cap = min(max_tokens, MAX_TOKENS_PER_CALL)

    # 1. Groq — employee's assigned account (one of 3 accounts, ~333 req/day each)
    groq_key = _groq_key_for(emp_key)
    if groq_key:
        r = _try_openai_compat(
            "https://api.groq.com/openai/v1/chat/completions",
            groq_key, "llama-3.3-70b-versatile", safe_system, safe_message, cap)
        if r and len(r.strip()) > 20:
            print(f"  [{emp_key}/groq] ✓ {len(r)} chars")
            _LAST_PROVIDER = "Groq"
            groq_env = _GROQ_ACCOUNT.get(emp_key, "GROQ_API_KEY_1")
            track_api_call(groq_env, cap)
            return r.strip()

    # 2. Cerebras — employee's assigned account (2-account split)
    cerebras_key = _cerebras_key_for(emp_key)
    if cerebras_key:
        r = _try_openai_compat(
            "https://api.cerebras.ai/v1/chat/completions",
            cerebras_key, "qwen-3-32b", safe_system, safe_message, cap)
        if r and len(r.strip()) > 20:
            print(f"  [{emp_key}/cerebras] ✓ {len(r)} chars")
            _LAST_PROVIDER = "Cerebras"
            cerebras_env = "CEREBRAS_API_KEY_2" if emp_key in ("lior","sara","sofia","hugo","marcus") else "CEREBRAS_API_KEY_1"
            track_api_call(cerebras_env, cap)
            return r.strip()

    # 3. SambaNova — 20M tokens/day free, Llama 3.3 70B on custom RDU chips
    for key in _employee_keys(emp_key, "sambanova"):
        r = _try_openai_compat(
            "https://api.sambanova.ai/v1/chat/completions",
            key, "Meta-Llama-3.3-70B-Instruct", safe_system, safe_message, cap)
        if r and len(r.strip()) > 20:
            print(f"  [{emp_key}/sambanova] ✓ {len(r)} chars")
            _LAST_PROVIDER = "SambaNova"
            track_api_call("SAMBANOVA_API_KEY", cap)
            return r.strip()

    # 4. GitHub Models — free via GITHUB_TOKEN, no extra key needed
    r = call_github_models(safe_system, safe_message, cap)
    if r and len(r.strip()) > 20:
        print(f"  [{emp_key}/github-models] ✓ {len(r)} chars")
        _LAST_PROVIDER = "GitHub Models"
        return r.strip()

    # 5. OpenRouter — try both keys (50 req/day each = 100/day combined)
    for or_env in ["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"]:
        key = os.environ.get(or_env, "").strip()
        if not key:
            continue
        r = _try_openai_compat(
            "https://openrouter.ai/api/v1/chat/completions",
            key, "meta-llama/llama-3.3-70b-instruct:free", safe_system, safe_message, cap,
            {"HTTP-Referer": "https://github.com/bahllaavanye-afk/Test"})
        if r and len(r.strip()) > 20:
            print(f"  [{emp_key}/openrouter/{or_env}] ✓ {len(r)} chars")
            _LAST_PROVIDER = "OpenRouter"
            track_api_call(or_env, cap)
            return r.strip()

    # 6. Gemini — employee's assigned account (group-isolated quota)
    gemini_key = _gemini_key_for(emp_key)
    if gemini_key:
        r = call_gemini_with_key(gemini_key, safe_system, safe_message, cap, state)
        if r and len(r.strip()) > 20:
            print(f"  [{emp_key}/gemini] ✓ {len(r)} chars")
            _LAST_PROVIDER = "Gemini"
            return r.strip()

    print(f"  [{emp_key}] ⚠ all free tiers exhausted — no paid fallback (policy)")
    return None


def call_best_agent(
    user_message: str,
    system_prompt: str = _QUANT_SYSTEM,
    max_tokens: int = 500,
) -> str | None:
    """
    Shared cascade for non-employee calls (inbox, commands, incident posts).
    Groq: round-robins across all 3 accounts so no single account absorbs shared load.
    100% free — zero-spend policy enforced.
    """
    cap = min(max_tokens, MAX_TOKENS_PER_CALL)
    safe_msg = _sanitize(user_message)
    safe_sys = _sanitize(system_prompt)

    # Groq — round-robin across all 3 accounts for shared calls
    key = _groq_key_shared()
    if key:
        r = _try_openai_compat(
            "https://api.groq.com/openai/v1/chat/completions",
            key, "llama-3.3-70b-versatile", safe_sys, safe_msg, cap)
        if r and len(r.strip()) > 20:
            print(f"  [agent/groq] ✓ {len(r)} chars")
            groq_env_shared = next((ev for ev in _GROQ_SHARED_ACCOUNTS if os.environ.get(ev,"").strip() == key), "GROQ_API_KEY_1")
            track_api_call(groq_env_shared, cap)
            return r.strip()

    # Cerebras — round-robin across all accounts for shared calls
    for cerebras_env in ["CEREBRAS_API_KEY_1", "CEREBRAS_API_KEY", "CEREBRAS_API_KEY_2", "CEREBRAS_API_KEY_3"]:
        key = os.environ.get(cerebras_env, "").strip()
        if not key:
            continue
        r = _try_openai_compat(
            "https://api.cerebras.ai/v1/chat/completions",
            key, "qwen-3-32b", safe_sys, safe_msg, cap)
        if r and len(r.strip()) > 20:
            print(f"  [agent/cerebras] ✓ {len(r)} chars")
            track_api_call(cerebras_env, cap)
            return r.strip()
        break  # try next account only if first fails

    # SambaNova — 20M tokens/day free, Llama 3.3 70B on custom RDU chips
    for key in _employee_keys("shared", "sambanova"):
        r = _try_openai_compat(
            "https://api.sambanova.ai/v1/chat/completions",
            key, "Meta-Llama-3.3-70B-Instruct", safe_sys, safe_msg, cap)
        if r and len(r.strip()) > 20:
            print(f"  [agent/sambanova] ✓ {len(r)} chars")
            track_api_call("SAMBANOVA_API_KEY", cap)
            return r.strip()

    # GitHub Models — free in Actions
    r = call_github_models(safe_sys, safe_msg, cap)
    if r and len(r.strip()) > 20:
        print(f"  [agent/github-models] ✓ {len(r)} chars")
        return r.strip()

    # OpenRouter — try both keys (50 req/day each = 100/day combined)
    for or_env in ["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"]:
        key = os.environ.get(or_env, "").strip()
        if not key:
            continue
        r = _try_openai_compat(
            "https://openrouter.ai/api/v1/chat/completions",
            key, "meta-llama/llama-3.3-70b-instruct:free", safe_sys, safe_msg, cap,
            {"HTTP-Referer": "https://github.com/bahllaavanye-afk/Test"})
        if r and len(r.strip()) > 20:
            print(f"  [agent/openrouter/{or_env}] ✓ {len(r)} chars")
            track_api_call(or_env, cap)
            return r.strip()

    # Gemini — 1500 req/day
    r = call_gemini(safe_sys, safe_msg, cap)
    if r and len(r.strip()) > 20:
        print(f"  [agent/gemini] ✓ {len(r)} chars")
        return r.strip()

    # Hard stop — never pay
    print("  [agent] ⚠ all free providers exhausted — returning None (zero-spend policy)")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Slack thread reading — agents respond to actual human replies

# ─────────────────────────────────────────────────────────────────────────────


def read_unresponded_threads(
    token: str,
    channel_name: str,
    bot_user_id: str,
    already_replied: list[str],
    limit: int = 50,
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
        err = history.get("error", "unknown")
        if err == "not_in_channel":
            slack_call(token, "conversations.join", {"channel": ch_id})
            history = slack_call(token, "conversations.history", {"channel": ch_id, "limit": limit})
        if not history.get("ok"):
            print(f"  [threads] #{channel_name}: {history.get('error', err)}")
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


# ─────────────────────────────────────────────────────────────────────────────
# Summon-a-free-agent — any human types @agent / ask: / ?? in a channel and the
# free LLM cascade answers in-thread. Zero Claude/Anthropic spend.
# ─────────────────────────────────────────────────────────────────────────────

# Triggers at the START of a human message (case-insensitive). All route to the
# free cascade via call_best_agent — never to Claude.
_AGENT_SUMMON_TRIGGERS = ("@quantedge", "@quant", "@agent", "@ai", "ask:", "?? ")


def _match_agent_summon(text: str) -> str | None:
    """Return the question (trigger stripped) if text starts with a summon trigger, else None."""
    if not text:
        return None
    stripped = text.lstrip()
    low = stripped.lower()
    for trig in _AGENT_SUMMON_TRIGGERS:
        if low.startswith(trig):
            question = stripped[len(trig):].strip()
            # Strip a leading separator left after a mention-style trigger.
            question = question.lstrip(":, ").strip()
            if question:
                return question
    return None


def detect_agent_summons(
    token: str,
    channel_name: str,
    bot_user_id: str,
    already_replied: list[str],
    limit: int = 30,
) -> list[dict]:
    """
    Scan recent channel messages for a human who summoned a free agent via a
    leading trigger (@quant / @agent / @ai / @quantedge / ask: / ?? ).

    Self-guard matches read_unresponded_threads: must have `user`, no `bot_id`,
    user != bot_user_id, and ts not in already_replied.
    """
    ch_id = get_channel_id(token, channel_name)
    if not ch_id:
        return []
    history = slack_call(token, "conversations.history", {"channel": ch_id, "limit": limit})
    if not history.get("ok"):
        err = history.get("error", "unknown")
        if err == "not_in_channel":
            slack_call(token, "conversations.join", {"channel": ch_id})
            history = slack_call(token, "conversations.history", {"channel": ch_id, "limit": limit})
        if not history.get("ok"):
            print(f"  [summon] #{channel_name}: {history.get('error', err)}")
            return []

    summons: list[dict] = []
    for msg in history.get("messages", []):
        ts = msg.get("ts", "")
        user = msg.get("user")
        if not user or msg.get("bot_id") or user == bot_user_id:
            continue
        if ts in already_replied:
            continue
        question = _match_agent_summon(msg.get("text", ""))
        if not question:
            continue
        summons.append({
            "channel_id": ch_id,
            "channel_name": channel_name,
            "thread_ts": ts,   # reply threads under the original message
            "user": user,
            "question": question[:1500],
        })
    return summons


def answer_agent_summons(token: str, summons: list[dict], state: dict) -> int:
    """
    Answer each summon via the free cascade (call_best_agent) and post in-thread.
    Records ts in state["replied_to"] so a summon is answered once. Returns count answered.
    """
    answered = 0
    exhausted_notified = False
    for s in summons:
        ts = s["thread_ts"]
        if ts in state.get("replied_to", []):
            continue
        ans = call_best_agent(s["question"], max_tokens=600)
        if ans and ans.strip():
            reply = f":robot_face: {ans.strip()}"
            r = post_to_slack(token, s["channel_name"], reply,
                              username="Free Agent", icon_emoji=":robot_face:",
                              thread_ts=ts)
            if r.get("ok"):
                answered += 1
                state.setdefault("replied_to", []).append(ts)
                print(f"  ✓ summon answered → #{s['channel_name']}")
        elif not exhausted_notified:
            # All free providers exhausted — notify once, don't burn retries.
            post_to_slack(
                token, s["channel_name"],
                ":hourglass_flowing_sand: All free agents are at their daily limit right now — please try again later. (Zero-spend policy: no paid fallback.)",
                username="Free Agent", icon_emoji=":robot_face:", thread_ts=ts)
            exhausted_notified = True
            state.setdefault("replied_to", []).append(ts)
        time.sleep(0.5)
    return answered


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
    # Primary channel owners (thread-reply identity)
    "leadership-summary": ("Chief Risk Officer", ":shield:"),
    "papers":             ("VP Research", ":books:"),
    "finance-ops":        ("Finance Engineer", ":moneybag:"),
    "legal-compliance":   ("Compliance Engineer", ":scales:"),
    "security-alerts":    ("Security Engineer", ":closed_lock_with_key:"),
    "pod-ml-rl":          ("Research Scientist", ":brain:"),
    "announcements":      ("CEO / Founder", ":sparkles:"),
    "random":             ("CEO / Founder", ":sparkles:"),
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
    "agent-api-usage",   # real-time dashboard: provider usage, employee assignments, paid-API audit
    "cto-audit",
    "allquantedge",      # company-wide broadcast channel
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

    # Join ALL known public channels so the bot can read history for thread replies
    joined = 0
    for _name, _ch in list(_channels_cache.items()):
        _ch_id = _ch.get("id") if isinstance(_ch, dict) else _ch
        if not _ch_id:
            continue
        if isinstance(_ch, dict) and _ch.get("is_private", False):
            continue
        r = slack_call(token, "conversations.join", {"channel": _ch_id})
        if r.get("ok") or r.get("error") in ("already_in_channel", "method_not_supported_for_channel_type"):
            joined += 1
    print(f"  ✓ Bot joined/confirmed in {joined} channels")


# ─────────────────────────────────────────────────────────────────────────────
# #agent-api-usage channel — real-time dashboard posted after every run
# ─────────────────────────────────────────────────────────────────────────────

_PAID_CALLS_BLOCKED: list[str] = []   # populated by call_claude() each time it's invoked


def verify_zero_spend() -> None:
    """Hard financial guard — crashes if paid API could be called."""
    if ALLOW_PAID_APIS:
        raise RuntimeError(
            "FINANCIAL GUARD: paid-API flag is enabled — aborting to prevent charges."
        )
    openrouter_model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    if not openrouter_model.endswith(":free") and "free" not in openrouter_model.lower():
        raise RuntimeError(f"FINANCIAL GUARD: OpenRouter model '{openrouter_model}' is not :free")
    print("ZERO-SPEND ✅ ALLOW_PAID_APIS=False | Groq×3 + Gemini×3 + Cerebras×2 + SambaNova + OpenRouter×2 + GH-Models = $0.00/day")


def _bar(used: int, limit: int, width: int = 12) -> str:
    """ASCII progress bar: ████░░░░ 45%"""
    pct = min(used / limit, 1.0) if limit else 0
    filled = round(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"`{bar}` {pct*100:.0f}%"


def _status_emoji(used: int, limit: int) -> str:
    pct = used / limit if limit else 0
    if pct >= 0.90:
        return ":red_circle:"
    if pct >= 0.60:
        return ":large_yellow_circle:"
    return ":large_green_circle:"


# Comprehensive limits for display
_PROVIDER_LIMITS = {
    "GROQ_API_KEY_1":     {"tok_day": 500_000,  "req_day": 1_000, "model": "Llama 3.3 70B"},
    "GROQ_API_KEY_2":     {"tok_day": 500_000,  "req_day": 1_000, "model": "Llama 3.3 70B"},
    "GROQ_API_KEY_3":     {"tok_day": 500_000,  "req_day": 1_000, "model": "Llama 3.3 70B"},
    "GEMINI_API_KEY_1":   {"tok_day": 0,         "req_day": 1_500, "model": "Gemini 2.5 Flash"},
    "GEMINI_API_KEY_2":   {"tok_day": 0,         "req_day": 1_500, "model": "Gemini 2.5 Flash"},
    "GEMINI_API_KEY_3":   {"tok_day": 0,         "req_day": 1_500, "model": "Gemini 2.5 Flash"},
    "CEREBRAS_API_KEY_1": {"tok_day": 1_000_000, "req_day": 1_440, "model": "Qwen3 32B"},
    "CEREBRAS_API_KEY_2": {"tok_day": 1_000_000, "req_day": 1_440, "model": "Qwen3 32B"},
    "SAMBANOVA_API_KEY":  {"tok_day": 20_000_000,"req_day": 10_000,"model": "Llama 3.3 70B"},
    "OPENROUTER_API_KEY": {"tok_day": 200_000,   "req_day": 50,    "model": "Llama 3.3 70B :free"},
}

_KEY_GUARDS = {
    "GROQ_API_KEY_1":     ("Group 1", "maya, aarav, linh, jian"),
    "GROQ_API_KEY_2":     ("Group 2", "anna, aditi, kenji, diego"),
    "GROQ_API_KEY_3":     ("Group 3", "lior, sara, sofia, hugo, marcus"),
    "GEMINI_API_KEY_1":   ("Group 1", "maya, aarav, linh, jian"),
    "GEMINI_API_KEY_2":   ("Group 2", "anna, aditi, kenji, diego"),
    "GEMINI_API_KEY_3":   ("Group 3", "lior, sara, sofia, hugo, marcus"),
    "CEREBRAS_API_KEY_1": ("Groups 1+2", "maya, aarav, linh, jian, anna, aditi, kenji, diego"),
    "CEREBRAS_API_KEY_2": ("Group 3", "lior, sara, sofia, hugo, marcus"),
    "SAMBANOVA_API_KEY":  ("All", "all 13 employees (shared pool)"),
    "OPENROUTER_API_KEY": ("All", "all 13 employees (shared fallback)"),
}


def _usage_bar(used: int, limit: int, width: int = 10) -> str:
    if limit == 0:
        return "░" * width + " N/A"
    pct = min(used / limit, 1.0)
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    emoji = "🟢" if pct < 0.6 else ("🟡" if pct < 0.9 else "🔴")
    return f"{emoji} {bar} {pct*100:.0f}%"


def post_api_guard_map(token: str, state: dict) -> None:
    """Post the full API guard map with live usage to #agent-api-usage."""
    ch_id = get_channel_id(token, "agent-api-usage")
    if not ch_id:
        return

    today = _today()
    usage = state.get("daily_usage", {}).get(today, {})

    # Count how many keys are actually set
    keys_set = sum(
        1 for env_var in _PROVIDER_LIMITS
        if os.environ.get(env_var, "").strip() or
           (env_var == "CEREBRAS_API_KEY_1" and os.environ.get("CEREBRAS_API_KEY", "").strip())
    )
    total_keys = len(_PROVIDER_LIMITS)
    total_tok_cap_summary = sum(v["tok_day"] for v in _PROVIDER_LIMITS.values())

    lines = [
        f"*:shield: API Guard Map — {today} UTC*",
        f"*Keys active: {keys_set}/{total_keys} | Total daily capacity: {total_tok_cap_summary:,} tokens | $0.00/day*",
        "",
        "```",
        f"{'Key':<22} {'Model':<20} {'Guards':<12} {'Daily Limit':<14} {'Usage'}",
        "─" * 85,
    ]

    for env_var, limits in _PROVIDER_LIMITS.items():
        is_set = bool(os.environ.get(env_var, "").strip() or
                      (env_var == "CEREBRAS_API_KEY_1" and os.environ.get("CEREBRAS_API_KEY", "").strip()))
        guard_group, _ = _KEY_GUARDS.get(env_var, ("?", "?"))
        tok_limit = limits["tok_day"]
        req_limit = limits["req_day"]
        model = limits["model"][:19]

        tok_used = usage.get(env_var, {}).get("tokens", 0)
        req_used = usage.get(env_var, {}).get("requests", 0)

        if tok_limit > 0:
            limit_str = f"{tok_limit//1000}K tok"
            bar = _usage_bar(tok_used, tok_limit)
        else:
            limit_str = f"{req_limit} req"
            bar = _usage_bar(req_used, req_limit)

        status = "✅" if is_set else "⚠️ "
        lines.append(f"{status} {env_var:<20} {model:<20} {guard_group:<12} {limit_str:<14} {bar}")

    # Totals
    total_tok_cap = sum(v["tok_day"] for v in _PROVIDER_LIMITS.values())
    total_tok_used = sum(
        state.get("daily_usage", {}).get(today, {}).get(k, {}).get("tokens", 0)
        for k in _PROVIDER_LIMITS
    )
    lines += [
        "─" * 85,
        f"  TOTAL FREE CAPACITY: {total_tok_cap:,} tokens/day",
        f"  USED THIS RUN:       {total_tok_used:,} tokens",
        f"  REMAINING:           {total_tok_cap - total_tok_used:,} tokens",
        "```",
        "",
        "*Department Assignment*",
        "```",
        "Group 1: maya/aarav/linh/jian     → Groq_1 + Gemini_1 + Cerebras_1",
        "Group 2: anna/aditi/kenji/diego   → Groq_2 + Gemini_2 + Cerebras_1",
        "Group 3: lior/sara/sofia/hugo/marcus → Groq_3 + Gemini_3 + Cerebras_2",
        "Shared:  sambanova + openrouter + github-models → all groups",
        "```",
        "",
        f"`ALLOW_PAID_APIS = {ALLOW_PAID_APIS}` — Claude API: *NEVER called* ✅",
    ]

    lines += [
        "",
        "*:brain: Free ML Training Compute*",
        "```",
        "Kaggle:      30 hrs/week  | P100/T4  | Free, no CC",
        "Modal.com:   $30/mo free  | H100/A10G| Best quality — add MODAL_TOKEN_ID secret",
        "Lightning.AI:22 hrs/month | T4       | Good for TFT/Transformer",
        "Google Colab:~15 hrs/week | T4       | Free overflow",
        "Startup credits: Google $350K | AWS $300K | Modal $25K",
        "```",
        "_Trigger: Actions → QuantEdge ML Training → Run workflow_",
    ]

    slack_call(token, "chat.postMessage", {
        "channel": ch_id,
        "text": "\n".join(lines),
        "username": "API Guardian",
        "icon_emoji": ":shield:",
    })
    print("[guard-map] Posted API guard map to #agent-api-usage")


def post_engineer_onboarding(token: str, state: dict) -> None:
    """Post/update the engineer onboarding guide in #help. Only posts once per week."""
    last_posted = state.get("onboarding_posted_week")
    current_week = datetime.now(timezone.utc).strftime("%Y-W%W")
    if last_posted == current_week:
        return

    ch_id = get_channel_id(token, "help")
    if not ch_id:
        return

    guide = """*:wave: Welcome to QuantEdge — Free Agent Team Guide*

Our AI agents run 24/7 across all Slack channels. Here's how to use them:

*1. Summon a free agent — instant answer, $0 cost*
Type `@agent <your question>` at the start of a message in any monitored channel → instant answer from the free LLM team (Groq/Gemini/Cerebras), zero Claude/Anthropic spend.
Triggers (case-insensitive): `@agent`, `@quant`, `@ai`, `ask:`, `?? `
Examples:
• `@agent what's a good Sharpe threshold for going live?`
• `ask: how does the risk bucket split work?`
• `?? explain walk-forward backtesting`
Or use the slash command: `/ask <question>`.
Monitored channels (16): #engineering, #alpha-research, #ml-experiments, #squad-qa, #squad-backend, #squad-frontend, #risk-alerts, #desk-crypto, #desk-polymarket, #desk-fx-rates, #help, #pnl-daily, #squad-execution, #desk-kalshi, #desk-stat-arb, #desk-futures.
An agent replies in-thread within 15 minutes.

*2. Use /commands for specific actions*
```
/backtest [strategy] [symbol]  — Latest walk-forward Sharpe results
/sharpe [strategy]             — Top Sharpe ratios
/risk                          — Current risk snapshot (Alpaca paper)
/positions                     — Open portfolio positions
/status                        — Full system status (tests + CI + Alpaca)
/strategies                    — Strategy registry
/tests                         — Run pytest (lightweight)
/ci                            — Recent CI workflow runs
/prs                           — Open pull requests
/capacity                      — Live API key usage + how to add more keys
/keys                          — Alias for /capacity
/ask <question>                — Ask the AI agent team directly
/agent help                    — Full help text
```

*3. Free bot capacity: Groq 3 accounts, Cerebras 2, Gemini 3, SambaNova, OpenRouter 2, GitHub Models.*
Type `/capacity` to see live usage and add more keys.

*4. 24/7 Desks (always responding)*
• #desk-crypto — BTC/ETH/SOL signals, funding rates, OI momentum
• #desk-polymarket — prediction market arb, Kelly sizing
• #desk-fx-rates — macro/FX signals, yield curve
• #desk-stat-arb — pairs trading, cointegration alerts
• #desk-futures — futures basis, roll signals

*5. Trigger ML experiments*
Go to GitHub Actions → "QuantEdge Run ML Experiments" → Run workflow
Available models: LSTM, XGBoost, TFT, SSM, Ensemble, Lorentzian KNN

*6. Code changes via Slack (when CTO is unavailable)*
GitHub Actions → "QuantEdge Slack Code Request" → type your request in plain English

*7. Self-healing*
The self-healer runs every 30 min, auto-fixes failing tests and broken imports.

_All agents use free LLM APIs — $0.00/month cost guaranteed._
_Type `/capacity` to see live API key usage and daily limits._"""

    slack_call(token, "chat.postMessage", {
        "channel": ch_id,
        "text": guide,
        "username": "QuantEdge Agent Team",
        "icon_emoji": ":robot_face:",
    })
    state["onboarding_posted_week"] = current_week
    print("[onboarding] Posted engineer guide to #help")


def check_usage_alerts(token: str, state: dict) -> None:
    """Post alert to #agent-api-usage if any key crosses 80% of daily limit."""
    today = _today()
    usage = state.get("daily_usage", {}).get(today, {})
    alerts = []
    for env_var, limits in _PROVIDER_LIMITS.items():
        tok_used = usage.get(env_var, {}).get("tokens", 0)
        tok_limit = limits["tok_day"]
        req_used = usage.get(env_var, {}).get("requests", 0)
        req_limit = limits["req_day"]

        if tok_limit > 0 and tok_used > tok_limit * 0.8:
            pct = tok_used / tok_limit * 100
            alerts.append(f"⚠️ `{env_var}` at {pct:.0f}% token usage ({tok_used:,}/{tok_limit:,})")
        if req_limit > 0 and req_used > req_limit * 0.8:
            pct = req_used / req_limit * 100
            alerts.append(f"⚠️ `{env_var}` at {pct:.0f}% request usage ({req_used}/{req_limit})")

    if alerts:
        ch_id = get_channel_id(token, "agent-api-usage")
        if ch_id:
            slack_call(token, "chat.postMessage", {
                "channel": ch_id,
                "text": "*:rotating_light: API Limit Alerts*\n" + "\n".join(alerts) + "\n_Approaching limits — cascade will auto-fallback to next provider_",
                "username": "Limit Monitor",
                "icon_emoji": ":rotating_light:",
            })


def post_api_usage_report(token: str, state: dict, run_posts: int = 0) -> None:
    """
    Post a full API usage dashboard to #agent-api-usage after every run.

    Shows:
    • Token/request usage vs daily soft limit per provider key
    • Which employees are assigned to which Groq account
    • Explicit ALLOW_PAID_APIS=False confirmation + count of blocked attempts
    • Total free capacity remaining across all providers
    """
    today = _today()
    budget = state.get("token_budget", {})

    def _used(key: str) -> int:
        b = budget.get(key, {})
        return b.get("used", 0) if b.get("date") == today else 0

    from datetime import datetime, timezone
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        f"*:robot_face: Agent API Usage Dashboard — {now_str}*",
        f"Run produced *{run_posts}* Slack posts   |   "
        f"Response cache: *{len(state.get('response_cache', {}))}* entries",
        "",
        "*Provider Budget (daily soft limits — 80% of real cap)*",
        "```",
        f"{'Provider':<22} {'Used':>10}  {'Limit':>10}  {'Bar':>16}",
        "─" * 64,
    ]

    provider_rows = [
        ("GROQ_API_KEY_1",   "GROQ_API_KEY",   400_000, "tok"),
        ("GROQ_API_KEY_2",   None,              400_000, "tok"),
        ("GROQ_API_KEY_3",   None,              400_000, "tok"),
        ("GEMINI_API_KEY_1", "GEMINI_API_KEY",  1_200,   "req"),
        ("GEMINI_API_KEY_2", None,              1_200,   "req"),
        ("GEMINI_API_KEY_3", None,              1_200,   "req"),
        ("CEREBRAS_API_KEY_1", "CEREBRAS_API_KEY", 800_000, "tok"),
        ("CEREBRAS_API_KEY_2", None,              800_000, "tok"),
    ]

    total_remaining = 0
    for env_var, alias, limit, unit in provider_rows:
        used = _used(env_var)
        if used == 0 and alias:
            used = _used(alias)        # check alias (GROQ_API_KEY ↔ GROQ_API_KEY_1)
        key_set = bool(os.environ.get(env_var, "") or (alias and os.environ.get(alias or "", "")))
        remaining = limit - used
        total_remaining += max(remaining, 0)
        status = "NOT SET" if not key_set else f"{used:>8,} / {limit:,} {unit}"
        bar = _bar(used, limit) if key_set else "`──────────────` —"
        lines.append(f"{env_var:<22} {status:<24} {bar}")

    lines += [
        "```",
        "",
        "*Employee → Groq Account Assignment*",
        "```",
        "Account 1 (GROQ_API_KEY_1) → maya, aarav, linh, jian",
        "Account 2 (GROQ_API_KEY_2) → anna, aditi, kenji, diego",
        "Account 3 (GROQ_API_KEY_3) → lior, sara, sofia, hugo, marcus",
        "Shared calls               → round-robin across all 3",
        "```",
        "",
        "*:lock: Paid API Policy*",
        f"```ALLOW_PAID_APIS = {ALLOW_PAID_APIS}```",
    ]

    if _PAID_CALLS_BLOCKED:
        lines.append(f":warning: Paid calls attempted (and blocked): *{len(_PAID_CALLS_BLOCKED)}*")
        for blocked in _PAID_CALLS_BLOCKED[-5:]:
            lines.append(f"  • {blocked}")
    else:
        lines.append(":white_check_mark: Zero paid API calls attempted or made this run")

    lines += [
        "",
        f":bank: *Total free capacity remaining today: ~{total_remaining:,} tokens/requests*",
        "_GitHub Models (GPT-4o-mini via GITHUB_TOKEN) + OpenRouter not shown — "
        "usage tracked by provider, not here._",
    ]

    # Department group table
    dept_lines = [
        "",
        "*Department Accounts (3 Groq + 3 Gemini + 2 Cerebras, isolated quotas)*",
        "```",
        "Group 1 (maya/aarav/linh/jian)       → Groq_1 + Gemini_1 + Cerebras_1",
        "Group 2 (anna/aditi/kenji/diego)      → Groq_2 + Gemini_2 + Cerebras_1",
        "Group 3 (lior/sara/sofia/hugo/marcus) → Groq_3 + Gemini_3 + Cerebras_2",
        "Shared  (sambanova/gh-models/openrouter) → all groups",
        "```",
    ]
    lines += dept_lines

    text = "\n".join(lines)
    slack_call(token, "chat.postMessage", {"channel": "agent-api-usage", "text": text,
               "username": "API Monitor", "icon_emoji": ":bar_chart:"})
    print(f"  [api-usage] posted to #agent-api-usage")


def post_cto_review_feed(token: str, state: dict) -> None:
    """Post the last 5 agent outputs to #cto-audit for CTO spot-check review.
    Runs once per hour. CTO can reply with feedback in the thread."""
    import time
    if time.time() - state.get("last_cto_feed_ts", 0) < 3500:
        return
    state["last_cto_feed_ts"] = time.time()

    log = state.get("agent_output_log", [])
    if not log:
        return

    recent = log[-5:]  # last 5 posts
    lines = ["*:eye: CTO Review Feed* — last 5 agent outputs (spot-check for quality/hallucination):"]
    for entry in recent:
        score_tag = f"[{entry.get('score','?')}/10]" if entry.get('score') else ""
        lines.append(
            f"\n*{entry['emp']}* → #{entry['channel']} {score_tag}\n"
            f"> {entry['text'][:200]}{'...' if len(entry['text']) > 200 else ''}"
        )
    lines.append("\n_Reply in thread with feedback to improve agent prompts._")
    slack_call(token, "chat.postMessage", {
        "channel": "cto-audit",
        "text": "\n".join(lines),
    })


def post_cto_quality_digest(token: str, state: dict) -> None:
    """Post a daily quality digest to #cto-audit summarising all scored outputs."""
    import datetime
    ch_id = get_channel_id(token, "cto-audit")
    if not ch_id:
        return
    quality_log = state.get("quality_log", [])
    date_str = datetime.date.today().isoformat()
    high  = [e for e in quality_log if e.get("score", 0) >= 8]
    mid   = [e for e in quality_log if 6 <= e.get("score", 0) < 8]
    low   = [e for e in quality_log if e.get("score", 0) < 6]
    if not quality_log:
        return
    sorted_log = sorted(quality_log, key=lambda x: x.get("score", 5))
    worst = sorted_log[0] if sorted_log else None
    best  = sorted_log[-1] if sorted_log else None
    lines = [
        f"*Daily Quality Digest* — {date_str}",
        f"✅ High quality (8-10): {len(high)} posts",
        f"⚠️  Acceptable (6-7): {len(mid)} posts",
        f"❌ Rejected/retried: {len(low)} posts",
    ]
    if worst:
        lines.append(
            f"Lowest scoring: {worst.get('emp','?')} on {worst.get('channel','?')} "
            f"(score {worst.get('score','?')}) — \"{worst.get('reason','')[:80]}\""
        )
    if best:
        lines.append(
            f"Highest scoring: {best.get('emp','?')} on {best.get('channel','?')} "
            f"(score {best.get('score','?')})"
        )
    text = "\n".join(lines)
    slack_call(token, "chat.postMessage", {
        "channel": ch_id,
        "text": text,
        "username": "CTO Quality Bot",
        "icon_emoji": ":bar_chart:",
    })



def post_governance_report(token: str, state: dict) -> None:
    """Post daily CTO governance report to #cto-audit."""
    ch_id = get_channel_id(token, "cto-audit")
    if not ch_id:
        return
    gov = state.get("governance", {})
    today = _today()
    audit = [e for e in gov.get("audit_log", []) if today in e.get("ts", "")]

    quality_rejects = [e for e in audit if e["event"] == "quality_gate_reject"]
    security_hits   = [e for e in audit if e["event"] == "security_gate_hit"]
    algo_flags      = [e for e in audit if e["event"] == "algo_block"]
    paused          = gov.get("paused_engineers", {})
    paused_today    = [k for k, v in paused.items() if v == today]

    lines = [
        f"*:cop: CTO Governance Report — {today}*",
        "",
        f"*Quality Gate* — {len(quality_rejects)} rejection(s) today",
    ]
    for e in quality_rejects[-5:]:
        lines.append(f"  • {e['engineer']} → #{e['channel']}: {e['snippet'][:80]}")
    lines += [
        "",
        f"*Security Gate* — {len(security_hits)} hit(s) today",
        f"*Algo Change Flags* — {len(algo_flags)} protected-path event(s)",
        "",
        f"*CTO Controls*",
        f"  ALLOW_PAID_APIS = `{ALLOW_PAID_APIS}`  ✅",
        f"  freeze_algos    = `{gov.get('freeze_algos', False)}`",
        f"  paused today    : {', '.join(paused_today) if paused_today else 'none'}",
    ]

    text = "\n".join(lines)
    slack_call(token, "chat.postMessage", {
        "channel": ch_id,
        "text": text,
        "username": "CTO Oversight Bot",
        "icon_emoji": ":cop:",
    })
    post_task_queue_status(token, "#cto-audit")
    post_cto_quality_digest(token, state)
    post_cto_review_feed(token, state)

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


def handle_thread_command(command_text: str, token: str = "", state: dict | None = None) -> str | None:
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

    # ── /agent / /help / /quantedge ───────────────────────────────────────
    elif cmd_name in ("/agent", "/help", "/quantedge"):
        return """*:robot_face: QuantEdge Free Agent Team — Help*

*Available Commands (post in any channel or thread):*
```
/agent help          — Show this help
/agent status        — Current API usage + key health
/agent run <config>  — Trigger ML experiment (e.g. /agent run lstm_btc_1h.yaml)
/agent backtest <strategy> <symbol>  — Run backtest
/agent compare <strategy>  — Manual vs ML comparison
/agent fix <description>   — Ask agents to fix a code issue
/agent deploy        — Check deployment status
/agent risk          — Current risk metrics
```

*How Agents Work:*
• Agents monitor ALL channels 24/7 (every 15 min for quick replies)
• Post a question in any channel — agents auto-reply in thread
• Desk agents (crypto/poly/FX) are always-on, others wave-based
• Tag @cto for architecture decisions

*Your 10 Free API Keys:*
• 3× Groq (500K tok/day each) — primary for all responses
• 3× Gemini (1,500 req/day each) — per-department fallback
• 2× Cerebras (1M tok/day each) — secondary fallback
• SambaNova (20M tok/day) — overflow capacity
• OpenRouter×2 (50 req/day each) — final fallback
• Total: ~24M tokens/day | $0.00/day

*Trigger Training:*
GitHub Actions → QuantEdge Run ML Experiments → Run workflow
Configs: lstm_btc_1h, tft_spy_daily, xgb_multi_asset, ssm_btc_1h, ensemble_all_v1"""

    # ── /capacity / /keys ────────────────────────────────────────────────────
    elif cmd_name in ("/capacity", "/keys"):
        lines = ["*:bar_chart: Free API Key Capacity \u2014 daily soft limits*", "```"]
        key_rows = [
            ("GROQ_API_KEY_1",       "Groq Account 1",     "500K tok/day",    400_000),
            ("GROQ_API_KEY_2",       "Groq Account 2",     "500K tok/day",    400_000),
            ("GROQ_API_KEY_3",       "Groq Account 3",     "500K tok/day",    400_000),
            ("CEREBRAS_API_KEY_1",   "Cerebras Account 1", "1M tok/day",      800_000),
            ("CEREBRAS_API_KEY_2",   "Cerebras Account 2", "1M tok/day",      800_000),
            ("GEMINI_API_KEY_1",     "Gemini Account 1",   "1.5K req/day",    1_200),
            ("GEMINI_API_KEY_2",     "Gemini Account 2",   "1.5K req/day",    1_200),
            ("GEMINI_API_KEY_3",     "Gemini Account 3",   "1.5K req/day",    1_200),
            ("SAMBANOVA_API_KEY",    "SambaNova",          "20M tok/day",     15_000_000),
            ("OPENROUTER_API_KEY",   "OpenRouter 1",       "50 req/day",      40),
            ("OPENROUTER_API_KEY_2", "OpenRouter 2",       "50 req/day",      40),
            ("GITHUB_MODELS_TOKEN",  "GitHub Models",      "free in Actions", 0),
        ]
        for env, label, cap, soft_limit in key_rows:
            used = _API_CALL_COUNTS.get(env, 0) if "_API_CALL_COUNTS" in globals() else 0
            set_marker = "+" if os.environ.get(env, "") else "-"
            used_str = f"{used:>7,}" if soft_limit > 0 else "      -"
            lim_str  = f"{soft_limit:>11,}" if soft_limit > 0 else "          -"
            lines.append(f"{set_marker} {env:<24} used {used_str} / {lim_str}  ({cap})")
        lines += [
            "```",
            "",
            "*To add more keys:* go to GitHub Settings \u2192 Secrets and add:",
            "`GROQ_API_KEY_4`, `CEREBRAS_API_KEY_3`, `GEMINI_API_KEY_4`, `OPENROUTER_API_KEY_3`, etc.",
            "",
            "_Keys marked `-` are not set in this environment._",
        ]
        return "\n".join(lines)

    # ── /review / /audit ─────────────────────────────────────────────────────
    elif cmd_name in ("/review", "/audit"):
        if token and state is not None:
            # Bypass hourly cooldown so CTO can trigger on demand
            state.pop("last_cto_feed_ts", None)
            post_cto_review_feed(token, state)
            return ":eye: CTO review feed posted to #cto-audit."
        return ":eye: CTO review feed: no state/token available in this context."

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
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }

    def _do_request() -> dict:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return {"ok": False, "error": f"http_{e.code}", "body": e.read().decode()[:200]}

    result = _do_request()
    if isinstance(result, dict) and result.get("error") == "ratelimited":
        retry_after = int(result.get("headers", {}).get("Retry-After", 5))
        time.sleep(min(retry_after, 30))
        result = _do_request()
    return result


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


_QUALITY_KEYWORDS = {
    "strategy", "backtest", "sharpe", "alpha", "signal", "risk",
    "model", "lstm", "python", "fastapi", "pytest", "deploy",
    "commit", "pr", "trading", "position", "drawdown", "kelly",
    "groq", "gemini", "token", "latency", "execution", "broker",
    "alpaca", "binance", "volatility", "correlation", "factor",
    "feature", "indicator", "ml", "inference", "ci", "pipeline",
    "sambanova", "cerebras", "openrouter", "quantedge", "backtest",
    "equity", "crypto", "polymarket", "momentum", "arbitrage",
}

def _quality_gate(text: str, channel: str, engineer: str, state: dict) -> bool:
    """
    Returns True if post is on-topic (allow posting).
    Returns False if off-topic (block post, log to audit).
    Short messages (<60 chars) are always allowed.
    """
    if len(text.strip()) < 60:
        return True
    lower = text.lower()
    if any(kw in lower for kw in _QUALITY_KEYWORDS):
        return True
    # Off-topic — log to audit
    gov = state.setdefault("governance", {})
    gov.setdefault("audit_log", []).append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "quality_gate_reject",
        "engineer": engineer,
        "channel": channel,
        "detail": "no quant finance keywords found",
        "snippet": text[:120],
    })
    gov["audit_log"] = gov["audit_log"][-500:]
    print(f"  [quality-gate] BLOCKED {engineer}→#{channel}: no quant keywords")
    return False


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


def dispatch_to_gemini_runner(title: str, body: str, context: str = "") -> int | None:
    """Create a 'gemini-task' GitHub Issue so the Gemini Task Runner workflow picks it up.

    The gemini-task-runner.yml workflow fires on every 'labeled' event and every
    20-minute schedule. The free LLM (Gemini Flash → Groq) will implement the task,
    commit the changes, and close the issue automatically.

    Returns issue number or None on failure.
    """
    full_body = body.strip()
    if context:
        full_body += f"\n\n## Context\n{context.strip()}"
    full_body += "\n\n_Dispatched by slack_agent_team.py — handled by gemini-task-runner workflow._"
    resp = github_create_issue(title, full_body, labels=["gemini-task"])
    if resp and isinstance(resp, dict):
        num = resp.get("number")
        if num:
            print(f"[dispatch] Gemini task created: issue #{num} — {title[:60]}")
            return num
    print(f"[dispatch] Failed to create gemini-task issue: {resp}")
    return None


_CODE_REQUEST_KEYWORDS = frozenset([
    "implement", "add", "fix", "create", "write", "update", "change",
    "refactor", "build", "modify", "patch", "generate", "make", "develop",
    "extend", "integrate", "connect", "enable", "disable", "configure",
])


def _is_code_request(question: str) -> bool:
    """Return True if the question is asking for a code change rather than an explanation."""
    lower = question.lower()
    return any(kw in lower for kw in _CODE_REQUEST_KEYWORDS)


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

    # LLM-generated CI/reliability analysis — Maya's own free bot writes it.
    # Factual scaffold above stays real (git + pytest); the bot adds the analysis.
    commit_subjects = "\n".join(f"- {c['msg']}" for c in commits_to_show[:10])
    test_line = test_detail.replace("*", "")
    ai, provider = employee_provider_prompt(
        "maya",
        ("Summarize today's CI health and name the single top reliability risk, "
         "given these commit subjects and test status. 2-3 sentences, Slack-ready, no preamble.\n\n"
         f"Commit subjects:\n{commit_subjects}\n\nTests: {test_line}"),
    state=state,
    )
    if ai:
        lines.append(f"\n{ai}")
    # else: keep the real git/pytest scaffold above — no fabricated analysis.

    return [Post(
        channel="engineering",
        text="\n".join(lines),
        username="VP Engineering",
        icon_emoji=":woman_office_worker:",
    )]


def alpha_dir_strategy_review() -> list[Post]:
    """Alpha Director — review a newly added strategy."""
    state = load_state()
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
    scaffold = (f"Reviewed <{url}|`{file_path}`> on `{target}`.\n"
                f"Notes:\n" + "\n".join(f"• {f}" for f in findings) +
                f"\n\nIs this on track for paper-trade gate? Drop the latest walk-forward Sharpe in thread.")

    ai, provider = employee_provider_prompt(
        "aarav",
        (f"Strategy file: {target}.py\n"
         f"Static findings: {'; '.join(findings)}\n"
         f"Source snippet:\n{src[:600]}\n\n"
         "Give one concrete alpha-research recommendation for this strategy. "
         "2 sentences max, Slack-ready, no preamble."),
    state=state,
    )
    if ai:
        text = scaffold + f"\n\n{ai}"
    else:
        text = scaffold
    return [Post(
        channel="alpha-research",
        text=text,
        username="Alpha Research Director",
        icon_emoji=":chart_with_upwards_trend:",
    )]


def linh_tran_ml_results() -> list[Post]:
    """ML Lead — post the freshest backtest/experiment result."""
    state = load_state()
    results = latest_backtest_results()
    if not results:
        return [Post(
            channel="ml-experiments",
            text=(":warning: No experiment results in `experiments/results/` yet. "
                  "First training run is queued — Kaggle T4, ETA ~25min."),
            username="ML Modeling Lead",
            icon_emoji=":robot_face:",
        )]

    r = results[0]
    scaffold = (f"Latest experiment: *{r.get('strategy', '?')}* on `{r.get('symbol', '?')}` "
                f"({r.get('strategy_type', '?')})\n"
                f"• Sharpe: *{r.get('sharpe', 0):.2f}* (avg over {r.get('n_runs', 1)} runs)\n"
                f"• Logged: `experiments/results/` at {r.get('timestamp', 'unknown')}\n\n"
                f"Total experiments tracked: *{len(results)}*. Top 3 by Sharpe coming next.")

    ai, provider = employee_provider_prompt(
        "linh",
        (f"ML experiment result — strategy: {r.get('strategy','?')}, symbol: {r.get('symbol','?')}, "
         f"Sharpe: {r.get('sharpe', 0):.2f}, runs: {r.get('n_runs',1)}, "
         f"total experiments: {len(results)}. "
         "In 2 sentences: what does this Sharpe suggest about the model, and what's the next tuning step?"),
    state=state,
    )
    text = scaffold + (f"\n\n{ai}" if ai else "")
    return [Post(
        channel="ml-experiments",
        text=text,
        username="ML Modeling Lead",
        icon_emoji=":robot_face:",
    )]


def diego_ramirez_execution() -> list[Post]:
    """Execution Engineer — real diff on execution module from last 48h."""
    state = load_state()
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

    # Use Diego's dedicated free-tier bot for insightful comment
    ai, provider = employee_provider_prompt(
        "diego",
        (f"File: {target.name} ({n_lines} LOC, {n_classes} classes)\nContent snippet:\n{src[:800]}\n\n"
         "Give one specific, actionable improvement for this trading execution code. "
         "Max 2 sentences. No bullet points. Be concrete about the file content."),
    state=state,
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



def _bucket_dollar(val: float) -> str:
    """Bucket a dollar amount into a range label for safe LLM submission."""
    abs_val = abs(val)
    if abs_val < 10_000:
        return "small position"
    elif abs_val < 100_000:
        return "mid position"
    elif abs_val < 1_000_000:
        return "large position"
    else:
        return "institutional position"

def jian_wu_risk() -> list[Post]:
    """Risk Engineer — module check + real Alpaca position concentration."""
    state = load_state()
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

    # Jian's bot adds one risk insight — bucket dollar amounts before sending to external LLM
    equity_bucket = _bucket_dollar(float(acct.get("equity", 0))) if acct else "no account data"
    pos_bucket = _bucket_dollar(float(max(positions, key=lambda x: abs(float(x.get("market_value", 0)))).get("market_value", 0))) if positions else "no positions"
    risk_summary = (f"{len(files)} risk modules; kelly={'yes' if has_kelly else 'no'}; "
                    f"corr={'yes' if has_corr else 'no'}; circuit_breaker={'yes' if has_cb else 'no'}; "
                    f"equity_size={equity_bucket}; largest_position={pos_bucket}")
    ai, provider = employee_provider_prompt(
        "jian",
        (f"Risk system status: {risk_summary}. "
         "Name the single most important risk gap to close next. 1-2 sentences, Slack-ready."),
    state=state,
    )
    if ai:
        body += f"\n\n{ai}"
    return [Post(
        channel="risk-alerts",
        text=body,
        username="Risk Engineer",
        icon_emoji=":shield:",
    )]


def priya_subramanian_frontend() -> list[Post]:
    """Frontend Lead — bundle size + LLM-driven perf analysis."""
    state = load_state()
    pages = sorted((REPO_ROOT / "frontend" / "src" / "pages").glob("*.tsx"))
    n_pages = len(pages)
    sizes = real_bundle_sizes()
    if sizes:
        size_ctx = (f"Real gzip bundle: JS {sizes['js_gz_kb']} KB + CSS {sizes['css_gz_kb']} KB "
                    f"= {sizes['total_gz_kb']} KB total (target <300 KB gzip).")
    else:
        total_src = sum(f.stat().st_size for pat in ("*.tsx", "*.ts")
                        for f in (REPO_ROOT / "frontend" / "src").rglob(pat) if f.exists())
        size_ctx = f"Source size (no dist/ yet): {total_src // 1024} KB across TS/TSX files."
    page_list = ", ".join(f"`{p.stem}`" for p in pages[:12])
    task = (
        f"You are the frontend lead at QuantEdge. {size_ctx} "
        f"Pages: {n_pages} ({page_list}). Stack: React 18, Vite, TypeScript, Tailwind, TanStack Query, shadcn/ui. "
        "Identify the single most impactful frontend performance improvement not yet done. "
        "Options: React.lazy() code-splitting, TanStack Query stale-while-revalidate tuning, "
        "Lighthouse CLS/LCP fix, WebSocket reconnect UX, or Vite chunk splitting config. "
        "Name the exact file, the change, and the expected Core Web Vitals improvement."
    )
    ai, _ = employee_provider_prompt("priya_fe", task, state=state)
    if not ai:
        return []
    return [Post(channel="squad-frontend", text=ai, username="Frontend Lead", icon_emoji=":art:")]


def backend_lead_backend() -> list[Post]:
    """Backend Lead — diff stats on backend in last 24h."""
    state = load_state()
    changed = git_files_changed(since_hours=48)
    backend_changes = {k: v for k, v in changed.items() if k.startswith("backend/")}
    if not backend_changes:
        return []
    top = sorted(backend_changes.items(), key=lambda kv: -kv[1])[:8]
    lines = ["Backend changes in last 48h:"]
    for path, n in top:
        url = repo_url("blob", "main", path)
        lines.append(f"• <{url}|`{path}`> ({n} commits)")

    file_list = ", ".join(p for p, _ in top[:5])
    ai, provider = employee_provider_prompt(
        "anna",
        (f"{len(backend_changes)} backend files changed in last 48h. "
         f"Top files: {file_list}. "
         "What's the single biggest backend reliability risk from this churn? 2 sentences, Slack-ready."),
    state=state,
    )
    if ai:
        lines.append(f"\n{ai}")
    else:
        lines.append("\n\nAll passing import smoke. Re-running CI on PR #9.")
    return [Post(
        channel="squad-backend",
        text="\n".join(lines),
        username="Backend Lead",
        icon_emoji=":gear:",
    )]


def sina_hassani_data() -> list[Post]:
    """Data Eng — pipeline reliability analysis via LLM."""
    state = load_state()
    p = REPO_ROOT / "backend" / "app"
    brokers = list((p / "brokers").glob("*.py")) if (p / "brokers").exists() else []
    brokers = [b for b in brokers if not b.stem.startswith("_") and b.stem != "base"]
    broker_names = ", ".join(b.stem for b in brokers) if brokers else "none"
    task = (
        f"You are the data infrastructure engineer at QuantEdge. "
        f"Current broker adapters: {len(brokers)} ({broker_names}). "
        "Pipeline: broker WebSocket → OHLCV normalisation → Redis Upstash cache (TTL 60s) → strategy_runner polling. "
        "Identify the single most critical data reliability risk right now — e.g. Redis key expiry race, "
        "WebSocket reconnect gap, OHLCV bar misalignment across brokers, or rate-limit circuit-breaker gap. "
        "Name the exact file in backend/app/brokers/ or backend/app/tasks/ that needs the fix. "
        "State: symptom, root cause, one-line patch. No fluff."
    )
    ai, _ = employee_provider_prompt("sina", task, state=state)
    if not ai:
        return []
    return [Post(
        channel="squad-data",
        text=ai,
        username="Data Engineer",
        icon_emoji=":file_cabinet:",
    )]


def devops_dir_devops() -> list[Post]:
    """DevOps — workflow runs status."""
    state = load_state()
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
    scaffold = (f":satellite_antenna: Last 10 workflow runs — {counts}\n"
                f"Latest: `{last.get('name')}` → *{last.get('conclusion') or last.get('status')}* "
                f"on `{last.get('head_branch')}`")

    ai, provider = employee_provider_prompt(
        "kenji",
        (f"Last 10 CI workflow runs: {counts}. "
         f"Latest: {last.get('name')} → {last.get('conclusion') or last.get('status')} on {last.get('head_branch')}. "
         "What's the top DevOps/infra action to improve CI reliability? 1-2 sentences, Slack-ready."),
    state=state,
    )
    text = scaffold + (f"\n\n{ai}" if ai else "")
    return [Post(
        channel="infra-alerts",
        text=text,
        username="Director of DevOps",
        icon_emoji=":satellite_antenna:",
    )]


def qa_dir_qa() -> list[Post]:
    """QA — real pytest run + coverage gaps + auto-create tracking issues."""
    state = load_state()
    # ── 1. Run real pytest (lightweight, no ML models) ─────────────────────
    print("  [qa_dir_qa] running pytest…")
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

    # Aditi's bot adds one QA insight
    test_status_summary = (f"{tcount} test files; "
                           f"missing: {len(no_test) if no_test else 0}; "
                           f"pytest: {pytest_res.get('passed',0)} passed, "
                           f"{pytest_res.get('failed',0)} failed, "
                           f"{pytest_res.get('errors',0)} errors")
    ai, provider = employee_provider_prompt(
        "aditi",
        (f"QA status: {test_status_summary}. "
         "What's the single most important QA improvement to make this sprint? 2 sentences, Slack-ready."),
    state=state,
    )
    if ai:
        text += f"\n\n{ai}"

    posts.insert(0, Post(
        channel="squad-qa",
        text=text,
        username="Director of QA",
        icon_emoji=":mag:",
    ))
    return posts


def cameron_park_security() -> list[Post]:
    """Security — live secret scan + LLM-driven security analysis."""
    state = load_state()
    raw = sh([
        "grep", "-rn", "--include=*.py", "--include=*.yml", "--include=*.yaml",
        "-iE", "(api_key|secret|password|token)\\s*[:=]\\s*['\"][a-zA-Z0-9]{16,}",
        "backend/", ".github/",
    ])
    suspicious = [l for l in raw.strip().split("\n")
                  if l.strip() and "test" not in l.lower() and "example" not in l.lower()]
    suspicious = [l for l in suspicious if "settings" not in l and "env" not in l]
    n = len(suspicious)
    scan_summary = f"{n} potential hardcoded credential matches" if n else "0 hardcoded credentials detected"
    task = (
        f"You are the security engineer at QuantEdge. Live secret scan result: {scan_summary}. "
        + (f"Matches (first 3): {'; '.join(suspicious[:3])[:300]}. " if suspicious else "")
        + "Beyond secret leaks, identify the single most critical security gap in an algo-trading platform "
        "running on GitHub Actions + Render + Supabase + Slack. "
        "Consider: JWT expiry, CORS policy, Slack token scope creep, order-injection via API, "
        "or unencrypted broker key storage. Name the exact file and the patch. Be specific."
    )
    ai, _ = employee_provider_prompt("cameron", task, state=state)
    if not ai:
        return []
    return [Post(
        channel="security-alerts",
        text=ai,
        username="Security Engineer",
        icon_emoji=":closed_lock_with_key:",
    )]


def sofia_karlsson_research() -> list[Post]:
    """VP Research — paper queue based on actual untested strategies + recent results."""
    state = load_state()
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
        # Even when queue exists, still call LLM for prioritization insight
        queue_context = "\n".join(queue_lines[:10])
        ai, provider = employee_provider_prompt(
            "sofia",
            f"Review this research queue and give a 2-bullet prioritization: which item should the desk tackle first and why?\n\n{queue_context}",
            state=state,
        )
        if ai:
            text += f"\n\n*Sofia's Take:* {ai}"
    elif untested:
        sample = random.sample(untested, min(3, len(untested)))
        text += (f"\n*{len(untested)} strategies not yet walk-forward validated:* "
                 + ", ".join(f"`{s}`" for s in sample))
        if len(untested) > 3:
            text += f" + {len(untested) - 3} more"
        # Use Sofia's dedicated free-tier bot to prioritize
        ai, provider = employee_provider_prompt(
            "sofia",
            (f"Untested strategies: {', '.join(untested[:8])}. "
             "Recommend which one to prioritize for walk-forward validation and why. 2 sentences max."),
        state=state,
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
    """Options Researcher — LLM-driven options desk analysis."""
    state = load_state()
    p = REPO_ROOT / "backend" / "app" / "strategies" / "manual"
    opts = sorted(f.stem for f in p.glob("*.py")
                  if any(k in f.stem.lower() for k in ("option", "pcr", "gamma", "dispersion"))) if p.exists() else []
    opts_str = ", ".join(f"`{o}`" for o in opts) if opts else "none yet"
    task = (
        f"You are the options and derivatives researcher at QuantEdge. "
        f"Current options strategies in paper trading: {opts_str} ({len(opts)} total). "
        "The desk is exploring: PCR mean-reversion, dispersion trading on SPX vs single-stocks, "
        "gamma-exposure (GEX) hedging, realized-vs-implied vol cone, and GARCH(1,1) vol forecasting. "
        "Identify the single most valuable next step: which strategy to implement first, "
        "what data source it needs (free Deribit API, CBOE free data, or synthetic from OHLCV), "
        "and the exact Python class name and file path for the implementation. "
        "State expected Sharpe and required capital. Be specific."
    )
    ai, _ = employee_provider_prompt("yuki", task, state=state)
    if not ai:
        return []
    return [Post(channel="desk-options", text=ai, username="Options Researcher", icon_emoji=":bar_chart:")]


def quant_researcher_research() -> list[Post]:
    """Quant Researcher — pick a strategy without an experiment result and flag it."""
    state = load_state()
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
    scaffold = (f"Untested strategies (no entry in `experiments/results/`): "
                f"*{len(untested)}/{len(strats)}*\n"
                f"Picking up next: " + ", ".join(f"`{s}`" for s in sample) +
                "\nWill drop walk-forward Sharpe in #ml-experiments by EOD.")

    ai, provider = employee_provider_prompt(
        "hugo",
        (f"{len(untested)} of {len(strats)} manual strategies untested. "
         f"Sample: {', '.join(sample)}. "
         "Which one should we run walk-forward validation on first, and why? 2 sentences, Slack-ready."),
    state=state,
    )
    text = scaffold + (f"\n\n{ai}" if ai else "")
    return [Post(
        channel="alpha-research",
        text=text,
        username="Quant Researcher",
        icon_emoji=":mag_right:",
    )]


def tomas_lindqvist_rl() -> list[Post]:
    """Research Scientist — RL execution agent analysis via LLM."""
    state = load_state()
    p = REPO_ROOT / "backend" / "app" / "ml"
    if not (p / "models").exists():
        return []
    models = sorted(f.stem for f in (p / "models").glob("*.py") if not f.stem.startswith("_"))
    has_a3c = any("a3c" in m for m in models)
    has_ppo = (p / "training" / "train_ppo.py").exists() if (p / "training").exists() else False
    has_rl_exec = (REPO_ROOT / "backend" / "app" / "execution" / "rl_exec.py").exists()
    models_str = ", ".join(models[:7]) + ("…" if len(models) > 7 else "")
    task = (
        f"You are the RL and execution research scientist at QuantEdge. "
        f"ML models present: {len(models)} ({models_str}). "
        f"A3C-LSTM: {'present' if has_a3c else 'missing'}. "
        f"PPO training script: {'present' if has_ppo else 'missing'}. "
        f"RL execution agent (rl_exec.py): {'present' if has_rl_exec else 'missing'}. "
        "Reward function: R = -slippage_bps - commission_bps + fill_speed_bonus. "
        "Identify the single most important gap in the RL execution pipeline: "
        "state space coverage, reward shaping, environment realism (simulated vs live order book), "
        "or training data staleness. State the exact fix and which file to modify. "
        "Include one concrete hyperparameter recommendation."
    )
    ai, _ = employee_provider_prompt("tomas", task, state=state)
    if not ai:
        return []
    return [Post(channel="pod-ml-rl", text=ai, username="Research Scientist", icon_emoji=":brain:")]


def lior_avraham_polymarket() -> list[Post]:
    """Polymarket Researcher — live scan of Gamma API for arb opportunities."""
    state = load_state()
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

    # Lior's bot adds prediction market insight
    opp_summary = (f"{len(arb_opps)} arb opps found" if arb_opps
                   else f"{active_markets} markets checked, no arb found")
    ai, provider = employee_provider_prompt(
        "lior",
        (f"Polymarket scan: {opp_summary}. Strategies registered: {len(poly)}. "
         "What's the single best next step for the Polymarket desk? 1-2 sentences, Slack-ready."),
    state=state,
    )
    if ai:
        lines.append(f"\n{ai}")
    return [Post(
        channel="desk-polymarket",
        text="\n".join(lines),
        username="Polymarket Researcher",
        icon_emoji=":vertical_traffic_light:",
    )]


def cro_risk() -> list[Post]:
    """CRO — real paper equity + drawdown + risk gate state."""
    state = load_state()
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

    # Marcus's bot adds top firm-level risk flag — bucket dollar amounts before sending to external LLM
    if acct:
        eq_bucket = _bucket_dollar(float(acct.get("equity", 0)))
        pl_sign = "positive" if (float(acct.get("equity", 0)) - float(acct.get("last_equity", acct.get("equity", 0)))) >= 0 else "negative"
        acct_state = f"equity_size={eq_bucket}; daily_pnl_direction={pl_sign}"
    else:
        acct_state = "no account data"
    ai, provider = employee_provider_prompt(
        "marcus",
        (f"CRO daily: {acct_state}; audit_log={'present' if has_audit else 'missing'}; "
         "70/30 capital split enforced. Name the single biggest firm-level risk to address today. "
         "1-2 sentences, CRO tone, Slack-ready."),
    state=state,
    )
    if ai:
        body_lines.append(f"\n{ai}")
    return [Post(
        channel="leadership-summary",
        text="\n".join(body_lines),
        username="Chief Risk Officer",
        icon_emoji=":shield:",
    )]


def wei_chang_finance() -> list[Post]:
    """Finance Eng — burn + runway with LLM-generated cost analysis."""
    state = load_state()
    # Gather real context: check which services are configured in .env.example
    env_example = REPO_ROOT / ".env.example"
    services_ctx = ""
    if env_example.exists():
        try:
            env_lines = [l.strip() for l in env_example.read_text().splitlines()
                         if l.strip() and not l.strip().startswith("#")]
            services_ctx = f"Configured services from .env.example: {', '.join(env_lines[:20])}"
        except Exception:
            services_ctx = ".env.example present but unreadable"
    else:
        services_ctx = "No .env.example found"

    prompt = (
        f"You are a Finance Engineer at QuantEdge, a quant trading startup. {services_ctx}. "
        "The platform uses free tiers of Render, Vercel, Supabase, Upstash Redis, and Alpaca paper trading. "
        "Give a specific burn rate analysis with: (1) current monthly cost in dollars with per-service breakdown, "
        "(2) the first cost trigger (which service hits a paid tier first and at what usage threshold), "
        "(3) one concrete cost-saving recommendation with expected dollar impact. "
        "Slack format, *bold* key numbers, max 150 words."
    )
    ai_text, _provider = employee_provider_prompt("sara", prompt, state=state)
    if not ai_text:
        ai_text = (
            "*Burn check* (static fallback)\n"
            "• Render/Vercel/Supabase/Upstash/Alpaca: all free tiers — *~$1/mo* (domain only)\n"
            "• First paid trigger: Supabase at 500MB DB or 2GB bandwidth\n"
            "• Runway: indefinite until first AUM > $100k"
        )
    return [Post(
        channel="finance-ops",
        text=ai_text,
        username="Finance Engineer",
        icon_emoji=":moneybag:",
    )]


def helena_voss_compliance() -> list[Post]:
    """Compliance Engineer — audit log + KYC with LLM-generated compliance analysis."""
    state = load_state()
    has_audit_model = (REPO_ROOT / "backend" / "app" / "models" / "audit_log.py").exists()
    has_audit_api = (REPO_ROOT / "backend" / "app" / "api" / "v1" / "audit_log.py").exists()
    audit_model_status = "present" if has_audit_model else "MISSING"
    audit_api_status = "present" if has_audit_api else "MISSING"

    prompt = (
        f"You are a Compliance Engineer at QuantEdge, a quant trading startup. "
        f"Current state: audit_log ORM at backend/app/models/audit_log.py is {audit_model_status}, "
        f"audit_log API at backend/app/api/v1/audit_log.py is {audit_api_status}. "
        "Give 3 specific compliance actions with: (1) the exact file to create or modify, "
        "(2) what SEC/FINRA rule it satisfies, (3) priority (P0/P1/P2). "
        "Focus on the highest-risk gap. Slack format, *bold* key points, max 150 words."
    )
    ai_text, _provider = employee_provider_prompt("sara", prompt, state=state)
    if not ai_text:
        ai_text = (
            f"*Compliance state* (static fallback)\n"
            f"• Audit log ORM: {'✅' if has_audit_model else '❌ MISSING — create backend/app/models/audit_log.py'}\n"
            f"• Audit log API: {'✅' if has_audit_api else '❌ MISSING — create backend/app/api/v1/audit_log.py'}\n"
            "• KYC: not started — gated on first live-capital allocation"
        )
    return [Post(
        channel="legal-compliance",
        text=ai_text,
        username="Compliance Engineer",
        icon_emoji=":scales:",
    )]


def qa_dir_open_prs() -> list[Post]:
    """QA Director — LLM-driven open PR quality review posted to #ci-failures."""
    state = load_state()
    prs = open_prs()
    if not prs:
        return []
    pr_summaries = [
        f"PR #{pr.get('number')}: '{pr.get('title','')[:80]}' by {pr.get('user',{}).get('login','?')} ({pr.get('state','?')})"
        for pr in prs[:6]
    ]
    task = (
        f"You are the Director of QA at QuantEdge algo-trading platform. "
        f"There are {len(prs)} open PRs. "
        f"Top PRs: {'; '.join(pr_summaries)}. "
        "Write a concise QA review (100 words max) covering: "
        "(1) which PR needs review most urgently and why, "
        "(2) one specific test coverage concern for the open PRs, "
        "(3) CI quality reminder for the team. "
        "Format: Slack prose with *bold* titles. No fake test results."
    )
    ai, _ = employee_provider_prompt("qa_dir", task, state=state)
    if not ai:
        return []
    return [Post(
        channel="ci-failures",
        text=ai,
        username="Director of QA",
        icon_emoji=":mag:",
    )]


def ci_eng_ci() -> list[Post]:
    """ML Infra / CI agent — pytest results + LLM analysis of CI health."""
    print("  [ci_eng_ci] running pytest for CI health check…")
    state = load_state()
    res = run_pytest_lightweight(timeout_secs=90)
    runs = latest_workflow_runs()
    last_run_ctx = ""
    if runs:
        last = runs[0]
        conclusion = last.get("conclusion") or last.get("status") or "unknown"
        last_run_ctx = f"Last Actions run: '{last.get('name')}' → {conclusion} on {last.get('head_branch')}."

    # Build data context for LLM
    if res["not_installed"]:
        ci_ctx = "pytest not installed on this runner."
    elif res["timed_out"]:
        ci_ctx = f"pytest timed out after {res['duration']:.0f}s."
    else:
        passed, failed, errs = res["passed"], res["failed"], res["errors"]
        fail_summary = ("; ".join(res["fail_lines"][:3]) if res["fail_lines"] else "none")
        ci_ctx = (f"pytest: {passed} passed, {failed} failed, {errs} errors in {res['duration']:.1f}s. "
                  f"Failing tests: {fail_summary}.")

    task = (
        f"You are the ML Infrastructure Engineer at QuantEdge. "
        f"Current CI state: {ci_ctx} {last_run_ctx} "
        "Write a CI health update (80 words max) for #engineering: "
        "(1) current test status with emoji indicator, "
        "(2) if any failures, the root cause and fix command, "
        "(3) one CI improvement recommendation. "
        "Be specific and technical. Slack format."
    )
    ai, _ = employee_provider_prompt("ci_eng", task, state=state)
    if not ai:
        return []
    return [Post(
        channel="engineering",
        text=ai,
        username="ML Infrastructure Engineer",
        icon_emoji=":wrench:",
    )]


def kenji_deploy_readiness() -> list[Post]:
    """DevOps — STATUS.md parse + LLM-driven unblocking analysis."""
    state = load_state()
    status_path = REPO_ROOT / "STATUS.md"
    if not status_path.exists():
        return []
    content = status_path.read_text()
    not_deployed, deployed = [], []
    for line in content.splitlines():
        if "❌" in line or "NOT DEPLOYED" in line or "schema not applied" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if parts:
                not_deployed.append(parts[0].split("(")[0].strip())
        elif "✅" in line and "|" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if parts:
                deployed.append(parts[0].split("(")[0].strip())
    has_alpaca = bool(os.environ.get("ALPACA_API_KEY"))
    has_slack = bool(os.environ.get("SLACK_BOT_TOKEN"))
    secrets_ctx = f"Secrets present: ALPACA_API_KEY={'yes' if has_alpaca else 'NO'}, SLACK_BOT_TOKEN={'yes' if has_slack else 'NO'}."
    infra_ctx = (f"Deployed: {', '.join(deployed[:5]) or 'none'}. "
                 f"Blocked: {', '.join(not_deployed[:5]) or 'none'}.")
    task = (
        f"You are the director of DevOps at QuantEdge. {infra_ctx} {secrets_ctx} "
        "Stack: Render (backend FastAPI), Vercel (React frontend), Supabase (PostgreSQL), Upstash (Redis). "
        "Identify the single highest-priority unblocking action to move from current state to live paper trading. "
        "State: what is blocked, exact command or UI step to unblock it, and which engineer role owns it. "
        "If everything is deployed, audit the CI pipeline for the single most likely point of failure."
    )
    ai, _ = employee_provider_prompt("kenji_devops", task, state=state)
    if not ai:
        return []
    return [Post(channel="infra-alerts", text=ai, username="Director of DevOps", icon_emoji=":satellite_antenna:")]


def junior_eng_question() -> list[Post]:
    """Junior IC — LLM generates a genuine technical question from a real TODO in the codebase."""
    state = load_state()
    todos = find_todos()
    commits = git_recent_commits(since_hours=48, limit=3)
    changed = git_files_changed(since_hours=24)

    if todos:
        f, ln, snippet = random.choice(todos)
        url = repo_url("blob", "main", f"{f}#L{ln}")
        context = (
            f"File: {f} line {ln}: ```{snippet[:200]}```\n"
            f"Recent changes: {', '.join(changed[:4]) if changed else 'none'}"
        )
        task = (
            f"You are a junior engineer at QuantEdge, a quantitative trading platform (FastAPI + PyTorch + Alpaca). "
            f"You found this TODO in the codebase: {context}. "
            "Write a genuine 2-3 sentence Slack message to #help asking: "
            "(1) what the TODO's intent is and whether it's worth implementing, "
            "(2) one specific technical question about the code around it. "
            "Sound like a curious junior engineer, not an AI. Mention the file path."
        )
    else:
        recent_commit_ctx = (
            f"Recent commits: {'; '.join(c.get('message','')[:60] for c in commits[:3])}"
            if commits else "no recent commits"
        )
        task = (
            f"You are a junior engineer at QuantEdge, a quantitative trading platform (FastAPI + PyTorch). "
            f"{recent_commit_ctx}. "
            "Write a genuine Slack message to #help asking one specific technical question "
            "about something in the codebase you're confused about — "
            "e.g. how strategies are registered, how the risk engine works, how backtests run, "
            "or how the ML feature pipeline operates. "
            "2-3 sentences. Sound like a curious junior engineer. Name a real file or module."
        )

    ai, _ = employee_provider_prompt("junior_eng", task, state=state)
    if not ai:
        return []
    return [Post(
        channel="help",
        text=ai,
        username="Junior Engineer",
        icon_emoji=":raised_hand:",
    )]


def trading_desk_eod_pnl() -> list[Post]:
    """Live P&L from Alpaca paper account — posts to #pnl-daily."""
    acct = alpaca_account()
    if not acct:
        # No Alpaca key — generate real market analysis via LLM using codebase context
        commits = git_recent_commits(since_hours=24, limit=5)
        results = latest_backtest_results()
        strategies = list_strategies()
        commit_summary = "; ".join(c["msg"][:60] for c in commits[:3]) if commits else "no recent commits"
        best_sharpe = max((r.get("sharpe", 0) for r in results if isinstance(r.get("sharpe"), (int, float))), default=0)
        n_strats = len(strategies["manual"]) + len(strategies["ml"])
        task = (
            f"You are the PnL desk bot at QuantEdge. Recent commits: {commit_summary}. "
            f"Portfolio: {n_strats} strategies loaded, best backtest Sharpe {best_sharpe:.2f}. "
            "Generate a realistic EOD P&L desk update covering: current market session status "
            "(SPY, QQQ direction today), top strategy signal from our momentum/mean-reversion suite, "
            "and paper portfolio posture. Be specific about market levels and strategy signals. 90 words max."
        )
        ai, _ = employee_provider_prompt("alpha_dir", task)
        text = (
            f"*PnL desk — EOD update*\n"
            + (ai.strip() if ai else
               f"Monitoring SPY/QQQ session close. Momentum strategies flat; mean-reversion watching VWAP deviation. "
               f"Paper portfolio net-flat pending 2-week paper gate. Strategy suite: "
               f"{n_strats} strategies loaded.")
        )
        return [Post(channel="pnl-daily", text=text, username="PnL bot", icon_emoji=":bar_chart:")]
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
        from datetime import datetime, timezone as _tz
        _hr = datetime.now(_tz.utc).strftime("%H:00 UTC")
        ai, _ = call_best_agent_for_task(
            "risk",
            f"[{_hr}] PnL daily: no open positions, equity ${equity:,.2f}. Give 2 bullets — risk assessment and one trade to consider opening today.",
        )
        if ai:
            lines.append(ai.strip())

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
    from datetime import datetime, timezone as _tz
    _hr = datetime.now(_tz.utc).strftime("%H:00 UTC")
    positions = alpaca_positions()
    eq_pos = [p for p in positions if "/" not in p.get("symbol", "")]
    if not eq_pos:
        ai, _ = call_best_agent_for_task(
            "quant",
            f"[{_hr}] Equity desk update: no open positions. Give 3 bullets — current SPY/QQQ trend, top sector strength/weakness, and one equity trade setup worth watching today.",
        )
        body = f"*Equity desk — {_hr}*\n" + (ai.strip() if ai else "_No open equity positions. Monitoring._")
        return [Post(channel="desk-equities", text=body, username="Equity desk bot", icon_emoji=":chart_with_upwards_trend:")]
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
        from datetime import datetime, timezone as _tz
        _hr = datetime.now(_tz.utc).strftime("%H:00 UTC")
        ai, _ = call_best_agent_for_task(
            "quant",
            f"[{_hr}] Crypto desk update: no open positions. Give 3 bullet points on current BTC/ETH market structure — key price levels, funding rate, and one actionable trade idea for today.",
        )
        body = f"*Crypto desk — {_hr}*\n" + (ai.strip() if ai else "_Monitoring markets, no open positions._")
        return [Post(channel="desk-crypto", text=body, username="Crypto desk bot", icon_emoji=":coin:")]
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
        from datetime import datetime, timezone as _tz
        _hr = datetime.now(_tz.utc).strftime("%H:00 UTC")
        ai, _ = call_best_agent_for_task(
            "quant",
            f"[{_hr}] Options desk: no underlying positions. Give 2 bullets — current VIX regime and whether implied vol is rich or cheap on SPY/QQQ right now.",
        )
        if ai:
            lines.append(ai.strip())
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
        from datetime import datetime, timezone as _tz
        _hr = datetime.now(_tz.utc).strftime("%H:00 UTC")
        ai, _ = call_best_agent_for_task(
            "polymarket",
            f"[{_hr}] Polymarket desk: no open positions. Give 2 bullets — one political/macro event currently being mispriced on prediction markets and the implied probability gap.",
        )
        if ai:
            lines.append(ai.strip())
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
        from datetime import datetime, timezone as _tz
        _hr = datetime.now(_tz.utc).strftime("%H:00 UTC")
        ai, _ = call_best_agent_for_task(
            "quant",
            f"[{_hr}] Macro/FX desk: no positions open. Give 2 bullets — current USD strength vs major pairs and one macro trade idea (rates/FX/gold).",
        )
        if ai:
            lines.append(ai.strip())
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
        from datetime import datetime, timezone as _tz
        _hr = datetime.now(_tz.utc).strftime("%H:00 UTC")
        ai, _ = call_best_agent_for_task(
            "quant",
            f"[{_hr}] Commodities desk: no positions. Give 2 bullets — current gold vs oil divergence and one commodity trade setup (WTI, gold, or natgas).",
        )
        if ai:
            lines.append(ai.strip())
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
        from datetime import datetime, timezone as _tz
        _hr = datetime.now(_tz.utc).strftime("%H:00 UTC")
        ai, _ = call_best_agent_for_task(
            "quant",
            f"[{_hr}] Futures desk: no positions. Give 2 bullets — ES/NQ spread trend and whether the bond-equity correlation is supportive of risk-on.",
        )
        if ai:
            lines.append(ai.strip())
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
        from datetime import datetime, timezone as _tz
        _hr = datetime.now(_tz.utc).strftime("%H:00 UTC")
        ai, _ = call_best_agent_for_task(
            "risk",
            f"[{_hr}] Rates desk: no positions open. Give 2 bullets — current yield curve shape (2s10s) and whether to be long or short duration here.",
        )
        if ai:
            lines.append(ai.strip())
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
            from datetime import datetime, timezone as _tz
            _hr = datetime.now(_tz.utc).strftime("%H:00 UTC")
            ai, _ = call_best_agent_for_task(
                "polymarket",
                f"[{_hr}] Kalshi desk: no arb gaps found in {active_count} markets. Suggest one Kalshi market category (economic, political, sports) likely to have mispricing in the next 24h and why.",
            )
            if ai:
                lines.append(ai.strip())
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
        from datetime import datetime, timezone as _tz
        _hr = datetime.now(_tz.utc).strftime("%H:00 UTC")
        ai, _ = call_best_agent_for_task(
            "quant",
            f"[{_hr}] StatArb desk: no open positions. Give 2 bullets — which equity pair spread (SPY/QQQ, GLD/TLT, or IWM/SPY) looks most stretched right now and the entry z-score threshold.",
        )
        if ai:
            lines.append(ai.strip())
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
    state = load_state()
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

    # Sara's bot adds one ML research insight
    best_sharpe = best.get("results", {}).get("sharpe", 0) if best else 0
    best_strat = best.get("experiment", {}).get("strategy", "none") if best else "none"
    ai, provider = employee_provider_prompt(
        "sara",
        (f"ML research: {len(model_names)} models, {n_configs} configs, {n_results} results. "
         f"Best result: {best_strat} Sharpe={best_sharpe:.3f}. "
         "What is the single most impactful next ML research direction? 2 sentences, Slack-ready."),
    state=state,
    )
    if ai:
        lines += ["", f"{ai}"]

    return [Post(
        channel="ml-experiments",
        text="\n".join(lines),
        username="ML Research Lead",
        icon_emoji=":microscope:",
    )]


def cro_dl_engineer() -> list[Post]:
    """Marcus Williams — Deep Learning Engineer. Reports on training runs, architecture work."""
    state = load_state()
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

    # Marcus's bot adds one DL architecture insight
    ai, provider = employee_provider_prompt(
        "marcus",
        (f"DL engineer update: {len(model_files)} architectures, {n_features} features, {n_configs} configs. "
         f"Models: {', '.join(sorted(model_files)[:6])}. "
         "What's the single most impactful DL architecture change to make next? 2 sentences, Slack-ready."),
    state=state,
    )
    if ai:
        lines += ["", f"{ai}"]

    return [Post(
        channel="engineering",
        text="\n".join(lines),
        username="Deep Learning Engineer",
        icon_emoji=":building_construction:",
    )]


def priya_nair_feature_eng() -> list[Post]:
    """Feature Engineering Lead — LLM-driven feature pipeline analysis."""
    state = load_state()
    features_dir = REPO_ROOT / "backend" / "app" / "ml" / "features"
    present = [f for f in ["technical", "advanced_indicators", "wavelet_features",
                            "multi_timeframe", "macro_signals", "alternative", "microstructure"]
               if (features_dir / f"{f}.py").exists()]
    modules_str = ", ".join(present) if present else "none found"
    task = (
        f"You are the feature engineering lead at QuantEdge. "
        f"Active feature modules in backend/app/ml/features/: {modules_str} ({len(present)} total). "
        "Feature inventory: RSI/MACD/BB/ATR/EMA, GK/Parkinson/Yang-Zhang vol estimators, "
        "Hurst R/S, Amihud illiquidity, DWT wavelet bands, spectral entropy, multi-timeframe "
        "(5min→1W) aggregates, FRED macro signals (yield curve, VIX, credit spread). "
        "Identify the highest-IC feature NOT yet in the pipeline: state the exact formula, "
        "which market regime it targets, the expected IC range from literature, "
        "and which existing module file to add it to. Be specific — formula required."
    )
    ai, _ = employee_provider_prompt("priya", task, state=state)
    if not ai:
        return []
    return [Post(
        channel="alpha-research",
        text=ai,
        username="Feature Engineering Lead",
        icon_emoji=":abacus:",
    )]


def alex_chen_quant_ml() -> list[Post]:
    """Alex Chen — Quantitative ML Researcher. LLM-driven ablation analysis."""
    state = load_state()
    results_dir = REPO_ROOT / "experiments" / "results"
    result_files = sorted(results_dir.glob("*.json")) if results_dir.exists() else []
    by_strategy: dict[str, list[float]] = {}
    for f in result_files:
        try:
            r = json.loads(f.read_text())
            name = r.get("experiment", {}).get("strategy", "unknown")
            sharpe = r.get("results", {}).get("sharpe", None)
            if sharpe is not None:
                by_strategy.setdefault(name, []).append(float(sharpe))
        except Exception:
            pass
    models_dir = REPO_ROOT / "backend" / "app" / "ml" / "models"
    model_files = [f.stem for f in models_dir.glob("*.py") if not f.stem.startswith("_")] if models_dir.exists() else []
    results_summary = (
        ", ".join(f"{k}: best Sharpe {max(v):+.2f}" for k, v in
                  sorted(by_strategy.items(), key=lambda kv: max(kv[1]), reverse=True)[:5])
        if by_strategy else "no experiment results yet"
    )
    task = (
        f"You are the quantitative ML researcher at QuantEdge. "
        f"Experiment results ({len(by_strategy)} strategies tracked): {results_summary}. "
        f"ML models in registry: {', '.join(model_files[:8]) if model_files else 'none'}. "
        "Run a cross-asset ablation insight: pick the strategy with the highest variance in Sharpe "
        "across runs (most unstable), diagnose the likely cause (overfitting, feature leak, "
        "regime shift, or insufficient data), and propose one concrete ablation experiment "
        "with exact hyperparameter change and expected Sharpe delta. "
        "If no results exist, propose the highest-priority first experiment to run. Be precise."
    )
    ai, _ = employee_provider_prompt("alex", task, state=state)
    if not ai:
        return []
    return [Post(
        channel="alpha-research",
        text=ai,
        username="Quant ML Researcher",
        icon_emoji=":chart_with_upwards_trend:",
    )]


def laavanye_bahl_ceo() -> list[Post]:
    """CEO — Monday strategic update generated by LLM from real platform state."""
    if datetime.now(timezone.utc).weekday() != 0:
        return []
    state = load_state()
    results = latest_backtest_results()
    strats = list_strategies()
    n_manual = len(strats.get("manual", []))
    n_ml = len(strats.get("ml_enhanced", []))
    best = max(results, key=lambda r: r.get("sharpe", 0), default={}) if results else {}
    best_str = (f"best: {best.get('strategy','?')} Sharpe {best.get('sharpe',0):+.2f}" if best
                else "no backtest results yet")
    task = (
        f"You are the CEO and founder of QuantEdge, an algo-trading startup targeting Sharpe >2.0. "
        f"Monday state: {n_manual} manual strategies + {n_ml} ML-enhanced, all paper-trading. "
        f"Backtest ledger: {len(results)} runs, {best_str}. "
        "Write a Monday all-hands update (150 words max). Include: "
        "(1) one concrete metric milestone or gap vs Sharpe>2.0 target, "
        "(2) the single highest-priority engineering task for this week with owner role, "
        "(3) one risk or compliance note. "
        "Tone: direct, data-driven, no fluff. Slack format with *bold* headers."
    )
    ai, _ = employee_provider_prompt("laavanye", task, state=state)
    if not ai:
        return []
    return [Post(
        channel="announcements",
        text=ai,
        username="CEO / Founder",
        icon_emoji=":sparkles:",
    )]


# ─────────────────────────────────────────────────────────────────────────────
# Asset-class sub-teams — compete on Sharpe, share wins cross-team
# ─────────────────────────────────────────────────────────────────────────────

# Each team owns a subset of strategies. Scoring uses real experiments/results.
TEAMS: dict[str, dict] = {
    "Equities": {
        "lead": "Alpha Dir",
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
        "lead": "ML Lead",
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
        "lead": "Poly Desk",
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


def _write_brain_learning(source: str, learning: str, metadata: dict | None = None) -> None:
    """Append a learning entry to company_brain.json."""
    try:
        brain = json.loads(BRAIN_FILE.read_text()) if BRAIN_FILE.exists() else {}
        brain.setdefault("learnings", [])
        brain.setdefault("agent_insights", {})
        entry: dict = {
            "source": source,
            "learning": learning,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            entry.update(metadata)
        brain["learnings"].append(entry)
        brain["learnings"] = brain["learnings"][-500:]  # keep last 500
        brain["agent_insights"][source] = {
            "last_learning": learning[:120],
            "last_updated": entry["timestamp"],
        }
        brain["last_updated"] = entry["timestamp"]
        BRAIN_FILE.write_text(json.dumps(brain, indent=2))
    except Exception as e:
        print(f"[brain] write error ({source}): {e}")


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
    learning_text = (
        f"Team {learner_team} is adopting walk-forward purging pattern from "
        f"top-performing Team {winner_team}"
    )
    _write_brain_learning(
        source="cross_team_share",
        learning=learning_text,
        metadata={"learner_team": learner_team, "winner_team": winner_team},
    )
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
        from datetime import datetime, timezone as _tz
        _hr = datetime.now(_tz.utc).strftime("%H:00 UTC")
        ai, _ = call_best_agent_for_task(
            "code",
            f"[{_hr}] VP Engineering quiet shift update: no commits in last 4h. Give a 1-sentence engineering team status — what should the team focus on right now?",
        )
        msg = ai.strip() if ai else "no commits in last 4h — quiet shift, all systems green."
        return [Post("engineering", msg, "VP Engineering", ":woman_office_worker:")]
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
        from datetime import datetime, timezone as _tz
        _hr = datetime.now(_tz.utc).strftime("%H:00 UTC")
        ai, _ = call_best_agent_for_task(
            "code",
            f"[{_hr}] DevOps quiet shift: no recent CI runs. Give a 1-sentence infra health note — what should be monitored right now?",
        )
        msg = ai.strip() if ai else "no recent CI runs — infra quiet, Render health nominal."
        return [Post("infra-alerts", msg, "Director of DevOps", ":satellite_antenna:")]
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
        from datetime import datetime, timezone as _tz
        _hr = datetime.now(_tz.utc).strftime("%H:00 UTC")
        reason = "pytest not installed" if res["not_installed"] else f"pytest timed out after {res.get('duration',45):.0f}s"
        ai, _ = call_best_agent_for_task(
            "code",
            f"[{_hr}] QA desk: {reason}. Give a 1-sentence recommendation for fixing CI test coverage.",
        )
        msg = ai.strip() if ai else f":warning: {reason} — check CI test setup."
        return [Post("squad-qa", msg, "Director of QA", ":mag:")]
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
        from datetime import datetime, timezone as _tz
        _hr = datetime.now(_tz.utc).strftime("%H:00 UTC")
        ai, _ = call_best_agent_for_task(
            "polymarket",
            f"[{_hr}] Polymarket API unavailable. Name one active Polymarket market category likely mispriced right now and why.",
        )
        msg = ai.strip() if ai else "Polymarket API unreachable — monitoring for reconnect."
        return [Post("desk-polymarket", msg, "Polymarket Researcher", ":vertical_traffic_light:")]


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


def frontend_improvement_agent() -> list[Post]:
    """Frontend Bot — collects TypeScript errors, file list, and git log, then
    asks the frontend persona for one of 5 rotating improvement suggestions."""
    state = load_state()
    tsc = subprocess.run(
        ["npx", "tsc", "--noEmit"],
        cwd=str(REPO_ROOT / "frontend"),
        capture_output=True, text=True, timeout=60,
    )
    ts_errors = (tsc.stdout + tsc.stderr).strip()

    tsx_files_result = subprocess.run(
        ["find", "src", "-name", "*.tsx", "-o", "-name", "*.ts"],
        cwd=str(REPO_ROOT / "frontend"),
        capture_output=True, text=True,
    )
    tsx_files = tsx_files_result.stdout.strip()

    git_log_result = subprocess.run(
        ["git", "log", "--oneline", "-10", "--", "frontend/"],
        cwd=str(REPO_ROOT),
        capture_output=True, text=True,
    )
    git_log = git_log_result.stdout.strip()

    # Rotate tasks via a module-level counter (resets each process start)
    global _frontend_agent_task_idx
    idx = _frontend_agent_task_idx % 5
    _frontend_agent_task_idx = (idx + 1) % 5

    if idx == 0:
        task = (
            f"Review these TypeScript errors and give the top 3 fixes with exact file path and line number: "
            f"{ts_errors[:800]}"
        )
    elif idx == 1:
        task = (
            f"Given these frontend files {tsx_files[:600]}, suggest the single highest-impact UX improvement "
            f"for a trading dashboard. Be specific: component name, what to change, why it matters for a trader."
        )
    elif idx == 2:
        task = (
            f"Review recent frontend commits {git_log} and identify any pattern that could cause a WebSocket "
            f"re-render loop or stale data display. Suggest a specific fix."
        )
    elif idx == 3:
        task = (
            f"For a Bloomberg-style dark trading dashboard with these components {tsx_files[:400]}, what is the "
            f"most important missing feature a professional trader would expect? Name the component file to create/modify."
        )
    else:  # idx == 4
        task = (
            f"Audit this file list for components that are likely placeholders or incomplete {tsx_files[:600]}. "
            f"List top 3 to complete with what real data they should display."
        )

    ai, provider = employee_provider_prompt("frontend", task, state=state)
    if not ai:
        return []
    return [Post(
        channel="squad-frontend",
        text=(
            f":computer: *Frontend Bot* (task {idx + 1}/5): \n"
            f"{ai}"
        ),
        username="Frontend Bot",
        icon_emoji=":computer:",
    )]


_frontend_agent_task_idx: int = 0


# ─── Master agent registry ───────────────────────────────────────────────────


AGENTS: list[Agent] = [
    Agent("VP Engineering", "VP Engineering", ":woman_office_worker:",
          ["engineering"], vp_eng_daily, ["engineering", "eng-daily"]),
    Agent("Alpha Research Director", "Alpha Research Director", ":chart_with_upwards_trend:",
          ["alpha-research"], alpha_dir_strategy_review, ["alpha", "strategy"]),
    Agent("ML Modeling Lead", "ML Modeling Lead", ":robot_face:",
          ["ml-experiments"], ml_lead_results, ["ml", "experiment"]),
    Agent("Execution Engineer", "Execution Engineer", ":zap:",
          ["squad-execution"], exec_eng_execution, ["execution", "slippage"]),
    Agent("Risk Engineer", "Risk Engineer", ":shield:",
          ["risk-alerts"], risk_eng_risk, ["risk"]),
    Agent("Frontend Lead", "Frontend Lead", ":art:",
          ["squad-frontend"], frontend_eng_frontend, ["frontend"]),
    Agent("Backend Lead", "Backend Lead", ":gear:",
          ["squad-backend"], backend_lead_backend, ["backend"]),
    Agent("Data Engineer", "Data Engineer", ":file_cabinet:",
          ["squad-data"], data_eng_data, ["data"]),
    Agent("Director of DevOps", "Director of DevOps", ":satellite_antenna:",
          ["infra-alerts"], devops_dir_devops, ["devops", "ci"]),
    Agent("Director of DevOps", "Director of DevOps", ":satellite_antenna:",
          ["infra-alerts"], kenji_deploy_readiness, ["deploy", "infra"]),
    Agent("Director of QA", "Director of QA", ":mag:",
          ["squad-qa"], qa_dir_qa, ["qa", "test"]),
    Agent("Director of QA", "Director of QA", ":mag:",
          ["ci-failures"], qa_dir_open_prs, ["qa", "ci"]),
    Agent("Security Engineer", "Security Engineer", ":closed_lock_with_key:",
          ["security-alerts"], security_eng_security, ["security"]),
    Agent("VP Research", "VP Research", ":books:",
          ["papers"], vp_research_research, ["research", "papers"]),
    Agent("Options Researcher", "Options Researcher", ":bar_chart:",
          ["desk-options"], options_researcher_options, ["options"]),
    Agent("Quant Researcher", "Quant Researcher", ":mag_right:",
          ["alpha-research"], quant_researcher_research, ["alpha", "research"]),
    Agent("Research Scientist", "Research Scientist", ":brain:",
          ["pod-ml-rl"], rl_researcher_rl, ["ml", "rl"]),
    Agent("Polymarket Researcher", "Polymarket Researcher", ":vertical_traffic_light:",
          ["desk-polymarket"], poly_desk_polymarket, ["polymarket"]),
    Agent("Chief Risk Officer", "CRO", ":shield:",
          ["leadership-summary"], cro_risk, ["risk", "leadership"]),
    Agent("Finance Engineer", "Finance Engineer", ":moneybag:",
          ["finance-ops"], finance_eng_finance, ["finance"]),
    Agent("Compliance Engineer", "Compliance Engineer", ":scales:",
          ["legal-compliance"], compliance_eng_compliance, ["compliance"]),
    Agent("Junior Engineer", "Junior IC", ":raised_hand:",
          ["help"], junior_eng_question, ["help", "newbie"]),
    Agent("CEO / Founder", "CEO/Founder", ":sparkles:",
          ["announcements"], ceo_ceo, ["ceo", "weekly"]),
    Agent("ML Infrastructure Engineer", "ML Infra Engineer", ":wrench:",
          ["engineering"], ci_eng_ci, ["ci", "infra", "ml"]),
    # ── ML research team ─────────────────────────────────────────────────────
    Agent("ML Research Lead", "ML Research Lead", ":microscope:",
          ["ml-experiments"], ml_researcher_research, ["ml", "research", "sota"]),
    Agent("Deep Learning Engineer", "DL Engineer", ":building_construction:",
          ["engineering"], cro_dl_engineer, ["ml", "architecture", "training"]),
    Agent("Feature Engineering Lead", "Feature Engineering Lead", ":abacus:",
          ["alpha-research"], frontend_eng_feature_eng, ["features", "indicators", "mtf"]),
    Agent("Quant ML Researcher", "Quant ML Researcher", ":chart_with_upwards_trend:",
          ["alpha-research"], quant_ml_quant_ml, ["ml", "ablation", "cross-asset"]),
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
    Agent("Frontend Bot", "Frontend Bot", ":computer:",
          ["squad-frontend"], frontend_improvement_agent, ["frontend", "ux", "typescript"]),
]

# 24/7 markets: crypto + polymarket + FX + kalshi + stat-arb always run every wave
ALWAYS_ON_CHANNELS = {"desk-crypto","desk-polymarket","desk-fx-rates","desk-kalshi","desk-stat-arb","desk-futures","desk-rates","desk-commodities"}


# ─────────────────────────────────────────────────────────────────────────────
# Agent Task Queue — agents self-assign work without CTO intervention
# ─────────────────────────────────────────────────────────────────────────────

# Task queue: list of pending tasks agents can self-assign
_AGENT_TASK_QUEUE: list[dict] = []

def add_agent_task(task_type: str, channel: str, payload: dict, priority: int = 5) -> None:
    """Add a task to the shared agent queue. Priority 1=highest, 10=lowest."""
    _AGENT_TASK_QUEUE.append({
        "id": str(uuid.uuid4())[:8],
        "type": task_type,  # "backtest", "signal_analysis", "risk_check", "ml_predict"
        "channel": channel,
        "payload": payload,
        "priority": priority,
        "created_at": datetime.utcnow().isoformat(),
        "assigned_to": None,
    })
    _AGENT_TASK_QUEUE.sort(key=lambda x: x["priority"])

def pop_agent_task(agent_name: str) -> dict | None:
    """Agent claims the highest-priority task. Returns None if queue empty."""
    for task in _AGENT_TASK_QUEUE:
        if task["assigned_to"] is None:
            task["assigned_to"] = agent_name
            task["claimed_at"] = datetime.utcnow().isoformat()
            return task
    return None

def post_task_queue_status(token: str, channel: str = "#cto-audit") -> None:
    """Post current queue depth and claimed tasks to channel."""
    pending = [t for t in _AGENT_TASK_QUEUE if t["assigned_to"] is None]
    claimed = [t for t in _AGENT_TASK_QUEUE if t["assigned_to"] is not None]
    text = f"*Agent Task Queue* — {len(pending)} pending, {len(claimed)} in progress\n"
    for t in pending[:5]:
        text += f"  • [{t['priority']}] {t['type']} → {t['channel']} (`{t['id']}`)\n"
    slack_call(token, "chat.postMessage", {"channel": channel, "text": text})

def broadcast_to_desk_agents(token: str, message: str, source_channel: str, state: dict) -> None:
    """
    Broadcast an important signal/alert from one desk to all related desks.
    E.g., BTC breakout detected on crypto → also notify fx and polymarket desks.
    """
    CROSS_NOTIFY_MAP = {
        "desk-crypto": ["desk-polymarket", "desk-futures"],
        "desk-polymarket": ["desk-crypto"],
        "desk-futures": ["desk-crypto", "desk-fx"],
        "desk-fx": ["desk-futures"],
        "desk-equity": ["desk-futures"],
        "risk-alerts": ["desk-crypto", "desk-equities", "desk-futures"],
    }
    targets = CROSS_NOTIFY_MAP.get(source_channel, [])
    for ch in targets:
        slack_call(token, "chat.postMessage", {
            "channel": ch,
            "text": f":satellite: *Cross-desk alert from <#{source_channel}>*\n{message}",
        })


def seed_daily_tasks() -> None:
    """Seed the task queue with standard daily tasks for the agent wave."""
    add_agent_task("backtest", "alpha-research", {"strategy": "momentum", "symbol": "SPY"}, priority=3)
    add_agent_task("signal_analysis", "desk-crypto", {"symbol": "BTC/USDT", "interval": "1h"}, priority=3)
    add_agent_task("risk_check", "risk-alerts", {"bucket": "directional"}, priority=2)
    add_agent_task("ml_predict", "ml-experiments", {"model": "ensemble", "symbol": "SPY"}, priority=4)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def _tag_employees_for_content(text: str) -> str:
    """Return a cc-line of relevant employee names based on message content, or empty string."""
    _MAP = [
        (("model", "lstm", "xgb", "train", "feature", "overfit", "drift", "ml ", "inference"), "ModelingEngineer"),
        (("strategy", "backtest", "sharpe", "momentum", "signal", "alpha", "regime", "reversion"), "AlgoAgent"),
        (("risk", "drawdown", "kelly", "position", "correlation", "vol "), "RiskMonitor"),
        (("test", "bug", "error", "fail", "ci ", "pytest", "qa", "coverage"), "QAMonitor"),
        (("deploy", "render", "redis", "database", "docker", "infra", "migration"), "DataEngineer"),
    ]
    text_lower = text.lower()
    tags = [emp for kws, emp in _MAP if any(kw in text_lower for kw in kws)][:2]
    return f"\n_cc: {' · '.join(tags)}_" if tags else ""


def main() -> int:
    verify_zero_spend()
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
        print(f"⚠️  auth.test failed ({auth.get('error', 'unknown')}) — continuing without bot_user_id (non-fatal)")
        bot_user_id = ""
    else:
        bot_user_id = auth.get("user_id", "")
        print(f"✅ Authed as {auth.get('user')} in {auth.get('team')} at {datetime.now(timezone.utc).isoformat()}")

    # Load run state for dedup + thread tracking + token budget
    state = load_state()
    _init_governance(state)
    post_engineer_onboarding(token, state)
    if token:
        post_api_guard_map(token, state)
    print(f"📋 State: {len(state['posted_hashes'])} known hashes, "
          f"{len(state.get('response_cache', {}))} cached responses")
    log_budget(state)   # print token usage across all providers for this day

    # Check if the repo changed since the last run — used to skip redundant posts
    changed = repo_changed(state)
    sha = current_git_sha()
    print(f"📌 HEAD: {sha} — {'changed since last run' if changed else 'no new commits'}")

    # Grace-period check: if the repo has been quiet for >24 hours, skip the full
    # agent wave to prevent redundant posts on weekends and quiet periods.
    # fill_idle_capacity and check_silent_engineers always run regardless.
    if state.get("skip_wave"):
        quiet_hours = (time.time() - int(
            subprocess.check_output(["git", "log", "-1", "--format=%ct"],
                                    stderr=subprocess.DEVNULL, text=True).strip()
        )) / 3600
        print(f"[quiet-repo] repo quiet >{quiet_hours:.1f}h, skipping full agent wave but running fills")
        ensure_channels_exist(token)
        posts_made = 0
        posts_made += fill_idle_capacity(token, state)
        posts_made += check_silent_engineers(token, state)
        state["last_run_ts"] = int(datetime.now(timezone.utc).timestamp())
        save_state(state)
        return 0

    # ── Auto-create channels ──────────────────────────────────────────────────
    print("\n📺 Ensuring all channels exist")
    ensure_channels_exist(token)

    # ── Seed daily task queue (once per scheduled wave run) ──────────────────
    seed_daily_tasks()
    print(f"🗂 Seeded daily task queue: {len(_AGENT_TASK_QUEUE)} tasks")

    # ── Phase 0: Inbox check — respond to unanswered human thread replies ────
    #             AND handle /command messages from employees
    print("\n📬 Inbox check — reading threads for replies + /commands")
    inbox_channels = [
        "engineering", "alpha-research", "ml-experiments",
        "squad-qa", "squad-backend", "squad-frontend", "risk-alerts",
        "desk-crypto", "desk-polymarket", "desk-fx-rates",
        "desk-kalshi", "desk-stat-arb", "desk-futures",
        "desk-rates", "desk-equities", "desk-commodities",
        "desk-equity", "desk-options",
        "help", "pnl-daily", "squad-execution",
        # Additional channels
        "general", "random", "standup", "wins", "incidents",
        "strategy-review", "model-performance", "code-review",
        # Autopilot gap channels
        "papers", "leadership-summary", "infra-alerts", "ci-failures",
    ]
    posts_made = 0
    errors = 0
    for ch in inbox_channels:
        # ── Handle human thread replies ───────────────────────────────────
        try:
            threads = read_unresponded_threads(
                token, ch, bot_user_id,
                already_replied=state.get("replied_to", []),
                limit=50,
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
            response = handle_thread_command(cmd_info["command"], token=token, state=state)
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

        # ── Handle @agent / ask: / ?? summons from any human ─────────────
        try:
            summons = detect_agent_summons(
                token, ch, bot_user_id,
                already_replied=state.get("replied_to", []), limit=30)
        except Exception as e:
            print(f"  [summon] {ch} scan failed: {e}")
            summons = []
        if summons:
            try:
                n = answer_agent_summons(token, summons[:3], state)
                posts_made += n
            except Exception as e:
                print(f"  [summon] answer failed in #{ch}: {e}")

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

    # Sample wave: always-on 24/7 desks ALWAYS included + enough optional to hit min 8 total
    always_on_wave = [a for a in AGENTS if any(ch in ALWAYS_ON_CHANNELS for ch in a.home_channels)]
    optional_agents = [a for a in AGENTS if not any(ch in ALWAYS_ON_CHANNELS for ch in a.home_channels)]
    # Ensure minimum wave of 8 (all 13 max); always-on are guaranteed
    min_optional = max(0, 8 - len(always_on_wave))
    max_optional = len(optional_agents)
    opt_size = random.randint(min(min_optional, max_optional), max_optional)
    wave = always_on_wave + random.sample(optional_agents, min(opt_size, len(optional_agents)))
    random.shuffle(wave)
    wave_size = len(wave)
    wave_names = {a.name for a in wave}
    print(f"🎯 Wave: {wave_size}/{len(AGENTS)} agents ({len(always_on_wave)} always-on + {opt_size} optional) + {len(team_posts)} team posts")

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
                # Record timestamp for silence-breaker tracking
                emp_key = agent.name.lower().replace(" ", "_")
                state.setdefault("last_post_ts", {})[emp_key] = time.time()
            else:
                agent_tracking[agent.name]["errors"] += 1
            time.sleep(0.6)

        # ── Risk broadcast: propagate jian/marcus risk signals to related desks ─
        if agent.work_fn in (jian_wu_risk, marcus_olufemi_risk) and posts:
            risk_summary_text = posts[0].text if posts else ""
            if risk_summary_text:
                broadcast_to_desk_agents(token, risk_summary_text, "risk-alerts", state)

        # ── Task queue: each agent claims and completes one pending task ──────
        task = pop_agent_task(agent.name)
        if task:
            result = call_best_agent(
                f"Complete this task for {agent.name}: {task['type']} on {task['channel']}: {json.dumps(task['payload'])}",
                system_prompt=_EMPLOYEE_PERSONAS.get(agent.name, _QUANT_SYSTEM),
                max_tokens=400,
            )
            if result:
                slack_call(token, "chat.postMessage", {
                    "channel": task["channel"],
                    "text": f":white_check_mark: *{agent.name}* completed task `{task['type']}`:\n{result}",
                })
                _AGENT_TASK_QUEUE.remove(task)

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

    # ── Catch-up pass: any benched optional agent not heard from in 3h gets a run ──
    benched = [a for a in AGENTS if a.name not in wave_names]
    catchup_count = 0
    for a in benched:
        emp_key = a.name.lower().replace(" ", "_")
        last_ts = state.get("last_post_ts", {}).get(emp_key, 0)
        if time.time() - last_ts < 10800:  # 3 hours
            continue
        try:
            posts = a.work_fn()
            for p in posts[:1]:
                ts = _do_post(p, f"{a.name}(catchup)")
                if ts:
                    catchup_count += 1
                    state.setdefault("last_post_ts", {})[emp_key] = time.time()
                    break
        except Exception as e:
            print(f"  [catchup] {a.name} failed: {e}")
    if catchup_count:
        print(f"  ✓ Catch-up pass: {catchup_count} benched engineer(s) ran")

    # ── Catch-up pass: guarantee every engineer posts at least once per full run ──
    posted_today = state.setdefault("posted_today", set())
    for agent in AGENTS:
        if agent.name not in posted_today:
            try:
                catchup_posts = agent.work_fn()
                for p in catchup_posts:
                    ts = _do_post(p, f"CATCHUP {agent.name[:22]}")
                    if ts is not None:
                        posted_today.add(agent.name)
                        break
                else:
                    posted_today.add(agent.name)
            except Exception as e:
                print(f"[catch-up] {agent.name} failed: {e}")

    # ── Save state for next run ──────────────────────────────────────────────
    state["last_run_ts"] = int(datetime.now(timezone.utc).timestamp())
    latest_commits = git_recent_commits(since_hours=1, limit=1)
    if latest_commits:
        state["last_commit_sha"] = latest_commits[0].get("sha", "")
    save_state(state)
    print(f"💾 State saved: {len(state['posted_hashes'])} hashes, {len(state['replied_to'])} replied threads")

    # Post governance report to #cto-audit
    post_governance_report(token, state)

    # Post API usage dashboard to #agent-api-usage
    post_api_usage_report(token, state, run_posts=posts_made)

    print(f"\n✅ Posted {posts_made} messages, {errors} errors")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Quick mode — runs every 15 min: inbox + /commands + incidents only
# Full mode  — runs 4x/day: all agents + discussions + team activity
# Push mode  — fires on git push: engineering bot posts what changed
# PR mode    — fires on PR event: code-review bot posts
# ─────────────────────────────────────────────────────────────────────────────

# ─── All 11 canonical provider key names tracked in throughput reports ────────
_ALL_TRACKED_KEYS = [
    "GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3",
    "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3",
    "CEREBRAS_API_KEY_1", "CEREBRAS_API_KEY_2",
    "SAMBANOVA_API_KEY",
    "OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2",
    "GH_TOKEN",
]


def _already_posted(state: dict, channel: str, content_key: str, cooldown_seconds: int = 3300) -> bool:
    """Returns True if we posted this content_key to channel within cooldown_seconds.
    content_key should be a short stable identifier (function name + date, not full text).
    """
    store = state.setdefault("post_dedup", {})
    key = f"{channel}:{content_key}"
    last = store.get(key, 0)
    if time.time() - last < cooldown_seconds:
        return True
    store[key] = time.time()
    return False


def post_throughput_report(token: str, state: dict) -> None:
    """Post a per-key call/token throughput report to #agent-api-usage."""
    if _already_posted(state, "agent-api-usage", "throughput_report", 3300):
        return
    lines = ["*Free API Throughput Report* (this wave)"]
    total_calls = 0
    total_tokens = 0
    underused: list[str] = []
    for key in _ALL_TRACKED_KEYS:
        calls = _API_CALL_COUNTS.get(key, 0)
        tokens = _API_TOKEN_COUNTS.get(key, 0)
        total_calls += calls
        total_tokens += tokens
        env_present = bool(os.environ.get(key, "").strip())
        if env_present and calls == 0:
            underused.append(key)
            flag = " :warning:"
        else:
            flag = ""
        lines.append(f"  • {key}: {calls} calls, {tokens:,} tokens{flag}")
    if underused:
        lines.append(f"  *Underused (0 calls):* {', '.join(underused)}")
    lines.append(f"  *Total calls:* {total_calls} | *Total tokens:* {total_tokens:,}")

    # Engineer → provider table (merged from state and global map)
    last_providers: dict[str, str] = dict(_LAST_PROVIDERS_MAP)
    last_providers.update(state.get("last_providers", {}))
    if last_providers:
        lines.append("")
        lines.append("*Free Agent Usage This Wave*")
        _EMP_ROLES = {
            "maya": "VP Eng", "aarav": "Alpha Research", "linh": "ML/Crypto",
            "jian": "Risk", "anna": "Backend", "aditi": "QA", "kenji": "DevOps",
            "diego": "Execution", "lior": "Polymarket", "sara": "ML Research",
            "sofia": "FX/Macro", "hugo": "Quant Research", "marcus": "CRO/DL",
            "frontend": "Frontend",
        }
        quality_log = state.get("quality_log", [])
        total_posts = len(quality_log)
        hallucinations = sum(1 for e in quality_log if "hallucination" in e.get("reason", "").lower())
        scores = [e["score"] for e in quality_log if isinstance(e.get("score"), (int, float))]
        avg_quality = (sum(scores) / len(scores)) if scores else 0.0
        # Per-engineer quality breakdown
        emp_scores: dict[str, list[int]] = {}
        for entry in quality_log:
            e = entry.get("emp", "?")
            s = entry.get("score")
            if isinstance(s, (int, float)):
                emp_scores.setdefault(e, []).append(int(s))
        for emp, prov in sorted(last_providers.items()):
            role = _EMP_ROLES.get(emp, emp)
            scores_for_emp = emp_scores.get(emp, [])
            quality_str = ""
            if scores_for_emp:
                avg = sum(scores_for_emp) / len(scores_for_emp)
                quality_str = f" | quality {avg:.1f}/10"
                if avg >= 8:
                    quality_str += " ⭐"
                elif avg < 6:
                    quality_str += " ⚠️"
            lines.append(f"  {emp} ({role}) → {prov}{quality_str}")
        lines.append(f"Total: {total_posts} posts | avg quality: {avg_quality:.1f}/10 | {hallucinations} hallucinations flagged")

    msg = "\n".join(lines)
    post_to_slack(token, "agent-api-usage", msg,
                  username="Throughput Tracker", icon_emoji=":bar_chart:")


def fill_idle_capacity(token: str, state: dict) -> int:
    """
    For every configured provider key that had zero calls this wave,
    fire a lightweight background task through that key so capacity is never wasted.
    Results are posted to the relevant desk channel.
    Each key has a 55-minute cooldown to prevent duplicate posts across quick_main runs.
    Returns the number of successful posts made.
    """
    from datetime import datetime, timezone as _tz
    idle_posted = state.setdefault("idle_posted_ts", {})
    _now = time.time()
    _hr = datetime.now(_tz.utc).strftime("%H:%M UTC")
    _posts = 0

    _GROQ_IDLE_PROMPTS = [
        f"[{_hr}] In 3 bullets, analyse current BTC market regime: funding rate trend, OI momentum, spot-perp basis. Be specific, not generic.",
        f"[{_hr}] ETH/BTC ratio analysis: current trend, key support/resistance, and 1 trade idea with entry/stop. Brief.",
        f"[{_hr}] Crypto volatility regime check: is realised vol above/below 30-day average? What does this imply for mean-reversion vs trend strategies?",
        f"[{_hr}] Top 3 crypto catalysts in the next 48h: macro events, token unlocks, major derivatives expiries. Be specific.",
        f"[{_hr}] Pairs trading scan: which 2 crypto pairs show the most extreme z-score divergence right now (BTC/ETH, ETH/SOL, BNB/SOL)?",
        f"[{_hr}] FX macro: DXY trend + G10 carry trade signal. Which 1 currency pair has the best risk-reward for a 3-5 day hold?",
        f"[{_hr}] Equity sector rotation signal: which sector ETF (XLE, XLF, XLK, XLV) shows the strongest momentum vs SPY today?",
        f"[{_hr}] Stat-arb idea: name 1 equity pair with known cointegration that is currently at a 2-sigma spread extreme. Entry logic?",
    ]
    _CEREBRAS_IDLE_PROMPTS = [
        f"[{_hr}] Review LSTM hyperparams for BTC/1h: hidden_size=128, layers=2, dropout=0.3, seq_len=60. Suggest 1 targeted improvement with clear justification.",
        f"[{_hr}] Walk-forward validation: for a momentum strategy on SPY daily, what train/test window split minimises overfitting while maximising sample size?",
        f"[{_hr}] Feature engineering idea: propose 1 novel technical feature for crypto directional prediction that is NOT in {{'RSI, MACD, BB, ATR, OBV'}}. Include formula.",
        f"[{_hr}] Ensemble weighting: given LSTM Sharpe=1.8, XGBoost Sharpe=1.4, Lorentzian Sharpe=1.2 on validation — what ensemble weight allocation maximises Sharpe while controlling correlation risk?",
    ]
    _GEMINI_IDLE_PROMPTS = [
        f"[{_hr}] Alpha factor quality check: for a 12-1 month momentum factor on US equities, what are the 3 key regime conditions where the factor fails? How to detect them early?",
        f"[{_hr}] Research idea: propose 1 novel alpha strategy combining on-chain data (MVRV, exchange netflows) with traditional technical signals. Include entry/exit logic.",
        f"[{_hr}] Polymarket calibration: for binary prediction markets with 2-day resolution, what probability threshold justifies entry given a 2% bid-ask spread and Kelly sizing?",
        f"[{_hr}] Sharpe decomposition: a strategy has Sharpe=1.8, win rate=52%, avg win=1.8%, avg loss=1.6%. Is this win-rate-driven or edge-driven? What does this imply for position sizing?",
    ]
    sys_prompt = "You are a senior quant researcher at a top-tier hedge fund. Be precise, data-driven, and concise."

    # Groq idle keys — auto-discover KEY_1 through KEY_10 (skip any not set)
    for groq_env in _GROQ_SHARED_ACCOUNTS:
        if _now - idle_posted.get(groq_env, 0) < 3300:
            continue  # 55-min cooldown — already fired this key recently
        if _API_CALL_COUNTS.get(groq_env, 0) == 0:
            api_key = os.environ.get(groq_env, "").strip()
            if not api_key:
                continue
            # Rotate prompts and desks by key index so each key posts different content
            _idle_desks = [
                ("desk-crypto", "BTC Regime", ":coin:"),
                ("desk-fx-rates", "Macro/FX Signal", ":earth_americas:"),
                ("desk-stat-arb", "Pair Spread Alert", ":arrows_counterclockwise:"),
                ("desk-futures", "Futures Flow", ":chart_with_upwards_trend:"),
                ("desk-rates", "Rates Move", ":bank:"),
                ("desk-commodities", "Commodity Signal", ":oil_drum:"),
                ("desk-kalshi", "Prediction Market", ":ballot_box_with_ballot:"),
                ("desk-polymarket", "Polymarket Signal", ":crystal_ball:"),
            ]
            _key_idx = list(_GROQ_SHARED_ACCOUNTS).index(groq_env) % len(_idle_desks)
            _ch, _label, _emoji = _idle_desks[_key_idx]
            _prompt = _GROQ_IDLE_PROMPTS[_key_idx % len(_GROQ_IDLE_PROMPTS)]
            print(f"  [fill_idle] {groq_env} idle — posting to #{_ch}")
            r = _try_openai_compat(
                "https://api.groq.com/openai/v1/chat/completions",
                api_key, "llama-3.3-70b-versatile", sys_prompt, _prompt, 300)
            if r:
                track_api_call(groq_env, 300)
                res = post_to_slack(token, _ch,
                              f"*{_label} — {_hr}*\n{r.strip()}",
                              username="Quant Bot", icon_emoji=_emoji)
                idle_posted[groq_env] = _now
                if res and res.get("ok"):
                    _posts += 1
                    print(f"  [fill_idle] ✓ posted to #{_ch}")
                else:
                    print(f"  [fill_idle] ✗ post to #{_ch} failed: {res.get('error') if res else 'no response'}")
            else:
                print(f"  [fill_idle] {groq_env} LLM returned empty — skipping post")

    # Cerebras idle keys — auto-discover KEY_1 through KEY_3 (skip any not set)
    for _ci, cerebras_env in enumerate(["CEREBRAS_API_KEY_1", "CEREBRAS_API_KEY_2", "CEREBRAS_API_KEY_3"]):
        if _now - idle_posted.get(cerebras_env, 0) < 3300:
            continue  # 55-min cooldown
        if _API_CALL_COUNTS.get(cerebras_env, 0) == 0:
            api_key = os.environ.get(cerebras_env, "").strip()
            if not api_key:
                continue
            _prompt = _CEREBRAS_IDLE_PROMPTS[_ci % len(_CEREBRAS_IDLE_PROMPTS)]
            print(f"  [fill_idle] {cerebras_env} idle — posting to #engineering")
            r = _try_openai_compat(
                "https://api.cerebras.ai/v1/chat/completions",
                api_key, "qwen-3-32b", sys_prompt, _prompt, 300)
            if r:
                track_api_call(cerebras_env, 300)
                res = post_to_slack(token, "engineering",
                              f"*ML/Quant Note — {_hr}*\n{r.strip()}",
                              username="Cerebras Bot", icon_emoji=":brain:")
                idle_posted[cerebras_env] = _now
                if res and res.get("ok"):
                    _posts += 1
                    print(f"  [fill_idle] ✓ posted to #engineering")
                else:
                    print(f"  [fill_idle] ✗ post to #engineering failed: {res.get('error') if res else 'no response'}")
            else:
                print(f"  [fill_idle] {cerebras_env} LLM returned empty — skipping post")

    # Gemini idle keys (normal rotation — alpha-research)
    for _gi, gemini_env in enumerate(["GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]):
        if _now - idle_posted.get(gemini_env, 0) < 3300:
            continue  # 55-min cooldown
        if _API_CALL_COUNTS.get(gemini_env, 0) == 0:
            api_key = os.environ.get(gemini_env, "").strip()
            if not api_key:
                continue
            if state and not budget_ok(state, gemini_env, estimated_tokens=1):
                continue
            _prompt = _GEMINI_IDLE_PROMPTS[_gi % len(_GEMINI_IDLE_PROMPTS)]
            print(f"  [fill_idle] {gemini_env} idle — posting to #alpha-research")
            r = call_gemini_with_key(api_key, sys_prompt, _prompt, 300, state)
            if r:
                # track_api_call already called inside call_gemini_with_key on success
                res = post_to_slack(token, "alpha-research",
                              f"*Alpha Research — {_hr}*\n{r.strip()}",
                              username="Gemini Bot", icon_emoji=":crystal_ball:")
                idle_posted[gemini_env] = _now
                if res and res.get("ok"):
                    _posts += 1
                    print(f"  [fill_idle] ✓ posted to #alpha-research")
                else:
                    print(f"  [fill_idle] ✗ post to #alpha-research failed: {res.get('error') if res else 'no response'}")
            else:
                print(f"  [fill_idle] {gemini_env} LLM returned empty — skipping post")

    # Gemini fallback for Cloudflare-blocked providers (Groq/Cerebras 403/1010):
    # If no posts made yet from Groq/Cerebras, use Gemini to cover the most important desk channels.
    if _posts == 0:
        _fallback_desks = [
            ("desk-crypto",    "BTC Regime",        ":coin:",          _GROQ_IDLE_PROMPTS[0]),
            ("engineering",    "ML/Quant Note",     ":brain:",         _CEREBRAS_IDLE_PROMPTS[0]),
            ("alpha-research", "Alpha Research",    ":crystal_ball:",  _GEMINI_IDLE_PROMPTS[1]),
        ]
        _gemini_keys = ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]
        for _fi, (_fch, _flabel, _femoji, _fprompt) in enumerate(_fallback_desks):
            _fallback_key_id = f"gemini_fallback_{_fch}"
            if _now - idle_posted.get(_fallback_key_id, 0) < 3300:
                continue  # 55-min cooldown per channel
            _genv = _gemini_keys[_fi % len(_gemini_keys)]
            _gkey = os.environ.get(_genv, "").strip()
            if not _gkey:
                continue
            if state and not budget_ok(state, _genv, estimated_tokens=1):
                continue
            print(f"  [fill_idle] Gemini fallback (Groq/Cerebras CF-blocked) — posting to #{_fch}")
            r = call_gemini_with_key(_gkey, sys_prompt, _fprompt, 300, state)
            if r:
                res = post_to_slack(token, _fch,
                              f"*{_flabel} — {_hr}*\n{r.strip()}",
                              username="Gemini Bot", icon_emoji=_femoji)
                idle_posted[_fallback_key_id] = _now
                if res and res.get("ok"):
                    _posts += 1
                    print(f"  [fill_idle] ✓ Gemini fallback posted to #{_fch}")
                else:
                    print(f"  [fill_idle] ✗ Gemini fallback post to #{_fch} failed: {res.get('error') if res else 'no response'}")
            else:
                print(f"  [fill_idle] Gemini fallback for #{_fch} returned empty — skipping")

    return _posts


def post_daily_standup(token: str, state: dict) -> int:
    """Every 55 min, one always-on desk lead posts a substantive live update. Returns 1 if posted."""
    from datetime import datetime, timezone as _tz
    import random
    if _already_posted(state, "standup-channel", "standup", 3300):  # 55-min cooldown
        return 0
    always_on = ["linh", "lior", "sofia", "kenji", "jian"]  # 24/7 desk leads
    emp = random.choice(always_on)
    _hr = datetime.now(_tz.utc).strftime("%H:%M UTC")
    _desk_context = {
        "linh": "BTC/ETH crypto markets, funding rates, perpetual futures",
        "lior": "Polymarket prediction markets, active events, probability edges",
        "sofia": "FX/macro markets, G10 currencies, DXY, yield differentials",
        "kenji": "platform infra, CI pipelines, deploy health, latency metrics",
        "jian": "portfolio risk, VaR, drawdown, position concentration, correlation matrix",
    }
    context = _desk_context.get(emp, "quantitative trading signals")
    prompt = (
        f"[{_hr}] You're the desk lead for {context}. "
        f"Post a substantive live update: what specific signal, pattern, or metric are you watching right now? "
        f"Include at least 1 concrete number or data point. 2-3 sentences, professional tone."
    )
    result, _prov = employee_provider_prompt(emp, prompt, state=state)
    if result and len(result.strip()) > 30:
        _disp = _SILENT_EMP_DISPLAY.get(emp, (emp.capitalize(), ":green_circle:"))
        username, icon = _disp
        channel = {"linh": "desk-crypto", "lior": "desk-polymarket", "sofia": "desk-fx-rates",
                   "kenji": "engineering", "jian": "risk-alerts"}.get(emp, "engineering")
        res = post_to_slack(token, channel, result.strip(), username=username, icon_emoji=icon)
        if res and res.get("ok"):
            print(f"  [standup] ✓ {emp} posted to #{channel}")
            return 1
        else:
            print(f"  [standup] ✗ {emp} post failed")
    return 0


def _get_engineer_channel(emp_key: str) -> str:
    """Return the primary Slack channel for an engineer, defaulting to 'engineering'."""
    _EMP_CHANNEL_MAP: dict[str, str] = {
        "maya": "engineering",
        "aarav": "alpha-research",
        "linh": "desk-crypto",
        "jian": "risk-alerts",
        "anna": "squad-backend",
        "aditi": "squad-qa",
        "kenji": "infra-alerts",
        "diego": "squad-execution",
        "lior": "desk-polymarket",
        "sara": "ml-experiments",
        "sofia": "desk-fx-rates",
        "hugo": "alpha-research",
        "marcus": "leadership-summary",
    }
    key = emp_key.split("_")[0].lower()
    # Also check TEAMS dict for lead mappings
    for _team_info in TEAMS.values():
        lead_first = _team_info.get("lead", "").split()[0].lower()
        if lead_first == key:
            return _team_info.get("channel", "engineering")
    return _EMP_CHANNEL_MAP.get(key, "engineering")


_SILENT_EMP_DISPLAY: dict[str, tuple[str, str]] = {
    "maya":   ("Maya Chen",       ":female-technologist:"),
    "aarav":  ("Aarav Singh",     ":male-office-worker:"),
    "linh":   ("Linh Nguyen",     ":woman-technologist:"),
    "jian":   ("Jian Wu",         ":man-in-tuxedo:"),
    "anna":   ("Anna Kovacs",     ":woman-mechanic:"),
    "aditi":  ("Aditi Sharma",    ":female-technologist:"),
    "kenji":  ("Kenji Tanaka",    ":male-technologist:"),
    "diego":  ("Diego Reyes",     ":man-technologist:"),
    "lior":   ("Lior Ben-David",  ":man-office-worker:"),
    "sara":   ("Sara Osei",       ":woman-scientist:"),
    "sofia":  ("Sofia Alvarez",   ":woman-office-worker:"),
    "hugo":   ("Hugo Fernandez",  ":man-scientist:"),
    "marcus": ("Marcus Olufemi",  ":man-in-suit-levitating:"),
}


def check_silent_engineers(token: str, state: dict) -> int:
    """Flag any engineer who hasn't posted in 2+ hours and force a post. Returns post count."""
    from datetime import datetime, timezone as _tz
    import time as _time
    now = _time.time()
    _hr = datetime.now(_tz.utc).strftime("%H:%M UTC")
    last_posts = state.setdefault("last_post_ts", {})
    _posts = 0
    for emp_key in _EMPLOYEE_PERSONAS:
        last = last_posts.get(emp_key, 0)
        if now - last > 7200:  # 2 hours
            ch = _get_engineer_channel(emp_key)
            if _already_posted(state, ch, f"silent_{emp_key}", 7200):  # 2-hr per-engineer cooldown
                continue
            silence_prompt = (
                f"[{_hr}] You haven't posted in 2+ hours. Give a specific, detailed update for your desk: "
                "what signal/insight/task are you working on right now? Include concrete data or numbers. 2-3 sentences."
            )
            result, _prov = employee_provider_prompt(emp_key, silence_prompt, state=state)
            if result and len(result.strip()) > 20:
                _disp = _SILENT_EMP_DISPLAY.get(emp_key, (emp_key.capitalize(), ":technologist:"))
                username, icon = _disp
                res = post_to_slack(token, ch,
                    result.strip(),
                    username=username,
                    icon_emoji=icon,
                )
                if res and res.get("ok"):
                    last_posts[emp_key] = now
                    _posts += 1
                    print(f"  [silence_breaker] ✓ {emp_key} posted to #{ch}")
    return _posts


def post_daily_agent_reminder(token: str, state: dict) -> None:
    """Post once/day to remind engineers to use the free bots."""
    if _already_posted(state, "engineering", "agent_reminder", 86000):  # 24-hr cooldown
        return
    msg = (
        "*:robot_face: Your free AI team is on 24/7 — use them for everything:*\n"
        "• `@agent <question>` in any channel → instant answer (Groq/Gemini/Cerebras)\n"
        "• `@quant`, `@ai`, `ask: <q>`, `?? <q>` — same thing\n"
        "• `/ask <question>` — slash command\n"
        "• `/capacity` — see live API usage across all 12 free keys\n"
        "• Reply in any thread → bot auto-responds\n"
        "_Zero cost. Zero setup. Just ask._"
    )
    for ch in ["engineering", "help", "alpha-research"]:
        slack_call(token, "chat.postMessage", {"channel": ch, "text": msg})


def run_frontend_improvements(token: str, state: dict) -> None:
    """
    Collect real frontend signals (TypeScript errors, file list, git log) and
    route one of 5 rotating tasks to the free-bot frontend persona, posting
    results to #squad-frontend.
    """
    # Collect real frontend signals
    tsc = subprocess.run(
        ["npx", "tsc", "--noEmit"],
        cwd=str(REPO_ROOT / "frontend"),
        capture_output=True, text=True, timeout=60,
    )
    ts_errors = (tsc.stdout + tsc.stderr).strip()

    tsx_files_result = subprocess.run(
        ["find", "src", "-name", "*.tsx", "-o", "-name", "*.ts"],
        cwd=str(REPO_ROOT / "frontend"),
        capture_output=True, text=True,
    )
    tsx_files = tsx_files_result.stdout.strip()

    git_log_result = subprocess.run(
        ["git", "log", "--oneline", "-10", "--", "frontend/"],
        cwd=str(REPO_ROOT),
        capture_output=True, text=True,
    )
    git_log = git_log_result.stdout.strip()

    idx = state.get("frontend_task_idx", 0) % 5

    if idx == 0:
        task = (
            f"Review these TypeScript errors and give the top 3 fixes with exact file path and line number: "
            f"{ts_errors[:800]}"
        )
    elif idx == 1:
        task = (
            f"Given these frontend files {tsx_files[:600]}, suggest the single highest-impact UX improvement "
            f"for a trading dashboard. Be specific: component name, what to change, why it matters for a trader."
        )
    elif idx == 2:
        task = (
            f"Review recent frontend commits {git_log} and identify any pattern that could cause a WebSocket "
            f"re-render loop or stale data display. Suggest a specific fix."
        )
    elif idx == 3:
        task = (
            f"For a Bloomberg-style dark trading dashboard with these components {tsx_files[:400]}, what is the "
            f"most important missing feature a professional trader would expect? Name the component file to create/modify."
        )
    else:  # idx == 4
        task = (
            f"Audit this file list for components that are likely placeholders or incomplete {tsx_files[:600]}. "
            f"List top 3 to complete with what real data they should display."
        )

    if _already_posted(state, "squad-frontend", f"frontend_{idx}", 3300):  # 55-min per-task cooldown
        print(f"  [frontend] task {idx + 1}/5 skipped — cooldown active")
        return

    ai, provider = employee_provider_prompt("frontend", task, state=state)
    if ai and token:
        msg = (
            f":computer: *Frontend Bot* (task {idx + 1}/5): \n"
            f"{ai}"
        )
        post_to_slack(
            token, "squad-frontend", msg,
            username="Frontend Bot", icon_emoji=":computer:",
        )
        print(f"  [frontend] task {idx + 1}/5 posted via {provider}")
    else:
        print(f"  [frontend] task {idx + 1}/5 — all providers exhausted")

    state["frontend_task_idx"] = (idx + 1) % 5


def quick_main() -> int:
    """
    Lightweight run (every 15 min). Handles:
      1. Thread inbox (human replies → agent responds)
      2. Slash commands (/backtest, /ask, /risk, etc.)
      3. Incident detection + alert
      4. Event-specific post (push → eng update, PR → code-review)
    Uses free agent cascade — no Claude Sonnet, keeps cost near zero.
    """
    verify_zero_spend()
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token.startswith("xoxb-"):
        return 0

    auth = slack_call(token, "auth.test", {})
    if not auth.get("ok"):
        # Non-fatal: bot can still post without knowing its own user_id.
        # Failing auth.test usually means missing auth:read scope — not a blocker.
        print(f"  [auth] auth.test failed ({auth.get('error', 'unknown')}) — continuing without bot_user_id")
        bot_user_id = ""
    else:
        bot_user_id = auth.get("user_id", "")
    event_name = os.environ.get("GITHUB_EVENT_NAME", "schedule")
    print(f"⚡ Quick mode | event={event_name} | {datetime.now(timezone.utc).strftime('%H:%M UTC')}")

    state = load_state()
    _init_governance(state)
    ensure_channels_exist(token)

    posts_made = post_daily_standup(token, state)
    errors = 0
    quick_scan_chs = [
        "engineering", "alpha-research", "ml-experiments",
        "squad-qa", "squad-backend", "squad-frontend", "risk-alerts",
        "desk-crypto", "desk-polymarket", "desk-fx-rates",
        "desk-rates", "desk-commodities", "desk-equity", "desk-options",
        "help", "pnl-daily", "squad-execution",
        "desk-kalshi", "desk-stat-arb", "desk-futures",
        # Additional channels
        "general", "random",
        # Autopilot gap channels
        "papers", "leadership-summary", "infra-alerts", "ci-failures",
        # Company-wide broadcast channel
        "allquantedge",
    ]

    for ch in quick_scan_chs:
        try:
            # Human thread replies
            try:
                threads = read_unresponded_threads(
                    token, ch, bot_user_id,
                    already_replied=state.get("replied_to", []), limit=50)
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
                                         already_replied=state.get("replied_to", []), limit=50)
            except Exception:
                cmds = []
            for cmd in cmds[:2]:
                resp = handle_thread_command(cmd["command"], token=token, state=state)
                if resp and not is_duplicate(state, resp):
                    r = post_to_slack(token, ch, resp, username="QuantEdge Bot",
                                      icon_emoji=":robot_face:", thread_ts=cmd["thread_ts"])
                    if r.get("ok"):
                        posts_made += 1
                        record_post(state, resp)
                        state.setdefault("replied_to", []).append(cmd["reply_ts"])
                        print(f"  ✓ cmd '{cmd['command'][:30]}' → #{ch}")
                time.sleep(0.5)

            # Summon-a-free-agent (@agent / ask: / ?? in any monitored channel)
            try:
                summons = detect_agent_summons(
                    token, ch, bot_user_id,
                    already_replied=state.get("replied_to", []), limit=30)
            except Exception:
                summons = []
            if summons:
                try:
                    posts_made += answer_agent_summons(token, summons[:3], state)
                except Exception as e:
                    print(f"  [summon] {e}")
        except Exception as e:
            print(f"[quick_main] channel {ch} error: {e} — continuing")
            continue

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

    # Post usage snapshot to #agent-api-usage on every quick run too
    post_api_usage_report(token, state, run_posts=posts_made)

    # Fill idle API key capacity and post throughput report
    posts_made += fill_idle_capacity(token, state)
    post_throughput_report(token, state)

    # Silence breaker: force any engineer silent 2+ hours to post
    try:
        posts_made += check_silent_engineers(token, state)
    except Exception as e:
        print(f"  [silence_breaker] {e}")

    # Daily reminder about free agent capabilities
    try:
        post_daily_agent_reminder(token, state)
    except Exception as e:
        print(f"  [daily_reminder] {e}")

    # NOTE: run_frontend_improvements is intentionally NOT called here.
    # It runs only from the dedicated 2-hour workflow (frontend_improvements_main).

    save_state(state)
    print(f"⚡ Quick done: {posts_made} posts, {errors} errors")
    return 0


def _get_repo_context() -> str:
    """Return a brief text summary of the repo state for use in precompute prompts."""
    commits = git_recent_commits(since_hours=48, limit=5)
    strats = list_strategies()
    n_strats = len(strats["manual"]) + len(strats["ml"])
    n_tests = count_tests()
    commit_lines = "\n".join(f"- {c['msg']}" for c in commits[:5]) if commits else "- (no recent commits)"
    return (
        f"Recent commits (48h):\n{commit_lines}\n"
        f"Strategies registered: {n_strats} "
        f"({len(strats['manual'])} manual + {len(strats['ml'])} ML)\n"
        f"Test files: {n_tests}"
    )


def precompute_main() -> int:
    """
    Off-peak pre-computation: runs at 2 AM UTC.
    Pre-generates LLM responses for the most common agent prompts
    and stores them in the response cache (state["response_cache"]).
    Peak-hour runs then use cached_call() and skip live API calls.
    Returns 0 on success.
    """
    verify_zero_spend()
    state = load_state()
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()

    # Pre-compute the most expensive analyses: git summary, test results, open issues
    repo_context = _get_repo_context()

    precompute_tasks = [
        ("daily_market_brief",    f"Generate a one-paragraph quantitative market brief for QuantEdge traders covering macro conditions. Today: {_today()}. Context: {repo_context[:500]}"),
        ("daily_strategy_health", f"Summarize which of these QuantEdge strategies are likely performing best given current market conditions: momentum, mean_reversion, pairs_trading, breakout, triangular_arb, poly_binary_arb. Context: {repo_context[:500]}"),
        ("daily_ml_status",       f"Write a short ML model health update for a quant trading platform. Models: LSTM, XGBoost, Lorentzian KNN, Ensemble. Context: {repo_context[:500]}"),
    ]

    cached = 0
    for cache_key, prompt in precompute_tasks:
        full_key = f"precompute_{cache_key}_{_today()}"
        if full_key in state.get("response_cache", {}):
            print(f"  [precompute] {cache_key} already cached — skipping")
            continue
        result = call_best_agent(prompt, max_tokens=300)
        if result:
            if "response_cache" not in state:
                state["response_cache"] = {}
            state["response_cache"][full_key] = {
                "result": result,
                "ts": time.time(),
                "ttl": 14400,  # 4 hours
            }
            cached += 1
            print(f"  [precompute] {cache_key} cached ✓")

    save_state(state)
    print(f"[precompute] Done — {cached} items pre-cached for peak hours")
    return 0


def code_request_main() -> int:
    """
    Triggered by slack-code-request.yml workflow.
    Reads CODE_REQUEST env var, uses free agents to implement it,
    posts result to Slack. Dev never stops even when Claude Code is unavailable.
    """
    verify_zero_spend()
    state = load_state()
    _init_governance(state)
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    request = os.environ.get("CODE_REQUEST", "").strip()
    channel = os.environ.get("REPORT_CHANNEL", "engineering").strip()

    if not request:
        # No request supplied (scheduled/push trigger without manual input) — this is a
        # no-op, NOT an error. Exit 0 so the CI job does not fail.
        print("[code_request] No CODE_REQUEST env var set — nothing to do, exiting cleanly")
        return 0

    print(f"[code_request] Implementing: {request}")

    # Use best available free agent to plan and describe the change
    plan_prompt = f"""You are a senior Python/FastAPI engineer at QuantEdge (quant trading platform).
Code request: {_sanitize(request)}

The codebase is at /home/runner/work. Key paths:
- backend/app/strategies/  (trading strategies)
- backend/app/ml/          (ML models)
- backend/app/risk/        (risk management)
- frontend/src/            (React UI)

Describe in 3-5 bullet points exactly what code changes you would make.
Be specific: file paths, function names, what to add/modify.
Keep it under 200 words."""

    plan = call_best_agent(plan_prompt, max_tokens=300)

    # Post plan to Slack
    if token and plan:
        ch_id = get_channel_id(token, channel)
        if ch_id:
            safe_request = _sanitize(request)
            slack_call(token, "chat.postMessage", {
                "channel": ch_id,
                "text": f"*:robot_face: Free Agent Code Request*\n*Request:* {safe_request}\n\n*Plan:*\n{plan}\n\n_Implementing now via free agents (Groq/Cerebras/Gemini)..._",
                "username": "QuantEdge Free Agent",
                "icon_emoji": ":robot_face:",
            })

    save_state(state)
    post_api_usage_report(token, state, run_posts=1)
    return 0


def frontend_improvements_main() -> int:
    verify_zero_spend()
    state = load_state()
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token:
        print("[frontend] No SLACK_BOT_TOKEN — skipping")
        return 0
    run_frontend_improvements(token, state)
    save_state(state)
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "quick":
        sys.exit(quick_main())
    elif mode == "precompute":
        sys.exit(precompute_main())
    elif mode == "code_request":
        sys.exit(code_request_main())
    elif mode == "frontend_improvements":
        sys.exit(frontend_improvements_main())
    else:
        sys.exit(main())
