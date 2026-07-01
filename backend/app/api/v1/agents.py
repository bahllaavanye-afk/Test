"""Agent management, monitoring, chat, and task assignment endpoints."""
from __future__ import annotations
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.models.user import User

# ─── Constants ─────────────────────────────────────────────────────────────

# File and directory names
EXPERIMENTS_DIR = "experiments"
RESULTS_SUBDIR = "results"
ALGO_AGENT_RESULTS_FILE = "algo_agent_results.json"
GITHUB_DIR = ".github"
STATE_SUBDIR = "state"
MEMORY_FILENAME = "agent_memory.json"
SKILL_FILENAME = "skill_library.json"
TASK_FILENAME = "task_registry.json"

# API endpoints and related settings
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_MAX_TOKENS = 600
GROQ_TIMEOUT = 20
DEEPSEEK_TIMEOUT = 25
GEMINI_TIMEOUT = 25

# Messages and static strings
LLM_UNAVAILABLE_MESSAGE = (
    "No LLM available — set GROQ_API_KEY_1, DEEPSEEK_API_KEY_1, or GEMINI_API_KEY_1 in environment."
)
WORKFLOW_FILE_SUFFIX = ".yml"
WORKFLOW_SCHEDULE_PLACEHOLDER = "varies"

router = APIRouter(prefix="/agents", tags=["agents"])

REPO_ROOT = Path(__file__).parents[4]
RESULTS_FILE = REPO_ROOT / EXPERIMENTS_DIR / RESULTS_SUBDIR / ALGO_AGENT_RESULTS_FILE
STATE_DIR = REPO_ROOT / GITHUB_DIR / STATE_SUBDIR
MEMORY_FILE = STATE_DIR / MEMORY_FILENAME
SKILL_FILE = STATE_DIR / SKILL_FILENAME
TASK_FILE = STATE_DIR / TASK_FILENAME

AGENT_ROLES = {
    "continuous_improver":   "Improves Python code quality across backend + scripts",
    "signal_runner":         "Generates trading signals every 5 min, all desks",
    "quick_backtest":        "Runs lightweight backtests, ranks strategies by Sharpe",
    "peer_reviewer":         "Reviews AI agent commits, opens issues for critical bugs",
    "frontend_design":       "Improves React/TypeScript UI components",
    "token_monitor":         "Tracks API usage, posts optimization suggestions",
    "strategy_generator":    "Generates new trading strategy ideas via LLM",
    "free_agent_engineer":   "General purpose: fixes bugs, adds features",
    "desk_trader":           "Paper trades across crypto/equity/polymarket desks",
    "system_watchdog":       "Health checks, self-heals state files every 5 min",
    "ml_trainer":            "Trains and evaluates ML models on historical data",
    "standup_agent":         "Posts daily standups and OKR tracking to Slack",
    "investor_pipeline":     "Tracks investor pipeline, auto-advances stages",
    "run_experiments":       "Runs strategy experiments, saves results to JSON",
    "algo_agent":            "UCB1 bandit exploration of strategy candidates",
    "self_improver":         "Autonomous code quality improver (continuous)",
    "research_scientist":    "Discovers new alpha ideas from research papers",
    "modeling_engineer":     "Monitors model drift and retraining decisions",
}

GITHUB_WORKFLOWS = [
    "continuous-improvement", "collective-learning", "signal-runner",
    "quick-backtest", "system-watchdog", "slack-agent-team",
    "daily-standup", "slack-pulse", "gemini-ml-training",
    "run-experiments-agent", "free-agent-engineer", "strategy-health",
    "peer-review", "agent-health-monitor", "token-usage-monitor",
]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


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


async def _call_llm(messages: list[dict], max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Try Groq → DeepSeek → Gemini for a chat completion."""
    groq_key = _resolve_key("GROQ_API_KEY")
    if groq_key:
        try:
            async with httpx.AsyncClient(timeout=GROQ_TIMEOUT) as client:
                r = await client.post(
                    GROQ_API_URL,
                    headers={"Authorization": f"Bearer {groq_key}"},
                    json={"model": "llama-3.1-8b-instant", "messages": messages, "max_tokens": max_tokens},
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    for key in [_resolve_key("DEEPSEEK_API_KEY"), os.environ.get("DEEPSEEK_API_KEY_2", ""),
                os.environ.get("DEEPSEEK_API_KEY_3", "")]:
        if not key:
            continue
        try:
            async with httpx.AsyncClient(timeout=DEEPSEEK_TIMEOUT) as client:
                r = await client.post(
                    DEEPSEEK_API_URL,
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": "deepseek-chat", "messages": messages, "max_tokens": max_tokens},
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    gemini_key = _resolve_key("GEMINI_API_KEY")
    if gemini_key:
        try:
            prompt = "\n".join(m["content"] for m in messages)
            async with httpx.AsyncClient(timeout=GEMINI_TIMEOUT) as client:
                r = await client.post(
                    f"{GEMINI_API_BASE_URL}/{GEMINI_MODEL}:generateContent?key={gemini_key}",
                    json={
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"maxOutputTokens": max_tokens},
                    },
                )
                if r.status_code == 200:
                    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            pass

    return LLM_UNAVAILABLE_MESSAGE


# ─── Read-only state endpoints ──────────────────────────────────────────────

@router.get("/memory")
async def get_memory(current_user: User = Depends(get_current_user)):
    """Full agent shared memory (agent_memory.json)."""
    mem = _read_json(MEMORY_FILE)
    return {
        "active_agents": mem.get("active_agents", {}),
        "improvement_stats": mem.get("improvement_stats", {}),
        "failure_traces": mem.get("failure_traces", [])[-20:],
        "peer_learnings": mem.get("peer_learnings", [])[-20:],
        "platform_metrics": mem.get("platform_metrics", {}),
        "last_updated": mem.get("last_updated"),
    }


@router.get("/skills")
async def get_skills(current_user: User = Depends(get_current_user)):
    """Shared Voyager skill library."""
    data = _read_json(SKILL_FILE)
    return {
        "skills": data.get("skills", []),
        "total": data.get("total", 0),
        "last_updated": data.get("last_updated"),
    }


@router.get("/tasks")
async def get_tasks(current_user: User = Depends(get_current_user)):
    """Current active task registry."""
    data = _read_json(TASK_FILE)
    return {
        "active": data.get("active", {}),
        "completed_count": len(data.get("completed", [])),
        "last_updated": data.get("last_updated"),
    }


@router.get("/roster")
async def get_roster(current_user: User = Depends(get_current_user)):
    """Full roster of all agents with roles, stats, and last seen."""
    mem = _read_json(MEMORY_FILE)
    stats = mem.get("improvement_stats", {})
    active = mem.get("active_agents", {})

    roster = []
    for name, role in AGENT_ROLES.items():
        agent_stats = stats.get(name, {})
        last_seen = active.get(name, {}).get("last_seen")
        roster.append({
            "name": name,
            "role": role,
            "runs": agent_stats.get("runs", 0),
            "successes": agent_stats.get("successes", 0),
            "last_success": agent_stats.get("last_success"),
            "last_summary": agent_stats.get("last_summary", ""),
            "last_seen": last_seen,
            "is_online": last_seen is not None,
        })
    return roster


@router.get("/workflows")
async def get_workflows(current_user: User = Depends(get_current_user)):
    """List all GitHub Actions workflows."""
    return [
        {"name": w, "file": f"{w}{WORKFLOW_FILE_SUFFIX}", "schedule": WORKFLOW_SCHEDULE_PLACEHOLDER}
        for w in GITHUB_WORKFLOWS
    ]


# ─── Task management ─────────────────────────────────────────────────────────

class TaskCreate(BaseModel):
    tas
# ... (truncated for brevity)