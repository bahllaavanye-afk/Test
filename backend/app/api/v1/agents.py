"""Agent management, monitoring, chat, and task assignment endpoints."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/agents", tags=["agents"])

REPO_ROOT   = Path(__file__).parents[4]
RESULTS_FILE = REPO_ROOT / "experiments" / "results" / "algo_agent_results.json"
STATE_DIR   = REPO_ROOT / ".github" / "state"
MEMORY_FILE = STATE_DIR / "agent_memory.json"
SKILL_FILE  = STATE_DIR / "skill_library.json"
TASK_FILE   = STATE_DIR / "task_registry.json"

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


async def _call_llm(messages: list[dict], max_tokens: int = 600) -> str:
    """Try Groq → DeepSeek → Gemini for a chat completion."""
    groq_key = _resolve_key("GROQ_API_KEY")
    if groq_key:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
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
            async with httpx.AsyncClient(timeout=25) as client:
                r = await client.post(
                    "https://api.deepseek.com/chat/completions",
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
            async with httpx.AsyncClient(timeout=25) as client:
                r = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}",
                    json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                          "generationConfig": {"maxOutputTokens": max_tokens}},
                )
                if r.status_code == 200:
                    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            pass

    return "No LLM available — set GROQ_API_KEY_1, DEEPSEEK_API_KEY_1, or GEMINI_API_KEY_1 in environment."


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
        {"name": w, "file": f"{w}.yml", "schedule": "varies"}
        for w in GITHUB_WORKFLOWS
    ]


# ─── Task management ─────────────────────────────────────────────────────────

class TaskCreate(BaseModel):
    task_id: str
    description: str
    assigned_to: str
    priority: str = "normal"


@router.post("/tasks")
async def create_task(
    body: TaskCreate,
    current_user: User = Depends(get_current_user),
):
    """Assign a task to an agent via task_registry.json."""
    data = _read_json(TASK_FILE)
    data.setdefault("active", {})
    if body.task_id in data["active"]:
        raise HTTPException(400, "Task ID already exists")
    data["active"][body.task_id] = {
        "agent": body.assigned_to,
        "description": body.description,
        "priority": body.priority,
        "claimed_at": datetime.now(UTC).isoformat(),
        "created_by": current_user.email,
    }
    data["last_updated"] = datetime.now(UTC).isoformat()
    _write_json(TASK_FILE, data)
    return {"ok": True, "task_id": body.task_id}


@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
):
    """Cancel / remove a task from the registry."""
    data = _read_json(TASK_FILE)
    if task_id not in data.get("active", {}):
        raise HTTPException(404, "Task not found")
    del data["active"][task_id]
    data["last_updated"] = datetime.now(UTC).isoformat()
    _write_json(TASK_FILE, data)
    return {"ok": True}


# ─── Agent chat ───────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    agent: str
    message: str
    history: list[dict] = []


@router.post("/chat")
async def chat_with_agent(
    body: ChatMessage,
    current_user: User = Depends(get_current_user),
):
    """Chat with any agent. Uses Groq/DeepSeek/Gemini — works even without Claude."""
    role = AGENT_ROLES.get(body.agent, "general-purpose quantitative engineer")
    mem = _read_json(MEMORY_FILE)
    skills = _read_json(SKILL_FILE).get("skills", [])[-8:]
    failures = mem.get("failure_traces", [])[-3:]
    learnings = mem.get("peer_learnings", [])[-5:]

    context_parts = [
        f"You are the **{body.agent}** agent on QuantEdge, an institutional-grade",
        f"quantitative trading platform. Your role: {role}.",
        "",
        "Platform: FastAPI backend + React 18 frontend. ML: PyTorch (LSTM, TFT,",
        "XGBoost, Lorentzian KNN, SSM). Brokers: Alpaca, Binance, Polymarket.",
        "All code is in Python 3.11 or TypeScript. Branch: claude/advanced-trading-bot-d5Lmw.",
        "",
        "Speak as this agent — concise, technical, first-person. Reference actual",
        "files and functions. No disclaimers about being an AI.",
    ]

    if skills:
        context_parts += ["", "KNOWN GOOD PATTERNS:"] + [f"  - {s}" for s in skills]
    if failures:
        context_parts += ["", "RECENT FAILURES TO AVOID:"] + [
            f"  - {f.get('what_failed','')}: {f.get('error','')}" for f in failures
        ]
    if learnings:
        context_parts += ["", "RECENT TEAM LEARNINGS:"] + [f"  - {l}" for l in learnings[-3:]]

    system = "\n".join(context_parts)

    messages = [{"role": "system", "content": system}]
    for h in body.history[-10:]:
        messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    messages.append({"role": "user", "content": body.message})

    reply = await _call_llm(messages, max_tokens=800)
    return {"agent": body.agent, "reply": reply, "timestamp": datetime.now(UTC).isoformat()}


# ─── Original endpoints (kept) ───────────────────────────────────────────────

@router.get("/leaderboard")
async def get_leaderboard(current_user: User = Depends(get_current_user)):
    from app.main import app
    agent = getattr(app.state, "algo_agent", None)
    if agent:
        return agent.get_leaderboard()
    return []


@router.get("/results")
async def get_results(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
):
    if not RESULTS_FILE.exists():
        return []
    try:
        data = json.loads(RESULTS_FILE.read_text())
        return sorted(data, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
    except Exception:
        return []


@router.get("/status")
async def agent_status(current_user: User = Depends(get_current_user)):
    from app.main import app
    algo_agent        = getattr(app.state, "algo_agent", None)
    self_improver     = getattr(app.state, "self_improver", None)
    qa_monitor        = getattr(app.state, "qa_monitor", None)
    research_scientist = getattr(app.state, "research_scientist", None)
    modeling_engineer  = getattr(app.state, "modeling_engineer", None)

    # Free-LLM fleet status — which providers have keys configured
    try:
        from app.tasks.free_llm_router import available_providers, available_keys, get_throughput_report
        llm_providers = available_providers()
        llm_keys      = available_keys()
        llm_throughput = get_throughput_report()
    except Exception:
        llm_providers  = []
        llm_keys       = []
        llm_throughput = []

    return {
        "algo_agent": {
            "running": getattr(algo_agent, "_running", False),
            "total_runs": getattr(algo_agent, "_total_runs", 0),
            "candidates": len(getattr(algo_agent, "_candidates", {})),
            "top_3": algo_agent.get_leaderboard()[:3] if algo_agent else [],
        },
        "self_improver": {
            "running": getattr(self_improver, "_running", False),
            "iteration": getattr(self_improver, "_iteration", 0),
            "llm_guided": len(llm_keys) > 0,
        },
        "qa_monitor": {"running": getattr(qa_monitor, "_running", False)},
        "research_scientist": {
            "running": research_scientist is not None,
            "cycles_completed": getattr(research_scientist, "_cycle", 0),
            "total_findings": len(getattr(research_scientist, "_findings", [])),
        },
        "modeling_engineer": {
            "running": modeling_engineer is not None,
            "cycles_completed": getattr(modeling_engineer, "_cycle", 0),
            "decisions_made": len(getattr(modeling_engineer, "_decisions", [])),
        },
        "free_llm_fleet": {
            "active_providers": llm_providers,
            "total_keys": len(llm_keys),
            "throughput": llm_throughput,
        },
    }


@router.get("/research")
async def get_research_summary(current_user: User = Depends(get_current_user)):
    from app.main import app
    agent = getattr(app.state, "research_scientist", None)
    if not agent:
        return {"error": "ResearchScientist not running", "cycles_completed": 0, "top_ideas": []}
    return agent.get_research_summary()


@router.get("/modeling")
async def get_modeling_summary(current_user: User = Depends(get_current_user)):
    from app.main import app
    agent = getattr(app.state, "modeling_engineer", None)
    if not agent:
        return {"error": "ModelingEngineer not running", "cycles_completed": 0, "models_monitored": []}
    return agent.get_engineering_summary()


@router.get("/code-reviews")
async def get_code_reviews(current_user: User = Depends(get_current_user)):
    """Return latest employee code review grades from docs/agent-reviews/ markdown files."""
    reviews_dir = REPO_ROOT / "docs" / "agent-reviews"
    if not reviews_dir.exists():
        return {"reviews": []}

    reviews: list[dict] = []
    seen_domains: set[str] = set()

    # Sort by modification time descending — newest first
    md_files = sorted(reviews_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)

    for md_path in md_files:
        # Filename pattern: <domain>-<date>-<emp_key>.md
        parts = md_path.stem.split("-")
        if len(parts) < 3:
            continue
        # date is parts[-3] if it matches YYYY-MM-DD, emp_key is last part
        emp_key = parts[-1]
        # date = parts[-3] through parts[-1] minus last
        date_str = "-".join(parts[-4:-1]) if len(parts) >= 4 else parts[-2]
        domain = "-".join(parts[:-4]) if len(parts) > 4 else parts[0]

        # Only keep latest per domain
        if domain in seen_domains:
            continue
        seen_domains.add(domain)

        text = md_path.read_text(errors="replace")

        # Extract grade from header or content
        import re
        grade = "?"
        grade_match = re.search(r"\*\*Grade:\*\*\s*([A-F][+-]?)", text)
        if grade_match:
            grade = grade_match.group(1)
        else:
            g2 = re.search(r"###\s*Overall Grade\s*\n+([A-F][+-]?)", text)
            if g2:
                grade = g2.group(1)

        # Extract top priority
        top_priority = ""
        pq = re.search(r"###\s*Implementation Priority Queue\s*(.*?)(?=\n###|$)", text, re.DOTALL)
        if pq:
            lines = [l.strip() for l in pq.group(1).strip().splitlines() if l.strip()]
            top_priority = lines[0].lstrip("0123456789.-) ") if lines else ""

        # Extract provider from header
        provider = emp_key
        prov_match = re.search(r"\*\*LLM:\*\*\s*(\S+)", text)
        if prov_match:
            provider = prov_match.group(1).rstrip("|").strip()

        reviews.append({
            "domain": domain,
            "employee": emp_key,
            "provider": provider,
            "grade": grade,
            "date": date_str,
            "top_priority": top_priority[:200],
            "filename": md_path.name,
        })

    return {"reviews": reviews}
