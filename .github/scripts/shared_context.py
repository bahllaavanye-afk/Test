"""
Shared Context System — the collective brain for all QuantEdge agents.

Every agent reads this at startup to get:
  - What other agents have learned (Reflexion traces)
  - Best prompts/patterns (Voyager skills)
  - Task registry (what's being worked on)
  - Collective performance metrics

Every agent writes back:
  - What it accomplished
  - What failed and why
  - New skills discovered

This implements a distributed Reflexion + Voyager architecture where all agents
share a single improving context rather than each reinventing the wheel.
"""
from __future__ import annotations
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[2]
STATE_FILE = REPO_ROOT / ".github" / "state" / "agent_memory.json"
SKILL_FILE = REPO_ROOT / ".github" / "state" / "skill_library.json"
TASK_FILE  = REPO_ROOT / ".github" / "state" / "task_registry.json"

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
}


class SharedContext:
    """
    Drop-in context manager for all agents.

    Usage:
        ctx = SharedContext(agent_name="signal_runner")
        ctx.load()
        skills = ctx.get_skills()
        failures = ctx.get_recent_failures(5)
        # ... do work ...
        ctx.record_success("Generated 7 signals", {"desk": "crypto"})
        ctx.record_skill("Always check if Binance API returns 200 before parsing JSON")
        ctx.save()
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._mem: dict = {}
        self._skills: list[str] = []
        self._tasks: dict = {}

    def load(self) -> "SharedContext":
        # Load memory
        try:
            self._mem = json.loads(STATE_FILE.read_text())
        except Exception:
            self._mem = {}

        # Load skills
        try:
            self._skills = json.loads(SKILL_FILE.read_text()).get("skills", [])
        except Exception:
            self._skills = []

        # Load task registry
        try:
            self._tasks = json.loads(TASK_FILE.read_text())
        except Exception:
            self._tasks = {}

        # Register agent as active
        self._mem.setdefault("active_agents", {})
        self._mem["active_agents"][self.agent_name] = {
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "role": AGENT_ROLES.get(self.agent_name, "unknown"),
        }
        return self

    def get_skills(self, max_skills: int = 10) -> list[str]:
        return self._skills[-max_skills:]

    def get_recent_failures(self, n: int = 5) -> list[dict]:
        traces = self._mem.get("failure_traces", [])
        # Get failures relevant to this agent or general failures
        relevant = [t for t in traces if t.get("agent") in (self.agent_name, "all")]
        return (relevant + traces)[-n:]

    def get_peer_learnings(self, max_items: int = 5) -> list[str]:
        """Get lessons learned by other agents."""
        learnings = self._mem.get("peer_learnings", [])
        return learnings[-max_items:]

    def get_active_tasks(self) -> dict:
        return self._tasks.get("active", {})

    def claim_task(self, task_id: str, description: str) -> bool:
        """Claim a task to prevent duplicate work. Returns True if claimed."""
        self._tasks.setdefault("active", {})
        if task_id in self._tasks["active"]:
            return False  # Already claimed
        self._tasks["active"][task_id] = {
            "agent": self.agent_name,
            "description": description,
            "claimed_at": datetime.now(timezone.utc).isoformat(),
        }
        return True

    def release_task(self, task_id: str):
        """Release a claimed task on completion or failure."""
        self._tasks.get("active", {}).pop(task_id, None)

    def record_success(self, summary: str, metadata: dict | None = None):
        """Record what this agent accomplished."""
        now = datetime.now(timezone.utc).isoformat()
        self._mem.setdefault("improvement_stats", {})
        self._mem["improvement_stats"].setdefault(self.agent_name, {"runs": 0, "successes": 0})
        stats = self._mem["improvement_stats"][self.agent_name]
        stats["runs"] = stats.get("runs", 0) + 1
        stats["successes"] = stats.get("successes", 0) + 1
        stats["last_success"] = now
        stats["last_summary"] = summary

        # Share learnings with peer agents
        self._mem.setdefault("peer_learnings", [])
        self._mem["peer_learnings"].append(f"[{self.agent_name} @ {now[:16]}] {summary}")
        self._mem["peer_learnings"] = self._mem["peer_learnings"][-100:]

    def record_failure(self, what_failed: str, error: str, what_to_try_next: str = ""):
        """Record a failure so other agents (and next run) can learn from it."""
        now = datetime.now(timezone.utc).isoformat()
        self._mem.setdefault("failure_traces", [])
        self._mem["failure_traces"].append({
            "agent": self.agent_name,
            "timestamp": now,
            "what_failed": what_failed,
            "error": error[:200],
            "suggestion": what_to_try_next,
        })
        self._mem["failure_traces"] = self._mem["failure_traces"][-200:]

        # Update stats
        self._mem.setdefault("improvement_stats", {})
        self._mem["improvement_stats"].setdefault(self.agent_name, {"runs": 0, "successes": 0})
        self._mem["improvement_stats"][self.agent_name]["runs"] = \
            self._mem["improvement_stats"][self.agent_name].get("runs", 0) + 1

    def record_skill(self, skill: str):
        """Add a new skill to the shared Voyager skill library."""
        if skill not in self._skills:
            self._skills.append(skill)

    def build_prompt_context(self) -> str:
        """Build context string to inject into LLM prompts."""
        parts = []

        # Recent failures (Reflexion)
        failures = self.get_recent_failures(3)
        if failures:
            parts.append("RECENT FAILURES TO AVOID:")
            for f in failures:
                parts.append(f"  - {f.get('what_failed', '')}: {f.get('error', '')} → {f.get('suggestion', '')}")

        # Skills (Voyager)
        skills = self.get_skills(8)
        if skills:
            parts.append("\nKNOWN GOOD PATTERNS:")
            for s in skills:
                parts.append(f"  - {s}")

        # Peer learnings
        learnings = self.get_peer_learnings(3)
        if learnings:
            parts.append("\nWHAT OTHER AGENTS LEARNED:")
            for l in learnings:
                parts.append(f"  - {l}")

        return "\n".join(parts)

    def save(self):
        """Write memory, skills, and task registry back to disk."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Save memory
        self._mem["last_updated"] = datetime.now(timezone.utc).isoformat()
        STATE_FILE.write_text(json.dumps(self._mem, indent=2))

        # Save skills
        skill_data = {"skills": self._skills, "last_updated": datetime.now(timezone.utc).isoformat()}
        SKILL_FILE.write_text(json.dumps(skill_data, indent=2))

        # Save task registry
        TASK_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._tasks["last_updated"] = datetime.now(timezone.utc).isoformat()
        TASK_FILE.write_text(json.dumps(self._tasks, indent=2))

    def get_collective_stats(self) -> dict:
        """Return stats across all agents."""
        stats = self._mem.get("improvement_stats", {})
        return {
            "total_runs":     sum(v.get("runs", 0) for v in stats.values()),
            "total_successes": sum(v.get("successes", 0) for v in stats.values()),
            "active_agents":  len(self._mem.get("active_agents", {})),
            "skills_count":   len(self._skills),
            "failure_traces": len(self._mem.get("failure_traces", [])),
            "agents":         {k: v.get("successes", 0) for k, v in stats.items()},
        }
