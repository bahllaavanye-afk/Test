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

# ── agent_memory.json size control ────────────────────────────────────────────
#
# `conversations` is a timestamp-keyed dict that all three writers
# (claude_conversations.py, employee_conversation_runner.py,
# multi_agent_discussion.py) append to and NONE of them trimmed. Measured
# 2026-08-05: 915 entries / 576KB inside a 933KB agent_memory.json, growing
# ~7KB per commit, and 200 of the last 200 commits rewrote the whole blob.
# **59.1 MB of the 125 MB .git — 47% of the repository — is 2340 historical
# copies of this one file.**
#
# The only reader is context_sync.py, which does `sorted(convs.items())[-20:]`
# and displays 10. So ~900 entries were retained to serve a consumer that wants
# 20. The same file already caps its other structures — `peer_learnings[-100:]`,
# `failure_traces[-200:]` — the discipline just never reached the largest and
# fastest-growing one.
#
# 300 keeps roughly six full conversation rounds (~48 employees each) and is 15x
# what any reader asks for. Trimming is by sorted key: the keys are ISO-8601 UTC
# timestamps, which sort lexicographically in chronological order.
#
# This does NOT shrink the existing 59 MB of history — that needs a rewrite,
# which is not done here. It stops the growth.
CONVERSATION_CAP = 300


def trim_conversations(mem: dict, cap: int = CONVERSATION_CAP) -> int:
    """Drop all but the newest `cap` conversation entries. Returns how many went.

    Safe on a missing or malformed `conversations` value: agent_memory.json is
    written by several unrelated scripts and repaired by system_watchdog, so a
    non-dict here must degrade rather than raise inside somebody's save path.
    """
    convs = mem.get("conversations")
    if not isinstance(convs, dict) or len(convs) <= cap:
        return 0
    keep = dict(sorted(convs.items())[-cap:])
    dropped = len(convs) - len(keep)
    mem["conversations"] = keep
    return dropped


# ── peer_learnings quality + outcome linkage ──────────────────────────────────
#
# Measured 2026-08-05: **56 of 200 peer_learnings entries (28%) were LLM prompt
# echoes** — the model restating its instruction instead of answering it:
#
#   [investor_pipeline @ ...] The user asks: "Give a one-sentence status update:
#                             what are you actively working on RIGHT NOW..."
#   [self_improver @ ...]     We need to respond as self_improver agent,
#                             autonomous, 2 sentences max, first person...
#
# `agent_status_checker.py:247` and `multi_agent_discussion.py:360` append the
# raw reply with no quality check, so a failed generation is stored as a
# "learning". That was merely wasteful until the retrieval fix landed; now these
# entries are RETRIEVED into other agents' prompts, so the noise compounds.
#
# The patterns below are deliberately narrow and anchored to observed pollution.
# A false positive silently discards a real learning, which is worse than
# keeping one echo — so this rejects only unambiguous instruction-restatement.
_ECHO_PATTERNS = (
    "the user asks",
    "we need to respond as",
    "respond as the",
    "one-sentence status update",
    "the instruction:",
    "as an ai language model",
    "i cannot fulfill",
)
_MIN_LEARNING_CHARS = 25


def is_low_quality_learning(text: str) -> bool:
    """True if `text` is an instruction echo rather than a learning.

    Applied at the WRITE boundary: once stored, an echo is indistinguishable
    from a real entry and gets retrieved into other agents' context.
    """
    if not isinstance(text, str):
        return True
    # Strip the "[agent @ timestamp] " prefix before judging length/content.
    body = text.split("] ", 1)[1] if text.startswith("[") and "] " in text else text
    body = body.strip()
    if len(body) < _MIN_LEARNING_CHARS:
        return True
    low = body.lower()
    return any(p in low for p in _ECHO_PATTERNS)


def clean_learnings(entries) -> list:
    """Filter a batch, preserving order. Non-strings are dropped."""
    return [e for e in (entries or []) if not is_low_quality_learning(e)]


_PERF_FILE = REPO_ROOT / "backend" / "performance_log" / "strategy_performance.json"


def outcome_learnings(top_n: int = 5, path: Path | None = None) -> list[str]:
    """Factual P&L attribution lines for peer_learnings — the anti-status-theater.

    `strategy_performance.json` is written by `fill_tracker.py` from real filled
    orders, attributed back to strategies via the client_order_id encoding, and
    it IS committed (verified 2026-08-05: 22 strategies, 247 tracked orders).
    Nothing consumed it into the agents' shared context, so the daily discussion
    ran on self-reported status while the actual results sat in a file.

    Ranked by total return, worst included: a losing strategy is the more useful
    thing to discuss, and reporting only winners is how status theater starts.
    """
    src = path or _PERF_FILE
    try:
        data = json.loads(Path(src).read_text())
    except Exception:  # noqa: BLE001 — no attribution is not an error
        return []
    strategies = data.get("strategies")
    if not isinstance(strategies, dict) or not strategies:
        return []
    ranked = sorted(
        ((n, s) for n, s in strategies.items()
         if isinstance(s, dict) and (s.get("trades") or 0) > 0),
        key=lambda kv: kv[1].get("total_return_pct", 0.0),
        reverse=True,
    )
    if not ranked:
        return []
    picks = ranked[:top_n]
    if len(ranked) > top_n:
        picks.append(ranked[-1])  # always surface the worst performer
    stamp = datetime.now(timezone.utc).isoformat()[:16]
    out = []
    for name, s in picks:
        out.append(
            "[attribution @ {ts}] {n}: {t} trades, {w:.0f}% win rate, "
            "{tot:+.2f}% total return, {avg:+.2f}% avg (period {d}d)".format(
                ts=stamp, n=name, t=s.get("trades", 0),
                w=100 * (s.get("win_rate") or 0.0),
                tot=s.get("total_return_pct", 0.0),
                avg=s.get("avg_return_pct", 0.0),
                d=data.get("period_days", "?"),
            )
        )
    return out


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
    "standup_agent":         "Posts daily standups and OKR tracking to Discord",
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

        # Share learnings with peer agents — but not instruction echoes.
        entry = f"[{self.agent_name} @ {now[:16]}] {summary}"
        if not is_low_quality_learning(entry):
            self._mem.setdefault("peer_learnings", [])
            self._mem["peer_learnings"].append(entry)
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
