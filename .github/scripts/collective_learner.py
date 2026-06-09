"""
Collective Learning Aggregator — runs every 30 minutes.
Distills learnings from all agents into compressed skills.
Implements collective intelligence: all agents get smarter together.

Architecture based on:
- Voyager (Wang et al., 2023): skill library accumulation
- Reflexion (Shinn et al., 2023): failure → improved next attempt
- AlphaCode feedback loops: aggregate failures to avoid repeat mistakes
- Constitutional AI: self-critique and revision
"""
from __future__ import annotations
import os, sys, json
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_common import llm, slack_post, memory_write

ALLOW_PAID_APIS = os.environ.get("ALLOW_PAID_APIS", "False")

if ALLOW_PAID_APIS.lower() == "true":
    sys.exit(1)

REPO_ROOT  = Path(__file__).resolve().parents[2]
STATE_FILE = REPO_ROOT / ".github" / "state" / "agent_memory.json"
SKILL_FILE = REPO_ROOT / ".github" / "state" / "skill_library.json"
TASK_FILE  = REPO_ROOT / ".github" / "state" / "task_registry.json"


def call_llm(prompt: str) -> str:
    """Delegate to shared llm_common infrastructure."""
    result = llm(prompt, max_tokens=600, inject_company_context=False)
    if result and not result.startswith("[LLM unavailable"):
        return result
    return ""


def distill_failures(failure_traces: list[dict], current_skills: list[str]) -> list[str]:
    """Use LLM to distill failure traces into actionable skills."""
    if not failure_traces:
        return []

    recent = failure_traces[-20:]
    failure_text = "\n".join(
        f"- [{f.get('agent','')}] {f.get('what_failed','')} → {f.get('error','')}"
        for f in recent
    )
    existing = "\n".join(f"- {s}" for s in current_skills[-10:])

    prompt = f"""You are a software architect analyzing failures across {len(failure_traces)} runs of an autonomous trading platform.

RECENT FAILURES:
{failure_text}

EXISTING SKILLS (already known):
{existing}

Generate 3-5 NEW, concise, actionable skills/rules that would prevent these failures in future runs.
Each skill must be:
1. Specific and implementable (not vague)
2. Not duplicate an existing skill
3. Under 80 characters

Format: one skill per line, no bullets or numbers."""

    response = call_llm(prompt)
    if not response:
        return []

    new_skills = [line.strip() for line in response.strip().split("\n")
                  if line.strip() and len(line.strip()) < 150
                  and line.strip() not in current_skills]
    return new_skills[:5]


def distill_peer_learnings(learnings: list[str], current_skills: list[str]) -> list[str]:
    """Extract patterns from peer learnings into reusable skills."""
    if not learnings:
        return []

    recent = learnings[-20:]
    learning_text = "\n".join(recent)

    prompt = f"""You are analyzing learnings from autonomous AI agents working on a trading platform.

RECENT LEARNINGS:
{learning_text}

Extract 2-3 reusable patterns or skills from these learnings.
Format: one skill per line, under 80 chars, actionable, not already in: {', '.join(current_skills[-5:])}"""

    response = call_llm(prompt)
    if not response:
        return []

    return [line.strip() for line in response.strip().split("\n")
            if line.strip() and len(line.strip()) < 150
            and line.strip() not in current_skills][:3]


def prune_task_registry(tasks: dict) -> dict:
    """Remove stale claimed tasks (older than 2 hours)."""
    now = datetime.now(timezone.utc)
    active = tasks.get("active", {})
    pruned = []
    for task_id, task in list(active.items()):
        try:
            claimed_at = datetime.fromisoformat(task["claimed_at"].replace("Z", "+00:00"))
            age_hours = (now - claimed_at).total_seconds() / 3600
            if age_hours > 2:
                del active[task_id]
                pruned.append(task_id)
        except Exception:
            del active[task_id]
            pruned.append(task_id)
    if pruned:
        print(f"  Pruned {len(pruned)} stale tasks: {pruned}")
    return tasks


def post_slack(channel: str, text: str):
    if not SLACK_TOKEN:
        print(f"[#{channel}] {text[:200]}")
        return
    try:
        requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_TOKEN}", "Content-Type": "application/json"},
            json={"channel": channel, "text": text, "mrkdwn": True},
            timeout=10
        )
    except Exception as e:
        print(f"Slack: {e}")


def main():
    now = datetime.now(timezone.utc)
    print(f"[{now.strftime('%H:%M UTC')}] Collective learning aggregator")

    # Load all state
    try:
        mem = json.loads(STATE_FILE.read_text())
    except Exception:
        mem = {}
    try:
        skill_data = json.loads(SKILL_FILE.read_text())
        current_skills = skill_data.get("skills", [])
    except Exception:
        current_skills = []
    try:
        tasks = json.loads(TASK_FILE.read_text())
    except Exception:
        tasks = {}

    failures = mem.get("failure_traces", [])
    peer_learnings = mem.get("peer_learnings", [])

    print(f"  Failures to distill: {len(failures)}")
    print(f"  Peer learnings: {len(peer_learnings)}")
    print(f"  Current skills: {len(current_skills)}")

    # Distill new skills from failures
    new_from_failures = distill_failures(failures, current_skills)
    print(f"  New skills from failures: {len(new_from_failures)}")

    # Distill new skills from peer learnings
    new_from_learnings = distill_peer_learnings(peer_learnings, current_skills + new_from_failures)
    print(f"  New skills from learnings: {len(new_from_learnings)}")

    # Merge and deduplicate
    all_new = new_from_failures + new_from_learnings
    added = []
    for skill in all_new:
        if skill and skill not in current_skills:
            current_skills.append(skill)
            added.append(skill)

    # Keep skill library bounded (max 200 skills — older ones pruned)
    if len(current_skills) > 200:
        # Keep last 200 (most recent = most relevant)
        current_skills = current_skills[-200:]

    # Prune stale task registry entries
    tasks = prune_task_registry(tasks)

    # Trim old failure traces (keep last 200)
    mem["failure_traces"] = failures[-200:]

    # Trim old peer learnings (keep last 200)
    mem["peer_learnings"] = peer_learnings[-200:]

    # Update collective stats
    agent_stats = mem.get("improvement_stats", {})
    total_runs = sum(v.get("runs", 0) for v in agent_stats.values())
    total_successes = sum(v.get("successes", 0) for v in agent_stats.values())

    # Save everything
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    mem["last_updated"] = now.isoformat()
    mem["platform_metrics"] = mem.get("platform_metrics", {})
    mem["platform_metrics"]["last_collective_learning"] = now.isoformat()
    mem["platform_metrics"]["total_agent_runs"] = total_runs
    mem["platform_metrics"]["total_successes"] = total_successes
    STATE_FILE.write_text(json.dumps(mem, indent=2))

    skill_data = {"skills": current_skills, "last_updated": now.isoformat(), "total": len(current_skills)}
    SKILL_FILE.write_text(json.dumps(skill_data, indent=2))

    TASK_FILE.parent.mkdir(parents=True, exist_ok=True)
    tasks["last_updated"] = now.isoformat()
    TASK_FILE.write_text(json.dumps(tasks, indent=2))

    # Post to Slack if new skills were learned
    if added:
        lines = [f"*Collective Learning Update — {now.strftime('%H:%M UTC')}*"]
        lines.append(f"Agents ran {total_runs} total tasks | {total_successes} successes")
        lines.append(f"Distilled {len(added)} new shared skills from {len(failures)} failure traces:")
        for skill in added:
            lines.append(f"  • {skill}")
        lines.append(f"Skill library: {len(current_skills)} total patterns")
        post_slack("engineering", "\n".join(lines))

    print(f"✓ Skills: {len(current_skills)} total, +{len(added)} new | Agent runs: {total_runs}")

    with open("/tmp/collective_learning_summary.json", "w") as f:
        json.dump({
            "timestamp": now.isoformat(),
            "skills_total": len(current_skills),
            "skills_added": added,
            "total_agent_runs": total_runs,
            "total_successes": total_successes,
            "failure_traces": len(failures),
        }, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
