"""
System Watchdog — runs every 5 minutes.
Checks all critical systems, self-heals what it can, posts health to Slack.
Ensures zero-downtime even if Claude Code session ends.
"""
from __future__ import annotations
import os, sys, json, subprocess, glob
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests

def _resolve_key(*names: str) -> str:
    for name in names:
        v = os.environ.get(name, "")
        if v: return v
        if not name[-1].isdigit():
            v = os.environ.get(name + "_1", "")
            if v: return v
    return ""

SLACK_TOKEN     = os.environ.get("SLACK_BOT_TOKEN", "")
GEMINI_API_KEY  = _resolve_key("GEMINI_API_KEY", "GEMINI_API_KEY_1")
GROQ_API_KEY    = _resolve_key("GROQ_API_KEY", "GROQ_API_KEY_1")
DEEPSEEK_KEYS   = [k for k in [
    _resolve_key("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_1"),
    os.environ.get("DEEPSEEK_API_KEY_2", ""),
    os.environ.get("DEEPSEEK_API_KEY_3", ""),
] if k]
ALLOW_PAID_APIS     = os.environ.get("ALLOW_PAID_APIS", "False")
GH_TOKEN            = os.environ.get("GH_TOKEN", "")
GH_REPO             = os.environ.get("GH_REPO", "bahllaavanye-afk/test")
BRANCH              = "claude/advanced-trading-bot-d5Lmw"
OPENROUTER_KEY      = _resolve_key("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2")

if ALLOW_PAID_APIS.lower() == "true":
    sys.exit(1)

# Scheduled workflows we want to guarantee are running — (filename_stem, max_silence_minutes)
CRITICAL_WORKFLOWS = [
    ("signal-runner",           10),   # every 5 min
    ("system-watchdog",         10),   # every 5 min
    ("continuous-improvement",  70),   # every 30 min
    ("agent-heartbeat",         70),   # every 30 min
    ("keep-alive",              25),   # every 10 min
    ("company-brain",           70),   # every 30 min
    ("market-scanner",          70),   # every 30 min
    ("claude-dispatch-router",  25),   # every 15 min
]

REPO_ROOT  = Path(__file__).resolve().parents[2]
STATE_FILE = REPO_ROOT / ".github" / "state" / "agent_memory.json"
SKILL_FILE = REPO_ROOT / ".github" / "state" / "skill_library.json"


# ── Health checks ──────────────────────────────────────────────────────────────

def check_gemini() -> tuple[bool, str]:
    if not GEMINI_API_KEY:
        return False, "no key"
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents": [{"role": "user", "parts": [{"text": "Say OK"}]}],
                  "generationConfig": {"maxOutputTokens": 5}},
            timeout=10
        )
        if r.status_code == 200:
            return True, "ok"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)[:40]


def check_groq() -> tuple[bool, str]:
    if not GROQ_API_KEY:
        return False, "no key"
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5},
            timeout=10
        )
        if r.status_code == 200:
            return True, "ok"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)[:40]


def check_deepseek() -> tuple[bool, str]:
    if not DEEPSEEK_KEYS:
        return False, "no key"
    key = DEEPSEEK_KEYS[0]
    try:
        r = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5},
            timeout=15
        )
        if r.status_code == 200:
            return True, f"ok ({len(DEEPSEEK_KEYS)} keys)"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)[:40]


def check_openrouter() -> tuple[bool, str]:
    if not OPENROUTER_KEY:
        return False, "no key"
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
            json={"model": "meta-llama/llama-3.3-70b-instruct:free",
                  "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5},
            timeout=15,
        )
        if r.status_code == 200:
            return True, "ok"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)[:40]


def check_binance() -> tuple[bool, str]:
    try:
        r = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
        return r.status_code == 200, "ok" if r.status_code == 200 else f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)[:40]


def check_state_files() -> tuple[bool, str]:
    issues = []
    for f in [STATE_FILE, SKILL_FILE]:
        if not f.exists():
            issues.append(f"{f.name} missing")
        else:
            try:
                json.loads(f.read_text())
            except Exception:
                issues.append(f"{f.name} corrupt")
    return (len(issues) == 0), (", ".join(issues) or "ok")


def reactivate_stale_workflows() -> list[str]:
    """
    Check each critical workflow's last run time via GitHub API.
    If a workflow hasn't run within its expected window, trigger it via workflow_dispatch.
    Returns list of workflows re-triggered.
    """
    if not GH_TOKEN:
        return []

    triggered = []
    now = datetime.now(timezone.utc)

    # Get all workflow IDs
    try:
        r = requests.get(
            f"https://api.github.com/repos/{GH_REPO}/actions/workflows",
            headers={"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json"},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        all_workflows = {w["path"].split("/")[-1].replace(".yml", "").replace(".yaml", ""): w
                        for w in r.json().get("workflows", [])}
    except Exception:
        return []

    for stem, max_silence_min in CRITICAL_WORKFLOWS:
        wf = all_workflows.get(stem)
        if not wf:
            continue
        wf_id = wf["id"]
        # Check last run
        try:
            runs_r = requests.get(
                f"https://api.github.com/repos/{GH_REPO}/actions/workflows/{wf_id}/runs",
                params={"per_page": 1, "branch": BRANCH},
                headers={"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json"},
                timeout=10,
            )
            runs = runs_r.json().get("workflow_runs", []) if runs_r.status_code == 200 else []
        except Exception:
            runs = []

        if runs:
            last_run_time = datetime.fromisoformat(runs[0]["created_at"].replace("Z", "+00:00"))
            silence_min = (now - last_run_time).total_seconds() / 60
            if silence_min <= max_silence_min * 1.5:  # within 1.5× expected window
                continue

        # Re-enable if disabled, then dispatch
        try:
            requests.put(
                f"https://api.github.com/repos/{GH_REPO}/actions/workflows/{wf_id}/enable",
                headers={"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json"},
                timeout=10,
            )
            dispatch_r = requests.post(
                f"https://api.github.com/repos/{GH_REPO}/actions/workflows/{wf_id}/dispatches",
                headers={"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json",
                         "Content-Type": "application/json"},
                json={"ref": BRANCH},
                timeout=10,
            )
            if dispatch_r.status_code in (204, 200):
                triggered.append(stem)
                print(f"  REACTIVATED: {stem}")
        except Exception as e:
            print(f"  Could not re-trigger {stem}: {e}")

    return triggered


def check_recent_commits() -> tuple[bool, str]:
    """Verify continuous improver is making commits."""
    try:
        r = requests.get(
            f"https://api.github.com/repos/{GH_REPO}/commits",
            params={"sha": BRANCH, "per_page": 10},
            headers={"Authorization": f"token {GH_TOKEN}"} if GH_TOKEN else {},
            timeout=10
        )
        if r.status_code != 200:
            return True, "github api unavailable"  # not a failure
        commits = r.json()
        if not commits:
            return False, "no commits found"
        latest = commits[0]
        commit_time = datetime.fromisoformat(latest["commit"]["committer"]["date"].replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - commit_time).total_seconds() / 3600
        if age_hours > 2:
            return False, f"last commit {age_hours:.1f}h ago — improver may be down"
        return True, f"last commit {age_hours*60:.0f}m ago"
    except Exception as e:
        return True, f"check skipped: {e}"[:40]


def check_workflows_enabled() -> tuple[bool, str]:
    """Check key workflows exist — checks local .github/workflows/ files first."""
    key_workflows = ["signal-runner", "continuous-improvement", "quick-backtest", "token-usage-monitor"]
    found = []

    # Primary check: local .github/workflows/ directory (always works, no token needed).
    # Workflow filenames use hyphens (e.g. signal-runner.yml) which match key_workflows exactly.
    workflows_dir = REPO_ROOT / ".github" / "workflows"
    if workflows_dir.exists():
        local_stems = {f.stem.lower() for f in workflows_dir.glob("*.yml")}
        local_stems.update(f.stem.lower() for f in workflows_dir.glob("*.yaml"))
        found = [k for k in key_workflows if k in local_stems]
        if len(found) == len(key_workflows):
            return True, f"{len(found)}/{len(key_workflows)} key workflows found (local)"

    # Secondary check: GitHub API — workflow name: field may use spaces; normalise to hyphens.
    if GH_TOKEN:
        try:
            r = requests.get(
                f"https://api.github.com/repos/{GH_REPO}/actions/workflows",
                headers={"Authorization": f"token {GH_TOKEN}"},
                timeout=10,
            )
            if r.status_code == 200:
                api_names = [
                    w["name"].lower().replace(" ", "-")
                    for w in r.json().get("workflows", [])
                ]
                api_found = [k for k in key_workflows if any(k in n for n in api_names)]
                found = list(set(found) | set(api_found))
        except Exception:
            pass

    all_ok = len(found) == len(key_workflows)
    missing = [k for k in key_workflows if k not in found]
    detail = f"{len(found)}/{len(key_workflows)} key workflows found"
    if missing:
        detail += f" (missing: {', '.join(missing)})"
    return all_ok, detail


def self_heal_state() -> list[str]:
    """Auto-fix issues found during health check."""
    actions = []
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not STATE_FILE.exists():
        STATE_FILE.write_text(json.dumps({
            "conversations": {}, "thread_state": {}, "employee_context": {},
            "platform_metrics": {}, "failure_traces": [], "improvement_stats": {},
            "signals": [], "backtest_results": [], "last_updated": datetime.now(timezone.utc).isoformat()
        }, indent=2))
        actions.append("created missing agent_memory.json")

    try:
        json.loads(STATE_FILE.read_text())
    except Exception:
        STATE_FILE.write_text(json.dumps({
            "conversations": {}, "thread_state": {}, "employee_context": {},
            "platform_metrics": {}, "failure_traces": [], "improvement_stats": {},
            "signals": [], "backtest_results": [], "last_updated": datetime.now(timezone.utc).isoformat()
        }, indent=2))
        actions.append("repaired corrupt agent_memory.json")

    if not SKILL_FILE.exists():
        SKILL_FILE.write_text(json.dumps({
            "skills": [
                "Files > 8000 chars cause LLM truncation — always truncate before sending",
                "git pull --rebase before every git push to avoid conflicts",
                "ALLOW_PAID_APIS must always be False — never set to True",
                "All API keys use numbered suffixes _1/_2/_3 — always use _resolve_key()",
                "Syntax check via compile() before writing Python files to disk",
                "State is stored in .github/state/agent_memory.json — read it at start of every run",
            ],
            "last_updated": datetime.now(timezone.utc).isoformat()
        }, indent=2))
        actions.append("created missing skill_library.json")

    return actions


def post_slack(channel: str, text: str) -> bool:
    if not SLACK_TOKEN:
        print(f"[#{channel}] {text[:200]}")
        return False
    try:
        r = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_TOKEN}", "Content-Type": "application/json"},
            json={"channel": channel, "text": text, "mrkdwn": True},
            timeout=10
        )
        return r.status_code == 200 and r.json().get("ok")
    except Exception as e:
        print(f"Slack error: {e}")
        return False


def fmt(ok: bool, detail: str) -> str:
    icon = ":white_check_mark:" if ok else ":x:"
    return f"{icon} {detail}"


def main():
    now = datetime.now(timezone.utc)
    print(f"[{now.strftime('%H:%M UTC')}] System watchdog running")

    # Self-heal first (state files)
    healed = self_heal_state()
    for action in healed:
        print(f"  HEALED: {action}")

    # Re-activate any stale scheduled workflows
    reactivated = reactivate_stale_workflows()
    healed.extend(f"re-triggered {w}" for w in reactivated)

    # Run all health checks
    checks = {
        "Gemini LLM":    check_gemini(),
        "Groq LLM":      check_groq(),
        "DeepSeek LLM":  check_deepseek(),
        "OpenRouter LLM":check_openrouter(),
        "Binance API":   check_binance(),
        "State files":   check_state_files(),
        "Recent commits":check_recent_commits(),
        "Workflows":     check_workflows_enabled(),
    }

    all_ok = all(ok for ok, _ in checks.values())
    failing = [(name, detail) for name, (ok, detail) in checks.items() if not ok]

    print(f"  Checks: {len(checks) - len(failing)}/{len(checks)} passing")

    # Update memory with watchdog status
    try:
        mem = json.loads(STATE_FILE.read_text())
        mem.setdefault("platform_metrics", {})
        mem["platform_metrics"]["last_watchdog_run"] = now.isoformat()
        mem["platform_metrics"]["health_status"] = "healthy" if all_ok else f"{len(failing)} failing"
        mem["platform_metrics"]["watchdog_checks"] = {k: {"ok": ok, "detail": d} for k, (ok, d) in checks.items()}
        mem["last_updated"] = now.isoformat()
        STATE_FILE.write_text(json.dumps(mem, indent=2))
    except Exception as e:
        print(f"Memory update error: {e}")

    # Post to Slack — always post full health, alert on failures
    lines = [f"*System Health — {now.strftime('%H:%M UTC')}*"]
    for name, (ok, detail) in checks.items():
        lines.append(f"  {fmt(ok, f'{name}: {detail}')}")

    if healed:
        lines.append(f"\n:wrench: *Auto-healed*: {', '.join(healed)}")

    if failing:
        lines.append(f"\n:rotating_light: *{len(failing)} system(s) need attention*")
        for name, detail in failing:
            lines.append(f"  • {name}: {detail}")

    msg = "\n".join(lines)

    # Always post to engineering, alert to incidents on failures
    post_slack("engineering", msg)
    if failing:
        post_slack("incidents", f":rotating_light: *SYSTEM ALERT — {len(failing)} service(s) down*\n{msg}")

    # Save summary
    summary = {
        "timestamp": now.isoformat(),
        "all_ok": all_ok,
        "passing": len(checks) - len(failing),
        "total": len(checks),
        "failing": [{"name": n, "detail": d} for n, d in failing],
        "healed": healed,
    }
    with open("/tmp/watchdog_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Watchdog done: {'HEALTHY' if all_ok else f'{len(failing)} FAILING'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
