"""
Session Handoff — posts a full system briefing to Slack #engineering when called.
Reads agent_memory.json + git log to reconstruct state.
Run this when Claude Code session is ending so the next session starts informed.
Also triggered by the watchdog if no recent commits are detected.
"""
from __future__ import annotations
import os, sys, json, subprocess
from datetime import datetime, timezone
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

SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
ALLOW_PAID_APIS = os.environ.get("ALLOW_PAID_APIS", "False")

if ALLOW_PAID_APIS.lower() == "true":
    sys.exit(1)

REPO_ROOT  = Path(__file__).resolve().parents[2]
STATE_FILE = REPO_ROOT / ".github" / "state" / "agent_memory.json"
BRANCH     = "main"


def get_git_log(n: int = 10) -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "log", "--oneline", f"-{n}", f"origin/{BRANCH}"],
            cwd=REPO_ROOT, text=True, timeout=10
        )
        return out.strip().split("\n")
    except Exception:
        return []


def load_memory() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def post_slack(channel: str, text: str) -> bool:
    if not SLACK_TOKEN:
        print(text)
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
        print(f"Slack: {e}")
        return False


def main():
    now = datetime.now(timezone.utc)
    mem = load_memory()
    git_log = get_git_log()
    metrics = mem.get("platform_metrics", {})
    signals = mem.get("signals", [])
    today = now.strftime("%Y-%m-%d")
    signals_today = [s for s in signals if s.get("timestamp", "")[:10] == today]

    # Build handoff message
    lines = [
        f"*QuantEdge Session Handoff — {now.strftime('%Y-%m-%d %H:%M UTC')}*",
        f"Branch: `{BRANCH}`",
        "",
        "*System Status*",
        f"  Last watchdog: {metrics.get('last_watchdog_run', 'unknown')[:16]}",
        f"  Health: {metrics.get('health_status', 'unknown')}",
        f"  Last signal run: {metrics.get('last_signal_run', 'unknown')[:16]}",
        f"  Signals today: {len(signals_today)}",
        "",
        "*Recent Commits (last 10)*",
        "```",
    ]
    for commit in git_log[:10]:
        lines.append(f"  {commit}")
    lines.append("```")

    # Active workflows
    lines += [
        "",
        "*Active Autonomous Workflows*",
        "  • `*/5 * * * *` — system-watchdog (health + self-heal)",
        "  • `*/5 * * * *` — signal-runner (crypto/equity/polymarket signals)",
        "  • `*/5 * * * *` — slack-summon-watcher (respond to Slack messages)",
        "  • `*/15 * * * *` — quick-backtest (strategy backtests)",
        "  • `*/15 * * * *` — token-usage-monitor (#token-usage posts)",
        "  • `*/30 * * * *` — continuous-improvement (5 files/run, RLVR+Reflexion+Voyager)",
        "  • `0 */3 * * *` — peer-review (Agent B reviews Agent A commits)",
        "  • `0 */4 * * *` — gemini-ml-training",
        "  • `0 13 * * *` — daily-standup",
        "",
        "*To continue:* Open a new Claude Code session. The system runs autonomously via GitHub Actions.",
        "*Key secrets needed:* GEMINI_API_KEY_1, GROQ_API_KEY_1, DEEPSEEK_API_KEY_1/2/3, SLACK_BOT_TOKEN",
        "",
        "_All employees continue working via GitHub Actions — no session required for core operations._",
    ]

    msg = "\n".join(lines)
    post_slack("engineering", msg)
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
