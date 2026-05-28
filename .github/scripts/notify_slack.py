"""
Slack job-lifecycle notifier for GitHub Actions.

Usage (in workflow steps):
  python .github/scripts/notify_slack.py start  "ML Experiments"    "#ml-experiments"
  python .github/scripts/notify_slack.py end    "ML Experiments"    "#ml-experiments" --exit-code 0
  python .github/scripts/notify_slack.py progress "Training epoch 10/100" "#ml-experiments"

Environment variables read:
  SLACK_BOT_TOKEN   required
  GITHUB_RUN_ID     injected by Actions
  GITHUB_WORKFLOW   injected by Actions
  GITHUB_REPOSITORY injected by Actions
  GITHUB_REF_NAME   injected by Actions
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
RUN_ID          = os.environ.get("GITHUB_RUN_ID", "local")
WORKFLOW        = os.environ.get("GITHUB_WORKFLOW", "unknown")
REPO            = os.environ.get("GITHUB_REPOSITORY", "unknown/unknown")
BRANCH          = os.environ.get("GITHUB_REF_NAME", "unknown")
RUN_URL         = f"https://github.com/{REPO}/actions/runs/{RUN_ID}"

# Persistent start-time file so "end" can compute elapsed time
_TIMER_FILE = f"/tmp/_qe_job_timer_{RUN_ID}.txt"


def _post(channel: str, text: str, blocks: list | None = None) -> None:
    if not SLACK_BOT_TOKEN:
        print(f"[notify_slack] No SLACK_BOT_TOKEN — would post to {channel}:\n{text}", flush=True)
        return
    payload: dict = {"channel": channel, "text": text}
    if blocks:
        payload["blocks"] = blocks
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            if not body.get("ok"):
                print(f"[notify_slack] Slack error: {body.get('error')}", flush=True)
    except Exception as exc:
        print(f"[notify_slack] HTTP error: {exc}", flush=True)


def _elapsed() -> str:
    try:
        start = float(open(_TIMER_FILE).read().strip())
        secs  = int(time.time() - start)
        return f"{secs // 60}m {secs % 60}s"
    except Exception:
        return "?"


def cmd_start(job_name: str, channel: str) -> None:
    # Record start time
    with open(_TIMER_FILE, "w") as f:
        f.write(str(time.time()))

    ts    = datetime.now(timezone.utc).strftime("%H:%M UTC")
    text  = f":rocket: *{job_name}* started  |  `{BRANCH}` branch  |  {ts}"
    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": (
                f":rocket: *{job_name}* started\n"
                f"Branch: `{BRANCH}`  |  Workflow: `{WORKFLOW}`  |  {ts}\n"
                f"<{RUN_URL}|View run →>"
            )},
        }
    ]
    _post(channel, text, blocks)


def cmd_end(job_name: str, channel: str, exit_code: int) -> None:
    elapsed = _elapsed()
    if exit_code == 0:
        icon, status = ":white_check_mark:", "completed successfully"
    else:
        icon, status = ":x:", f"FAILED (exit {exit_code})"

    text   = f"{icon} *{job_name}* {status}  |  {elapsed}  |  `{BRANCH}`"
    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": (
                f"{icon} *{job_name}* {status}\n"
                f"Duration: `{elapsed}`  |  Branch: `{BRANCH}`\n"
                f"<{RUN_URL}|View run →>"
            )},
        }
    ]
    _post(channel, text, blocks)


def cmd_progress(message: str, channel: str) -> None:
    elapsed = _elapsed()
    text    = f":hourglass_flowing_sand: `{elapsed}` — {message}"
    _post(channel, text)


def cmd_cache_hit(asset: str, channel: str) -> None:
    text = f":floppy_disk: *Cache hit* for `{asset}` — skipping download, using cached OHLCV"
    _post(channel, text)


def cmd_cache_miss(asset: str, channel: str) -> None:
    text = f":arrow_down: *Cache miss* for `{asset}` — downloading fresh OHLCV data"
    _post(channel, text)


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        print(f"Usage: notify_slack.py <start|end|progress|cache_hit|cache_miss> <job/msg> [channel] [--exit-code N]")
        sys.exit(1)

    command  = args[0]
    subject  = args[1]
    channel  = args[2] if len(args) > 2 and not args[2].startswith("--") else "#ml-experiments"
    exit_code = 0
    for i, a in enumerate(args):
        if a == "--exit-code" and i + 1 < len(args):
            exit_code = int(args[i + 1])

    if command == "start":
        cmd_start(subject, channel)
    elif command == "end":
        cmd_end(subject, channel, exit_code)
    elif command == "progress":
        cmd_progress(subject, channel)
    elif command == "cache_hit":
        cmd_cache_hit(subject, channel)
    elif command == "cache_miss":
        cmd_cache_miss(subject, channel)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
