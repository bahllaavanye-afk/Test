"""
Render Health Monitor + Auto-Fix

1. Polls Render API for latest deploy status.
2. On failure: fetches deploy logs.
3. Calls QuantEdge AI to diagnose and generate a fix.
4. Applies the fix (writes files), commits, pushes → triggers Render re-deploy.
5. Posts status to Slack.

Required secrets:
  RENDER_API_KEY      — from render.com/dashboard → Account Settings → API Keys
  RENDER_SERVICE_ID   — from Render service URL: render.com/web/<SERVICE_ID>
  ANTHROPIC_API_KEY   — for QuantEdge AI diagnosis + fix generation
  SLACK_BOT_TOKEN     — for Slack notifications
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import anthropic
import httpx

RENDER_API = "https://api.render.com/v1"
REPO_ROOT  = Path(__file__).parent.parent

RENDER_API_KEY    = os.environ.get("RENDER_API_KEY", "")
RENDER_SERVICE_ID = os.environ.get("RENDER_SERVICE_ID", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SLACK_TOKEN       = os.environ.get("SLACK_BOT_TOKEN", "")

BRANCH = "claude/advanced-trading-bot-d5Lmw"
SLACK_CHANNEL = "#risk-alerts"

# Files that QuantEdge AI is allowed to modify during auto-fix
SAFE_TO_MODIFY = [
    "backend/pyproject.toml",
    "backend/start.sh",
    "backend/alembic/env.py",
    "backend/alembic.ini",
    "backend/app/config.py",
    "backend/app/database.py",
    "backend/app/main.py",
    "render.yaml",
    "pyproject.toml",
]


def slack(msg: str) -> None:
    if not SLACK_TOKEN:
        return
    try:
        httpx.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
            json={"channel": SLACK_CHANNEL, "text": msg, "mrkdwn": True},
            timeout=10,
        )
    except Exception as e:
        print(f"Slack error: {e}")


def render_get(path: str) -> dict | list | None:
    if not RENDER_API_KEY:
        return None
    try:
        r = httpx.get(
            f"{RENDER_API}{path}",
            headers={"Authorization": f"Bearer {RENDER_API_KEY}", "Accept": "application/json"},
            timeout=20,
        )
        if r.status_code == 200:
            return r.json()
        print(f"Render API {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"Render API error: {e}")
    return None


def get_latest_deploy() -> dict | None:
    data = render_get(f"/services/{RENDER_SERVICE_ID}/deploys?limit=3")
    if not data or not isinstance(data, list):
        return None
    item = data[0]
    return item.get("deploy", item)


def get_deploy_logs(deploy_id: str) -> str:
    data = render_get(f"/services/{RENDER_SERVICE_ID}/deploys/{deploy_id}/logs")
    if not data or not isinstance(data, list):
        return ""
    return "\n".join(item.get("message", "") for item in data[-100:])


def read_key_files() -> str:
    """Read the most likely-to-be-broken files for QuantEdge AI's context."""
    parts = []
    for rel in SAFE_TO_MODIFY:
        p = REPO_ROOT / rel
        if p.exists():
            content = p.read_text()[:3000]
            parts.append(f"=== {rel} ===\n{content}")
    return "\n\n".join(parts)


def apply_fix(fix_json: str) -> bool:
    """
    Expects QuantEdge AI to return JSON like:
    {
      "root_cause": "...",
      "files": [
        {"path": "relative/path", "content": "full new file content"}
      ]
    }
    Returns True if at least one file was patched.
    """
    try:
        fix = json.loads(fix_json)
    except Exception:
        # QuantEdge AI might have wrapped JSON in markdown code blocks
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", fix_json, re.DOTALL)
        if m:
            try:
                fix = json.loads(m.group(1))
            except Exception:
                print("Could not parse QuantEdge AI fix JSON")
                return False
        else:
            print("QuantEdge AI did not return valid JSON")
            return False

    files = fix.get("files", [])
    if not files:
        print("QuantEdge AI returned no file changes")
        return False

    patched = []
    for item in files:
        rel_path = item.get("path", "")
        content  = item.get("content", "")
        if not rel_path or not content:
            continue
        if rel_path not in SAFE_TO_MODIFY:
            print(f"Skipping unsafe path: {rel_path}")
            continue
        full_path = REPO_ROOT / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        patched.append(rel_path)
        print(f"  Patched: {rel_path}")

    return len(patched) > 0


def git_commit_and_push(reason: str) -> bool:
    try:
        subprocess.run(["git", "add", "-A"], cwd=REPO_ROOT, check=True)
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=REPO_ROOT
        )
        if result.returncode == 0:
            print("No changes to commit")
            return False

        subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, check=True)
        subprocess.run(
            ["git", "push", "origin", BRANCH],
            cwd=REPO_ROOT, check=True,
        )
        print("Pushed auto-fix commit")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Git error: {e}")
        return False


def main() -> None:
    if not RENDER_API_KEY or not RENDER_SERVICE_ID:
        print("RENDER_API_KEY / RENDER_SERVICE_ID not set — skipping")
        slack("⚠️ Render monitor: API key or service ID not set. Add `RENDER_API_KEY` and `RENDER_SERVICE_ID` to GitHub Secrets.")
        return

    deploy = get_latest_deploy()
    if not deploy:
        print("Could not fetch deploy info")
        return

    status     = deploy.get("status", "unknown")
    deploy_id  = deploy.get("id", "")
    commit_msg = deploy.get("commit", {}).get("message", "")[:80] if deploy.get("commit") else ""

    print(f"Deploy status: {status} | id={deploy_id} | commit={commit_msg}")

    FAILED_STATUSES = ("failed", "build_failed", "canceled", "update_failed")
    if status not in FAILED_STATUSES:
        print(f"Service is {status} — healthy, no action needed")
        return

    # ── Service is failing ────────────────────────────────────────────────────
    logs = get_deploy_logs(deploy_id) if deploy_id else ""

    slack(
        f"🔴 *Render Deploy FAILED* · `{status}`\n"
        f"Deploy: `{deploy_id[:12]}` · Commit: {commit_msg}\n"
        f"```{logs[-2000:]}```\n"
        f"🤖 Running auto-fix..."
    )

    if not ANTHROPIC_API_KEY:
        print("No ANTHROPIC_API_KEY — cannot auto-fix")
        return

    # ── QuantEdge AI diagnosis + fix ────────────────────────────────────────────────
    context_files = read_key_files()
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    system = textwrap.dedent("""
        You are QuantEdge DevOps AI. A Render.com deployment failed.
        Analyze the logs and the key config files, then generate an exact fix.

        Return ONLY a JSON object in this format (no markdown, no prose):
        {
          "root_cause": "one sentence describing the root cause",
          "files": [
            {
              "path": "relative/path/from/repo/root",
              "content": "COMPLETE new file content — do not use placeholders"
            }
          ]
        }

        Rules:
        - Only modify files from this allowed list:
          backend/pyproject.toml, backend/start.sh, backend/alembic/env.py,
          backend/alembic.ini, backend/app/config.py, backend/app/database.py,
          backend/app/main.py, render.yaml, pyproject.toml
        - Fix only the specific error — do not refactor unrelated code
        - Preserve all existing functionality
        - If unsure, change the minimal thing (e.g. pip install flag, timeout value)
        - If root cause is "Network is unreachable" during alembic: use psycopg2 sync in env.py
        - If root cause is pip/uv install failure: simplify buildCommand in render.yaml
    """).strip()

    user_msg = (
        f"Render deploy status: {status}\n"
        f"Deploy ID: {deploy_id}\n"
        f"Commit: {commit_msg}\n\n"
        f"=== FAILURE LOGS (last 4000 chars) ===\n{logs[-4000:]}\n\n"
        f"=== KEY REPO FILES ===\n{context_files}"
    )

    print("Calling QuantEdge AI for auto-fix...")
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        fix_text = response.content[0].text
        print(f"QuantEdge AI response:\n{fix_text[:500]}...")
    except Exception as e:
        print(f"QuantEdge AI API error: {e}")
        slack(f"❌ Auto-fix failed: QuantEdge AI API error — {e}")
        return

    # ── Apply and push ────────────────────────────────────────────────────────
    if apply_fix(fix_text):
        # Parse root cause for commit message
        try:
            fix_data = json.loads(fix_text)
            reason = fix_data.get("root_cause", "Render deploy failure")
        except Exception:
            reason = "Render deploy failure auto-fix"

        pushed = git_commit_and_push(reason)
        if pushed:
            slack(
                f"✅ *Auto-fix applied & pushed*\n"
                f"Root cause: {reason}\n"
                f"Render will re-deploy automatically."
            )
        else:
            slack("⚠️ Auto-fix generated no changes — manual intervention needed")
    else:
        # Just post the diagnosis
        try:
            fix_data = json.loads(fix_text)
            reason = fix_data.get("root_cause", fix_text[:300])
        except Exception:
            reason = fix_text[:300]
        slack(
            f"🔍 *Render Failure Diagnosis:*\n{reason}\n\n"
            f"_Auto-fix could not apply changes to safe files — check manually._"
        )


if __name__ == "__main__":
    main()
