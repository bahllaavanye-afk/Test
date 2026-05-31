"""
Frontend AI Team — reviews UI files and applies targeted improvements.
Focused on: animations, visual polish, component quality.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import anthropic
import httpx

REPO_ROOT    = Path(__file__).parent.parent
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"
BRANCH       = os.environ.get("GITHUB_REF_NAME", os.environ.get("GITHUB_HEAD_REF", "main"))
SLACK_TOKEN  = os.environ.get("SLACK_BOT_TOKEN", "")
API_KEY      = os.environ.get("ANTHROPIC_API_KEY", "")
FOCUS        = os.environ.get("FOCUS", "overall UI quality, animations, and UX polish")

SAFE_FRONTEND = [
    "frontend/src/styles/animations.css",
    "frontend/src/index.css",
    "frontend/src/pages/Landing.tsx",
    "frontend/src/pages/Login.tsx",
    "frontend/src/pages/Dashboard.tsx",
    "frontend/src/components/layout/TopBar.tsx",
    "frontend/src/components/layout/Sidebar.tsx",
]


def slack(msg: str) -> None:
    if not SLACK_TOKEN:
        return
    try:
        httpx.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
            json={"channel": "#desk-options", "text": msg, "mrkdwn": True},
            timeout=10,
        )
    except Exception as e:
        print(f"Slack error: {e}")


def read_files() -> str:
    parts = []
    for rel in SAFE_FRONTEND:
        p = REPO_ROOT / rel
        if p.exists():
            content = p.read_text()[:3000]
            parts.append(f"=== {rel} ===\n{content}")
    return "\n\n".join(parts)


def apply_and_push(files: list[dict]) -> bool:
    patched = []
    for item in files:
        rel = item.get("path", "")
        content = item.get("content", "")
        if not rel or not content or rel not in SAFE_FRONTEND:
            continue
        (REPO_ROOT / rel).write_text(content)
        patched.append(rel)
    if not patched:
        return False

    subprocess.run(["git", "add", "-A"], cwd=REPO_ROOT, check=True)
    r = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT)
    if r.returncode == 0:
        return False

    subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "push", "origin", BRANCH], cwd=REPO_ROOT, check=True)
    return True


def main() -> None:
    if not API_KEY:
        print("No ANTHROPIC_API_KEY")
        return

    context = read_files()
    client = anthropic.Anthropic(api_key=API_KEY)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8096,
        system="""You are QuantEdge's senior frontend engineer. Review the UI files and make targeted improvements.
Focus on: animations, micro-interactions, visual polish, sci-fi terminal aesthetic.

Return ONLY JSON (no prose):
{
  "summary": "what was improved and why",
  "files": [{"path": "relative/path", "content": "COMPLETE new file content"}]
}

Rules:
- Only improve files listed in the allowed set
- Make conservative, high-impact improvements
- Preserve all existing TypeScript types and logic
- Keep animations smooth (CSS transforms, opacity, not layout)
- No external library imports — only what's already in package.json
- CSS variables from index.css only (--bg, --surface, --green, --accent, --red, --blue, --purple, --muted)""",
        messages=[{"role": "user", "content": f"Focus: {FOCUS}\n\nCurrent files:\n{context}"}],
    )
    text = response.content[0].text

    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except Exception:
                print("Could not parse JSON"); return
        else:
            print("No JSON found"); return

    pushed = apply_and_push(data.get("files", []))
    summary = data.get("summary", "UI improvements")

    if pushed:
        slack(f"🎨 *Frontend AI Team:* {summary}")
    else:
        print("No changes applied")


if __name__ == "__main__":
    main()
