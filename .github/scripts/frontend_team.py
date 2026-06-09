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
BRANCH       = "claude/advanced-trading-bot-d5Lmw"
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

    commit_msg = "style(auto): frontend team UI improvements"
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "push", "origin", BRANCH], cwd=REPO_ROOT, check=True)
    return True


FRONTEND_SYSTEM = """You are QuantEdge's senior frontend engineer. Review the UI files and make targeted improvements.
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
- CSS variables from index.css only (--bg, --surface, --green, --accent, --red, --blue, --purple, --muted)"""


def _run_gemini_review() -> None:
    """Use Gemini (free) to review frontend when Anthropic API key is disabled."""
    import urllib.request

    gemini_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY_1", ""))
    if not gemini_key:
        print("No GEMINI_API_KEY either — frontend review skipped")
        slack("⚠️ *Frontend AI Team:* Skipped (no LLM key available)")
        return

    print("Frontend AI Team starting review via Gemini...")
    context = read_files()
    prompt = (
        f"You are QuantEdge's senior frontend engineer.\n"
        f"Focus: {FOCUS}\n\n"
        f"Review these UI files and return a brief JSON report.\n"
        f"Format: {{\"summary\": \"2 sentence assessment\", \"findings\": [{{\"file\": \"...\", \"issue\": \"...\"}}]}}\n\n"
        f"Files:\n{context[:6000]}"
    )
    try:
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.2},
        }).encode()
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL) or re.search(r"(\{.*\})", text, re.DOTALL)
        result = json.loads(m.group(1)) if m else {"findings": [], "summary": text[:200]}
        summary = result.get("summary", "Gemini review complete")
        findings = result.get("findings", [])
        print(f"Summary: {summary}\nFindings: {len(findings)}")
        lines = [f"🎨 *Frontend AI Audit (Gemini)* — {len(findings)} findings", f"_{summary}_"]
        for f in findings[:5]:
            lines.append(f"• `{f.get('file', '?')}`: {f.get('issue', '?')}")
        slack("\n".join(lines))
    except Exception as exc:
        print(f"Gemini review failed: {exc}")
        slack(f"⚠️ *Frontend AI Team:* Gemini review failed: {exc}")


def main() -> None:
    if not API_KEY or API_KEY == "disabled":
        print("No ANTHROPIC_API_KEY (or set to 'disabled') — frontend team uses Gemini fallback")
        _run_gemini_review()
        return

    context = read_files()
    client = anthropic.Anthropic(api_key=API_KEY)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8096,
        system=FRONTEND_SYSTEM,
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
