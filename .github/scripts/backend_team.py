"""
Backend AI Engineering Team

Reviews backend code for bugs, security issues, and performance problems.
Auto-fixes safe issues (config, imports, error handling) and posts a report to Slack.

Focused areas:
  - API endpoint correctness
  - Strategy logic (new additions)
  - ML model integration bugs
  - Database query patterns
  - Security (missing auth, rate limits)
  - Dependency issues (pyproject.toml)

Run: ANTHROPIC_API_KEY=... SLACK_BOT_TOKEN=... python .github/scripts/backend_team.py
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
BRANCH       = "claude/advanced-trading-bot-d5Lmw"
SLACK_TOKEN  = os.environ.get("SLACK_BOT_TOKEN", "")
API_KEY      = os.environ.get("ANTHROPIC_API_KEY", "")
FOCUS        = os.environ.get("FOCUS", "bugs, security, performance, correctness")

# Files the backend team can safely auto-fix
SAFE_TO_FIX = [
    "backend/app/config.py",
    "backend/app/database.py",
    "backend/app/main.py",
    "backend/start.sh",
    "backend/pyproject.toml",
    "render.yaml",
]

# Key files to review (read-only audit)
AUDIT_FILES = [
    "backend/app/api/v1/auth.py",
    "backend/app/api/v1/orders.py",
    "backend/app/api/v1/risk.py",
    "backend/app/risk/manager.py",
    "backend/app/risk/kelly.py",
    "backend/app/tasks/strategy_runner.py",
    "backend/app/tasks/algo_agent.py",
    "backend/app/brokers/alpaca.py",
    "backend/app/execution/smart_router.py",
]


def slack(channel: str, msg: str) -> None:
    if not SLACK_TOKEN:
        return
    try:
        httpx.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
            json={"channel": channel, "text": msg, "mrkdwn": True},
            timeout=10,
        )
    except Exception as e:
        print(f"Slack error: {e}")


def read_files(paths: list[str]) -> str:
    parts = []
    for rel in paths:
        p = REPO_ROOT / rel
        if p.exists():
            content = p.read_text()[:4000]
            parts.append(f"=== {rel} ===\n{content}")
    return "\n\n".join(parts)


def apply_and_push(files: list[dict]) -> list[str]:
    patched = []
    for item in files:
        rel = item.get("path", "")
        content = item.get("content", "")
        if not rel or not content or rel not in SAFE_TO_FIX:
            continue
        (REPO_ROOT / rel).write_text(content)
        patched.append(rel)

    if not patched:
        return []

    subprocess.run(["git", "add"] + patched, cwd=REPO_ROOT, check=True)
    r = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT)
    if r.returncode == 0:
        return []

    subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "push", "origin", BRANCH], cwd=REPO_ROOT, check=True)
    return patched


def _run_gemini_audit() -> None:
    """Use Gemini (free) to audit the backend when Anthropic API key is disabled."""
    gemini_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY_1", ""))
    if not gemini_key:
        print("No GEMINI_API_KEY either — backend audit skipped")
        slack("#engineering", "⚠️ *Backend AI Team:* Skipped (no LLM key available)")
        return

    print("Backend AI Team starting audit via Gemini...")
    context = read_files(AUDIT_FILES + SAFE_TO_FIX)
    prompt = (
        f"You are QuantEdge's senior backend engineer.\n"
        f"Focus: {FOCUS}\n\n"
        f"Review these files and return a brief JSON audit report with findings.\n"
        f"Format: {{\"findings\": [{{\"severity\": \"high|medium|low\", \"file\": \"...\", \"issue\": \"...\"}}], "
        f"\"summary\": \"2 sentence assessment\"}}\n\n"
        f"Files:\n{context[:6000]}"
    )
    try:
        import urllib.request
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
        findings = result.get("findings", [])
        summary = result.get("summary", "Gemini audit complete")
        print(f"Summary: {summary}\nFindings: {len(findings)}")
        lines = [f"🔍 *Backend AI Audit (Gemini)* — {len(findings)} findings", f"_{summary}_"]
        for f in findings[:6]:
            sev = f.get("severity", "medium").upper()
            icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}.get(sev, "❓")
            lines.append(f"{icon} `{f.get('file', '?')}`: {f.get('issue', '?')}")
        slack("#engineering", "\n".join(lines))
    except Exception as exc:
        print(f"Gemini audit failed: {exc}")
        slack("#engineering", f"⚠️ *Backend AI Team:* Gemini audit failed: {exc}")


def main() -> None:
    if not API_KEY or API_KEY == "disabled":
        print("No ANTHROPIC_API_KEY (or set to 'disabled') — backend team uses Gemini fallback")
        _run_gemini_audit()
        return

    print("Backend AI Team starting audit...")

    context = read_files(AUDIT_FILES + SAFE_TO_FIX)
    client = anthropic.Anthropic(api_key=API_KEY)

    system = """You are QuantEdge's senior backend engineer conducting a code audit.
Review the provided backend files for:
1. Security issues (missing auth, SQL injection, exposed secrets, missing rate limits)
2. Correctness bugs (wrong logic, off-by-one, unhandled exceptions, race conditions)
3. Performance issues (N+1 queries, missing indexes, sync calls in async context)
4. Configuration problems (missing env vars, wrong defaults, IPv6 issues)
5. Dependency issues (conflicting versions, missing packages)

Return ONLY valid JSON (no markdown, no prose):
{
  "findings": [
    {
      "severity": "critical|high|medium|low",
      "file": "path/to/file.py",
      "issue": "one-sentence description",
      "fix": "how to fix it"
    }
  ],
  "auto_fix_files": [
    {
      "path": "relative/path/from/repo/root",
      "content": "COMPLETE new file content"
    }
  ],
  "summary": "2-3 sentence overall assessment"
}

Rules:
- Only include auto_fix_files for files in: backend/app/config.py, backend/app/database.py,
  backend/app/main.py, backend/start.sh, backend/pyproject.toml, render.yaml
- Never auto-fix strategy or ML model files (too risky)
- Include up to 3 auto_fix_files maximum
- If no fixes are needed, set auto_fix_files to []"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8000,
        system=system,
        messages=[{"role": "user", "content": f"Focus: {FOCUS}\n\nFiles:\n{context}"}],
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
                print("Could not parse response JSON")
                return
        else:
            print("No JSON found in response")
            return

    findings = data.get("findings", [])
    summary  = data.get("summary", "Backend audit complete")
    patches  = data.get("auto_fix_files", [])

    # Print findings
    print(f"\n=== Backend Audit Results ===")
    print(f"Summary: {summary}")
    print(f"Findings: {len(findings)}")
    for f in findings:
        sev = f.get("severity", "?").upper()
        icon = {"CRITICAL": "🚨", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}.get(sev, "❓")
        print(f"  {icon} [{sev}] {f.get('file', '?')}: {f.get('issue', '?')}")

    # Apply auto-fixes
    patched = apply_and_push(patches) if patches else []

    # Report to Slack
    if not findings and not patched:
        slack("#engineering", f"✅ *Backend AI Team:* All systems clean\n_{summary}_")
        return

    ICONS = {"critical": "🚨", "high": "🔴", "medium": "🟡", "low": "🔵"}
    critical_count = sum(1 for f in findings if f.get("severity") == "critical")
    high_count     = sum(1 for f in findings if f.get("severity") == "high")

    lines = [f"🔍 *Backend AI Audit* — {len(findings)} findings", f"_{summary}_", ""]
    for f in findings[:8]:
        icon = ICONS.get(f.get("severity", ""), "❓")
        lines.append(f"{icon} `{f.get('file', '?')}`: {f.get('issue', '?')}")

    if patched:
        lines.append(f"\n✅ Auto-fixed: {', '.join(patched)}")

    channel = "#risk-alerts" if critical_count > 0 else "#engineering"
    slack(channel, "\n".join(lines))

    if critical_count > 0 or high_count > 1:
        slack("#engineering", f"⚠️ Backend team found {critical_count} critical + {high_count} high severity issues. Check #risk-alerts")


if __name__ == "__main__":
    main()
