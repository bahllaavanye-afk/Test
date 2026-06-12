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
import time
from pathlib import Path

import anthropic
import httpx

REPO_ROOT    = Path(__file__).parent.parent
BRANCH       = "claude/advanced-trading-bot-d5Lmw"
SLACK_TOKEN  = os.environ.get("SLACK_BOT_TOKEN", "")
API_KEY      = os.environ.get("ANTHROPIC_API_KEY", "")
FOCUS        = os.environ.get("FOCUS", "bugs, security, performance, correctness")

_DEDUP_FILE  = REPO_ROOT / ".github" / "state" / "backend_team_dedup.json"
_CLEAN_COOLDOWN_SECS = 14400  # 4 hours — only post "all clean" once per 4h


def _already_posted_clean() -> bool:
    """Return True if we posted 'all systems clean' within the cooldown window."""
    try:
        if _DEDUP_FILE.exists():
            d = json.loads(_DEDUP_FILE.read_text())
            last = d.get("last_clean_post", 0)
            return (time.time() - last) < _CLEAN_COOLDOWN_SECS
    except Exception:
        pass
    return False


def _record_clean_post() -> None:
    _DEDUP_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        d = json.loads(_DEDUP_FILE.read_text()) if _DEDUP_FILE.exists() else {}
    except Exception:
        d = {}
    d["last_clean_post"] = time.time()
    _DEDUP_FILE.write_text(json.dumps(d))

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

    commit_msg = f"fix(auto): backend team auto-fix [{', '.join(patched)}]"
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "push", "origin", BRANCH], cwd=REPO_ROOT, check=True)
    return patched


def _call_free_llm(prompt: str, max_tokens: int = 1024) -> str | None:
    """Try free LLM providers in cascade order; return first successful response text."""
    providers = [
        ("gemini", os.environ.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY_1", "")),
         "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "gemini-2.0-flash"),
        ("groq", os.environ.get("GROQ_API_KEY", ""),
         "https://api.groq.com/openai/v1/chat/completions", "llama-3.3-70b-versatile"),
        ("deepseek", os.environ.get("DEEPSEEK_API_KEY", ""),
         "https://api.deepseek.com/v1/chat/completions", "deepseek-chat"),
        ("together", os.environ.get("TOGETHER_API_KEY", ""),
         "https://api.together.xyz/v1/chat/completions", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        ("cerebras", os.environ.get("CEREBRAS_API_KEY", ""),
         "https://api.cerebras.ai/v1/chat/completions", "llama-3.3-70b"),
        ("sambanova", os.environ.get("SAMBANOVA_API_KEY", ""),
         "https://api.sambanova.ai/v1/chat/completions", "Meta-Llama-3.3-70B-Instruct"),
    ]
    import urllib.request
    for name, key, url, model in providers:
        if not key or key in ("disabled", ""):
            continue
        try:
            payload = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            }).encode()
            req = urllib.request.Request(url, data=payload,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            print(f"Free LLM response from {name}")
            return content
        except Exception as e:
            print(f"Provider {name} failed: {e}")
            continue
    return None


def _run_gemini_audit() -> None:
    """Use free LLM cascade to audit the backend when Anthropic API key is disabled."""
    gemini_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY_1", ""))
    has_any_key = any(os.environ.get(k, "") not in ("", "disabled") for k in [
        "GEMINI_API_KEY", "GEMINI_API_KEY_1", "GROQ_API_KEY", "DEEPSEEK_API_KEY",
        "TOGETHER_API_KEY", "CEREBRAS_API_KEY", "SAMBANOVA_API_KEY",
    ])
    if not has_any_key:
        print("No free LLM keys configured — backend audit skipped")
        slack("#engineering", "⚠️ *Backend AI Team:* Skipped (no LLM key available)")
        return

    print("Backend AI Team starting audit via free LLM cascade...")
    prompt = (
        f"You are QuantEdge's senior backend engineer.\n"
        f"Focus: {FOCUS}\n\n"
        f"Review these files and return a brief JSON audit report with findings.\n"
        f'Format: {{"findings": [{{"severity": "high|medium|low", "file": "...", "issue": "..."}}], '
        f'"summary": "2 sentence assessment"}}\n\n'
        f"Files:\n{context[:6000]}"
    )
    text = _call_free_llm(prompt, max_tokens=1024)
    if not text:
        print("All free LLM providers failed — audit skipped")
        slack("#engineering", "⚠️ *Backend AI Team:* All LLM providers failed")
        return
    try:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL) or re.search(r"(\{.*\})", text, re.DOTALL)
        result = json.loads(m.group(1)) if m else {"findings": [], "summary": text[:200]}
        findings = result.get("findings", [])
        summary = result.get("summary", "Free LLM audit complete")
        print(f"Summary: {summary}\nFindings: {len(findings)}")
        lines = [f"🔍 *Backend AI Audit (Free LLM)* — {len(findings)} findings", f"_{summary}_"]
        for f in findings[:6]:
            sev = f.get("severity", "medium").upper()
            icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}.get(sev, "❓")
            lines.append(f"{icon} `{f.get('file', '?')}`: {f.get('issue', '?')}")
        slack("#engineering", "\n".join(lines))
    except Exception as exc:
        print(f"Free LLM audit parse failed: {exc}")
        slack("#engineering", f"⚠️ *Backend AI Team:* Audit parse failed: {exc}")


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
        if not _already_posted_clean():
            slack("#engineering", f"✅ *Backend AI Team:* All systems clean\n_{summary}_")
            _record_clean_post()
        else:
            print("[backend-team] All clean — skipping Slack post (cooldown active)")
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
