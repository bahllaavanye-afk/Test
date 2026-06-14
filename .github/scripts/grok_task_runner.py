"""
QuantEdge Grok Task Runner — fast coding via xAI Grok API.

Reads GitHub Issues labeled "grok-task" and executes them using Grok Mini
(xAI's fast coding model). Grok is optimized for quick, precise edits —
bug fixes, config changes, small features. For complex multi-file work,
tasks get routed to codex-task (GPT-4o) instead.

After completing a task, commits + posts to Slack. Claude acceptance gate
then reviews and closes the issue.

Environment variables:
  GH_TOKEN            — GitHub token (repo + issues write)
  GH_REPO             — owner/repo
  XAI_API_KEY         — xAI Grok API key (https://console.x.ai)
  GROK_API_KEY        — fallback key name
  GROQ_API_KEY        — fallback LLM if Grok unavailable (note: Groq ≠ Grok)
  GEMINI_API_KEY      — second fallback
  SLACK_BOT_TOKEN     — optional Slack posting

Usage:
  python .github/scripts/grok_task_runner.py               # batch
  python .github/scripts/grok_task_runner.py --issue 42    # specific issue
  python .github/scripts/grok_task_runner.py --task "Fix X"  # inline
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    from llm_common import slack_post as _slack_post
except ImportError:
    def _slack_post(ch: str, msg: str) -> None: print(f"[slack] {ch}: {msg[:200]}")

AGENT_LABEL = "grok-task"
REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_TASKS_PER_RUN = 3

SAFE_PATHS = ("backend/", "frontend/", ".github/scripts/", ".github/workflows/", "experiments/")
FORBIDDEN_PATHS = (".env", "secrets", "credentials", "private_key")

GH_TOKEN = os.environ.get("GH_TOKEN", os.environ.get("GITHUB_TOKEN", ""))
GH_REPO = os.environ.get("GH_REPO", "")

# xAI API key — try multiple env names
XAI_KEY = next(
    (os.environ.get(k, "") for k in ["XAI_API_KEY", "GROK_API_KEY", "XAI_KEY"] if os.environ.get(k, "")),
    ""
)
GROQ_KEY = os.environ.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY_1", ""))
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY_1", ""))

SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")


# ── LLM callers ───────────────────────────────────────────────────────────────

class _RateLimited(Exception):
    pass


def _call_grok(prompt: str, max_tokens: int = 6000) -> str:
    if not XAI_KEY:
        raise RuntimeError("No xAI key (XAI_API_KEY or GROK_API_KEY)")
    body = json.dumps({
        "model": "grok-3-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Grok, a fast and precise coding assistant for QuantEdge, "
                    "an institutional-grade quantitative trading platform. "
                    "You write clean Python and TypeScript. You never hardcode secrets. "
                    "You always use SQLAlchemy ORM, never raw SQL."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }).encode()
    req = urllib.request.Request(
        "https://api.x.ai/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {XAI_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise _RateLimited("Grok 429")
        raise


def _call_groq_fallback(prompt: str, max_tokens: int = 4000) -> str:
    if not GROQ_KEY:
        raise RuntimeError("No Groq fallback key")
    body = json.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are an expert Python/TypeScript engineer for QuantEdge."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"].strip()


def _call_gemini_fallback(prompt: str) -> str:
    if not GEMINI_KEY:
        raise RuntimeError("No Gemini fallback key")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"maxOutputTokens": 4000}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = json.loads(resp.read())
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def llm(prompt: str, max_tokens: int = 6000) -> str:
    for fn in [_call_grok, _call_groq_fallback, _call_gemini_fallback]:
        try:
            return fn(prompt, max_tokens) if fn == _call_grok else fn(prompt)
        except _RateLimited:
            print(f"[grok-runner] {fn.__name__} rate-limited, trying next", flush=True)
        except Exception as e:
            print(f"[grok-runner] {fn.__name__} failed: {e}", flush=True)
    raise RuntimeError("All LLM providers failed")


# ── GitHub helpers ─────────────────────────────────────────────────────────────

def _gh(method: str, path: str, body: dict | None = None) -> dict | list:
    url = f"https://api.github.com/repos/{GH_REPO}/{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read()) if resp.status not in (204,) else {}
    except urllib.error.HTTPError as e:
        print(f"[grok-runner] GH {method} {path}: HTTP {e.code}", flush=True)
        return {}


def fetch_open_tasks() -> list[dict]:
    issues = _gh("GET", f"issues?labels={AGENT_LABEL}&state=open&per_page=10")
    return [i for i in (issues if isinstance(issues, list) else [])
            if i.get("state") == "open"]


# ── Code change helpers ────────────────────────────────────────────────────────

_FILE_CHANGE_RE = re.compile(
    r"```(?:python|typescript|tsx?|js|jsx|yaml|yml|bash|sh|json)?\s*\n"
    r"# FILE: (.+?)\n(.*?)```",
    re.DOTALL,
)


def _safe_path(rel: str) -> Path | None:
    rel = rel.lstrip("/")
    if any(rel.startswith(p) for p in SAFE_PATHS) and not any(f in rel for f in FORBIDDEN_PATHS):
        return REPO_ROOT / rel
    return None


def _validate_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"[grok-runner] Python syntax error: {e}", flush=True)
        return False


def apply_changes(response: str) -> list[str]:
    changed: list[str] = []
    for rel_path, content in _FILE_CHANGE_RE.findall(response):
        rel_path = rel_path.strip()
        path = _safe_path(rel_path)
        if path is None:
            print(f"[grok-runner] Skipping unsafe path: {rel_path}", flush=True)
            continue
        if path.suffix == ".py" and not _validate_python(content):
            print(f"[grok-runner] Skipping {rel_path} — Python syntax errors", flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        changed.append(rel_path)
        print(f"[grok-runner] Wrote {rel_path}", flush=True)
    return changed


def git_commit_push(issue_number: int, title: str, changed: list[str]) -> bool:
    try:
        subprocess.run(["git", "config", "user.email", "grok@quantedge.ai"],
                       cwd=REPO_ROOT, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "QuantEdge Grok Agent"],
                       cwd=REPO_ROOT, check=True, capture_output=True)
        for f in changed:
            subprocess.run(["git", "add", f], cwd=REPO_ROOT, check=True, capture_output=True)
        msg = f"feat(grok-agent): #{issue_number} {title[:60]}"
        subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, check=True, capture_output=True)
        for i in range(1, 5):
            result = subprocess.run(
                ["git", "push", "-u", "origin", "claude/advanced-trading-bot-d5Lmw"],
                cwd=REPO_ROOT, capture_output=True)
            if result.returncode == 0:
                return True
            subprocess.run(
                ["git", "pull", "--rebase", "--quiet", "origin", "claude/advanced-trading-bot-d5Lmw"],
                cwd=REPO_ROOT, capture_output=True)
            time.sleep(i * 2)
        return False
    except subprocess.CalledProcessError as e:
        print(f"[grok-runner] git error: {e}", flush=True)
        return False


# ── Task execution ─────────────────────────────────────────────────────────────

_TASK_PROMPT = """\
You are Grok (xAI), a fast and precise coding assistant for QuantEdge.
Implement the following task for the QuantEdge quantitative trading platform.

## Task
{title}

{body}

## Platform Context
- Backend: FastAPI + SQLAlchemy async + Pydantic v2
- Frontend: React 18 + TypeScript + Vite + TanStack Query
- Never hardcode secrets or mock data
- Always use SQLAlchemy ORM (no raw SQL)
- JWT auth via Depends(get_current_user) on all new endpoints
- TypeScript must compile cleanly

## Output Format
For EACH file to create or modify, use EXACTLY this format:

```python
# FILE: backend/app/path/to/file.py
<complete file content>
```

```typescript
# FILE: frontend/src/path/to/file.tsx
<complete file content>
```

Provide complete working code. No TODOs. No truncation.
"""


def execute_task(issue: dict) -> bool:
    number = issue["number"]
    title = issue["title"]
    body = (issue.get("body") or "")[:3000]

    print(f"\n[grok-runner] Task #{number}: {title}", flush=True)

    prompt = _TASK_PROMPT.format(title=title, body=body)
    try:
        response = llm(prompt)
    except Exception as e:
        _gh("POST", f"issues/{number}/comments",
            {"body": f"❌ Grok task runner failed: {e}"})
        return False

    changed = apply_changes(response)
    if not changed:
        print(f"[grok-runner] #{number}: no file changes found in response", flush=True)
        _gh("POST", f"issues/{number}/comments",
            {"body": "⚠️ Grok produced no file changes. Re-labeling for manual review."})
        return False

    ok = git_commit_push(number, title, changed)
    status = "✅ committed" if ok else "⚠️ commit failed"

    _gh("POST", f"issues/{number}/comments", {"body": (
        f"🤖 **Grok Agent** completed task\n\n"
        f"**Status:** {status}\n"
        f"**Files changed:** {', '.join(f'`{f}`' for f in changed)}\n\n"
        f"_Claude acceptance gate will review and close this issue._"
    )})

    _slack_post("#engineering",
        f"🚀 *Grok Agent* completed #{number}: {title}\n"
        f"Files: {', '.join(changed[:5])}\n"
        f"Status: {status}")

    return ok


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int)
    parser.add_argument("--task", type=str)
    args = parser.parse_args()

    if args.task:
        # Inline task — no GitHub issue
        fake_issue = {"number": 0, "title": args.task, "body": ""}
        execute_task(fake_issue)
        return

    if args.issue:
        issue = _gh("GET", f"issues/{args.issue}")
        if issue:
            execute_task(issue)
        return

    # Batch: process open grok-task issues
    tasks = fetch_open_tasks()
    print(f"[grok-runner] {len(tasks)} open grok-task issues", flush=True)
    done = 0
    for issue in tasks[:MAX_TASKS_PER_RUN]:
        if execute_task(issue):
            done += 1
        time.sleep(5)
    print(f"[grok-runner] Done: {done}/{min(len(tasks), MAX_TASKS_PER_RUN)}", flush=True)


if __name__ == "__main__":
    main()
