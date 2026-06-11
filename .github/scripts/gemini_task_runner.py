"""
QuantEdge Gemini/Groq Task Runner — offloads complex multi-step work to free LLMs.

Reads GitHub Issues labeled "gemini-task", sends each task to Gemini Flash
(or Groq fallback), applies multi-file changes, verifies syntax, commits,
and posts progress to #engineering.

Unlike free_agent_engineer.py (50-line bug fixes), this handles:
  - Multi-file feature implementations
  - Refactoring tasks
  - Test writing
  - Documentation
  - Config changes

Environment variables:
  GH_TOKEN        — GitHub token (repo + issues write)
  GH_REPO         — owner/repo
  GEMINI_API_KEY  — primary (free 1500 req/day)
  GEMINI_API_KEY_2, GEMINI_API_KEY_3 — fallback keys
  GROQ_API_KEY    — fallback (free 500k tok/day)
  SLACK_BOT_TOKEN — optional Slack posting
  ALLOW_PAID_APIS — must be "False"

Usage:
  python .github/scripts/gemini_task_runner.py
  python .github/scripts/gemini_task_runner.py --issue 42
  python .github/scripts/gemini_task_runner.py --task "Add X to Y"  # inline task
"""

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


def _resolve_key(*names: str) -> str:
    for name in names:
        v = os.environ.get(name, "")
        if v: return v
        if not name[-1].isdigit():
            v = os.environ.get(name + "_1", "")
            if v: return v
    return ""


# ── Safety ────────────────────────────────────────────────────────────────────
ALLOW_PAID_APIS: bool = False
_env = os.environ.get("ALLOW_PAID_APIS", "False").strip().lower()
if _env in ("true", "1", "yes"):
    print("[gemini-runner] ALLOW_PAID_APIS must stay False. Aborting.")
    sys.exit(1)

MAX_TASKS_PER_RUN = 2
AGENT_LABEL = "gemini-task"
REPO_ROOT = Path(__file__).resolve().parents[2]

SAFE_PATHS = ("backend/", "frontend/", ".github/scripts/", ".github/workflows/", "experiments/")
FORBIDDEN_PATHS = (".env", "secrets", "credentials", "private_key", "passlib")

# ── Env ───────────────────────────────────────────────────────────────────────
GH_TOKEN = os.environ.get("GH_TOKEN", "")
GH_REPO = os.environ.get("GH_REPO", "")
GEMINI_KEYS = [
    os.environ.get(k, "").strip()
    for k in ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3",
              "GEMINI_API_KEY", "GEMINI_API_KEY_4", "GEMINI_API_KEY_5"]
    if os.environ.get(k, "").strip()
]
GROQ_KEYS = [
    os.environ.get(k, "").strip()
    for k in ["GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3",
              "GROQ_API_KEY", "GROQ_API_KEY_4", "GROQ_API_KEY_5"]
    if os.environ.get(k, "").strip()
]
SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "").strip()

# ── LLM callers ───────────────────────────────────────────────────────────────

class _RateLimited(Exception):
    """All providers returned 429 — retry next scheduled run."""


def _call_gemini(prompt: str, api_key: str, max_tokens: int = 8000) -> str:
    if not api_key:
        raise RuntimeError("empty api_key")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2},
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise _RateLimited("Gemini 429")
        raise
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_groq(prompt: str, api_key: str, max_tokens: int = 4000) -> str:
    body = json.dumps({
        "model": "llama-3.3-70b-versatile",
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": "You are an expert Python/TypeScript engineer for QuantEdge."},
            {"role": "user", "content": prompt},
        ],
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise _RateLimited("Groq 429")
        raise
    return data["choices"][0]["message"]["content"]


def call_free_llm(prompt: str, max_tokens: int = 8000) -> tuple[str, str]:
    """Try Gemini keys then Groq. On 429 waits 15s and retries once.
    Raises _RateLimited (not RuntimeError) when all providers throttled — caller exits 0."""
    throttled = 0
    active_keys = [k for k in GEMINI_KEYS if k]

    for key in active_keys:
        try:
            text = _call_gemini(prompt, key, max_tokens)
            if text and len(text.strip()) > 50:
                return text.strip(), "gemini-flash"
        except _RateLimited as e:
            print(f"  [gemini] {e} — trying next key")
            throttled += 1
            time.sleep(3)
        except Exception as e:
            print(f"  [gemini] failed: {e}")

    # All keys rate-limited — wait 15s and retry once per key
    if throttled == len(active_keys) and active_keys:
        print("  [gemini] all keys throttled — waiting 15s before retry")
        time.sleep(15)
        for key in active_keys:
            try:
                text = _call_gemini(prompt, key, max_tokens)
                if text and len(text.strip()) > 50:
                    return text.strip(), "gemini-flash-retry"
            except (_RateLimited, Exception) as e:
                print(f"  [gemini-retry] {e}")

    for groq_key in GROQ_KEYS:
        try:
            text = _call_groq(prompt, groq_key, min(max_tokens, 4000))
            if text and len(text.strip()) > 50:
                return text.strip(), "groq-llama3"
        except _RateLimited as e:
            print(f"  [groq] {e} — trying next key")
            time.sleep(2)
        except Exception as e:
            print(f"  [groq] failed: {e}")

    if not GEMINI_KEYS and not GROQ_KEYS:
        raise _RateLimited("No API keys configured — add GEMINI_API_KEY_1/GROQ_API_KEY_1 to secrets")
    raise _RateLimited("All free LLM providers throttled — issues stay open for next run")


# ── GitHub API ─────────────────────────────────────────────────────────────────

def _gh(method: str, path: str, body: dict | None = None) -> dict:
    url = f"https://api.github.com{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.reason, "status": e.code}


def fetch_open_tasks() -> list[dict]:
    """Fetch open issues labeled 'gemini-task'."""
    if not GH_TOKEN or not GH_REPO:
        return []
    data = _gh("GET", f"/repos/{GH_REPO}/issues?labels={AGENT_LABEL}&state=open&per_page=10")
    if isinstance(data, list):
        return data[:MAX_TASKS_PER_RUN]
    return []


def close_issue(issue_num: int, comment: str) -> None:
    _gh("POST", f"/repos/{GH_REPO}/issues/{issue_num}/comments", {"body": comment})
    _gh("PATCH", f"/repos/{GH_REPO}/issues/{issue_num}", {"state": "closed"})


def create_issue(title: str, body: str) -> int | None:
    """Create a gemini-task issue. Returns issue number."""
    if not GH_TOKEN or not GH_REPO:
        print(f"[dispatch] No GH_TOKEN — would create issue: {title}")
        return None
    # Ensure label exists
    _gh("POST", f"/repos/{GH_REPO}/labels", {
        "name": AGENT_LABEL, "color": "0075ca", "description": "Task for Gemini/Groq agents"
    })
    resp = _gh("POST", f"/repos/{GH_REPO}/issues", {
        "title": title,
        "body": body,
        "labels": [AGENT_LABEL],
    })
    num = resp.get("number")
    if num:
        print(f"[dispatch] Created issue #{num}: {title}")
    return num


# ── Slack ──────────────────────────────────────────────────────────────────────

def _slack_post(channel: str, text: str) -> None:
    if not SLACK_TOKEN:
        return
    body = json.dumps({"channel": channel, "text": text}).encode()
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage", data=body,
        headers={"Authorization": f"Bearer {SLACK_TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


# ── Task execution ─────────────────────────────────────────────────────────────

_TASK_PROMPT = """
You are an autonomous engineering agent for QuantEdge (FastAPI backend + React/TypeScript frontend).
Implement the following task by producing file changes.

## Task
{title}

## Description
{body}

## Rules
- Only modify files under: {safe_paths}
- NEVER modify .env, secrets, credentials, or private_key files
- Python files must be syntactically valid (ast.parse must pass)
- TypeScript must not introduce obvious type errors
- Prefer targeted edits over full rewrites
- If a file doesn't exist yet, set old_snippet to ""

## Output format
Return a JSON object (no markdown fences) with this exact structure:
{{
  "summary": "one paragraph explaining what you did and why",
  "files": [
    {{
      "file_path": "relative/path/from/repo/root",
      "old_snippet": "exact text to replace (empty string to append/create)",
      "new_snippet": "replacement text",
      "description": "what this change does"
    }}
  ]
}}

If this task is truly too complex for a single LLM response, return:
{{
  "summary": "too_complex: <reason>",
  "files": []
}}
""".strip()


def _build_prompt(title: str, body: str) -> str:
    return _TASK_PROMPT.format(
        title=title,
        body=body[:3000],
        safe_paths=", ".join(SAFE_PATHS),
    )


def _parse_response(raw: str) -> dict:
    clean = re.sub(r"```(?:json)?\s*", "", raw)
    clean = re.sub(r"```\s*$", "", clean).strip()
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start == -1 or end <= 0:
        raise ValueError("No JSON in LLM response")
    return json.loads(clean[start:end])


def _validate_file_change(fc: dict) -> str | None:
    """Returns error string or None if valid."""
    path = fc.get("file_path", "")
    if not path:
        return "empty file_path"
    if not any(path.startswith(p) for p in SAFE_PATHS):
        return f"unsafe path: {path}"
    if any(fp in path.lower() for fp in FORBIDDEN_PATHS):
        return f"forbidden path: {path}"
    return None


def _apply_file_change(fc: dict) -> tuple[bool, str]:
    """Apply a single file change. Returns (success, message)."""
    rel_path = fc["file_path"]
    old = fc.get("old_snippet", "")
    new = fc.get("new_snippet", "")
    full_path = REPO_ROOT / rel_path

    if old:
        if not full_path.exists():
            return False, f"{rel_path}: file not found"
        content = full_path.read_text(encoding="utf-8")
        if old not in content:
            return False, f"{rel_path}: old_snippet not found in file"
        full_path.write_text(content.replace(old, new, 1), encoding="utf-8")
    else:
        # Create or append
        full_path.parent.mkdir(parents=True, exist_ok=True)
        if full_path.exists():
            full_path.write_text(full_path.read_text(encoding="utf-8") + "\n" + new, encoding="utf-8")
        else:
            full_path.write_text(new, encoding="utf-8")

    # Syntax check for Python files
    if rel_path.endswith(".py"):
        try:
            ast.parse(full_path.read_text(encoding="utf-8"))
        except SyntaxError as e:
            # Revert
            if old:
                full_path.write_text(full_path.read_text().replace(new, old, 1), encoding="utf-8")
            return False, f"{rel_path}: syntax error after patch — {e}"

    return True, f"{rel_path}: applied"


def verify_fix_applied(report: str) -> tuple[bool, str]:
    """Verify the committed patch is sound: syntax-clean, non-empty, no LLM exhaustion.

    Returns (ok, reason).  Called after a successful commit so HEAD~1 is the pre-fix state.
    """
    # 1. Detect LLM exhaustion markers — means no real work was done
    _EXHAUSTION_MARKERS = [
        "LLM exhausted",
        "All free LLM providers exhausted",
        "No changes made",
        "too_complex",
        "Rate limited",
    ]
    for marker in _EXHAUSTION_MARKERS:
        if marker.lower() in report.lower():
            return False, f"Report contains exhaustion marker: '{marker}'"

    # 2. Ensure at least one file was committed
    diff_stat = subprocess.run(
        ["git", "diff", "--stat", "HEAD~1", "HEAD"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    ).stdout.strip()
    if not diff_stat:
        return False, "No files differ between HEAD~1 and HEAD — patch produced no real changes"

    # 3. Python syntax check on every .py file touched in this commit
    changed_py = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~1", "HEAD", "--", "*.py"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    ).stdout.splitlines()
    for rel in changed_py:
        full = REPO_ROOT / rel
        if not full.exists():
            continue  # deleted file — skip
        try:
            ast.parse(full.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            return False, f"Syntax error in {rel} after patch: {exc}"

    summary = diff_stat.splitlines()[0] if diff_stat else "ok"
    return True, f"Fix verified: {summary}"


def execute_task(title: str, body: str) -> tuple[bool, str]:
    """Run a task through Gemini/Groq, apply changes, return (success, summary)."""
    print(f"[task] Running: {title[:80]}")
    prompt = _build_prompt(title, body)

    try:
        raw, provider = call_free_llm(prompt, max_tokens=8000)
        print(f"  [task] LLM response from {provider}: {len(raw)} chars")
    except _RateLimited as e:
        return None, str(e)  # None = rate-limited, leave issue open
    except Exception as e:
        return False, f"LLM error: {e}"

    try:
        result = _parse_response(raw)
    except Exception as e:
        return False, f"JSON parse failed: {e}\n\nRaw:\n{raw[:500]}"

    summary = result.get("summary", "")
    if summary.startswith("too_complex"):
        return False, summary

    files = result.get("files", [])
    if not files:
        return False, "LLM returned no file changes"

    applied = []
    failed = []
    for fc in files:
        err = _validate_file_change(fc)
        if err:
            failed.append(f"SKIP {fc.get('file_path', '?')}: {err}")
            continue
        ok, msg = _apply_file_change(fc)
        (applied if ok else failed).append(msg)

    if not applied:
        return False, f"No changes applied. Failures: {'; '.join(failed)}"

    # Commit
    changed_files = [fc["file_path"] for fc in files if any(fc["file_path"] in m for m in applied)]
    if changed_files:
        subprocess.run(["git", "config", "user.email", "gemini-runner@quantedge.ai"], cwd=REPO_ROOT)
        subprocess.run(["git", "config", "user.name", "QuantEdge Gemini Runner"], cwd=REPO_ROOT)
        subprocess.run(["git", "add"] + [fc["file_path"] for fc in files], cwd=REPO_ROOT)
        result_commit = subprocess.run(
            ["git", "commit", "-m", f"feat(gemini): {title[:72]}\n\n{summary[:500]}"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if result_commit.returncode == 0:
            subprocess.run(
                ["git", "push", "origin", "HEAD:main"],
                cwd=REPO_ROOT, capture_output=True,
            )
            print(f"  [task] committed + pushed {len(applied)} file(s)")

    report = f"✅ Applied {len(applied)} change(s) via {provider}\n"
    report += "\n".join(f"  • {m}" for m in applied)
    if failed:
        report += f"\n⚠️  Skipped: {'; '.join(failed)}"
    return True, report


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"[gemini-runner] Starting — {datetime.now(timezone.utc).isoformat()}")

    # ── Inline task mode: --task "description"
    if "--task" in sys.argv:
        idx = sys.argv.index("--task")
        if idx + 1 < len(sys.argv):
            task_text = sys.argv[idx + 1]
            ok, report = execute_task(task_text, "Dispatched inline via --task flag.")
            print(report)
            if ok is None:
                print("[gemini-runner] Rate-limited — will retry next run.")
                return 0
            _slack_post("engineering", f":robot_face: *Gemini runner* — `{task_text[:60]}`\n{report}")
            return 0 if ok else 1

    # ── Single issue mode: --issue 42
    if "--issue" in sys.argv:
        idx = sys.argv.index("--issue")
        if idx + 1 < len(sys.argv):
            issue_num = int(sys.argv[idx + 1])
            data = _gh("GET", f"/repos/{GH_REPO}/issues/{issue_num}")
            title = data.get("title", "")
            body = data.get("body", "") or ""
            ok, report = execute_task(title, body)
            if ok is None:
                print(f"[gemini-runner] Rate-limited on issue #{issue_num} — leaving open for retry.")
                return 0  # don't close issue, don't fail workflow
            if ok:
                ok, verify_msg = verify_fix_applied(report)
                if ok:
                    report = f"{verify_msg}\n\n" + report
                    close_issue(issue_num, f"## ✅ Done\n\n{report}")
                    _slack_post("engineering", f":robot_face: *Gemini runner* closed issue #{issue_num}\n{report[:300]}")
                else:
                    report = f"Verification failed: {verify_msg}\n\n" + report
                    _gh("POST", f"/repos/{GH_REPO}/issues/{issue_num}/comments",
                        {"body": f"## ⚠️ Verification failed — left open for retry\n\n{report}"})
                    _slack_post("engineering", f":warning: *Gemini runner* — verification failed for #{issue_num}: {verify_msg}")
            else:
                _gh("POST", f"/repos/{GH_REPO}/issues/{issue_num}/comments",
                    {"body": f"## ⚠️ Attempt failed — left open for retry\n\n{report}"})
                _slack_post("engineering", f":warning: *Gemini runner* could not fix #{issue_num} — left open\n{report[:200]}")
            return 0  # always exit 0; failures are tracked via issue comments

    # ── Batch mode: process all open gemini-task issues
    tasks = fetch_open_tasks()
    if not tasks:
        print("[gemini-runner] No open gemini-task issues found.")
        return 0

    results = []
    for issue in tasks:
        num = issue["number"]
        title = issue.get("title", "")
        body = issue.get("body", "") or ""
        print(f"\n[gemini-runner] Processing issue #{num}: {title[:80]}")
        ok, report = execute_task(title, body)
        if ok is None:
            print(f"  [gemini-runner] Rate-limited on #{num} — leaving open for retry.")
            results.append((num, None, report[:200]))
            break  # stop processing; all keys throttled, no point continuing
        verify_msg = ""
        if ok:
            ok, verify_msg = verify_fix_applied(report)
            if not ok:
                report = f"Verification failed: {verify_msg}\n\n" + report
            else:
                report = f"{verify_msg}\n\n" + report
        if ok:
            close_issue(num, f"## ✅ Done\n\n{report}")
            print(f"  [gemini-runner] Closed #{num} — {verify_msg or 'ok'}")
        else:
            # Add a failure comment but keep issue OPEN for the next run to retry
            _gh("POST", f"/repos/{GH_REPO}/issues/{num}/comments",
                {"body": f"## ⚠️ Attempt failed — left open for retry\n\n{report}"})
            print(f"  [gemini-runner] Issue #{num} left OPEN — fix failed or unverified")
        results.append((num, ok, report[:200]))
        time.sleep(2)

    # Post summary to #engineering
    lines = [f":robot_face: *Gemini Task Runner* — {len(results)} task(s) processed"]
    for num, ok, rep in results:
        icon = "✅" if ok else ("⏳" if ok is None else "❌")
        lines.append(f"{icon} Issue #{num}: {rep[:100]}")
    _slack_post("engineering", "\n".join(lines))

    # Only hard-fail if a task actually errored (not rate-limited)
    failed = sum(1 for _, ok, _ in results if ok is False)
    return 1 if failed == len(results) and failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
