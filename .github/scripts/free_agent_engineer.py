"""
QuantEdge Free-Agent Engineer — autonomous issue fixer.

Reads open GitHub Issues labeled "agent-fix-needed", calls free LLMs
(Gemini Flash → Groq fallback) to generate targeted code fixes, applies
them, verifies imports, commits, closes the issue, and posts to Slack.

Environment variables required:
  GH_TOKEN        — GitHub token with repo + issues write permissions
  GH_REPO         — owner/repo  e.g.  "org/quantedge"
  GEMINI_API_KEY  — Google AI Studio key
  GROQ_API_KEY    — Groq console key
  SLACK_BOT_TOKEN — Slack bot token (optional; skipped if absent)
  ALLOW_PAID_APIS — must be "False" (enforced)

Usage:
  python .github/scripts/free_agent_engineer.py
"""

import json
import os
import re
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path


def _resolve_key(*names: str) -> str:
    for name in names:
        v = os.environ.get(name, "")
        if v: return v
        if not name[-1].isdigit():
            v = os.environ.get(name + "_1", "")
            if v: return v
    return ""


# ─── Safety constants ─────────────────────────────────────────────────────────

ALLOW_PAID_APIS: bool = False          # NEVER set to True
MAX_ISSUES_PER_RUN: int = 3
MAX_ISSUE_AGE_DAYS: int = 7
MAX_LINES_CHANGED: int = 50            # skip if fix is too large
AGENT_LABEL: str = "agent-fix-needed"

# Allowed file prefixes for automated edits
SAFE_PATHS = (
    "backend/",
    "frontend/",
    ".github/scripts/",
)

# Paths that must never be touched
FORBIDDEN_PATHS = (
    ".env",
    ".env.",
    "secrets",
    "credentials",
    "private_key",
)

# LLM endpoints (free tier only)
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

GITHUB_API = "https://api.github.com"

# ─── Environment ──────────────────────────────────────────────────────────────

GH_TOKEN: str = os.environ.get("GH_TOKEN", "")
GH_REPO: str = os.environ.get("GH_REPO", "")
GEMINI_API_KEY: str = _resolve_key("GEMINI_API_KEY", "GEMINI_API_KEY_1")
GROQ_API_KEY: str = _resolve_key("GROQ_API_KEY", "GROQ_API_KEY_1")
SLACK_BOT_TOKEN: str = os.environ.get("SLACK_BOT_TOKEN", "")

# Validate safety flag — abort if someone tries to enable paid APIs
_env_allow = os.environ.get("ALLOW_PAID_APIS", "False").strip().lower()
if _env_allow in ("true", "1", "yes"):
    print("[free-agent] ALLOW_PAID_APIS must stay False. Aborting.")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ─── HTTP helpers (urllib only — no external libs) ────────────────────────────

def _http_json(
    url: str,
    payload: dict | None = None,
    headers: dict | None = None,
    method: str | None = None,
    timeout: int = 30,
) -> dict:
    """Make an HTTP request and return parsed JSON. Raises on HTTP errors."""
    data = json.dumps(payload).encode() if payload is not None else None
    if method is None:
        method = "POST" if data is not None else "GET"
    req = urllib.request.Request(url, data=data, method=method,
                                 headers=headers or {})
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {body[:400]}") from e


def _gh_headers() -> dict:
    return {
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


# ─── Secrets sanitiser ───────────────────────────────────────────────────────

_SECRET_PATTERN = re.compile(
    r"(AKIA|sk-|ghp_|gho_|xoxb-|xoxp-|AIza)[A-Za-z0-9_\-]{6,}",
    re.IGNORECASE,
)


def _sanitize(text: str) -> str:
    """Replace anything that looks like a secret with a placeholder."""
    return _SECRET_PATTERN.sub("[REDACTED]", text)


# ─── GitHub API helpers ───────────────────────────────────────────────────────

def _list_open_issues() -> list[dict]:
    """Return open issues labelled AGENT_LABEL, max age MAX_ISSUE_AGE_DAYS."""
    if not GH_TOKEN or not GH_REPO:
        print("[free-agent] GH_TOKEN / GH_REPO not set — skipping issue fetch")
        return []

    url = (
        f"{GITHUB_API}/repos/{GH_REPO}/issues"
        f"?state=open&labels={urllib.parse.quote(AGENT_LABEL)}&per_page=20"
    )
    try:
        issues = _http_json(url, headers=_gh_headers(), method="GET")
    except Exception as e:
        print(f"[free-agent] Failed to list issues: {e}")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_ISSUE_AGE_DAYS)
    fresh = []
    for iss in issues:
        created = datetime.fromisoformat(
            iss["created_at"].replace("Z", "+00:00")
        )
        if created >= cutoff:
            fresh.append(iss)

    print(f"[free-agent] Found {len(fresh)} fresh issue(s) labelled '{AGENT_LABEL}'")
    return fresh[:MAX_ISSUES_PER_RUN]


def _close_issue(issue_number: int, comment: str) -> None:
    """Post a comment then close the issue."""
    url_comment = f"{GITHUB_API}/repos/{GH_REPO}/issues/{issue_number}/comments"
    try:
        _http_json(url_comment, payload={"body": comment}, headers=_gh_headers())
    except Exception as e:
        print(f"[free-agent] Comment post failed: {e}")

    url_close = f"{GITHUB_API}/repos/{GH_REPO}/issues/{issue_number}"
    try:
        _http_json(url_close, payload={"state": "closed"}, headers=_gh_headers(),
                   method="PATCH")
        print(f"[free-agent] Closed issue #{issue_number}")
    except Exception as e:
        print(f"[free-agent] Issue close failed: {e}")


def _comment_on_issue(issue_number: int, comment: str) -> None:
    url = f"{GITHUB_API}/repos/{GH_REPO}/issues/{issue_number}/comments"
    try:
        _http_json(url, payload={"body": comment}, headers=_gh_headers())
    except Exception as e:
        print(f"[free-agent] Failed to comment on issue #{issue_number}: {e}")


# ─── LLM callers ─────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    """Call Gemini Flash. Returns text or raises."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 4096,
            "temperature": 0.2,
        },
    }
    resp = _http_json(url, payload=payload)
    candidates = resp.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {resp}")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


def _call_groq(prompt: str) -> str:
    """Call Groq (Llama-3.3-70b). Returns text or raises."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.2,
    }
    resp = _http_json(
        GROQ_URL,
        payload=payload,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    choices = resp.get("choices", [])
    if not choices:
        raise RuntimeError(f"Groq returned no choices: {resp}")
    return choices[0]["message"]["content"].strip()


def _call_llm(prompt: str) -> tuple[str, str]:
    """
    Try Gemini Flash first, fall back to Groq.
    Returns (text, agent_name).
    """
    for caller, name in [(_call_gemini, "gemini-flash"), (_call_groq, "groq-llama3")]:
        try:
            text = caller(prompt)
            print(f"[free-agent] LLM response from {name} ({len(text)} chars)")
            return text, name
        except Exception as e:
            print(f"[free-agent] {name} failed: {e}")
    raise RuntimeError("All free LLM providers failed")


# ─── Fix prompt builder ───────────────────────────────────────────────────────

def _build_fix_prompt(issue_title: str, issue_body: str) -> str:
    return textwrap.dedent(f"""
        You are an autonomous engineering agent for QuantEdge, an institutional
        quantitative trading platform built with Python (FastAPI) and
        React/TypeScript. Your job is to analyse a GitHub issue and produce a
        minimal, targeted code fix.

        ## Issue
        Title: {issue_title}
        Body:
        {issue_body}

        ## Instructions
        1. Identify the most likely root cause.
        2. Produce a fix that touches as few lines as possible (maximum {MAX_LINES_CHANGED} lines changed).
        3. Only modify files under: {", ".join(SAFE_PATHS)}
        4. NEVER modify .env files, secret files, or credential files.
        5. Return your answer in EXACTLY the following JSON format (no markdown fences, no extra text):

        {{
          "root_cause": "one-sentence explanation",
          "file_path": "relative/path/to/file.py",
          "old_snippet": "exact lines to replace (empty string if appending)",
          "new_snippet": "replacement lines",
          "explanation": "2-3 sentence explanation of what was changed and why"
        }}

        If the issue is too complex or ambiguous for a {MAX_LINES_CHANGED}-line fix, return:
        {{
          "root_cause": "too_complex",
          "file_path": "",
          "old_snippet": "",
          "new_snippet": "",
          "explanation": "Reason this cannot be auto-fixed"
        }}
    """).strip()


# ─── Fix parser & validator ───────────────────────────────────────────────────

def _parse_fix_response(raw: str) -> dict:
    """
    Extract JSON from LLM response.
    The LLM sometimes wraps in markdown fences — strip them first.
    """
    # Remove possible markdown code fences
    clean = re.sub(r"```(?:json)?\s*", "", raw)
    clean = re.sub(r"```\s*$", "", clean).strip()

    # Try to find the outermost JSON object
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(clean[start:end])


def _validate_fix(fix: dict) -> str | None:
    """
    Returns an error string if fix is invalid/unsafe, else None.
    Returns "too_complex" if LLM flagged it as such.
    """
    file_path: str = fix.get("file_path", "")
    old_snippet: str = fix.get("old_snippet", "")
    new_snippet: str = fix.get("new_snippet", "")

    if fix.get("root_cause") == "too_complex":
        return "too_complex"

    if not file_path:
        return "LLM returned empty file_path"

    # Must be under a safe prefix
    if not any(file_path.startswith(p) for p in SAFE_PATHS):
        return f"File '{file_path}' is not under a safe path ({SAFE_PATHS})"

    # Must not be a forbidden file
    if any(fp in file_path.lower() for fp in FORBIDDEN_PATHS):
        return f"File '{file_path}' matches a forbidden pattern"

    # Count changed lines
    old_lines = old_snippet.count("\n") + (1 if old_snippet else 0)
    new_lines = new_snippet.count("\n") + (1 if new_snippet else 0)
    delta = abs(new_lines - old_lines) + max(old_lines, new_lines)
    if delta > MAX_LINES_CHANGED:
        return f"Fix changes ~{delta} lines (limit {MAX_LINES_CHANGED})"

    return None


# ─── File patcher ─────────────────────────────────────────────────────────────

def _apply_fix(fix: dict) -> bool:
    """
    Write the fix directly to the file.
    Returns True if successfully applied.
    """
    file_path = REPO_ROOT / fix["file_path"]
    old_snippet: str = fix["old_snippet"]
    new_snippet: str = fix["new_snippet"]

    if not file_path.exists():
        print(f"[free-agent] Target file not found: {file_path}")
        return False

    original = file_path.read_text(encoding="utf-8")

    if old_snippet:
        if old_snippet not in original:
            print(f"[free-agent] old_snippet not found in {fix['file_path']}")
            return False
        patched = original.replace(old_snippet, new_snippet, 1)
    else:
        # Append mode
        patched = original + "\n" + new_snippet

    file_path.write_text(patched, encoding="utf-8")
    print(f"[free-agent] Patched {fix['file_path']}")
    return True


# ─── Import verifier ─────────────────────────────────────────────────────────

def _verify_imports() -> bool:
    """Run `python -c "from app.main import app"` inside backend/."""
    backend_dir = REPO_ROOT / "backend"
    if not (backend_dir / "app" / "main.py").exists():
        print("[free-agent] backend/app/main.py not found — skipping import check")
        return True  # don't block if the file doesn't exist yet
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from app.main import app"],
            cwd=str(backend_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print("[free-agent] Import verification passed")
            return True
        print(f"[free-agent] Import verification FAILED:\n{result.stderr[:500]}")
        return False
    except subprocess.TimeoutExpired:
        print("[free-agent] Import check timed out")
        return False
    except Exception as e:
        print(f"[free-agent] Import check error: {e}")
        return False


# ─── Git helpers ──────────────────────────────────────────────────────────────

def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=check,
    )


def _commit_fix(file_path: str, issue_title: str, agent_name: str) -> bool:
    """Stage the changed file and commit."""
    try:
        _git("config", "user.email", "free-agent@quantedge.ai")
        _git("config", "user.name", "QuantEdge Free Agent")
        _git("add", file_path)
        commit_msg = f"fix(auto): {issue_title} [agent: {agent_name}]"
        _git("commit", "-m", commit_msg)
        print(f"[free-agent] Committed: {commit_msg}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[free-agent] Git commit failed: {e.stderr}")
        return False


def _push_branch() -> bool:
    """Push current branch to origin."""
    try:
        branch_result = _git("rev-parse", "--abbrev-ref", "HEAD")
        branch = branch_result.stdout.strip()
        _git("push", "origin", f"HEAD:{branch}")
        print(f"[free-agent] Pushed to origin/{branch}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[free-agent] Git push failed: {e.stderr}")
        return False


def _revert_file(file_path: str) -> None:
    """Revert a file to its HEAD state (undo bad fix)."""
    try:
        _git("checkout", "HEAD", "--", file_path)
        print(f"[free-agent] Reverted {file_path}")
    except Exception as e:
        print(f"[free-agent] Revert failed: {e}")


# ─── Slack notification ───────────────────────────────────────────────────────

def _slack_post(channel: str, text: str) -> None:
    if not SLACK_BOT_TOKEN:
        print(f"[free-agent] [dry-slack] #{channel}: {text[:200]}")
        return
    try:
        payload = json.dumps({"channel": channel, "text": text}).encode()
        req = urllib.request.Request(
            "https://slack.com/api/chat.postMessage",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read())
        if not resp.get("ok"):
            print(f"[free-agent] Slack error: {resp.get('error')}")
    except Exception as e:
        print(f"[free-agent] Slack post error: {e}")


# ─── Per-issue workflow ───────────────────────────────────────────────────────

def _process_issue(issue: dict) -> str:
    """
    Process a single GitHub issue.
    Returns one of: "fixed" | "skipped:<reason>" | "failed:<reason>"
    """
    number: int = issue["number"]
    title: str = issue["title"]
    body: str = issue.get("body", "") or ""

    print(f"\n[free-agent] Processing issue #{number}: {title}")

    # Sanitize before sending to LLM
    safe_title = _sanitize(title)
    safe_body = _sanitize(body)[:3000]   # trim to avoid token waste

    # 1. Ask LLM for a fix
    prompt = _build_fix_prompt(safe_title, safe_body)
    try:
        raw, agent_name = _call_llm(prompt)
    except RuntimeError as e:
        msg = f"All LLM providers failed: {e}"
        _comment_on_issue(number, f":robot: Free-agent engineer could not process this issue: {msg}")
        return f"failed:{msg}"

    # 2. Parse the fix
    try:
        fix = _parse_fix_response(raw)
    except (ValueError, json.JSONDecodeError) as e:
        msg = f"Could not parse LLM response: {e}"
        _comment_on_issue(number, f":robot: {msg}\n\nRaw LLM output (truncated):\n```\n{raw[:500]}\n```")
        return f"failed:{msg}"

    # 3. Validate
    validation_error = _validate_fix(fix)
    if validation_error == "too_complex":
        comment = (
            f":robot: **Auto-fix skipped — too complex for automated resolution.**\n\n"
            f"Reason: {fix.get('explanation', 'No explanation provided')}\n\n"
            f"This issue requires manual engineering attention. Removing the "
            f"`{AGENT_LABEL}` label and leaving open for human review."
        )
        _comment_on_issue(number, comment)
        # Remove the label so it doesn't loop forever
        try:
            label_url = (
                f"{GITHUB_API}/repos/{GH_REPO}/issues/{number}/labels/"
                f"{urllib.parse.quote(AGENT_LABEL)}"
            )
            req = urllib.request.Request(label_url, method="DELETE",
                                         headers=_gh_headers())
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass
        return "skipped:too_complex"

    if validation_error:
        comment = f":robot: **Auto-fix aborted** — validation failed: {validation_error}"
        _comment_on_issue(number, comment)
        return f"skipped:{validation_error}"

    # 4. Apply the fix
    if not _apply_fix(fix):
        comment = ":robot: **Auto-fix failed** — could not apply patch to file."
        _comment_on_issue(number, comment)
        return "failed:apply_fix"

    # 5. Verify imports
    if not _verify_imports():
        # Revert the bad fix
        _revert_file(fix["file_path"])
        comment = (
            ":robot: **Auto-fix reverted** — import verification failed after applying patch.\n\n"
            "The agent's proposed change broke the application. Manual review required."
        )
        _comment_on_issue(number, comment)
        return "failed:import_check"

    # 6. Commit
    if not _commit_fix(fix["file_path"], title, agent_name):
        _revert_file(fix["file_path"])
        return "failed:git_commit"

    # 7. Push
    _push_branch()

    # 8. Close the issue with a success comment
    try:
        branch = _git("rev-parse", "--abbrev-ref", "HEAD", check=False).stdout.strip()
    except Exception:
        branch = "unknown"

    close_comment = (
        f":white_check_mark: **Auto-fixed by {agent_name}**\n\n"
        f"**Root cause:** {fix.get('root_cause', 'N/A')}\n\n"
        f"**What was changed:** {fix.get('explanation', 'N/A')}\n\n"
        f"**File patched:** `{fix.get('file_path', 'N/A')}`\n\n"
        f"Commit pushed to `{branch}`."
    )
    _close_issue(number, close_comment)

    # 9. Notify Slack #incidents
    slack_msg = (
        f":robot: *Free-agent engineer fixed issue #{number}*\n"
        f"*Issue:* {title}\n"
        f"*Agent:* {agent_name}\n"
        f"*File:* `{fix['file_path']}`\n"
        f"*Root cause:* {fix.get('root_cause', 'N/A')}\n"
        f"*Fix:* {fix.get('explanation', 'N/A')}"
    )
    _slack_post("incidents", slack_msg)

    return "fixed"


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    print(
        f"[free-agent] QuantEdge Free-Agent Engineer — "
        f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}"
    )
    print(f"[free-agent] ALLOW_PAID_APIS={ALLOW_PAID_APIS}  (must be False)")
    print(f"[free-agent] Repo: {GH_REPO}")

    if not GH_TOKEN or not GH_REPO:
        print("[free-agent] GH_TOKEN or GH_REPO not configured. Exiting.")
        return 1

    issues = _list_open_issues()
    if not issues:
        print("[free-agent] No actionable issues found. All clear.")
        return 0

    results: dict[str, str] = {}
    for issue in issues:
        number = issue["number"]
        outcome = _process_issue(issue)
        results[f"#{number}"] = outcome
        time.sleep(2)   # brief pause between issues

    # Summary
    print("\n[free-agent] === Run summary ===")
    fixed = sum(1 for v in results.values() if v == "fixed")
    for issue_ref, outcome in results.items():
        icon = "OK" if outcome == "fixed" else "!!"
        print(f"  [{icon}] {issue_ref}: {outcome}")

    if fixed > 0:
        _slack_post(
            "incidents",
            f":robot: Free-agent engineer run complete — *{fixed}/{len(results)}* issue(s) auto-fixed.",
        )

    return 0  # partial fixes are still progress; don't fail the workflow


if __name__ == "__main__":
    sys.exit(main())
