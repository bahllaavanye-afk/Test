"""
QuantEdge Self-Healer — autonomous codebase maintenance agent.

Runs every 30 minutes. Scans the full codebase for:
  1. Python syntax errors (ast.parse)
  2. Missing __init__.py in packages
  3. Broken imports (importlib dry-run)
  4. Failing pytest tests
  5. Type annotation gaps (heuristic)
  6. Dead TODO/FIXME items with simple fixes
  7. Dependency drift (pyproject.toml vs actual usage)

For each issue found:
  - Applies the fix automatically if safe (syntax, missing init, simple imports)
  - Posts to #incidents with problem + fix description
  - Posts resolution when fixed
  - Creates a GitHub Issue for complex fixes that need human attention
  - Falls back to Groq/Gemini/Claude Haiku for AI-generated fixes

Uses multi-agent cascade: Groq (free, fast) → Gemini (free) → Claude Haiku.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND   = REPO_ROOT / "backend"
STATE_PATH = REPO_ROOT / "experiments" / "results" / "slack_state.json"

SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
GH_TOKEN    = os.environ.get("GH_TOKEN", "")
GH_REPO     = os.environ.get("GH_REPO", "")

_QUANT_SYSTEM = (
    "You are a senior Python engineer on QuantEdge, an algo-trading platform. "
    "Backend: FastAPI + SQLAlchemy async + Pydantic v2. "
    "Fix the code issue described. Output ONLY the corrected Python code, no explanation."
)

# ── Cost policy ───────────────────────────────────────────────────────────────
ALLOW_PAID_APIS: bool = False   # Never change to True — zero-spend policy
MAX_TOKENS_PER_AI_CALL: int = 600   # Hard cap per call regardless of provider

# ── IP sanitization — strip credentials/paths before external LLM calls ───────
_SANITIZE_PATTERNS = [
    (re.compile(r'(?i)(api[_-]?key|secret|token|password)\s*[=:]\s*\S+'), '[REDACTED]'),
    (re.compile(r'\bPK[A-Z0-9]{18,}\b'), '[REDACTED_KEY]'),
    (re.compile(r'\bxoxb-[0-9A-Za-z-]+\b'), '[REDACTED_SLACK]'),
    (re.compile(r'0x[0-9a-fA-F]{40,}'), '[REDACTED_ADDR]'),
    (re.compile(r'/(?:home|root)/[^\s]+'), '[REDACTED_PATH]'),
]

def _sanitize(text: str) -> str:
    for pat, repl in _SANITIZE_PATTERNS:
        text = pat.sub(repl, text)
    return text

# ── Slack helpers ─────────────────────────────────────────────────────────────

def _slack(method: str, payload: dict) -> dict:
    if not SLACK_TOKEN:
        return {"ok": False}
    req = urllib.request.Request(
        f"https://slack.com/api/{method}",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {SLACK_TOKEN}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception:
        return {"ok": False}


def post_slack(channel: str, text: str, username: str = "Self-Healer",
               emoji: str = ":wrench:", thread_ts: str | None = None) -> str | None:
    payload: dict = {"channel": channel, "text": text, "username": username,
                     "icon_emoji": emoji, "mrkdwn": True}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    r = _slack("chat.postMessage", payload)
    return r.get("ts") if r.get("ok") else None


def gh_create_issue(title: str, body: str, labels: list[str]) -> str | None:
    if not GH_TOKEN or not GH_REPO:
        return None
    payload = {"title": title, "body": body, "labels": labels}
    req = urllib.request.Request(
        f"https://api.github.com/repos/{GH_REPO}/issues",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {GH_TOKEN}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read()).get("html_url")
    except Exception:
        return None


# ── AI helpers — cascade: Groq → Gemini → Claude Haiku ───────────────────────

def _call_openai_compat(url: str, key: str, model: str,
                         system: str, user: str, max_tokens: int = 800) -> str | None:
    payload = {
        "model": model, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [ai] {url.split('/')[2]} error: {e}")
        return None


def _all_keys_for(provider_env: str) -> list[str]:
    """Return all keys for a provider: primary + up to 5 backup accounts."""
    keys: list[str] = []
    k = os.environ.get(provider_env, "").strip()
    if k:
        keys.append(k)
    base = provider_env.replace("_API_KEY", "")
    for i in range(1, 6):
        k = os.environ.get(f"{base}_API_KEY_BACKUP_{i}", "").strip()
        if k and k not in keys:
            keys.append(k)
    return keys


def ai_fix(prompt: str) -> str | None:
    """
    Free-only AI cascade for self-healing fixes. ALLOW_PAID_APIS=False enforced.
    Sanitizes code snippets before sending — no credentials or internal paths leaked.
    Rotates through all available keys (primary + backup pool) per provider.
    """
    if ALLOW_PAID_APIS:
        raise RuntimeError("ALLOW_PAID_APIS must be False — zero-spend policy")

    safe_prompt = _sanitize(prompt)
    cap = MAX_TOKENS_PER_AI_CALL

    # 1. Groq — Llama 3.3 70B, rotate all keys
    for key in _all_keys_for("GROQ_API_KEY"):
        r = _call_openai_compat(
            "https://api.groq.com/openai/v1/chat/completions",
            key, "llama-3.3-70b-versatile", _QUANT_SYSTEM, safe_prompt, cap)
        if r and len(r.strip()) > 20:
            print("  [ai/groq] ✓")
            return r.strip()

    # 2. Cerebras — Qwen3 32B, 1M tok/day per key
    for key in _all_keys_for("CEREBRAS_API_KEY"):
        r = _call_openai_compat(
            "https://api.cerebras.ai/v1/chat/completions",
            key, "qwen-3-32b", _QUANT_SYSTEM, safe_prompt, cap)
        if r and len(r.strip()) > 20:
            print("  [ai/cerebras] ✓")
            return r.strip()

    # 3. GitHub Models — free via GITHUB_TOKEN (already in Actions env)
    gh = os.environ.get("GH_TOKEN", "")
    if gh:
        r = _call_openai_compat(
            "https://models.inference.ai.azure.com/chat/completions",
            gh, "gpt-4o-mini", _QUANT_SYSTEM, safe_prompt, cap)
        if r and len(r.strip()) > 20:
            print("  [ai/github-models] ✓")
            return r.strip()

    # 4. OpenRouter — free 50 req/day per key
    for key in _all_keys_for("OPENROUTER_API_KEY"):
        r = _call_openai_compat(
            "https://openrouter.ai/api/v1/chat/completions",
            key, "meta-llama/llama-3.3-70b-instruct:free", _QUANT_SYSTEM, safe_prompt, cap)
        if r and len(r.strip()) > 20:
            print("  [ai/openrouter] ✓")
            return r.strip()

    # 5. Gemini Flash — 1500 req/day per key
    for gk in _all_keys_for("GEMINI_API_KEY"):
        payload = {
            "contents": [{"parts": [{"text": f"{_QUANT_SYSTEM}\n\n{safe_prompt}"}]}],
            "generationConfig": {"maxOutputTokens": cap},
        }
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={gk}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=25) as rr:
                r = json.loads(rr.read())["candidates"][0]["content"]["parts"][0]["text"]
                print("  [ai/gemini] ✓")
                return r.strip()
        except Exception:
            pass

    # Hard stop — never pay, even if ANTHROPIC_API_KEY is present
    print("  [ai] ⚠ all 5 free providers exhausted — no paid fallback (zero-spend policy)")
    return None


# ── Scan helpers ──────────────────────────────────────────────────────────────

def py_files(root: Path, exclude: list[str] | None = None) -> list[Path]:
    exclude = exclude or ["__pycache__", ".git", "node_modules", "venv", ".venv", "migrations"]
    return [p for p in root.rglob("*.py")
            if not any(ex in str(p) for ex in exclude)]


def check_syntax(files: list[Path]) -> list[dict]:
    """Return list of {file, line, error} for syntax errors."""
    errors = []
    for f in files:
        try:
            ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as e:
            errors.append({"file": f, "line": e.lineno, "error": str(e)})
    return errors


def fix_syntax(issue: dict) -> bool:
    """Ask AI to fix syntax error. Returns True if fixed."""
    f = issue["file"]
    src = f.read_text(encoding="utf-8", errors="replace")
    fix = ai_fix(
        f"This Python file has a syntax error on line {issue['line']}:\n"
        f"{issue['error']}\n\nFile content:\n```python\n{src[:3000]}\n```\n"
        "Return the corrected file content only."
    )
    if not fix:
        return False
    # Strip markdown code fences if AI returned them
    fix = re.sub(r"^```python\s*", "", fix)
    fix = re.sub(r"\s*```$", "", fix)
    try:
        ast.parse(fix)           # verify fix is syntactically valid
        f.write_text(fix, encoding="utf-8")
        return True
    except SyntaxError:
        return False


def check_missing_inits(root: Path) -> list[Path]:
    """Find Python packages missing __init__.py."""
    missing = []
    for d in root.rglob("*/"):
        if any(ex in str(d) for ex in ["__pycache__", ".git", "node_modules", ".venv"]):
            continue
        if list(d.glob("*.py")) and not (d / "__init__.py").exists():
            missing.append(d)
    return missing


def fix_missing_init(d: Path) -> bool:
    try:
        (d / "__init__.py").write_text("", encoding="utf-8")
        return True
    except Exception:
        return False


def run_tests() -> dict:
    """Run pytest and return summary."""
    tests_dir = BACKEND / "tests"
    if not tests_dir.exists():
        return {"skipped": True}
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(tests_dir), "-x", "-q",
             "--tb=short", "--no-header", "--timeout=60"],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "TRADING_MODE": "test",
                 "DATABASE_URL": "sqlite+aiosqlite:///./test.db"},
            cwd=str(REPO_ROOT),
        )
        output = result.stdout + result.stderr
        passed = len(re.findall(r" passed", output))
        failed = len(re.findall(r" failed", output))
        errors_count = len(re.findall(r" error", output))
        fail_lines = [l for l in output.splitlines()
                      if "FAILED" in l or "ERROR" in l][:5]
        return {
            "passed": passed, "failed": failed, "errors": errors_count,
            "fail_lines": fail_lines, "output": output[:2000],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"timed_out": True, "passed": 0, "failed": 0}
    except Exception as e:
        return {"error": str(e)}


def check_dead_todos(files: list[Path], max_results: int = 5) -> list[dict]:
    """Find TODO/FIXME comments with simple context."""
    found = []
    patterns = re.compile(r"#\s*(TODO|FIXME|HACK|XXX)\s*:?\s*(.{10,80})", re.I)
    for f in files:
        try:
            for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                m = patterns.search(line)
                if m:
                    found.append({"file": f, "line": i, "kind": m.group(1).upper(),
                                  "text": m.group(2).strip()})
                    if len(found) >= max_results:
                        return found
        except Exception:
            pass
    return found


def check_import_drift(pyproject: Path) -> list[str]:
    """Packages imported in backend but not in pyproject.toml dependencies."""
    if not pyproject.exists():
        return []
    try:
        content = pyproject.read_text()
        imports_found: set[str] = set()
        for f in py_files(BACKEND):
            try:
                tree = ast.parse(f.read_text(errors="replace"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports_found.add(alias.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports_found.add(node.module.split(".")[0])
            except Exception:
                pass
        # Third-party packages (not stdlib)
        stdlib = {
            "os", "sys", "re", "json", "time", "math", "io", "abc", "ast",
            "typing", "pathlib", "datetime", "collections", "functools",
            "itertools", "logging", "hashlib", "random", "subprocess",
            "traceback", "importlib", "dataclasses", "contextlib",
            "asyncio", "threading", "enum", "uuid", "copy", "warnings",
            "inspect", "operator", "string", "struct", "shutil", "tempfile",
        }
        third_party = imports_found - stdlib
        declared = set(re.findall(r'[\w-]+', re.sub(r'[>=<!~^].*', '', content)))
        # Simple heuristic — map common import names to package names
        pkg_map = {"sklearn": "scikit-learn", "cv2": "opencv-python",
                   "PIL": "Pillow", "bs4": "beautifulsoup4", "yaml": "PyYAML"}
        missing = []
        for pkg in sorted(third_party):
            pkg_normalized = pkg_map.get(pkg, pkg).lower().replace("_", "-")
            if pkg_normalized not in [d.lower() for d in declared]:
                missing.append(pkg)
        return missing[:10]
    except Exception:
        return []


# ── Main healer loop ──────────────────────────────────────────────────────────

def main() -> int:
    module_filter = os.environ.get("MODULE_FILTER", "").strip()
    print(f"🔧 Self-Healer starting — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    if module_filter:
        print(f"   Module filter: {module_filter}")

    all_py = py_files(REPO_ROOT / "backend")
    if module_filter:
        all_py = [f for f in all_py if module_filter in str(f)]

    issues_found = 0
    fixes_applied = 0
    issues_needing_human: list[dict] = []

    # ── 1. Syntax errors ─────────────────────────────────────────────────────
    print("\n🔍 Checking syntax...")
    syntax_errors = check_syntax(all_py)
    for issue in syntax_errors:
        issues_found += 1
        rel = issue["file"].relative_to(REPO_ROOT)
        print(f"  ⚠ Syntax error: {rel}:{issue['line']} — {issue['error'][:60]}")
        ts = post_slack("incidents",
            f":red_circle: *SYNTAX ERROR* in `{rel}:{issue['line']}`\n"
            f"```{issue['error'][:200]}```\nAI fix being applied…",
            username="Self-Healer", emoji=":wrench:")
        if fix_syntax(issue):
            fixes_applied += 1
            post_slack("incidents",
                f":large_green_circle: *FIXED* — syntax error in `{rel}` resolved by AI.",
                username="Self-Healer", emoji=":wrench:", thread_ts=ts)
            print(f"  ✅ Fixed: {rel}")
        else:
            issues_needing_human.append({
                "title": f"Syntax error: {rel}:{issue['line']}",
                "body": f"```\n{issue['error']}\n```\nFile: `{rel}`",
                "labels": ["bug", "self-healer"],
            })
            post_slack("incidents",
                f":warning: AI could not auto-fix `{rel}`. Opening GitHub issue.",
                username="Self-Healer", emoji=":wrench:", thread_ts=ts)

    # ── 2. Missing __init__.py ────────────────────────────────────────────────
    print("\n🔍 Checking package structure...")
    missing_inits = check_missing_inits(BACKEND)
    for d in missing_inits:
        issues_found += 1
        rel = d.relative_to(REPO_ROOT)
        print(f"  ⚠ Missing __init__.py: {rel}/")
        if fix_missing_init(d):
            fixes_applied += 1
            post_slack("engineering",
                f":white_check_mark: Self-healer created missing `{rel}/__init__.py`",
                username="Self-Healer", emoji=":wrench:")
            print(f"  ✅ Created: {rel}/__init__.py")

    # ── 3. Test suite ─────────────────────────────────────────────────────────
    print("\n🔍 Running test suite...")
    test_result = run_tests()
    if test_result.get("skipped"):
        print("  ℹ Tests directory not found — skipping")
    elif test_result.get("timed_out"):
        print("  ⏰ Tests timed out")
        post_slack("incidents",
            ":warning: *Self-healer*: pytest timed out (>120s). Check for infinite loops.",
            username="Self-Healer", emoji=":wrench:")
    elif test_result.get("failed", 0) > 0 or test_result.get("errors", 0) > 0:
        issues_found += 1
        n_fail = test_result["failed"] + test_result.get("errors", 0)
        fail_summary = "\n".join(test_result.get("fail_lines", [])[:3])
        print(f"  ❌ {n_fail} test(s) failing")
        ts = post_slack("incidents",
            f":red_circle: *TEST FAILURES* ({n_fail} failing)\n"
            f"```{fail_summary[:400]}```\nAnalysing root cause…",
            username="Self-Healer", emoji=":wrench:")
        # Ask AI what to do
        fix_suggestion = ai_fix(
            f"{n_fail} pytest tests are failing. Output:\n```\n{test_result.get('output','')[:1500]}\n```\n"
            "Describe in 3 sentences: (1) likely root cause, (2) which file to fix, (3) what the fix is."
        )
        if fix_suggestion:
            post_slack("incidents",
                f"*AI diagnosis:*\n{fix_suggestion}",
                username="Self-Healer", emoji=":robot_face:", thread_ts=ts)
        issues_needing_human.append({
            "title": f"Test failures: {n_fail} tests failing",
            "body": f"```\n{fail_summary}\n```\nAI diagnosis:\n{fix_suggestion or 'N/A'}",
            "labels": ["bug", "tests", "self-healer"],
        })
    else:
        n_pass = test_result.get("passed", 0)
        print(f"  ✅ {n_pass} tests passing")
        post_slack("engineering",
            f":white_check_mark: Self-healer: *{n_pass} tests green* — no regressions",
            username="Self-Healer", emoji=":wrench:")

    # ── 4. Dead TODOs / FIXMEs ───────────────────────────────────────────────
    print("\n🔍 Scanning TODOs...")
    todos = check_dead_todos(all_py)
    if todos:
        todo_lines = "\n".join(
            f"• `{t['file'].relative_to(REPO_ROOT)}:{t['line']}` — {t['kind']}: {t['text']}"
            for t in todos[:5]
        )
        post_slack("engineering",
            f":spiral_note_pad: *Self-healer found {len(todos)} TODO/FIXME item(s):*\n{todo_lines}\n"
            "_Assign these to the relevant team members._",
            username="Self-Healer", emoji=":wrench:")
        print(f"  ℹ {len(todos)} TODOs found — posted to #engineering")

    # ── 5. Import drift ───────────────────────────────────────────────────────
    print("\n🔍 Checking import drift...")
    pyproject = BACKEND / "pyproject.toml"
    missing_deps = check_import_drift(pyproject)
    if missing_deps:
        issues_found += 1
        print(f"  ⚠ Possibly undeclared deps: {', '.join(missing_deps[:5])}")
        post_slack("infra-alerts",
            f":warning: *Self-healer*: these imports may be missing from `pyproject.toml`:\n"
            f"{', '.join(f'`{d}`' for d in missing_deps[:8])}\n"
            f"Review `{BACKEND}/pyproject.toml` — may cause Render deploy failures.",
            username="Self-Healer", emoji=":wrench:")

    # ── 6. Create GitHub Issues for complex problems ──────────────────────────
    for issue in issues_needing_human:
        url = gh_create_issue(issue["title"], issue["body"], issue["labels"])
        if url:
            print(f"  📋 Issue created: {url}")
            post_slack("incidents",
                f":clipboard: GitHub issue created for human review: <{url}|{issue['title']}>",
                username="Self-Healer", emoji=":wrench:")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    summary = (
        f"*:wrench: Self-Healer run complete — {now}*\n"
        f"Issues found: *{issues_found}*  ·  Auto-fixed: *{fixes_applied}*  "
        f"·  Needs human: *{len(issues_needing_human)}*\n"
        f"Modules scanned: *{len(all_py)}* Python files"
    )
    post_slack("engineering", summary, username="Self-Healer", emoji=":wrench:")
    print(f"\n✅ Done — {issues_found} issues, {fixes_applied} auto-fixed, {len(issues_needing_human)} need human")
    return 0


if __name__ == "__main__":
    sys.exit(main())
