"""
Continuous Improvement Agent — runs every 2 hours.
SOTA self-improvement: RLVR test loop + Reflexion failure memory + skill library.
Drives CTO OKR: ≥ 50 commits/day across org.
"""
from __future__ import annotations
import os, sys, json, random, glob, subprocess
from datetime import datetime, timezone
from pathlib import Path
import requests

# ── Key resolver ──────────────────────────────────────────────────────────────
def _resolve_key(*names: str) -> str:
    for name in names:
        v = os.environ.get(name, "")
        if v: return v
        if not name[-1].isdigit():
            v = os.environ.get(name + "_1", "")
            if v: return v
    return ""

GEMINI_API_KEY = _resolve_key("GEMINI_API_KEY", "GEMINI_API_KEY_1")
GROQ_API_KEY   = _resolve_key("GROQ_API_KEY", "GROQ_API_KEY_1")
ALLOW_PAID_APIS = os.environ.get("ALLOW_PAID_APIS", "False")

if ALLOW_PAID_APIS.lower() == "true":
    print("SECURITY VIOLATION: ALLOW_PAID_APIS must be False")
    sys.exit(1)

# ── Shared memory (Reflexion + skill library) ─────────────────────────────────
STATE_FILE  = Path(__file__).resolve().parents[2] / ".github" / "state" / "agent_memory.json"
SKILLS_FILE = Path(__file__).resolve().parents[2] / ".github" / "state" / "skill_library.json"

def load_memory() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"conversations": {}, "thread_state": {}, "employee_context": {},
                "platform_metrics": {}, "failure_traces": [], "improvement_stats": {}}

def save_memory(mem: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    mem["last_updated"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(mem, indent=2))

def load_skills() -> list[str]:
    try:
        return json.loads(SKILLS_FILE.read_text()).get("skills", [])
    except Exception:
        return []

def save_skill(skill: str):
    SKILLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(SKILLS_FILE.read_text())
    except Exception:
        data = {"skills": [], "last_updated": ""}
    if skill not in data["skills"]:
        data["skills"].append(skill)
        data["last_updated"] = datetime.now(timezone.utc).isoformat()
        SKILLS_FILE.write_text(json.dumps(data, indent=2))

def record_failure(mem: dict, file_path: str, reason: str, improvement_type: str):
    traces = mem.setdefault("failure_traces", [])
    traces.append({
        "file": file_path,
        "reason": reason,
        "improvement_type": improvement_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    mem["failure_traces"] = traces[-50:]  # keep last 50

def record_success(mem: dict, file_path: str, improvement_type: str, tests_passed: bool):
    stats = mem.setdefault("improvement_stats", {})
    key = improvement_type
    s = stats.setdefault(key, {"successes": 0, "failures": 0, "test_pass": 0})
    s["successes"] += 1
    if tests_passed:
        s["test_pass"] += 1

# ── LLM helpers ──────────────────────────────────────────────────────────────

def call_gemini(prompt: str, max_tokens: int = 2048) -> str:
    if not GEMINI_API_KEY: return ""
    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                  "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.3}},
            timeout=45
        )
        if resp.status_code == 200:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"Gemini error: {e}")
    return ""

def call_groq(prompt: str, max_tokens: int = 2048) -> str:
    if not GROQ_API_KEY: return ""
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant",
                  "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Groq error: {e}")
    return ""

def llm(prompt: str, max_tokens: int = 2048) -> str:
    return call_gemini(prompt, max_tokens) or call_groq(prompt, max_tokens) or ""

# ── File selection ────────────────────────────────────────────────────────────

CANDIDATE_PATTERNS = [
    "backend/app/strategies/manual/*.py",
    "backend/app/strategies/ml_enhanced/*.py",
    "backend/app/ml/models/*.py",
    "backend/app/ml/features/*.py",
    "backend/app/execution/*.py",
    "backend/app/risk/*.py",
    "backend/app/brokers/*.py",
    "backend/app/backtest/*.py",
    "backend/app/comparison/*.py",
    "backend/app/tasks/*.py",
    "backend/app/api/v1/*.py",
    "backend/tests/unit/*.py",
]

def pick_target_file(hour: int, skip_files: set[str]) -> str | None:
    pattern_idx = hour % len(CANDIDATE_PATTERNS)
    pattern = CANDIDATE_PATTERNS[pattern_idx]
    files = [f for f in glob.glob(pattern)
             if not f.endswith("__init__.py") and f not in skip_files]
    if not files:
        all_files = glob.glob("backend/app/**/*.py", recursive=True)
        files = [f for f in all_files
                 if "__init__" not in f and "__pycache__" not in f and f not in skip_files]
    return random.choice(files) if files else None

# ── Improvement types ─────────────────────────────────────────────────────────

IMPROVEMENT_TYPES = [
    ("docstrings",     "Add or improve docstrings and type hints. Add missing parameter/return type annotations. Do not change any logic."),
    ("error_handling", "Add proper error handling with specific exception types. Add structured logging for error cases. Do not change business logic."),
    ("test_cases",     "Add 2-3 new unit test cases for edge cases not currently tested. Focus on boundary conditions."),
    ("refactor",       "Refactor the most complex function to be more readable. Extract helper functions if appropriate. No behavior changes."),
    ("validation",     "Add input validation for public functions. Raise ValueError with descriptive messages for invalid inputs."),
    ("optimization",   "Identify and optimize the most expensive operation. Use caching, vectorization, or early exit where appropriate."),
    ("constants",      "Extract magic numbers and hardcoded strings into named constants at the top of the file."),
    ("schemas",        "Improve Pydantic schema definitions: add field descriptions, examples, and validators where missing."),
    ("edge_cases",     "Add handling for None inputs, empty collections, and off-by-one conditions in the existing logic."),
    ("cleanup",        "Remove dead code, fix TODO/FIXME comments by implementing them, remove unused imports."),
    ("strategy_logic", "Improve the strategy's signal quality: tighten entry conditions, add confirmation filters, improve exit logic."),
    ("monitoring",     "Add structured logging with key metrics (signal count, execution time, P&L) at INFO level."),
]

def get_improvement_type(hour: int) -> tuple[str, str]:
    return IMPROVEMENT_TYPES[hour % len(IMPROVEMENT_TYPES)]

# ── Core improvement flow ─────────────────────────────────────────────────────

SYSTEM_CONTEXT = """You are a senior quantitative software engineer at QuantEdge, a production trading platform.
You are improving existing code files. Rules:
1. Output ONLY the complete improved Python file — no markdown, no explanation, no ```python blocks
2. Never add mock data or hardcoded test values
3. Never change ALLOW_PAID_APIS or TRADING_MODE settings
4. Never add external paid API calls
5. Preserve all existing behavior — only improve quality
6. Keep changes minimal and focused on the specified improvement type
7. The output must be syntactically valid Python"""

def improve_file(file_path: str, content: str, improvement_type: str,
                 improvement_desc: str, failure_context: str = "", skills: list[str] = []) -> str | None:
    if len(content) > 8000:
        content = content[:8000] + "\n# ... (truncated for brevity)"

    skill_hint = ""
    if skills:
        skill_hint = "\nKnown patterns from past runs:\n" + "\n".join(f"- {s}" for s in skills[-5:])

    failure_hint = ""
    if failure_context:
        failure_hint = f"\nReflexion — past failures on this file:\n{failure_context}\nAvoid repeating these mistakes."

    prompt = f"""{SYSTEM_CONTEXT}{skill_hint}{failure_hint}

File: {file_path}
Improvement type: {improvement_type}
Task: {improvement_desc}

Current file content:
{content}

Output the complete improved file:"""

    improved = llm(prompt, max_tokens=4096)
    if not improved:
        return None

    if improved.startswith("```"):
        lines = improved.split("\n")
        improved = "\n".join(lines[1:])
        if improved.rstrip().endswith("```"):
            improved = improved.rstrip()[:-3]

    return improved.strip()

def syntax_check(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError as e:
        print(f"Syntax error: {e}")
        return False

def run_tests() -> tuple[bool, str]:
    """RLVR: run pytest on backend tests. Returns (passed, output)."""
    result = subprocess.run(
        ["python", "-m", "pytest", "backend/tests/", "-x", "-q", "--tb=short",
         "--timeout=30", "--no-header"],
        capture_output=True, text=True, timeout=120
    )
    passed = result.returncode == 0
    output = (result.stdout + result.stderr)[-2000:]
    return passed, output

def git_commit(file_path: str, message: str) -> bool:
    subprocess.run(["git", "add", file_path], check=True)
    result = subprocess.run(["git", "diff", "--cached", "--stat"], capture_output=True, text=True)
    if not result.stdout.strip():
        print("No changes to commit")
        return False
    subprocess.run(
        ["git", "commit", "-m", message],
        check=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "QuantEdge AI", "GIT_AUTHOR_EMAIL": "ai@quantedge.ai",
             "GIT_COMMITTER_NAME": "QuantEdge AI", "GIT_COMMITTER_EMAIL": "ai@quantedge.ai"}
    )
    print(f"✓ Committed: {message}")
    return True

def git_revert_file(file_path: str, original_content: str):
    with open(file_path, "w") as f:
        f.write(original_content)
    print(f"  ↩ Reverted {file_path} (tests failed)")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    hour = datetime.now(timezone.utc).hour
    improvement_type, improvement_desc = get_improvement_type(hour)
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M UTC')}] Improvement type: {improvement_type}")

    mem = load_memory()
    skills = load_skills()
    n_files = int(os.environ.get("N_FILES", "3"))
    improved_count = 0
    tried = set()
    attempts = 0

    # Pull latest state first
    subprocess.run(["git", "pull", "--rebase", "--quiet"], capture_output=True)

    while improved_count < n_files and attempts < 10:
        attempts += 1
        target = pick_target_file((hour + attempts) % 24, tried)
        if not target:
            continue
        tried.add(target)

        original_content = ""
        try:
            with open(target) as f:
                original_content = f.read()
        except Exception:
            continue

        if len(original_content) < 100:
            continue

        # Reflexion: build failure context for this file
        past_failures = [t for t in mem.get("failure_traces", [])
                         if t.get("file") == target]
        failure_ctx = "\n".join(
            f"[{t['timestamp'][:10]}] {t['improvement_type']}: {t['reason']}"
            for t in past_failures[-3:]
        )

        print(f"  Improving: {target}")
        improved = improve_file(target, original_content, improvement_type,
                                improvement_desc, failure_ctx, skills)

        if not improved:
            print(f"  ✗ LLM returned nothing for {target}")
            record_failure(mem, target, "LLM returned empty", improvement_type)
            continue

        if not syntax_check(improved):
            print(f"  ✗ Syntax check failed for {target}")
            record_failure(mem, target, "syntax check failed", improvement_type)
            continue

        if improved.strip() == original_content.strip():
            print(f"  = No change for {target}")
            continue

        with open(target, "w") as f:
            f.write(improved)

        # RLVR: run tests — revert if they break
        tests_passed = True
        test_output = ""
        if os.path.exists("backend/tests"):
            tests_passed, test_output = run_tests()
            if not tests_passed:
                git_revert_file(target, original_content)
                record_failure(mem, target, f"tests failed: {test_output[:200]}", improvement_type)
                save_skill(f"File {target}: changes caused test failures — be more conservative")
                continue

        short_path = target.replace("backend/app/", "").replace("backend/tests/", "test/")
        commit_msg = f"improve({improvement_type}): {short_path} — {improvement_desc[:60]}"
        if git_commit(target, commit_msg):
            improved_count += 1
            record_success(mem, target, improvement_type, tests_passed)
            if tests_passed:
                save_skill(f"{improvement_type} on {short_path}: success — tests green")

    save_memory(mem)

    # Commit updated memory
    try:
        subprocess.run(["git", "add", str(STATE_FILE), str(SKILLS_FILE)], capture_output=True)
        subprocess.run(["git", "commit", "-m", f"state: continuous_improver memory update — {improved_count} improvements",
                        "--allow-empty"],
                       capture_output=True,
                       env={**os.environ, "GIT_AUTHOR_NAME": "QuantEdge AI",
                            "GIT_AUTHOR_EMAIL": "ai@quantedge.ai",
                            "GIT_COMMITTER_NAME": "QuantEdge AI",
                            "GIT_COMMITTER_EMAIL": "ai@quantedge.ai"})
        subprocess.run(["git", "push"], capture_output=True)
    except Exception as e:
        print(f"Memory push error: {e}")

    print(f"\n✓ Committed {improved_count} improvements (type: {improvement_type})")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "improvement_type": improvement_type,
        "files_improved": improved_count,
        "hour": hour,
        "improvement_stats": mem.get("improvement_stats", {}),
    }
    with open("/tmp/continuous_improvement_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return 0

if __name__ == "__main__":
    sys.exit(main())
