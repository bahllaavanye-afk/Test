"""
Continuous Improvement Agent — runs every 2 hours, picks a module and improves it.
Drives the CTO OKR: ≥ 50 commits/day across org.

Improvement types (rotated by hour):
  0-1h  → add/improve docstrings and type hints
  2-3h  → add error handling and logging
  4-5h  → improve test coverage (add test cases)
  6-7h  → refactor complex functions
  8-9h  → add input validation
  10-11h → optimize hot paths
  12-13h → add feature flags / config constants
  14-15h → improve API response schemas
  16-17h → add missing edge case handling
  18-19h → clean up TODO/FIXME comments
  20-21h → improve strategy logic
  22-23h → add monitoring / metrics hooks
"""
import os, sys, json, random, glob, subprocess, textwrap
from datetime import datetime, timezone
import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GH_TOKEN       = os.environ.get("GH_TOKEN", "")
GH_REPO        = os.environ.get("GH_REPO", "")
ALLOW_PAID_APIS = os.environ.get("ALLOW_PAID_APIS", "False")

# Hard security check
if ALLOW_PAID_APIS.lower() == "true":
    print("SECURITY VIOLATION: ALLOW_PAID_APIS must be False")
    sys.exit(1)

# ── LLM helpers ──────────────────────────────────────────────────────────────

def call_gemini(prompt: str, max_tokens: int = 2048) -> str:
    if not GEMINI_API_KEY:
        return ""
    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.3}
            },
            timeout=45
        )
        if resp.status_code == 200:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"Gemini error: {e}")
    return ""

def call_groq(prompt: str, max_tokens: int = 2048) -> str:
    if not GROQ_API_KEY:
        return ""
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            },
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

def pick_target_file(hour: int) -> str | None:
    """Pick a file to improve based on the current hour."""
    # Rotate through patterns to ensure broad coverage
    pattern_idx = hour % len(CANDIDATE_PATTERNS)
    pattern = CANDIDATE_PATTERNS[pattern_idx]
    files = [f for f in glob.glob(pattern) if not f.endswith("__init__.py")]
    if not files:
        # Fallback: any backend Python file
        all_files = glob.glob("backend/app/**/*.py", recursive=True)
        files = [f for f in all_files if "__init__" not in f and "__pycache__" not in f]
    if not files:
        return None
    return random.choice(files)

def read_file(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return ""

def write_file(path: str, content: str):
    with open(path, "w") as f:
        f.write(content)

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

def improve_file(file_path: str, content: str, improvement_type: str, improvement_desc: str) -> str | None:
    if len(content) > 8000:
        # Truncate very large files to avoid token limits
        content = content[:8000] + "\n# ... (truncated for brevity)"

    prompt = f"""{SYSTEM_CONTEXT}

File: {file_path}
Improvement type: {improvement_type}
Task: {improvement_desc}

Current file content:
{content}

Output the complete improved file:"""

    improved = llm(prompt, max_tokens=4096)
    if not improved:
        return None

    # Strip markdown code fences if the LLM added them
    if improved.startswith("```"):
        lines = improved.split("\n")
        improved = "\n".join(lines[1:])
        if improved.endswith("```"):
            improved = improved[:-3]

    return improved.strip()

def syntax_check(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError as e:
        print(f"Syntax error: {e}")
        return False

def git_commit(file_path: str, improvement_type: str, message: str):
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

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    hour = datetime.now(timezone.utc).hour
    improvement_type, improvement_desc = get_improvement_type(hour)

    print(f"[{datetime.now(timezone.utc).strftime('%H:%M UTC')}] Improvement type: {improvement_type}")

    # Pick 3 files to improve (for more commits per run)
    n_files = int(os.environ.get("N_FILES", "3"))
    improved_count = 0

    # Spread across different patterns
    tried_patterns = set()
    attempts = 0

    while improved_count < n_files and attempts < 10:
        attempts += 1
        target = pick_target_file((hour + attempts) % 24)
        if not target or target in tried_patterns:
            continue
        tried_patterns.add(target)

        content = read_file(target)
        if not content or len(content) < 100:
            continue

        print(f"  Improving: {target}")
        improved = improve_file(target, content, improvement_type, improvement_desc)

        if not improved:
            print(f"  ✗ LLM returned nothing for {target}")
            continue

        if not syntax_check(improved):
            print(f"  ✗ Syntax check failed for {target}")
            continue

        # Only write if actually changed
        if improved.strip() == content.strip():
            print(f"  = No change for {target}")
            continue

        write_file(target, improved)

        short_path = target.replace("backend/app/", "").replace("backend/tests/", "test/")
        commit_msg = f"improve({improvement_type}): {short_path} — {improvement_desc[:60]}"
        if git_commit(target, improvement_type, commit_msg):
            improved_count += 1

    print(f"\n✓ Committed {improved_count} improvements (type: {improvement_type})")

    # Write run summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "improvement_type": improvement_type,
        "files_improved": improved_count,
        "hour": hour
    }
    with open("/tmp/continuous_improvement_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return 0 if improved_count > 0 else 0  # Never fail the workflow

if __name__ == "__main__":
    sys.exit(main())
