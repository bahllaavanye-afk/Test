"""
Frontend Design Agent — Priya Iyer (VP Frontend, ex-Bloomberg)
Continuously improves the QuantEdge dashboard: UX, accessibility, animations,
Bloomberg dark-theme consistency, and component quality.

Runs every 6 hours via GitHub Actions. Commits improvements directly.
"""
from __future__ import annotations

import glob
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone

import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
ALLOW_PAID_APIS = os.environ.get("ALLOW_PAID_APIS", "False")
COMPONENT_OVERRIDE = os.environ.get("COMPONENT_OVERRIDE", "").strip()

if ALLOW_PAID_APIS.lower() == "true":
    print("SECURITY: ALLOW_PAID_APIS must be False")
    sys.exit(1)

# Bloomberg dark theme spec
THEME_SPEC = """
Bloomberg dark theme:
- Background: #0a0a0a (page), #111111 (cards), #1a1a1a (inputs)
- Border: #1e1e1e (default), #2a2a2a (hover), #f5a623 (accent)
- Text: #e8e8e8 (primary), #888 (muted), #f5a623 (highlighted)
- Green: #00c853 (profit/long/positive), Red: #ff1744 (loss/short/negative)
- Blue: #2196F3 (info/benchmark), Purple: var(--purple)
- Font: JetBrains Mono for numbers, Inter for UI labels
- No rounded corners on data tables — sharp borders like Bloomberg Terminal
- Loading state: skeleton shimmer (#1a1a1a → #222), never spinner
"""

# ── LLM ──────────────────────────────────────────────────────────────────────

def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return ""
    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 4096, "temperature": 0.2}
            },
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        if resp.status_code == 429:
            print(f"Gemini quota hit (429) — falling back to Groq")
    except Exception as e:
        print(f"Gemini error: {e}")
    return ""

def call_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        return ""
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 3000
            },
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Groq error: {e}")
    return ""

def llm(prompt: str) -> str:
    return call_gemini(prompt) or call_groq(prompt) or ""

# ── File picker ───────────────────────────────────────────────────────────────

FRONTEND_PATTERNS = [
    "frontend/src/pages/Landing.tsx",
    "frontend/src/pages/Dashboard.tsx",
    "frontend/src/pages/Analytics.tsx",
    "frontend/src/pages/Comparison.tsx",
    "frontend/src/components/charts/*.tsx",
    "frontend/src/components/trading/*.tsx",
    "frontend/src/components/analytics/*.tsx",
    "frontend/src/components/risk/*.tsx",
    "frontend/src/components/layout/*.tsx",
    "frontend/src/components/ml/*.tsx",
    "frontend/src/components/strategies/*.tsx",
]

IMPROVEMENT_TASKS = [
    "Add hover states and micro-animations (0.15s ease transitions) to interactive elements",
    "Improve loading skeleton placeholders — use shimmer effect matching Bloomberg dark theme",
    "Improve accessibility: add aria-label, role attributes, keyboard navigation support",
    "Add empty-state placeholders when data is null/empty (no data messages, action buttons)",
    "Improve responsive layout for smaller screens (1280px minimum width)",
    "Improve number formatting: always show 2 decimal places for prices, + prefix for gains",
    "Add tooltip to complex metrics explaining what they measure (Sharpe, Sortino, etc.)",
    "Improve color consistency with theme spec: ensure all profit/loss uses exact green/red hex",
    "Improve table row hover effects with subtle #1a2a1a (green) or #2a1a1a (red) background",
    "Add real-time sparkline mini-charts to strategy cards and position rows",
    "Improve typography hierarchy: use JetBrains Mono for all numbers, Inter for labels",
    "Add export CSV button to all data tables",
]

def pick_file() -> str | None:
    if COMPONENT_OVERRIDE:
        matches = glob.glob(f"frontend/src/**/*{COMPONENT_OVERRIDE}*", recursive=True)
        if matches:
            return matches[0]

    hour = datetime.now(timezone.utc).hour
    pattern = FRONTEND_PATTERNS[hour % len(FRONTEND_PATTERNS)]
    files = glob.glob(pattern)
    if not files:
        # Fallback to any TSX file
        files = glob.glob("frontend/src/**/*.tsx", recursive=True)
    if not files:
        return None
    return random.choice(files)

def pick_task() -> str:
    hour = datetime.now(timezone.utc).hour
    return IMPROVEMENT_TASKS[hour % len(IMPROVEMENT_TASKS)]

# ── Improvement ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are Priya Iyer, VP Frontend at QuantEdge (ex-Bloomberg Terminal team).
You are improving a React 18 + TypeScript + Tailwind trading dashboard.

Design spec:
{THEME_SPEC}

Rules:
1. Output ONLY the complete improved TypeScript/TSX file — no markdown, no ``` fences, no explanation
2. Never add hardcoded/mock data
3. Never change API endpoint URLs
4. Never remove existing functionality
5. Keep all existing imports that are actually used
6. The output must be syntactically valid TypeScript/TSX
7. Never use emojis in the actual UI (Bloomberg terminals don't have emojis)
8. Always use const arrow functions for React components
"""

def improve_component(file_path: str, content: str, task: str) -> str | None:
    if len(content) > 10000:
        content = content[:10000] + "\n// ... (truncated)"

    prompt = f"""{SYSTEM_PROMPT}

File: {file_path}
Improvement task: {task}

Current file:
{content}

Output the complete improved file:"""

    result = llm(prompt)
    if not result:
        return None

    # Strip fences if present
    if result.startswith("```"):
        lines = result.split("\n")
        result = "\n".join(lines[1:])
        if result.rstrip().endswith("```"):
            result = result.rstrip()[:-3]

    return result.strip()

def syntax_check_tsx(code: str) -> bool:
    """Basic validity check: must have import React or 'use client', must have export."""
    has_structure = ("export" in code and ("import" in code or "'use client'" in code))
    has_jsx = ("<" in code and ">" in code)
    return has_structure and has_jsx and len(code) > 100

def git_commit(file_path: str, message: str) -> bool:
    subprocess.run(["git", "add", file_path], check=True)
    result = subprocess.run(["git", "diff", "--cached", "--stat"], capture_output=True, text=True)
    if not result.stdout.strip():
        print("No changes to commit")
        return False
    subprocess.run(["git", "commit", "-m", message], check=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "Frontend Design Agent",
                        "GIT_AUTHOR_EMAIL": "frontend-agent@quantedge.ai",
                        "GIT_COMMITTER_NAME": "Frontend Design Agent",
                        "GIT_COMMITTER_EMAIL": "frontend-agent@quantedge.ai"})
    return True

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    task = pick_task()
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M UTC')}] Frontend task: {task}")

    improved = 0
    for attempt in range(5):
        target = pick_file()
        if not target:
            print("No frontend files found")
            break

        try:
            with open(target) as f:
                content = f.read()
        except Exception:
            continue

        if len(content) < 50:
            continue

        print(f"  Improving: {target}")
        result = improve_component(target, content, task)

        if not result:
            print(f"  ✗ No LLM response for {target}")
            continue

        if not syntax_check_tsx(result):
            print(f"  ✗ TSX validation failed for {target}")
            continue

        if result.strip() == content.strip():
            print(f"  = No change for {target}")
            continue

        with open(target, "w") as f:
            f.write(result)

        short = target.replace("frontend/src/", "")
        msg = f"ui(design): {short} — {task[:70]}"
        if git_commit(target, msg):
            improved += 1
            print(f"  ✓ Committed: {msg}")
            break  # One improvement per run keeps diffs clean

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "files_improved": improved,
    }
    with open("/tmp/frontend_design_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Frontend design agent: {improved} file(s) improved")
    return 0

if __name__ == "__main__":
    sys.exit(main())
