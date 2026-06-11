"""
TV Indicator Improvement Agent
================================
Reads the current tv_indicators.py, uses free LLM cascade to propose
SOTA upgrades (multi-timeframe confluence, adaptive periods, volume confirmation,
regime filtering), validates the output compiles cleanly, then commits & pushes.

Runs as a GitHub Action every 6 hours.
"""
from __future__ import annotations
import json, os, re, subprocess, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from llm_common import llm, slack_post, memory_write

REPO_ROOT = Path(__file__).parent.parent
BRANCH    = "main"
TV_FILE   = REPO_ROOT / "backend/app/strategies/manual/tv_indicators.py"
SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")

ALLOW_PAID = os.environ.get("ALLOW_PAID_APIS", "False")
if ALLOW_PAID.lower() == "true":
    sys.exit(1)


def slack(channel: str, msg: str) -> None:
    if not SLACK_TOKEN:
        return
    try:
        import urllib.request as req
        payload = json.dumps({"channel": channel, "text": msg, "mrkdwn": True}).encode()
        r = req.Request(
            "https://slack.com/api/chat.postMessage",
            data=payload,
            headers={"Authorization": f"Bearer {SLACK_TOKEN}",
                     "Content-Type": "application/json"},
        )
        urllib.request.urlopen(r, timeout=10)
    except Exception as e:
        print(f"Slack error: {e}")


# ── Improvement prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior quantitative strategist improving TradingView indicator strategies.
The current strategies are basic. Your job is to make them SOTA-level by adding:

1. **Volume confirmation**: Only trade when volume > 1.5x 20-period MA
2. **Multi-signal confluence**: Require 2+ aligned signals before entry
3. **Adaptive periods**: Use ATR volatility to scale lookback windows (e.g., high ATR = shorter period)
4. **Regime gating**: Skip directional trades when ADX < 20 (no trend)
5. **Momentum filter**: Require MACD or RSI alignment with primary signal
6. **Better exit logic**: Trail stop using ATR × 2 instead of simple EMA cross

Rules you MUST follow:
- Return ONLY valid Python code — no prose, no markdown, no explanations
- Keep the same class names and `name` attributes
- Keep the same imports: `import app.ml.features.pandas_ta_compat as ta`
- All `backtest_signals()` must use `shift(1)` — no lookahead
- Return `BacktestSignals(entries, exits, short_entries, short_exits)`
- `analyze()` must return `Signal | None` (async method)
- If data is insufficient, return `_EMPTY(df.index)` / `None`
- Do NOT add new imports outside stdlib + numpy/pandas + the existing ta import
- The full file must be self-contained and syntactically valid Python
"""


def build_prompt(current_code: str) -> str:
    return f"""{SYSTEM_PROMPT}

Here is the current tv_indicators.py file. Rewrite it with SOTA improvements.
Return the COMPLETE improved file — same structure, same class names, better logic.

CURRENT FILE:
```python
{current_code}
```

IMPROVED FILE (return ONLY the Python code, starting with the module docstring or import):"""


# ── Validation ────────────────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    """Extract Python code from LLM output (strip markdown fences)."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # If no fences, assume entire text is code
    return text.strip()


def validate_code(code: str) -> tuple[bool, str]:
    """Compile-check the code and verify required class names exist."""
    required_classes = [
        "EMAStackStrategy", "SqueezeProStrategy", "WaveTrendStrategy",
        "HullSuiteStrategy", "SupertrendRsiComboStrategy", "KamaRocStrategy",
        "VwapBandsStrategy", "IchimokuCloudStrategy", "MacdDivergenceStrategy",
        "AdxDmiStrategy", "StochRsiMacdStrategy", "ElliottWaveProxyStrategy",
    ]
    try:
        compile(code, "<tv_indicators>", "exec")
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    missing = [c for c in required_classes if f"class {c}" not in code]
    if missing:
        return False, f"Missing classes: {missing}"

    if "shift(1)" not in code:
        return False, "shift(1) not found — lookahead bias risk"

    return True, "ok"


# ── Git helpers ───────────────────────────────────────────────────────────────

def git_config() -> None:
    subprocess.run(["git", "config", "user.email", "agents@quantedge.ai"],
                   cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "config", "user.name", "TV Indicator Agent"],
                   cwd=REPO_ROOT, check=True)


def commit_and_push(message: str) -> None:
    subprocess.run(["git", "add", str(TV_FILE.relative_to(REPO_ROOT))],
                   cwd=REPO_ROOT, check=True)
    r = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT)
    if r.returncode == 0:
        print("[tv-improve] No changes to commit")
        return
    subprocess.run(["git", "commit", "-m", message], cwd=REPO_ROOT, check=True)
    for delay in [2, 4, 8, 16]:
        result = subprocess.run(
            ["git", "push", "-u", "origin", BRANCH],
            cwd=REPO_ROOT,
        )
        if result.returncode == 0:
            break
        import time; time.sleep(delay)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not TV_FILE.exists():
        print(f"[tv-improve] {TV_FILE} not found — aborting")
        return

    current_code = TV_FILE.read_text()
    print(f"[tv-improve] Current file: {len(current_code)} chars")

    prompt = build_prompt(current_code[:12000])  # keep within token budget
    raw = llm(prompt, max_tokens=8000)
    if not raw:
        msg = "⚠️ *TV Indicator Agent:* All LLM providers failed — no improvement this cycle"
        print(msg)
        slack("#desk-tv-indicators", msg)
        return

    improved_code = extract_code(raw)
    ok, reason = validate_code(improved_code)

    if not ok:
        msg = f"⚠️ *TV Indicator Agent:* Generated code failed validation: `{reason}` — keeping current version"
        print(msg)
        slack("#desk-tv-indicators", msg)
        return

    # Check if there are meaningful changes (not just whitespace)
    if improved_code.strip() == current_code.strip():
        print("[tv-improve] No meaningful changes in improved code")
        slack("#desk-tv-indicators", "ℹ️ *TV Indicator Agent:* Strategies already at SOTA level — no changes needed")
        return

    # Count improvements
    improvements = []
    if "volume" in improved_code.lower() and "volume" not in current_code.lower():
        improvements.append("volume confirmation")
    if "adx" in improved_code.lower():
        improvements.append("ADX regime gate")
    if "atr" in improved_code.lower() and improved_code.lower().count("atr") > current_code.lower().count("atr"):
        improvements.append("ATR adaptive stops")
    if "confluence" in improved_code.lower() or improved_code.lower().count("&") > current_code.lower().count("&") + 5:
        improvements.append("multi-signal confluence")

    TV_FILE.write_text(improved_code)
    print(f"[tv-improve] Wrote improved code ({len(improved_code)} chars)")

    git_config()
    improvements_str = ", ".join(improvements) if improvements else "signal logic"
    commit_msg = f"feat(tv-indicators): SOTA upgrade — {improvements_str}"
    commit_and_push(commit_msg)

    summary = f"✅ *TV Indicator Agent:* Upgraded 12 strategies\n"
    if improvements:
        summary += "Improvements: " + " | ".join(f"`{i}`" for i in improvements) + "\n"
    summary += f"_{len(improved_code)} chars_ → committed to `{BRANCH}`"
    print(summary)
    slack("#desk-tv-indicators", summary)


if __name__ == "__main__":
    main()
