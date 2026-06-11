"""
Autonomous Strategy Generator — uses Gemini/Claude to write new trading strategies.

Reads existing strategies to avoid duplication, generates novel ones,
writes them to backend/app/strategies/manual/ and registers them.

Run by: autonomous-strategy-generator.yml (weekly)
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from datetime import datetime, timezone


def _resolve_key(*names: str) -> str:
    for name in names:
        v = os.environ.get(name, "")
        if v: return v
        if not name[-1].isdigit():
            v = os.environ.get(name + "_1", "")
            if v: return v
    return ""


REPO_ROOT = Path(__file__).parent.parent
STRATEGIES_DIR = REPO_ROOT / "backend" / "app" / "strategies" / "manual"
INIT_FILE = REPO_ROOT / "backend" / "app" / "strategies" / "__init__.py"
CONFIGS_DIR = REPO_ROOT / "experiments" / "configs"

MARKET_TYPE = os.environ.get("MARKET_TYPE", "auto")
N_STRATEGIES = int(os.environ.get("N_STRATEGIES", "1"))

# ─── Existing strategy inventory ─────────────────────────────────────────────

def _existing_strategy_names() -> list[str]:
    names = []
    for f in STRATEGIES_DIR.glob("*.py"):
        if f.stem.startswith("_"):
            continue
        m = re.search(r'name\s*=\s*["\']([^"\']+)["\']', f.read_text())
        if m:
            names.append(m.group(1))
    return names

def _existing_class_names() -> list[str]:
    names = []
    for f in STRATEGIES_DIR.glob("*.py"):
        if f.stem.startswith("_"):
            continue
        m = re.search(r'^class\s+(\w+)\s*\(', f.read_text(), re.MULTILINE)
        if m:
            names.append(m.group(1))
    return names

# ─── LLM prompt ──────────────────────────────────────────────────────────────

_BASE_PROMPT = """You are a quantitative researcher at QuantEdge, a professional algorithmic trading firm.

Existing strategies (do NOT duplicate these):
{existing}

Your task: write ONE new, unique, academically-grounded trading strategy for {market_type} markets.

Requirements:
1. Must be genuinely novel — not a copy or minor variation of existing ones
2. Must cite a real academic paper or documented market anomaly
3. Must implement the AbstractStrategy interface exactly
4. market_type must be one of: "equity", "crypto", "polymarket"
5. strategy_type must be "manual"
6. risk_bucket must be "directional" or "arbitrage"
7. The analyze() method must return a Signal or None — never crash
8. The backtest_signals() method must use .shift(1) to prevent lookahead bias
9. No external API calls in analyze() — only operate on the data DataFrame provided
10. No ML models — this is a pure indicator-based manual strategy

Output ONLY the complete Python file content (no markdown, no explanation).
The file must be valid Python that can be directly written to a .py file.
Start with the module docstring, then imports, then the class.

The class must follow this exact pattern:
```python
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

class MyStrategy(AbstractStrategy):
    name = "my_strategy_name"           # snake_case, unique
    display_name = "My Strategy Name"
    market_type = "equity"              # equity | crypto | polymarket
    strategy_type = "manual"
    risk_bucket = "directional"         # directional | arbitrage
    tick_interval_seconds = 3600.0

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        # ... params

    def description(self) -> str:
        return "..."

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        # ... return Signal(...) or None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        # ... use .shift(1), return BacktestSignals(entries, exits, short_entries, short_exits)
```
"""

# ─── LLM calls ───────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    key = _resolve_key("GEMINI_API_KEY", "GEMINI_API_KEY_1")
    if not key:
        return ""
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config={"temperature": 0.7, "max_output_tokens": 4096},
        )
        resp = model.generate_content(prompt)
        return resp.text or ""
    except Exception as e:
        print(f"Gemini call failed: {e}", file=sys.stderr)
        return ""


def _call_claude(prompt: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return ""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text if msg.content else ""
    except Exception as e:
        print(f"Claude call failed: {e}", file=sys.stderr)
        return ""


def _generate_strategy_code(market_type: str, existing: list[str]) -> str:
    prompt = _BASE_PROMPT.format(
        existing=", ".join(existing[:20]),
        market_type=market_type,
    )
    # Try Gemini first, fall back to Claude
    code = _call_gemini(prompt)
    if not code:
        code = _call_claude(prompt)
    return code


# ─── Code validation ─────────────────────────────────────────────────────────

def _extract_clean_python(raw: str) -> str:
    """Strip markdown fences if present."""
    # Remove ```python ... ``` wrappers
    m = re.search(r'```(?:python)?\s*([\s\S]+?)```', raw)
    if m:
        return m.group(1).strip()
    return raw.strip()


def _validate_strategy_code(code: str) -> tuple[bool, str]:
    """Parse and validate the generated strategy code."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    # Must have a class
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    if not classes:
        return False, "No class definition found"

    cls = classes[0]
    methods = {n.name for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)}
    required = {"__init__", "analyze", "backtest_signals", "description"}
    missing = required - methods
    if missing:
        return False, f"Missing methods: {missing}"

    # Must have name = "..."
    if "name = " not in code:
        return False, "Missing name = '...' class attribute"

    # Must have shift(1) in backtest_signals to prevent lookahead
    if "shift(1)" not in code and "shift( 1 )" not in code:
        return False, "backtest_signals must use .shift(1) to prevent lookahead bias"

    return True, "OK"


def _extract_class_name(code: str) -> str:
    m = re.search(r'^class\s+(\w+)\s*\(', code, re.MULTILINE)
    return m.group(1) if m else "UnknownStrategy"


def _extract_strategy_name(code: str) -> str:
    m = re.search(r'name\s*=\s*["\']([^"\']+)["\']', code)
    return m.group(1) if m else "unknown_strategy"


# ─── File writing ─────────────────────────────────────────────────────────────

def _write_strategy_file(code: str, strategy_name: str) -> Path:
    filename = strategy_name.lower().replace(" ", "_").replace("-", "_") + ".py"
    path = STRATEGIES_DIR / filename
    path.write_text(code + "\n")
    print(f"Written: {path}")
    return path


def _register_strategy(class_name: str, strategy_name: str, module_stem: str) -> None:
    """Add import and registry entry to strategies/__init__.py."""
    content = INIT_FILE.read_text()

    import_line = f"from app.strategies.manual.{module_stem} import {class_name}"
    registry_line = f'    "{strategy_name}": {class_name},'

    # Skip if already registered
    if import_line in content or strategy_name in content:
        print(f"Strategy {strategy_name} already registered — skipping")
        return

    # Add import near the top (after existing manual imports)
    # Find last manual import line
    lines = content.splitlines()
    last_import_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("from app.strategies.manual."):
            last_import_idx = i

    lines.insert(last_import_idx + 1, import_line)

    # Add to STRATEGY_REGISTRY dict
    content_with_import = "\n".join(lines)
    content_with_import = re.sub(
        r'(STRATEGY_REGISTRY\s*=\s*\{[^}]*)',
        lambda m: m.group(0) + f"\n{registry_line}",
        content_with_import,
        count=1,
    )

    INIT_FILE.write_text(content_with_import)
    print(f"Registered: {strategy_name} ({class_name})")


def _write_experiment_config(strategy_name: str, market_type: str) -> None:
    """Write a basic experiment config for the new strategy."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    symbol = "BTC-USD" if market_type == "crypto" else "SPY"
    interval = "1d"
    cfg = {
        "experiment": {
            "name": f"{strategy_name}_v1",
            "model": "gemini",
            "symbol": symbol,
            "interval": interval,
            "strategy": strategy_name,
            "description": f"Auto-generated experiment for {strategy_name}",
        },
        "data": {
            "train_start": "2021-01-01",
            "train_end": "2023-12-31",
            "test_start": "2024-01-01",
            "test_end": "2025-12-31",
        },
        "results": {
            "test_sharpe": None,
            "trained_at": None,
        },
    }
    import yaml
    out = CONFIGS_DIR / f"{strategy_name}_v1.yaml"
    out.write_text(yaml.dump(cfg, default_flow_style=False))
    print(f"Experiment config: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    existing_names = _existing_strategy_names()
    existing_classes = _existing_class_names()

    market_types = ["equity", "crypto", "polymarket"]
    if MARKET_TYPE != "auto":
        market_types = [MARKET_TYPE]

    generated = 0
    for i in range(N_STRATEGIES):
        mt = market_types[i % len(market_types)]
        print(f"\n=== Generating strategy {i+1}/{N_STRATEGIES} ({mt}) ===")

        raw_code = _generate_strategy_code(mt, existing_names)
        if not raw_code:
            print("No LLM response — skipping (set GEMINI_API_KEY or ANTHROPIC_API_KEY)")
            continue

        code = _extract_clean_python(raw_code)
        valid, reason = _validate_strategy_code(code)
        if not valid:
            print(f"Validation failed: {reason}")
            print("First 300 chars of generated code:")
            print(code[:300])
            continue

        class_name = _extract_class_name(code)
        strategy_name = _extract_strategy_name(code)

        # Dedup check
        if strategy_name in existing_names:
            print(f"Duplicate strategy name '{strategy_name}' — skipping")
            continue

        # Write files
        path = _write_strategy_file(code, strategy_name)
        module_stem = path.stem
        _register_strategy(class_name, strategy_name, module_stem)
        _write_experiment_config(strategy_name, mt)

        existing_names.append(strategy_name)
        generated += 1
        print(f"✅ Generated: {strategy_name} ({class_name})")

    print(f"\n=== Done: {generated} new strategies generated ===")
    if generated == 0:
        print("No strategies generated. Check API keys.")
        # Don't fail the workflow — just produce no output
        sys.exit(0)


if __name__ == "__main__":
    main()
