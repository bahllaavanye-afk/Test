#!/usr/bin/env python3
"""Import Options Alpha bots from a JSON dump into backend/app/bots/templates.py.

Closes the loop from docs/playbooks/OA_BOT_COPY.md: a Cowork/browser session
extracts bots to oa_dump.json; this script turns them into BOT_TEMPLATES
entries (1-min / 2.5% / no_position defaults), deduping against existing keys.

Usage:
    python scripts/import_oa_bots.py oa_dump.json [--dry-run]

Input: JSON array of bots. Two shapes accepted:
  1. Full schema (preferred, from the playbook extraction spec):
     {"name", "symbol", "legs":[{"side","option_type"|"type","delta","dte","ratio"}],
      "trigger":{"interval","time_window":{"start","end"}}, "conditions":[...],
      "exits":{"tp_pct","sl_pct","time_hours"}, "allocation_pct", "source_url"}
     Bots WITHOUT legs are treated as equity bots and need an "action" dict.
  2. Bookmarklet shape ({"name","source_url","raw_text"}): SKIPPED with a
     message — raw_text needs human/LLM parsing; file it as an [oa-import]
     issue instead.

Exit codes: 0 = ok (including nothing to do), 1 = bad input, 2 = write failed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "backend" / "app" / "bots" / "templates.py"

DEFAULT_INTERVAL = "1m"
DEFAULT_SIZE_PCT = 2.5
DEFAULT_TP = 25
DEFAULT_SL = 100
DEFAULT_TIME_HOURS = 240
DEFAULT_WINDOW = {"start": "14:35", "end": "20:00"}  # US cash session, UTC


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"oa_{s[:40]}" if not s.startswith("oa_") else s[:43]


def build_entry(bot: dict) -> tuple[str, dict] | None:
    """One dump record -> (key, BOT_TEMPLATES entry). None = skip (with reason)."""
    name = str(bot.get("name", "")).strip()
    if not name:
        print("  ✗ skipped: record has no name")
        return None
    if "raw_text" in bot and "legs" not in bot and "action" not in bot:
        print(f"  ✗ skipped '{name}': bookmarklet raw_text needs parsing — "
              f"file an [oa-import] issue with it instead")
        return None

    key = slugify(name)
    symbol = str(bot.get("symbol", "SPY")).upper()
    legs = bot.get("legs") or []
    exits = bot.get("exits") or {}
    tp = float(exits.get("tp_pct", DEFAULT_TP))
    sl = float(exits.get("sl_pct", DEFAULT_SL))
    hours = int(exits.get("time_hours", DEFAULT_TIME_HOURS))
    alloc = float(bot.get("allocation_pct", DEFAULT_SIZE_PCT))
    trig = bot.get("trigger") or {}
    interval = str(trig.get("interval", DEFAULT_INTERVAL))
    window = trig.get("time_window") or DEFAULT_WINDOW

    conditions: list[dict] = [
        {"type": "time_window",
         "start_time": str(window.get("start", DEFAULT_WINDOW["start"])),
         "end_time": str(window.get("end", DEFAULT_WINDOW["end"]))},
    ]
    for c in bot.get("conditions") or []:
        if isinstance(c, dict) and c.get("type"):
            conditions.append(c)
    if bot.get("no_position_guard", True):
        conditions.append({"type": "no_position"})

    if legs:
        norm_legs = []
        for leg in legs:
            side = str(leg.get("side", "")).lower()
            otype = str(leg.get("option_type") or leg.get("type") or "").lower()
            if side not in ("buy", "sell") or otype not in ("put", "call"):
                print(f"  ✗ skipped '{name}': bad leg {leg}")
                return None
            norm_legs.append({
                "side": side, "option_type": otype,
                "delta": float(leg.get("delta", 0.3)),
                "dte": int(leg.get("dte", 7)),
                "ratio": int(leg.get("ratio", 1)),
            })
        action = {"type": "open_option_spread", "size_pct": alloc,
                  "take_profit_pct": tp, "legs": norm_legs}
        market_type = "options"
    else:
        action = bot.get("action")
        if not isinstance(action, dict) or not action.get("type"):
            print(f"  ✗ skipped '{name}': no legs and no action")
            return None
        action.setdefault("size_pct", alloc)
        market_type = str(bot.get("market_type", "equity"))

    desc = str(bot.get("description") or f"OA import: {name}").strip()
    if bot.get("source_url"):
        desc += f" [{bot['source_url']}]"
    entry = {
        "name": name, "description": desc[:300], "symbol": symbol,
        "market_type": market_type,
        "trigger": {"type": "schedule", "interval": interval},
        "conditions": conditions, "condition_logic": "ALL",
        "action": action,
        "exit_rules": [
            {"type": "take_profit", "value": tp},
            {"type": "stop_loss", "value": sl},
            {"type": "time_exit", "hours": hours},
        ],
    }
    return key, entry


def insert_entries(src: str, entries: dict[str, dict]) -> str:
    """Insert entries before BOT_TEMPLATES' closing brace (depth-tracked)."""
    start = src.index("BOT_TEMPLATES")
    open_i = src.index("{", start)
    depth, i = 0, open_i
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    if depth != 0:
        raise ValueError("unbalanced braces in templates.py")
    blob = ""
    for key, entry in entries.items():
        body = json.dumps(entry, indent=8, ensure_ascii=False)
        body = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False",
                      re.sub(r"\bnull\b", "None", body)))
        # align json's closing brace with dict-value indentation
        body = body[:-1] + "    }"
        blob += f'    "{key}": {body},\n'
    return src[:i] + blob + src[i:]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    dump_path = Path(sys.argv[1])
    dry = "--dry-run" in sys.argv
    try:
        bots = json.loads(dump_path.read_text())
        assert isinstance(bots, list)
    except Exception as exc:  # noqa: BLE001
        print(f"✗ cannot read {dump_path}: {exc}")
        return 1

    src = TEMPLATES.read_text()
    existing_keys = set(re.findall(r'^\s{4}"([a-z0-9_]+)":', src, re.M))
    existing_names = set(re.findall(r'"name":\s*"([^"]+)"', src))

    new: dict[str, dict] = {}
    for bot in bots:
        built = build_entry(bot if isinstance(bot, dict) else {})
        if not built:
            continue
        key, entry = built
        if key in existing_keys or entry["name"] in existing_names:
            print(f"  = duplicate, skipped: {entry['name']} ({key})")
            continue
        new[key] = entry
        print(f"  + {key}: {entry['name']} ({entry['market_type']}, "
              f"{len(entry['action'].get('legs', []))} legs)")

    if not new:
        print("Nothing new to import.")
        return 0
    out = insert_entries(src, new)
    # validate before touching disk: the result must be importable
    ns: dict = {"__annotations__": {}}
    exec(compile(out, str(TEMPLATES), "exec"), ns)  # noqa: S102 — our own file
    assert all(k in ns["BOT_TEMPLATES"] for k in new)
    if dry:
        print(f"[dry-run] would add {len(new)} bot(s)")
        return 0
    try:
        TEMPLATES.write_text(out)
    except OSError as exc:
        print(f"✗ write failed: {exc}")
        return 2
    try:
        shown = TEMPLATES.relative_to(REPO)
    except ValueError:  # e.g. sandboxed copy in tests
        shown = TEMPLATES
    print(f"✓ added {len(new)} bot(s) to {shown}")
    print("Next: commit on a branch, push — the automerge gate lands it and "
          "the weekly backtester scores the new bots.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
