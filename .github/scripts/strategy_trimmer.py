"""
Continuous strategy trimmer — the demotion counterpart to strategy_promotion.py.
================================================================================
Option-Alpha-style "always trimming": the promotion gate promotes winners; this
retires persistent losers so the desk doesn't keep bleeding on dead strategies.

Reads paper performance from backend/performance_log/strategy_performance.json
(written by fill_tracker.py: per strategy → trades, wins, win_rate,
avg_return_pct, total_return_pct). Flags strategies to TRIM, records them in
.github/state/strategy_trims.json, and posts a summary to Discord #alpha-research.
The desk placer can read the trims file to skip retired strategies.

Only judges strategies with enough trades (statistical significance) — never
trims a fresh strategy on a tiny sample. Pure/offline gate logic in
`evaluate_trim`, covered by tests/test_strategy_trimmer.py.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # .github/scripts/x.py → repo root
PERF_FILE = REPO_ROOT / "backend" / "performance_log" / "strategy_performance.json"
STATE_DIR = REPO_ROOT / ".github" / "state"
TRIMS_FILE = STATE_DIR / "strategy_trims.json"

# ── Trim thresholds (a strategy is retired if it clears the bar for losing) ────
MIN_TRADES        = 10      # need a real sample before judging — never trim a fresh one
RETURN_FLOOR_PCT  = -5.0    # cumulative return at/below this = bleeding
WIN_RATE_FLOOR    = 0.35    # combined with negative expectancy = no edge
AVG_RETURN_FLOOR  = -0.50   # consistently negative per-trade expectancy


def evaluate_trim(stats: dict, min_trades: int = MIN_TRADES) -> tuple[bool, str]:
    """Decide whether a strategy should be retired. Pure + testable.

    Returns (trim, reason). Never trims below `min_trades` (insufficient sample).
    """
    trades = int(stats.get("trades", 0) or 0)
    if trades < min_trades:
        return False, f"insufficient sample ({trades} < {min_trades} trades)"

    total_ret = float(stats.get("total_return_pct", 0.0) or 0.0)
    win_rate = float(stats.get("win_rate", 0.0) or 0.0)
    avg_ret = float(stats.get("avg_return_pct", 0.0) or 0.0)

    if total_ret <= RETURN_FLOOR_PCT:
        return True, f"cumulative return {total_ret:.1f}% ≤ {RETURN_FLOOR_PCT}% over {trades} trades"
    if win_rate < WIN_RATE_FLOOR and avg_ret < 0:
        return True, f"no edge: win_rate {win_rate:.0%} < {WIN_RATE_FLOOR:.0%} and avg_return {avg_ret:.2f}% < 0"
    if avg_ret <= AVG_RETURN_FLOOR:
        return True, f"negative expectancy: avg_return {avg_ret:.2f}% ≤ {AVG_RETURN_FLOOR}% over {trades} trades"
    return False, "performing within tolerance"


def load_perf() -> dict:
    """The per-strategy stats map, unwrapped from fill_tracker's envelope.

    fill_tracker.py writes {generated_at, period_days, strategies,
    tracked_order_ids}; the stats live under "strategies". This used to return
    the WHOLE document, so run() iterated the envelope: it evaluated keys like
    "generated_at" and "period_days" as if each were a strategy's stats, and
    the one dict-valued key ("strategies") was handed to evaluate_trim as a
    single blob whose .get("trades") is 0 — "insufficient sample". The trimmer
    could therefore never retire anything at any level of performance.

    Invisible until the artifact existed: with no perf file, load_perf()
    returned {} and the loop did nothing either way. strategy_auto_tuner.py
    reads the same file and always unwrapped it correctly — the two consumers
    simply disagreed about the schema.
    """
    if not PERF_FILE.exists():
        return {}
    try:
        saved = json.loads(PERF_FILE.read_text())
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(saved, dict):
        return {}
    strategies = saved.get("strategies", {})
    return strategies if isinstance(strategies, dict) else {}


def load_trims() -> dict:
    if TRIMS_FILE.exists():
        try:
            return json.loads(TRIMS_FILE.read_text())
        except Exception:
            pass
    return {}


def save_trims(trims: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    TRIMS_FILE.write_text(json.dumps(trims, indent=2))


def run() -> list[dict]:
    """Evaluate all strategies; record newly-trimmed ones. Returns new-trim events."""
    perf = load_perf()
    trims = load_trims()
    now = datetime.now(timezone.utc).isoformat()
    events: list[dict] = []

    for name, stats in perf.items():
        if not isinstance(stats, dict):
            continue
        trim, reason = evaluate_trim(stats)
        if trim and name not in trims:
            trims[name] = {
                "trimmed_at": now,
                "reason": reason,
                "stats_at_trim": {k: stats.get(k) for k in
                                  ("trades", "win_rate", "avg_return_pct", "total_return_pct")},
            }
            events.append({"name": name, "reason": reason})
            print(f"[TRIM] {name}: {reason}")
        elif not trim and name in trims:
            # Recovered (e.g. re-tuned) → un-trim so it can trade again.
            print(f"[UNTRIM] {name}: recovered — {reason}")
            del trims[name]

    save_trims(trims)
    print(f"trimmed total: {len(trims)} | newly trimmed this run: {len(events)}")
    return events


def _post_chat(events: list[dict]) -> None:
    """Announce retired strategies in Discord (Slack removed 2026-07-25)."""
    if not events:
        return
    lines = ["✂️ **Strategy Trim** — retired underperformers (paper):"]
    for e in events:
        lines.append(f"• `{e['name']}` — {e['reason']}")
    lines.append("_They stay archived (restorable) and won't be re-traded until they recover._")
    try:
        import notify
        notify.post("alpha-research", "\n".join(lines), username="Strategy Trimmer")
    except Exception as exc:
        print(f"[notify] trim post failed: {exc}", file=sys.stderr)


def main() -> None:
    events = run()
    _post_chat(events)


if __name__ == "__main__":
    main()
