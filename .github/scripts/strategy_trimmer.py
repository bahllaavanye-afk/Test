"""
Continuous strategy trimmer — the demotion counterpart to strategy_promotion.py.
================================================================================
Option-Alpha-style "always trimming": the promotion gate promotes winners; this
retires persistent losers so the desk doesn't keep bleeding on dead strategies.

Reads paper performance from backend/performance_log/strategy_performance.json
(written by fill_tracker.py: per strategy → trades, wins, win_rate,
avg_return_pct, total_return_pct). Flags strategies to TRIM, records them in
.github/state/strategy_trims.json, and posts a summary to Slack #alpha-research.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
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
    if not PERF_FILE.exists():
        return {}
    try:
        return json.loads(PERF_FILE.read_text())
    except Exception:
        return {}


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


def _post_slack(events: list[dict]) -> None:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token or not events:
        return
    import urllib.request
    lines = [":scissors: *Strategy Trim* — retired underperformers (paper):"]
    for e in events:
        lines.append(f"• `{e['name']}` — {e['reason']}")
    lines.append("_They stay archived (restorable) and won't be re-traded until they recover._")
    body = json.dumps({"channel": "#alpha-research", "text": "\n".join(lines)}).encode()
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage", data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0"}, method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=15)
    except Exception as exc:
        print(f"[slack] trim post failed: {exc}", file=sys.stderr)


def main() -> None:
    events = run()
    _post_slack(events)


if __name__ == "__main__":
    main()
