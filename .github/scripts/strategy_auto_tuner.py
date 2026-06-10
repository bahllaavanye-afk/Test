"""
Strategy Auto-Tuner — reads fill-tracker output and adjusts confidence thresholds.

Reads:  backend/performance_log/strategy_performance.json
Writes: backend/performance_log/tuned_thresholds.json

Rules (applied per strategy, requires >= 5 trades):
  win_rate < 40%  → raise confidence_min by 0.03  (tighten signal quality)
  win_rate > 65%  → lower confidence_min by 0.02  (allow more signals through)
  win_rate 40-65% → no change
  < 5 trades      → no change (insufficient data)

Bounds: confidence_min stays in [0.60, 0.90].

desk_order_placer.py reads tuned_thresholds.json at startup and applies it as
an override on top of DESKS.confidence_min (floored at desk minimum, never below).

Run nightly at 22:30 UTC via strategy-auto-tune.yml.
The workflow commits + pushes the updated files with [skip ci] tag.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
TRADING_MODE    = os.environ.get("TRADING_MODE", "paper")
ALLOW_PAID      = os.environ.get("ALLOW_PAID_APIS", "False")

REPO_ROOT        = Path(__file__).parent.parent.parent
PERF_FILE        = REPO_ROOT / "backend" / "performance_log" / "strategy_performance.json"
THRESHOLDS_FILE  = REPO_ROOT / "backend" / "performance_log" / "tuned_thresholds.json"

# Safety blocks
if ALLOW_PAID.lower() == "true":
    sys.exit(1)
if TRADING_MODE == "live":
    sys.exit(1)

# Tuning constants
DEFAULT_THRESHOLD = 0.65
MIN_THRESHOLD     = 0.60
MAX_THRESHOLD     = 0.90
RAISE_STEP        = 0.03   # raise when underperforming
LOWER_STEP        = 0.02   # lower when outperforming
MIN_TRADES        = 5      # minimum sample size


def _post_slack(channel: str, text: str) -> None:
    if not SLACK_BOT_TOKEN:
        return
    try:
        import urllib.request
        payload = json.dumps({"channel": channel, "text": text}).encode()
        req = urllib.request.Request(
            "https://slack.com/api/chat.postMessage",
            data=payload,
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as e:
        print(f"  ⚠ Slack post failed: {e}", flush=True)


def main() -> None:
    print(f"QuantEdge Strategy Auto-Tuner — {datetime.now(timezone.utc).isoformat()}", flush=True)

    if not PERF_FILE.exists():
        print("⚠ strategy_performance.json not found — no data to tune from", flush=True)
        return

    try:
        saved = json.loads(PERF_FILE.read_text())
        perf  = saved.get("strategies", {})
    except Exception as e:
        print(f"✗ failed to read performance file: {e}", flush=True)
        return

    if not perf:
        print("⚠ No strategy performance data yet — nothing to tune", flush=True)
        return

    # Load current thresholds as baseline
    current: dict[str, float] = {}
    if THRESHOLDS_FILE.exists():
        try:
            saved_t = json.loads(THRESHOLDS_FILE.read_text())
            current = {k: float(v) for k, v in saved_t.get("thresholds", {}).items()}
        except Exception:
            pass

    new_thresholds: dict[str, float] = dict(current)
    rationale:      dict[str, str]   = {}
    changes:        list[str]        = []

    for sname, data in perf.items():
        trades   = data.get("trades", 0)
        win_rate = data.get("win_rate", 0.0)
        avg_ret  = data.get("avg_return_pct", 0.0)

        if trades < MIN_TRADES:
            rationale[sname] = f"only {trades} trades — need >= {MIN_TRADES} to tune"
            continue

        prev = current.get(sname, DEFAULT_THRESHOLD)

        if win_rate < 0.40:
            new_t = round(min(prev + RAISE_STEP, MAX_THRESHOLD), 3)
            if new_t != prev:
                new_thresholds[sname] = new_t
                msg = f"`{sname}`: {prev:.2f} → {new_t:.2f} ↑  (win={win_rate:.0%} avg_ret={avg_ret:+.2f}%)"
                changes.append(msg)
                rationale[sname] = f"win_rate {win_rate:.0%} < 40% → raised +{RAISE_STEP}"
            else:
                rationale[sname] = f"already at max ({MAX_THRESHOLD})"

        elif win_rate > 0.65:
            new_t = round(max(prev - LOWER_STEP, MIN_THRESHOLD), 3)
            if new_t != prev:
                new_thresholds[sname] = new_t
                msg = f"`{sname}`: {prev:.2f} → {new_t:.2f} ↓  (win={win_rate:.0%} avg_ret={avg_ret:+.2f}%)"
                changes.append(msg)
                rationale[sname] = f"win_rate {win_rate:.0%} > 65% → lowered -{LOWER_STEP}"
            else:
                rationale[sname] = f"already at min ({MIN_THRESHOLD})"

        else:
            rationale[sname] = f"win_rate {win_rate:.0%} in [40%, 65%] — no change"

    # Always write the file (even if no changes, to update timestamps)
    THRESHOLDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds":   new_thresholds,
        "rationale":    rationale,
        "strategies_evaluated": len(perf),
        "strategies_changed":   len(changes),
    }
    THRESHOLDS_FILE.write_text(json.dumps(output, indent=2))
    print(f"✓ Written {THRESHOLDS_FILE} ({len(new_thresholds)} thresholds)", flush=True)

    if changes:
        print(f"\n📐 Auto-tuned {len(changes)} confidence thresholds:", flush=True)
        for c in changes:
            print(f"  {c}", flush=True)

        msg = (
            f"*🎛️ Strategy Auto-Tuner — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}*\n"
            f"Adjusted *{len(changes)}* confidence thresholds based on live P&L:\n"
            + "\n".join(f"  • {c}" for c in changes)
            + f"\n\n_{len(perf)} strategies evaluated, {len(perf) - len(changes)} unchanged_"
        )
        _post_slack("#pnl-daily", msg)
    else:
        print("✓ All strategies within acceptable performance — no changes needed", flush=True)
        if len(perf) > 0:
            qualified = {k: v for k, v in perf.items() if v["trades"] >= MIN_TRADES}
            print(f"  Evaluated {len(qualified)} strategies with >= {MIN_TRADES} trades", flush=True)

    print("Auto-tuner complete.", flush=True)


if __name__ == "__main__":
    main()
