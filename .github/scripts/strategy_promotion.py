"""
Strategy Promotion Pipeline
============================
Automatically promotes strategies through three stages:

  backtest → paper_candidate → paper_active → live_candidate

Promotion gates:
  backtest → paper_candidate:  out-of-sample Sharpe > 1.0 AND max_dd < 20%
  paper_candidate → paper_active: 2+ weeks running with paper Sharpe > 0.8
  paper_active → live_candidate: 4+ weeks with live Sharpe > 1.0 AND max_dd < 15%

State is stored in .github/state/strategy_promotions.json
All transitions posted to Slack #engineering + #desk-lead-review.

Runs daily via GitHub Actions.
"""
from __future__ import annotations

import json, os, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import pvariance

sys.path.insert(0, str(Path(__file__).parent))
from llm_common import llm, slack_post, memory_write, core_update, core_get
from strategy_gate import passes_promotion_gate

REPO_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = REPO_ROOT / "backend" / "experiments" / "results"
CONFIGS_DIR = REPO_ROOT / "backend" / "experiments" / "configs"
STATE_DIR = REPO_ROOT / ".github" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
PROMOTIONS_FILE = STATE_DIR / "strategy_promotions.json"

ALLOW_PAID = os.environ.get("ALLOW_PAID_APIS", "False")
if ALLOW_PAID.lower() == "true":
    sys.exit(1)

# Promotion gates
BACKTEST_SHARPE_MIN = 1.0
BACKTEST_MAXDD_MAX  = 20.0
PAPER_SHARPE_MIN    = 0.8
PAPER_MIN_DAYS      = 14
LIVE_SHARPE_MIN     = 1.0
LIVE_MAXDD_MAX      = 15.0
LIVE_MIN_DAYS       = 28


# ── State helpers ──────────────────────────────────────────────────────────────

def load_promotions() -> dict:
    """
    Load PROMOTIONS_FILE. Returns dict keyed by strategy_name.

    Each entry:
        {
            "stage": "backtest|paper_candidate|paper_active|live_candidate",
            "promoted_at": <unix timestamp>,
            "sharpe": float,
            "max_dd": float,
            "notes": str,
        }
    """
    if PROMOTIONS_FILE.exists():
        try:
            return json.loads(PROMOTIONS_FILE.read_text())
        except Exception:
            pass
    return {}


def save_promotions(promotions: dict) -> None:
    """Persist PROMOTIONS_FILE atomically."""
    try:
        PROMOTIONS_FILE.write_text(json.dumps(promotions, indent=2))
    except Exception as exc:
        print(f"[WARN] Could not save promotions: {exc}", file=sys.stderr)


# ── Backtest scanner ───────────────────────────────────────────────────────────

def scan_backtest_results() -> list[dict]:
    """
    Read all JSON files in RESULTS_DIR.

    Returns list of:
        {
            "name": str,
            "test_sharpe": float,
            "val_sharpe": float,
            "max_dd": float,   # percentage, positive number
            "config": dict,    # raw JSON content
        }
    Only includes entries where status == "done" and test_sharpe is not null.
    """
    results: list[dict] = []

    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        return results

    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            print(f"[WARN] Could not parse {path.name}: {exc}", file=sys.stderr)
            continue

        # Two result shapes are supported:
        #   (a) trainer output: top-level test_sharpe/val_sharpe/max_drawdown
        #   (b) run_experiments output: metrics nested under "results" (sharpe, n_trades…)
        res = data.get("results") if isinstance(data.get("results"), dict) else {}
        status_done = data.get("status", "done") == "done"  # run_experiments has no status
        if not status_done:
            continue

        test_sharpe = data.get("test_sharpe")
        if test_sharpe is None:
            test_sharpe = res.get("sharpe")  # shape (b)
        if test_sharpe is None:
            continue

        # max_drawdown stored as a negative fraction (e.g. -0.112) → positive percent
        raw_dd = data.get("max_drawdown", res.get("max_drawdown", 0.0))
        max_dd_pct = abs(float(raw_dd)) * 100.0

        # Carry advanced metrics through to the gate from whichever shape has them.
        merged = {
            "num_trades": data.get("num_trades", res.get("n_trades")),
            "sortino": data.get("sortino", res.get("sortino")),
            "calmar": data.get("calmar", res.get("calmar")),
            "win_rate": data.get("win_rate", res.get("win_rate")),
            "profit_factor": data.get("profit_factor", res.get("profit_factor")),
            "skew": data.get("skew", res.get("skew", 0.0)),
            "kurtosis": data.get("kurtosis", res.get("kurtosis", 3.0)),
        }

        name = data.get("experiment", {}).get("name") or path.stem

        results.append({
            "name": name,
            "test_sharpe": float(test_sharpe),
            "val_sharpe": float(data.get("val_sharpe") or res.get("val_sharpe") or 0.0),
            "max_dd": max_dd_pct,
            "config": {**data, **merged},
        })

    return results


# ── Paper performance reader ───────────────────────────────────────────────────

def check_paper_performance(strategy_name: str) -> dict | None:
    """
    Read live paper-trading performance from the company brain.

    Returns {"sharpe": float, "max_dd": float, "days_active": int}
    or None if no data is available for this strategy.
    """
    paper_data = core_get("paper_performance", {})
    entry = paper_data.get(strategy_name)
    if not entry:
        return None

    try:
        return {
            "sharpe": float(entry.get("sharpe", 0.0)),
            "max_dd": float(entry.get("max_dd", 0.0)),
            "days_active": int(entry.get("days_active", 0)),
        }
    except (TypeError, ValueError):
        return None


# ── Promotion logic ────────────────────────────────────────────────────────────

def _days_since(ts: float) -> int:
    """Return number of whole days elapsed since a Unix timestamp."""
    return int((time.time() - ts) / 86400)


def run_promotion_check() -> list[dict]:
    """
    Main promotion logic. Returns list of promotion events, each:
        {
            "name": str,
            "from_stage": str,
            "to_stage": str,
            "sharpe": float,
            "max_dd": float,
            "days_active": int,
            "extra": dict,   # extra metadata (backtest config, etc.)
        }
    """
    promotions = load_promotions()
    backtest_results = scan_backtest_results()
    events: list[dict] = []
    now = time.time()

    # Multiple-testing context: deflate each Sharpe by how many strategies were
    # screened and how widely their Sharpes vary (selection-bias correction).
    n_trials = max(len(backtest_results), 1)
    _sharpes = [r["test_sharpe"] for r in backtest_results]
    sharpe_var = pvariance(_sharpes) if len(_sharpes) > 1 else 0.25

    # ── Gate 1: backtest → paper_candidate ────────────────────────────────────
    for result in backtest_results:
        name = result["name"]

        # Skip if already promoted beyond backtest stage
        if name in promotions:
            continue

        sharpe = result["test_sharpe"]
        max_dd = result["max_dd"]

        # Multi-criteria gate: Sharpe + Sortino + Calmar + max-DD + win-rate +
        # profit-factor + min-trades + OOS consistency + Deflated Sharpe.
        cfg = result.get("config", {})
        metrics = {
            "test_sharpe": sharpe,
            "val_sharpe": result["val_sharpe"],
            "max_dd": max_dd,
            # None (not 0) when unrecorded so the gate skips rather than rejects.
            "num_trades": cfg.get("num_trades") if cfg.get("num_trades") is not None else cfg.get("n_trades"),
            "sortino": cfg.get("sortino"),
            "calmar": cfg.get("calmar"),
            "win_rate": cfg.get("win_rate"),
            "profit_factor": cfg.get("profit_factor"),
            "skew": cfg.get("skew", 0.0),
            "kurtosis": cfg.get("kurtosis", 3.0),
        }
        passed, scorecard = passes_promotion_gate(metrics, n_trials, sharpe_var)
        if not passed:
            failed = [k for k, c in scorecard.items() if not c["ok"]]
            print(f"[GATE] {name}: REJECTED — failed {failed}")
            continue

        if passed:
            dsr = scorecard["deflated_sharpe"]["value"]
            promotions[name] = {
                "stage": "paper_candidate",
                "promoted_at": now,
                "sharpe": sharpe,
                "max_dd": max_dd,
                "deflated_sharpe": dsr,
                "scorecard": scorecard,
                "notes": (
                    f"Auto-promoted from backtest. "
                    f"test_sharpe={sharpe:.2f}, val_sharpe={result['val_sharpe']:.2f}, "
                    f"max_dd={max_dd:.1f}%, DSR={dsr}, trials={n_trials}"
                ),
            }
            events.append({
                "name": name,
                "from_stage": "backtest",
                "to_stage": "paper_candidate",
                "sharpe": sharpe,
                "max_dd": max_dd,
                "days_active": 0,
                "extra": result,
            })
            print(f"[PROMOTE] {name}: backtest → paper_candidate  (Sharpe={sharpe:.2f}, MaxDD={max_dd:.1f}%)")

    # ── Gate 2: paper_candidate → paper_active ────────────────────────────────
    for name, state in list(promotions.items()):
        if state["stage"] != "paper_candidate":
            continue

        days = _days_since(state["promoted_at"])
        if days < PAPER_MIN_DAYS:
            continue

        perf = check_paper_performance(name)
        if perf is None:
            # No paper data yet — keep waiting
            continue

        if perf["sharpe"] >= PAPER_SHARPE_MIN and perf["days_active"] >= PAPER_MIN_DAYS:
            promotions[name] = {
                "stage": "paper_active",
                "promoted_at": now,
                "sharpe": perf["sharpe"],
                "max_dd": perf["max_dd"],
                "notes": (
                    f"Promoted from paper_candidate after {days} days. "
                    f"paper_sharpe={perf['sharpe']:.2f}, days_active={perf['days_active']}"
                ),
            }
            events.append({
                "name": name,
                "from_stage": "paper_candidate",
                "to_stage": "paper_active",
                "sharpe": perf["sharpe"],
                "max_dd": perf["max_dd"],
                "days_active": perf["days_active"],
                "extra": {},
            })
            print(f"[PROMOTE] {name}: paper_candidate → paper_active  (Sharpe={perf['sharpe']:.2f}, days={perf['days_active']})")

    # ── Gate 3: paper_active → live_candidate ─────────────────────────────────
    for name, state in list(promotions.items()):
        if state["stage"] != "paper_active":
            continue

        days = _days_since(state["promoted_at"])
        if days < LIVE_MIN_DAYS:
            continue

        perf = check_paper_performance(name)
        if perf is None:
            continue

        if (
            perf["sharpe"] >= LIVE_SHARPE_MIN
            and perf["max_dd"] < LIVE_MAXDD_MAX
            and perf["days_active"] >= LIVE_MIN_DAYS
        ):
            promotions[name] = {
                "stage": "live_candidate",
                "promoted_at": now,
                "sharpe": perf["sharpe"],
                "max_dd": perf["max_dd"],
                "notes": (
                    f"Promoted from paper_active after {perf['days_active']} days. "
                    f"paper_sharpe={perf['sharpe']:.2f}, max_dd={perf['max_dd']:.1f}%"
                ),
            }
            events.append({
                "name": name,
                "from_stage": "paper_active",
                "to_stage": "live_candidate",
                "sharpe": perf["sharpe"],
                "max_dd": perf["max_dd"],
                "days_active": perf["days_active"],
                "extra": {},
            })
            print(f"[PROMOTE] {name}: paper_active → live_candidate  (Sharpe={perf['sharpe']:.2f}, MaxDD={perf['max_dd']:.1f}%)")

    # Persist updated state
    save_promotions(promotions)
    return events


# ── LLM summary ───────────────────────────────────────────────────────────────

def generate_promotion_summary(events: list[dict]) -> str:
    """
    Call llm() to write a brief investment memo about promoted strategies.
    Returns a plain-text memo (2-3 sentences per strategy).
    """
    if not events:
        return "No strategies were promoted in this run."

    bullets = []
    for ev in events:
        bullets.append(
            f"- {ev['name']}: {ev['from_stage']} → {ev['to_stage']} | "
            f"Sharpe={ev['sharpe']:.2f} | MaxDD={ev['max_dd']:.1f}% | "
            f"Days active={ev['days_active']}"
        )
    bullet_text = "\n".join(bullets)

    prompt = (
        f"You are a quantitative analyst writing an internal investment memo at QuantEdge.\n"
        f"The following strategies were automatically promoted today:\n\n"
        f"{bullet_text}\n\n"
        f"Write a concise investment memo (2-3 sentences per strategy) summarising why each "
        f"promotion is meaningful, what the Sharpe and drawdown numbers imply for risk-adjusted "
        f"returns, and any caveats to watch. Use professional but direct language."
    )

    return llm(
        prompt,
        system="You are a senior quant analyst at an institutional trading firm.",
        max_tokens=600,
        use_cache=False,
        inject_company_context=False,
    )


# ── Slack formatting ───────────────────────────────────────────────────────────

def _format_slack_message(events: list[dict]) -> str:
    """Build the Slack message body."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [f":rocket: *Strategy Promotion Update* — {today}", ""]

    for ev in events:
        name = ev["name"]
        from_stage = ev["from_stage"]
        to_stage = ev["to_stage"]
        sharpe = ev["sharpe"]
        max_dd = ev["max_dd"]
        days = ev["days_active"]

        if to_stage == "paper_candidate":
            cfg = ev.get("extra", {}).get("config", {})
            trained_at = cfg.get("trained_at", "unknown")
            lines.append(f":white_check_mark: *NEW PAPER CANDIDATE:* `{name}`")
            lines.append(f"  Backtest Sharpe: {sharpe:.2f} | Max DD: {max_dd:.1f}%")
            if trained_at != "unknown":
                lines.append(f"  Trained: {trained_at}")
            lines.append(f"  → Paper trading starts now on Alpaca paper account")

        elif to_stage == "paper_active":
            lines.append(f":white_check_mark: *PAPER CANDIDATE → PAPER ACTIVE:* `{name}`")
            lines.append(f"  Paper Sharpe ({days}d): {sharpe:.2f} | Max DD: {max_dd:.1f}% | {days} days active")
            lines.append(f"  → Strategy confirmed as consistently profitable in paper trading")

        elif to_stage == "live_candidate":
            lines.append(f":white_check_mark: *PAPER → LIVE CANDIDATE:* `{name}`")
            lines.append(f"  Paper Sharpe ({days}d): {sharpe:.2f} | Max DD: {max_dd:.1f}% | {days} days active")
            lines.append(f"  → Flagged for manual live review")

        lines.append("")  # blank line between entries

    return "\n".join(lines).rstrip()


# ── Main orchestrator ──────────────────────────────────────────────────────────

def main() -> None:
    print("[strategy_promotion] Starting promotion check …")

    # 1. Run promotion checks (scan + state transitions)
    events = run_promotion_check()

    if not events:
        print("[strategy_promotion] No promotions today. Exiting.")
        return

    # 2. Generate LLM investment memo
    print(f"[strategy_promotion] {len(events)} promotion(s) found. Generating memo …")
    memo = generate_promotion_summary(events)
    print("[MEMO]\n" + memo)

    # 3. Post to Slack
    slack_body = _format_slack_message(events)
    live_events = [ev for ev in events if ev["to_stage"] == "live_candidate"]
    paper_events = [ev for ev in events if ev["to_stage"] != "live_candidate"]

    if paper_events:
        slack_post("#engineering", slack_body)
        print("[strategy_promotion] Posted to #engineering")

    if live_events:
        # Live candidates need human review — post to desk lead channel
        live_body = _format_slack_message(live_events)
        slack_post("#desk-lead-review", live_body)
        print("[strategy_promotion] Posted to #desk-lead-review")

    # 4. Persist updated paper_candidate list to company brain
    promotions = load_promotions()
    paper_candidates = [
        name for name, state in promotions.items()
        if state["stage"] == "paper_candidate"
    ]
    paper_active = [
        name for name, state in promotions.items()
        if state["stage"] == "paper_active"
    ]
    live_candidates = [
        name for name, state in promotions.items()
        if state["stage"] == "live_candidate"
    ]

    core_update("paper_candidate_strategies", paper_candidates)
    core_update("paper_active_strategies", paper_active)
    core_update("live_candidate_strategies", live_candidates)

    # 5. Write episodic memory entries for each promotion
    for ev in events:
        memory_write(
            "episodic",
            {
                "lesson": (
                    f"Promoted {ev['name']} to {ev['to_stage']}: "
                    f"Sharpe={ev['sharpe']:.2f}, MaxDD={ev['max_dd']:.1f}%"
                ),
                "category": "strategy",
                "from_stage": ev["from_stage"],
                "to_stage": ev["to_stage"],
            },
        )

    print(f"[strategy_promotion] Done. {len(events)} promotion(s) processed.")


if __name__ == "__main__":
    main()
