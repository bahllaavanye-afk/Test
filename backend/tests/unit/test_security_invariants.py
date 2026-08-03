"""Brain-independent security invariants — structural guards that fail if a
safety-critical control is removed again.

These exist because two Codex security fixes silently regressed (the REST order risk
gate and the /slack/events signature check) and the LLM-based red team missed it (the
brain was unavailable). These are pure source-structure assertions: no LLM, no network,
so they catch regressions even when the brain is down.
"""
from pathlib import Path

API = Path(__file__).resolve().parents[2] / "app" / "api" / "v1"
CONFIG = Path(__file__).resolve().parents[2] / "app" / "config.py"


def test_rest_order_submission_is_risk_gated():
    src = (API / "orders.py").read_text()
    assert "risk_manager" in src, "risk manager reference missing from orders.py"
    # submit_order AND submit_bracket must each gate through check_order before the broker.
    assert src.count("check_order") >= 2, (
        "REST order submission must call risk_manager.check_order() in BOTH submit_order "
        "and submit_bracket before reaching the broker (regression guard)"
    )
    # The gate must precede the broker call in the file.
    assert src.index("check_order") < src.rindex("submit_alpaca_order"), (
        "risk check must come before the Alpaca submission"
    )


def test_every_notifications_route_requires_auth():
    """No unauthenticated inbound endpoint on the notifications router.

    Replaces the old /slack/events signature test: that endpoint was an
    UNAUTHENTICATED inbound webhook, so it needed an HMAC signature check to be
    safe. It was removed with the rest of the Slack integration (2026-07-25), and
    the invariant that matters now is the stronger one — every route here is
    behind `get_current_user`, so there is no unauthenticated entry point at all.
    If someone re-adds a public webhook, this fails and they must add verification.
    """
    src = (API / "notifications.py").read_text()
    routes = [ln for ln in src.splitlines() if ln.strip().startswith("@router.")]
    assert routes, "expected at least one route in notifications.py"
    # Count the auth dependency once per route handler.
    assert src.count("Depends(get_current_user)") >= len(routes), (
        f"{len(routes)} routes but only {src.count('Depends(get_current_user)')} "
        "auth dependencies — every notifications route must require auth, or carry "
        "its own request-signature verification if it is a public webhook"
    )


def test_slack_integration_stays_removed():
    """Slack was removed completely on 2026-07-25 (user directive).

    The autonomous improver rewrites these files unattended, so without a guard
    it can reintroduce a Slack path — which would silently swallow notifications
    again (the dead SLACK_BOT_TOKEN made `slack_post` a no-op for weeks).
    """
    backend_app = CONFIG.parent
    offenders = []
    for path in backend_app.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        text = path.read_text(errors="ignore")
        if "slack.com/api" in text or "SLACK_BOT_TOKEN" in text:
            offenders.append(str(path.relative_to(backend_app)))
    assert not offenders, f"Slack must stay removed from the backend; found in: {offenders}"


# ----------------------------------------------------------------------
# Strategy logic invariants – tighten entry, confirmation, and exit checks
# ----------------------------------------------------------------------


def _read_strategy_files():
    """Helper to collect all Python files under the strategies package."""
    strategies_dir = Path(__file__).resolve().parents[2] / "app" / "strategies"
    if not strategies_dir.is_dir():
        return []
    return [p for p in strategies_dir.rglob("*.py") if "__pycache__" not in str(p)]


def test_strategy_entry_conditions_are_tight():
    """Entry functions should include basic risk filters before signalling."""
    files = _read_strategy_files()
    if not files:
        # No strategy code present – nothing to enforce at the moment.
        return
    for path in files:
        src = path.read_text(errors="ignore")
        # Look for a function that appears to generate a signal (heuristic)
        if "def generate_signal" in src or "def signal_" in src:
            # Ensure the function checks volatility, price, and volume thresholds.
            # Simple keyword presence check; more sophisticated static analysis is out of scope.
            required_checks = ["volatility", "price", "volume"]
            missing = [c for c in required_checks if c not in src.lower()]
            assert not missing, (
                f"{path.relative_to(path.parents[2])} entry logic missing checks: {missing}"
            )


def test_strategy_confirmation_filters_present():
    """Signals must be confirmed before order placement."""
    files = _read_strategy_files()
    if not files:
        return
    for path in files:
        src = path.read_text(errors="ignore")
        # If an order placement call exists, ensure a confirmation call precedes it.
        if "place_order" in src:
            # Find the first occurrence of place_order
            idx_place = src.find("place_order")
            # Look backwards for a confirmation call within the same function (simple heuristic)
            snippet = src[:idx_place]
            assert "confirm_signal" in snippet, (
                f"{path.relative_to(path.parents[2])} places orders without confirming the signal"
            )


def test_strategy_exit_logic_includes_protective_measures():
    """Exit functions should contain stop‑loss and take‑profit logic."""
    files = _read_strategy_files()
    if not files:
        return
    for path in files:
        src = path.read_text(errors="ignore")
        # Heuristic: functions named exit_ or close_position
        if "def exit_" in src or "def close_position" in src:
            required_logic = ["stop_loss", "take_profit"]
            missing = [c for c in required_logic if c not in src.lower()]
            assert not missing, (
                f"{path.relative_to(path.parents[2])} exit logic missing: {missing}"
            )