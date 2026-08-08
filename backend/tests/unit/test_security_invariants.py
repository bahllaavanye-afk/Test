"""Brain-independent security invariants — structural guards that fail if a
safety-critical control is removed again.

These exist because two Codex security fixes silently regressed (the REST order
risk gate and the /slack/events signature check) and the LLM-based red team missed it (the
brain was unavailable). These are pure source-structure assertions: no LLM, no network,
so they catch regressions even when the brain is down.
"""
from pathlib import Path
import re

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


def test_rest_order_submission_gate_order_boundary():
    """Ensure that *every* occurrence of the broker call appears after a risk check."""
    src = (API / "orders.py").read_text()
    # Find all indices of the risk check and broker call.
    check_idxs = [m.start() for m in re.finditer(r"\bcheck_order\b", src)]
    broker_idxs = [m.start() for m in re.finditer(r"\bsubmit_alpaca_order\b", src)]
    assert check_idxs, "no risk check found in orders.py"
    assert broker_idxs, "no broker call found in orders.py"
    # The earliest broker index must be greater than the earliest check index.
    assert min(broker_idxs) > min(check_idxs), (
        "first broker call occurs before any risk check"
    )
    # Additionally, ensure no broker call appears before the *last* risk check,
    # which would indicate a missing gate in a later function.
    assert max(broker_idxs) > max(check_idxs), (
        "last broker call occurs before the final risk check"
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


def test_notifications_routes_have_dependency():
    """Each route handler must explicitly depend on `get_current_user`."""
    src = (API / "notifications.py").read_text()
    lines = src.splitlines()
    # Build a map from decorator line index to the next function definition.
    for i, line in enumerate(lines):
        if line.strip().startswith("@router."):
            # Look ahead for the function definition.
            func_def = None
            for j in range(i + 1, len(lines)):
                if lines[j].lstrip().startswith("def "):
                    func_def = lines[j]
                    break
            assert func_def is not None, "route decorator without a function definition"
            # The function signature must include the auth dependency.
            assert "Depends(get_current_user)" in func_def, (
                f"route handler '{func_def.strip()}' missing Depends(get_current_user)"
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


def test_no_slack_imports_present():
    """Ensure no source file imports the Slack SDK or related modules."""
    backend_app = CONFIG.parent
    offenders = []
    slack_import_patterns = [
        r"^\s*import\s+slack\b",
        r"^\s*from\s+slack\b",
        r"^\s*import\s+slack_sdk\b",
        r"^\s*from\s+slack_sdk\b",
    ]
    compiled = [re.compile(p) for p in slack_import_patterns]
    for path in backend_app.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        lines = path.read_text(errors="ignore").splitlines()
        for line in lines:
            if any(c.search(line) for c in compiled):
                offenders.append(str(path.relative_to(backend_app)))
                break
    assert not offenders, f"Slack imports detected in backend files: {offenders}"