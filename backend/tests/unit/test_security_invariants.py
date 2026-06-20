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


def test_slack_events_verifies_signature():
    src = (API / "notifications.py").read_text()
    assert "_verify_slack_signature" in src, "Slack signature verifier missing"
    assert "X-Slack-Signature" in src, "Slack signature header not read"
    assert "slack_signing_secret" in src, "signing-secret gate missing from /slack/events"
    assert "hmac" in src and "compare_digest" in src, "must use constant-time HMAC compare"


def test_config_exposes_slack_signing_secret():
    assert "slack_signing_secret" in CONFIG.read_text(), (
        "config must expose slack_signing_secret so /slack/events can verify requests"
    )
