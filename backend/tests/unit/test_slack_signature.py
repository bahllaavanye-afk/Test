"""Slack request-signature verification (restored Codex P1 security fix)."""
import hashlib
import hmac
import time

from app.api.v1.notifications import _verify_slack_signature


def _sign(secret: str, ts: str, body: bytes) -> str:
    base = b"v0:" + ts.encode() + b":" + body
    return "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()


def test_valid_signature_passes():
    secret, ts, body = "s3cr3t", str(int(time.time())), b'{"type":"event_callback"}'
    assert _verify_slack_signature(body, ts, _sign(secret, ts, body), secret) is True


def test_forged_signature_fails():
    secret, ts, body = "s3cr3t", str(int(time.time())), b'{"x":1}'
    assert _verify_slack_signature(body, ts, "v0=deadbeef", secret) is False
    # right signature but wrong secret
    assert _verify_slack_signature(body, ts, _sign("other", ts, body), secret) is False


def test_missing_secret_or_headers_fails_closed():
    ts, body = str(int(time.time())), b"{}"
    assert _verify_slack_signature(body, ts, _sign("s", ts, body), "") is False
    assert _verify_slack_signature(body, "", "v0=x", "s") is False


def test_stale_timestamp_rejected_replay_protection():
    secret, body = "s3cr3t", b"{}"
    old = str(int(time.time()) - 9999)
    assert _verify_slack_signature(body, old, _sign(secret, old, body), secret) is False


def test_none_inputs_are_handled_gracefully():
    # All None inputs should result in a safe False return rather than an exception
    assert _verify_slack_signature(None, None, None, None) is False
    assert _verify_slack_signature(b"{}", None, "v0=deadbeef", "s") is False
    assert _verify_slack_signature(None, str(int(time.time())), "v0=deadbeef", "s") is False
    assert _verify_slack_signature(b"{}", str(int(time.time())), None, "s") is False
    assert _verify_slack_signature(b"{}", str(int(time.time())), "v0=deadbeef", None) is False


def test_empty_collections_and_boundary_timestamp():
    # Empty body should be rejected
    secret, ts = "s3cr3t", str(int(time.time()))
    empty_body = b""
    assert _verify_slack_signature(empty_body, ts, _sign(secret, ts, empty_body), secret) is False

    # Timestamp exactly at the allowed boundary (e.g., 5 minutes old) should be accepted.
    # Assuming Slack's tolerance is 300 seconds; adjust if implementation differs.
    boundary_ts = str(int(time.time()) - 300)
    body = b'{"type":"event_callback"}'
    assert _verify_slack_signature(body, boundary_ts, _sign(secret, boundary_ts, body), secret) is True