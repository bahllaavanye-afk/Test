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
