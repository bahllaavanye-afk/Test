"""Unit tests for render_probe_pooler pure logic (no network).

The connection test (`_try_connect`) needs a live pooler + asyncpg and only runs
in CI's workflow; here we lock down the URL parsing / candidate generation / URL
building that decide WHAT gets probed and patched.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip("httpx")

os.environ.setdefault("SUPABASE_REGION", "us-west-1")

_MOD = Path(__file__).parent / "render_probe_pooler.py"
_spec = importlib.util.spec_from_file_location("rpp_test", _MOD)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)  # type: ignore[union-attr]

_REF = "vexzwnfbmznvxoxxktax"


def test_ref_and_password_from_pooler_url():
    url = f"postgresql+asyncpg://postgres.{_REF}:s3cret@aws-0-us-west-1.pooler.supabase.com:6543/postgres"
    ref, pw = m.ref_and_password(url)
    assert ref == _REF
    assert pw == "s3cret"


def test_ref_and_password_url_encoded_password():
    url = f"postgresql+asyncpg://postgres.{_REF}:S3cr3t%21%40@aws-1-us-west-1.pooler.supabase.com:5432/postgres"
    ref, pw = m.ref_and_password(url)
    assert ref == _REF
    assert pw == "S3cr3t!@"  # decoded


def test_ref_and_password_from_direct_host():
    url = f"postgresql+asyncpg://postgres:pw@db.{_REF}.supabase.co:5432/postgres"
    ref, pw = m.ref_and_password(url)
    assert ref == _REF
    assert pw == "pw"


def test_candidate_hosts_prefers_aws1_then_aws0():
    hosts = m.candidate_hosts("us-west-1")
    assert hosts == [
        "aws-1-us-west-1.pooler.supabase.com",
        "aws-0-us-west-1.pooler.supabase.com",
    ]


def test_build_url_roundtrips_driver_and_encodes_password():
    url = m.build_url("asyncpg", _REF, "p@ss:word", "aws-1-us-west-1.pooler.supabase.com", "6543")
    assert url == (
        f"postgresql+asyncpg://postgres.{_REF}:p%40ss%3Aword"
        "@aws-1-us-west-1.pooler.supabase.com:6543/postgres"
    )
    # alembic variant keeps the psycopg2 driver
    alu = m.build_url("psycopg2", _REF, "pw", "aws-0-us-west-1.pooler.supabase.com", "5432")
    assert alu.startswith("postgresql+psycopg2://") and ":5432/postgres" in alu


def test_current_host_port_parses_pooler():
    url = f"postgresql+asyncpg://postgres.{_REF}:pw@aws-0-us-west-1.pooler.supabase.com:6543/postgres"
    host, port = m.current_host_port(url)
    assert host == "aws-0-us-west-1.pooler.supabase.com"
    assert port == "6543"


def test_ports_env_default():
    assert m.PORTS == ["6543", "5432"] or m.PORTS == ["5432", "6543"] or "6543" in m.PORTS


# ---- error classification: the distinction the whole fix turns on ----------
# "tenant or user not found" = WRONG cluster. An auth failure = RIGHT cluster
# (it knows this tenant) with a stale password. Confusing the two is what kept
# the real cause hidden for weeks, so pin it down.


class _InvalidPasswordError(Exception):
    """Stands in for asyncpg.exceptions.InvalidPasswordError (matched by name)."""


def test_invalid_password_is_positive_cluster_identification():
    exc = _InvalidPasswordError('password authentication failed for user "postgres"')
    assert m.classify_error(exc) == m.BAD_PASSWORD


def test_invalid_password_detected_by_message_alone():
    # a generic exception type still classifies via its message
    assert m.classify_error(Exception('password authentication failed for user "postgres"')) == m.BAD_PASSWORD


def test_tenant_not_found_is_wrong_cluster():
    exc = Exception("(ENOTFOUND) tenant/user postgres.vexzwnfbmznvxoxxktax not found")
    assert m.classify_error(exc) == m.NO_TENANT


def test_tenant_or_user_not_found_wording_also_detected():
    assert m.classify_error(Exception("Tenant or user not found")) == m.NO_TENANT


def test_timeout_is_unreachable_not_a_verdict_about_the_tenant():
    assert m.classify_error(TimeoutError("timed out")) == m.UNREACHABLE
    assert m.classify_error(OSError("Network is unreachable")) == m.UNREACHABLE


def test_verdict_constants_are_distinct():
    assert len({m.OK, m.BAD_PASSWORD, m.NO_TENANT, m.UNREACHABLE}) == 4


# ---- no-flap regression -----------------------------------------------------
# With the tenant on aws-1 and the password stale, BOTH aws-1 ports answer
# bad_password. The original loop skipped the CURRENT host/port when probing, so
# it always "found" the other port and patched to it — oscillating
# :6543 <-> :5432 on every scheduled run. Pin the fix: when the current host
# already reports bad_password, probe_service must make NO patch at all.

import asyncio


def _run(coro):
    # asyncio.run(), not get_event_loop(): when the whole suite runs together
    # another test can leave the ambient loop closed, which made these two
    # silently skip with "coroutine was never awaited".
    return asyncio.run(coro)


def test_no_patch_when_current_host_already_recognises_the_tenant(monkeypatch):
    url = (f"postgresql+asyncpg://postgres.{_REF}:pw"
           "@aws-1-us-west-1.pooler.supabase.com:6543/postgres")
    monkeypatch.setattr(m, "get_env_vars", lambda sid: [{"key": "DATABASE_URL", "value": url}])

    patched: list = []
    monkeypatch.setattr(m, "patch_env_var", lambda sid, k, v: patched.append((k, v)) or True)
    monkeypatch.setattr(m, "trigger_deploy", lambda sid: patched.append(("DEPLOY", sid)))

    async def fake_connect(host, port, ref, password):
        # the stale-password reality: aws-1 knows the tenant, aws-0 does not
        return m.BAD_PASSWORD if host.startswith("aws-1") else m.NO_TENANT

    monkeypatch.setattr(m, "_try_connect", fake_connect)

    changed = _run(m.probe_service("srv-test"))
    assert changed is False
    assert patched == [], f"must not patch/redeploy when only the password is stale, got {patched}"


def test_still_corrects_a_genuinely_wrong_cluster(monkeypatch):
    url = (f"postgresql+asyncpg://postgres.{_REF}:pw"
           "@aws-0-us-west-1.pooler.supabase.com:6543/postgres")
    monkeypatch.setattr(m, "get_env_vars", lambda sid: [{"key": "DATABASE_URL", "value": url}])

    patched: list = []
    monkeypatch.setattr(m, "patch_env_var", lambda sid, k, v: patched.append((k, v)) or True)
    monkeypatch.setattr(m, "trigger_deploy", lambda sid: patched.append(("DEPLOY", sid)))

    async def fake_connect(host, port, ref, password):
        return m.BAD_PASSWORD if host.startswith("aws-1") else m.NO_TENANT

    monkeypatch.setattr(m, "_try_connect", fake_connect)

    changed = _run(m.probe_service("srv-test"))
    assert changed is True
    assert any("aws-1-us-west-1" in v for k, v in patched if k == "DATABASE_URL"), patched
    # host-only correction must NOT redeploy — the password is still wrong
    assert not any(k == "DEPLOY" for k, _ in patched), "must not redeploy on a host-only fix"


def _url(password: str, encode: bool = True) -> str:
    """Build a Supabase pooler URL with the password encoded or raw."""
    from urllib.parse import quote
    pw = quote(password, safe="") if encode else password
    return (
        f"postgresql+asyncpg://postgres.vexzwnfbmznvxoxxktax:{pw}"
        f"@aws-1-us-west-1.pooler.supabase.com:6543/postgres"
    )


# ── Credential-corruption guard ───────────────────────────────────────────────
# This script REWRITES a production credential. If the stored DATABASE_URL holds
# an UNENCODED password containing '#' or '?', urlparse treats them as
# fragment/query markers and truncates it — `Abc!@#$%^&*()` parses back as
# `Abc!`. Patching then writes those 4 characters to Render and permanently
# destroys a working credential, presenting as "password authentication failed":
# indistinguishable from a rotated password, and self-inflicted.

def test_encoded_password_roundtrips_and_may_be_patched():
    for pw in ("simple123", "p@ssw0rd", "a:b@c#d?e", "Abc!@#$%^&*()", "pä55"):
        url = _url(pw, encode=True)
        ref, parsed = m.ref_and_password(url)
        assert parsed == pw, f"encoded password must survive parsing: {pw!r}"
        assert m.password_roundtrips(url, parsed) is True


def test_unencoded_password_with_fragment_marker_is_refused():
    """The corrupting case — must be refused, not silently truncated."""
    for pw in ("a:b@c#d?e", "pa#ss", "pass?x", "Abc!@#$%^&*()"):
        url = _url(pw, encode=False)
        ref, parsed = m.ref_and_password(url)
        # parsed is either None or a TRUNCATED prefix — never the real password
        assert parsed != pw, f"expected truncation for unencoded {pw!r}"
        if parsed:
            assert m.password_roundtrips(url, parsed) is False, (
                f"patching {pw!r} would write back the truncated {parsed!r}"
            )


def test_plain_unencoded_password_still_patches():
    """The guard must not block the ordinary case."""
    url = _url("simplepass123", encode=False)
    ref, parsed = m.ref_and_password(url)
    assert parsed == "simplepass123"
    assert m.password_roundtrips(url, parsed) is True


def test_truncation_is_a_real_prefix_not_a_parse_error():
    """Pin the exact mechanism so the reason stays legible."""
    url = _url("Abc!@#$%^&*()", encode=False)
    _ref, parsed = m.ref_and_password(url)
    assert parsed == "Abc!", (
        "urlparse splits at the last '@' before the '#'-fragment, leaving a "
        "4-character prefix of a 13-character password"
    )
