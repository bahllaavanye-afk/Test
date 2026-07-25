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
