"""render_fix_db_url.to_pooler_region — correct a wrong-region Supabase pooler URL.

A pooler URL in the wrong region (aws-0-<region>.pooler.supabase.com) is what
makes Supavisor answer "Tenant or user not found" for a HEALTHY project — the
exact failure the live backend showed for weeks. The old script skipped any
already-pooler URL; this locks in the region correction.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip("httpx")

os.environ.setdefault("SUPABASE_REGION", "us-west-1")
os.environ.setdefault("SUPABASE_POOLER_PORT", "6543")

_MOD = Path(__file__).parent / "render_fix_db_url.py"
_spec = importlib.util.spec_from_file_location("rfdu_test", _MOD)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)  # type: ignore[union-attr]

_REF = "vexzwnfbmznvxoxxktax"


def test_wrong_region_pooler_is_rewritten():
    url = f"postgresql+asyncpg://postgres.{_REF}:pw@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
    fixed = m.to_pooler_region(url)
    assert fixed == (
        f"postgresql+asyncpg://postgres.{_REF}:pw@aws-0-us-west-1.pooler.supabase.com:6543/postgres"
    )


def test_correct_region_is_noop():
    url = f"postgresql+asyncpg://postgres.{_REF}:pw@aws-0-us-west-1.pooler.supabase.com:6543/postgres"
    assert m.to_pooler_region(url) is None


def test_alembic_psycopg2_variant_is_rewritten():
    url = f"postgresql+psycopg2://postgres.{_REF}:pw@aws-0-eu-central-1.pooler.supabase.com:5432/postgres"
    fixed = m.to_pooler_region(url)
    assert "aws-0-us-west-1.pooler.supabase.com" in fixed
    assert "psycopg2" in fixed and ":5432/" in fixed   # driver + port preserved


def test_direct_ipv6_host_is_not_a_pooler():
    url = f"postgresql+asyncpg://postgres:pw@db.{_REF}.supabase.co:5432/postgres"
    assert m.to_pooler_region(url) is None   # handled by the direct→pooler path, not this one


def test_password_and_user_preserved():
    url = f"postgresql+asyncpg://postgres.{_REF}:S3cr3t%21@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"
    fixed = m.to_pooler_region(url)
    assert f"postgres.{_REF}:S3cr3t%21@" in fixed
