"""The post-deploy smoke was green through a week of paused database.

`/health` returns `{"status": "ok"}` unconditionally — deliberately, so the
Render keep-alive ping does no DB work. `smoke_test_live.py` asserted on that
and nothing else, so every subsystem check the backend already computes
(`database_primary`, `redis`, `scheduler`, `alpaca`, `ml_models`,
`background_tasks`) was invisible to the only automated post-deploy gate.

There is a trap inside the fix, and it is the reason these tests are
behavioural rather than a source grep. `IMPROVEMENTS.md:843` specifies the guard
as "fail deploy on `database.ok=false`". **That guard can never fire.**
`app/main.py:487` sets `database.ok = True` whenever `SELECT 1` succeeds, and on
the SQLite fallback it succeeds — the fallback is functional, just ephemeral.
The live payload right now, with Supabase paused:

    "database":         {"ok": true,  "latency_ms": 5.2, "fallback": "sqlite"}
    "database_primary": {"ok": false, "error": "(ENOTFOUND) tenant/user
                         postgres.vexzwnfbmznvxoxxktax not found | ..."}

So `database.ok` is **true during the exact outage the item was written to
catch**. `database_primary` (emitted at `main.py:502`, only when
`db_fallback_active`) is the field that reports it. A guard keyed on the wrong
one would ship, pass CI, page nobody, and read as done — the "green-looking
absence" pattern this repo keeps rediscovering.

`test_a_functional_sqlite_fallback_is_still_a_failure` is the one that
distinguishes the two. It is the test the naive implementation fails.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture
def smoke():
    """Fresh module with the failure/notes accumulators cleared per test."""
    mod = importlib.import_module("smoke_test_live")
    mod.failures.clear()
    mod.notes.clear()
    return mod


def _payload(**overrides) -> dict:
    """The shape `/health/detailed` actually returns, healthy by default."""
    checks = {
        "database": {"ok": True, "latency_ms": 5.2},
        "redis": {"ok": True, "latency_ms": 20.6},
        "scheduler": {"ok": True, "jobs_total": 73},
        "alpaca": {"ok": True, "note": "connected"},
    }
    checks.update(overrides.pop("checks", {}))
    body = {"status": "ok", "version": "2.0.0", "mode": "paper", "checks": checks}
    body.update(overrides)
    return body


def test_a_healthy_payload_raises_nothing(smoke):
    smoke.health_detailed_checks(200, _payload(), strict_db=True)
    assert smoke.failures == [], f"clean payload produced failures: {smoke.failures}"


def test_a_functional_sqlite_fallback_is_still_a_failure(smoke):
    """The load-bearing test: `database.ok` stays TRUE during the outage.

    This is the live shape observed 2026-08-05 with Supabase paused. An
    implementation keyed on `checks["database"]["ok"]` — i.e. the guard exactly
    as IMPROVEMENTS.md specifies it — passes this payload and is a no-op.
    """
    body = _payload(checks={
        "database": {"ok": True, "latency_ms": 5.2, "fallback": "sqlite"},
        "database_primary": {"ok": False, "error": "(ENOTFOUND) tenant/user not found"},
    })
    smoke.health_detailed_checks(200, body, strict_db=True)
    assert any("durable database" in f for f in smoke.failures), (
        "a paused primary behind a WORKING SQLite fallback did not fail the "
        "smoke. The guard is almost certainly keyed on checks['database']['ok'], "
        "which is True here — it cannot catch the condition it exists for."
    )


def test_the_same_payload_does_not_fail_the_scheduled_run(smoke):
    """Deploy-time gate, not a 48×/day pager for an operator-blocked condition."""
    body = _payload(checks={
        "database": {"ok": True, "fallback": "sqlite"},
        "database_primary": {"ok": False, "error": "unreachable at boot"},
    })
    smoke.health_detailed_checks(200, body, strict_db=False)
    assert smoke.failures == [], (
        "the non-strict (scheduled) path failed. It must report only — a known "
        "paused database paging #ci-failures every 30 minutes buries the "
        "failures that are actually actionable."
    )
    assert any("database_primary DOWN" in n for n in smoke.notes), (
        "the scheduled path went silent instead of reporting. Non-fatal must "
        "not mean invisible — that is the bug this whole file is about."
    )


def test_an_absent_database_primary_is_not_treated_as_down(smoke):
    """`database_primary` is emitted ONLY when the fallback is active.

    Its absence is the healthy case. Defaulting a missing key to False would
    fail every healthy deploy.
    """
    smoke.health_detailed_checks(200, _payload(), strict_db=True)
    assert not any("durable database" in f for f in smoke.failures), (
        "a payload with no `database_primary` key was read as down. That key is "
        "absent whenever the primary is fine, so this fails every good deploy."
    )


def test_an_unreachable_endpoint_fails(smoke):
    smoke.health_detailed_checks(0, "connection refused", strict_db=False)
    assert any("health/detailed reachable" in f for f in smoke.failures), (
        "a dead /health/detailed produced no failure — an unreachable check "
        "and a passing check must not look the same."
    )


def test_a_non_paper_trading_mode_fails(smoke):
    """TRADING_MODE drifting off paper is the highest-consequence config change."""
    smoke.health_detailed_checks(200, _payload(mode="live"), strict_db=False)
    assert any("trading mode is paper" in f for f in smoke.failures), (
        "the smoke did not flag mode='live'. This is the only automated check "
        "positioned to see it."
    )


def test_main_actually_calls_it(smoke):
    """A tested helper nothing invokes is the same absence in a new shape."""
    import ast
    src = Path(smoke.__file__).read_text()
    called = {
        n.func.id
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "health_detailed_checks" in called, (
        "health_detailed_checks is defined and tested but never called from "
        "main() — the smoke still only sees /health."
    )
    assert "/health/detailed" in src, "the endpoint is no longer requested"


def _outputs(smoke, tmp_path, monkeypatch) -> dict:
    """Run _emit_outputs against a temp GITHUB_OUTPUT and parse it back."""
    f = tmp_path / "gh_output"
    f.write_text("")
    monkeypatch.setenv("GITHUB_OUTPUT", str(f))
    smoke._emit_outputs()
    text = f.read_text()
    out = {}
    if "only_known_degraded=" in text:
        out["only_known_degraded"] = text.split("only_known_degraded=")[1].split("\n")[0]
    if "failed_checks<<SMOKE_EOF" in text:
        out["failed_checks"] = text.split("failed_checks<<SMOKE_EOF\n")[1].split("\nSMOKE_EOF")[0]
    return out


def test_a_lone_db_failure_is_flagged_as_known(smoke, tmp_path, monkeypatch):
    """~10 push-triggered runs/day must not each page an operator-blocked pause."""
    smoke.health_detailed_checks(200, _payload(checks={
        "database": {"ok": True, "fallback": "sqlite"},
        "database_primary": {"ok": False, "error": "unreachable"},
    }), strict_db=True)
    out = _outputs(smoke, tmp_path, monkeypatch)
    assert out["only_known_degraded"] == "1", (
        "a run whose only failure is the paused primary DB did not set "
        "only_known_degraded, so it will page #ci-failures on every push."
    )


def test_any_second_failure_clears_the_flag(smoke, tmp_path, monkeypatch):
    """Suppression must never swallow a real, actionable failure."""
    smoke.health_detailed_checks(200, _payload(checks={
        "database": {"ok": True, "fallback": "sqlite"},
        "database_primary": {"ok": False, "error": "unreachable"},
    }), strict_db=True)
    smoke.check("GET /positions/", False, "HTTP 500")
    out = _outputs(smoke, tmp_path, monkeypatch)
    assert out["only_known_degraded"] == "0", (
        "a 500 on /positions/ was suppressed alongside the known DB failure. "
        "Only a LONE known-degraded failure may skip the page."
    )
    assert "/positions/" in out["failed_checks"], (
        "the real failure is missing from failed_checks, so the page would not "
        "name it even when it does fire."
    )


def test_a_clean_run_is_not_marked_known_degraded(smoke, tmp_path, monkeypatch):
    """`all()` over an empty list is True — the trap this test exists for."""
    smoke.health_detailed_checks(200, _payload(), strict_db=True)
    out = _outputs(smoke, tmp_path, monkeypatch)
    assert out["only_known_degraded"] == "0", (
        "a passing run reported only_known_degraded=1. all([]) is True, so the "
        "flag needs the `bool(failures) and` guard."
    )


def test_outputs_are_written_before_the_nonzero_exit(smoke, tmp_path, monkeypatch):
    """_summary() returns 1 — the outputs must already be on disk by then."""
    smoke.check("GET /trades/", False, "HTTP 500")
    f = tmp_path / "gh_output"
    f.write_text("")
    monkeypatch.setenv("GITHUB_OUTPUT", str(f))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    rc = smoke._summary()
    assert rc == 1, "a failing run no longer exits non-zero"
    assert "failed_checks" in f.read_text(), (
        "_summary returned 1 without writing outputs, so the page step has "
        "nothing to report and falls back to a bare link."
    )


def test_no_github_output_env_does_not_crash(smoke, monkeypatch):
    """Local runs have no GITHUB_OUTPUT."""
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    smoke.check("x", False, "y")
    smoke._emit_outputs()  # must be a no-op, not a crash


def test_the_workflow_consumes_both_outputs(smoke):
    wf = (Path(__file__).resolve().parents[1] / "workflows" / "smoke-test.yml").read_text()
    assert "steps.smoke.outputs.only_known_degraded != '1'" in wf, (
        "the page step no longer gates on only_known_degraded — every push "
        "during the pause re-pages #ci-failures."
    )
    assert "steps.smoke.outputs.failed_checks" in wf, (
        "the page no longer receives failed_checks, so it is back to a fixed "
        "string that names the wrong subsystem."
    )
    # The phrase survives in a comment explaining why it was removed, so match
    # only executable lines — otherwise this test fails on its own rationale.
    live = [ln for ln in wf.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("a deployed endpoint is broken or serving fake data" in ln for ln in live), (
        "the old fixed page text is back. It describes one failure mode out of "
        "nine and misreports the database gate."
    )


def test_the_strictness_flag_is_wired_to_the_environment(smoke):
    """The workflow passes SMOKE_FAIL_ON_DEGRADED_DB only on push."""
    src = Path(smoke.__file__).read_text()
    assert 'os.environ.get("SMOKE_FAIL_ON_DEGRADED_DB") == "1"' in src, (
        "strict_db is no longer read from SMOKE_FAIL_ON_DEGRADED_DB, so the "
        "workflow's push-only strictness has nothing to control."
    )
    wf = (Path(__file__).resolve().parents[1] / "workflows" / "smoke-test.yml").read_text()
    assert "SMOKE_FAIL_ON_DEGRADED_DB" in wf, (
        "smoke-test.yml no longer sets the flag — every run would be non-strict "
        "and the deploy-time gate would silently not exist."
    )
    assert "github.event_name == 'push'" in wf, (
        "the flag is no longer scoped to push events: either the deploy gate is "
        "gone, or the 30-min schedule now pages on a known condition."
    )
