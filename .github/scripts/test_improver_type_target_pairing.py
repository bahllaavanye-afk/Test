"""The improvement TYPE and the target FILE were chosen from two parallel lists.

The type came from `hour % 12` (fixed for a run); the pattern from
`((hour + attempts) % 24) % 12` (rotating per attempt). Attempt 0 therefore
paired type i with pattern i permanently, and every retry drifted the pattern
while the prompt stayed put. Measured 2026-08-06, attempt 0 gave:

    test_cases     -> backend/app/ml/models/*.py   tests written INTO source
    strategy_logic -> backend/app/api/v1/*.py      route handlers have no
                                                   entry/exit logic at all
    monitoring     -> backend/tests/unit/*.py      P&L logging into unit tests

And **6 of the 12 patterns yielded zero usable files** once PROTECTED_PREFIXES
and the 8000-char guard applied, so half of all runs fell through to a glob over
the whole backend — which is how a `strategy_logic` prompt reached
`models/account.py` (#1510) and produced two unreferenced methods.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("SECRET_KEY", "x" * 64)
os.environ.setdefault("DATABASE_URL", "")
import continuous_improver as ci  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _usable(pattern: str) -> list[str]:
    return [f for f in glob.glob(pattern)
            if not f.endswith("__init__.py")
            and not ci._is_protected(f) and not ci._too_large(f)]


def test_every_improvement_type_has_a_live_target():
    """The guard that makes the whole fix hold. A type whose patterns yield
    nothing degrades to the whole-backend glob, which is the silent failure
    this replaces — so that must fail CI, not pass quietly."""
    os.chdir(REPO)
    dead = {}
    for name, _desc in ci.IMPROVEMENT_TYPES:
        patterns = ci.TYPE_TARGETS.get(name)
        assert patterns, f"{name} has no TYPE_TARGETS entry"
        if not any(_usable(p) for p in patterns):
            dead[name] = patterns
    assert not dead, (
        f"improvement type(s) with no usable target file: {dead}. Each would fall "
        f"back to a glob over the entire backend, which is exactly the silent "
        f"mis-targeting this mapping exists to stop.")


def test_every_type_in_the_rotation_is_mapped_and_vice_versa():
    rotation = {n for n, _ in ci.IMPROVEMENT_TYPES}
    assert rotation == set(ci.TYPE_TARGETS), (
        "IMPROVEMENT_TYPES and TYPE_TARGETS have drifted apart; an unmapped type "
        "silently falls back to the whole backend")


def test_test_cases_writes_into_the_test_tree():
    """The most obviously wrong pairing: the prompt asks for unit test cases and
    used to point at `backend/app/ml/models/*.py`."""
    for pattern in ci.TYPE_TARGETS["test_cases"]:
        assert "/tests/" in pattern, f"test_cases targets non-test path {pattern}"


def test_monitoring_does_not_add_pnl_logging_to_unit_tests():
    for pattern in ci.TYPE_TARGETS["monitoring"]:
        assert "/tests/" not in pattern, f"monitoring targets the test tree: {pattern}"


def test_strategy_logic_is_gone_because_its_only_target_is_protected():
    """Its prompt only means something inside `backend/app/strategies/`, and that
    prefix is protected on purpose. A type whose only valid target is off-limits
    cannot do its job and spends one run in twelve doing something else."""
    assert "strategy_logic" not in {n for n, _ in ci.IMPROVEMENT_TYPES}
    assert "strategy_logic" not in ci.TYPE_TARGETS
    for name, patterns in ci.TYPE_TARGETS.items():
        for p in patterns:
            assert not ci._is_protected(p.split("*")[0]), (
                f"{name} targets protected prefix {p}")


def test_no_type_can_reach_a_protected_file():
    os.chdir(REPO)
    for name, patterns in ci.TYPE_TARGETS.items():
        for p in patterns:
            for f in glob.glob(p):
                assert not ci._is_protected(f), f"{name} can reach protected {f}"


def test_the_picker_stays_inside_the_type_s_locations(monkeypatch, tmp_path):
    """Given a type, every retry must stay in that type's directories rather
    than rotating into a neighbouring pattern as the old index did."""
    os.chdir(REPO)
    for hour in range(24):
        picked = ci.pick_target_file(hour, set(), improvement_type="test_cases")
        assert picked is not None
        assert "/tests/" in picked.replace("\\", "/"), (
            f"hour {hour}: test_cases picked {picked}, outside the test tree")


def test_the_fallback_announces_itself(capsys):
    """A silent fallback is what made the targeting look like it worked."""
    os.chdir(REPO)
    ci.pick_target_file(0, set(), improvement_type="nonexistent_type")
    assert "falling back to the whole backend" in capsys.readouterr().out


def test_main_passes_the_improvement_type_to_the_picker():
    """Without this the mapping exists but is never consulted."""
    src = (Path(__file__).resolve().parent / "continuous_improver.py").read_text()
    call = src.split("target = pick_target_file(", 1)[1][:160]
    assert "improvement_type=improvement_type" in call, (
        "main() calls pick_target_file without the type; the pairing is inert")


# ── The broker gap, found 2026-08-06 ─────────────────────────────────────────

def test_broker_files_are_protected():
    """#1547 rewrote `brokers/alpaca_headers.py` — the Alpaca CREDENTIAL path —
    in an autonomous "optimization" run.

    The change itself was benign and its claim was even true (measured 108ns vs
    128ns per call). That is not the point: `brokers/*.py` is declared
    Do-Not-Modify in THREE CLAUDE.md files (tasks/, risk/, and
    strategies/options/ for alpaca.py), and `grep -rl alpaca_headers
    backend/tests/` returns NOTHING — so a green CI said nothing whatever about
    a change to where the API keys are assembled. The money-path argument that
    already protects execution/ and risk/ applies here with more force.
    """
    assert "backend/app/brokers/" in ci.PROTECTED_PREFIXES
    assert ci._is_protected("backend/app/brokers/alpaca_headers.py")
    assert ci._is_protected("backend/app/brokers/alpaca.py")


def test_protection_covers_every_money_path_directory():
    """The four money-path directories, named so a future edit that drops one
    fails here rather than in production."""
    for prefix in ("backend/app/brokers/", "backend/app/execution/",
                   "backend/app/risk/", "backend/app/strategies/"):
        assert ci._is_protected(prefix + "anything.py"), f"{prefix} is reachable"


def test_no_type_targets_a_protected_directory_after_the_broker_change():
    """Adding a prefix to PROTECTED_PREFIXES without pruning TYPE_TARGETS would
    leave types pointing at patterns that can now only yield nothing — which is
    precisely the silent-fallback failure this module exists to prevent."""
    os.chdir(REPO)
    for name, patterns in ci.TYPE_TARGETS.items():
        for pat in patterns:
            assert not ci._is_protected(pat.split("*")[0]), (
                f"{name} still targets protected {pat}")
