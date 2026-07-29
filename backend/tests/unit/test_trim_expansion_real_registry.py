"""The trim-key expansion was only ever tested against an 8-name fake registry.

`_trimmed_strategies()` expands a legacy truncated attribution key
(`avellaneda`) to its full registry name (`avellaneda_stoikov_mm`) so a trim
written from pre-fix fill data actually matches what the desk checks. The rule
is "expand only when EXACTLY ONE registry entry shares the prefix".

Whether a prefix is unambiguous is a property of the REAL registry — 116
strategies — not of the eight names in
`.github/scripts/test_trim_key_expansion.py`. A prefix that resolves uniquely
in a fake set can easily be ambiguous in the real one, in which case the
expansion silently stops happening and the trim goes back to being a phantom.
Those unit tests would stay green throughout.

This is the same failure shape as the guards earlier in this work that passed
against the code they were meant to catch: verify against the real substrate,
not a convenient stand-in.

Lives under backend/tests because that is where STRATEGY_REGISTRY is
guaranteed importable — a `skip` when the registry cannot be loaded would make
this vacuous, which is exactly what it exists to prevent.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_DESK = Path(__file__).resolve().parents[3] / ".github" / "scripts" / "desk_order_placer.py"


@pytest.fixture(scope="module")
def dop():
    assert _DESK.is_file(), f"missing {_DESK}"
    spec = importlib.util.spec_from_file_location("dop_real_registry", _DESK)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dop_real_registry"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def registry() -> set[str]:
    from app.strategies import STRATEGY_REGISTRY
    names = set(STRATEGY_REGISTRY)
    assert len(names) > 50, f"registry looks wrong ({len(names)} names)"
    return names


# ── the live case ────────────────────────────────────────────────────────────

def test_the_key_the_artifact_actually_landed_with_expands(dop, registry):
    """`avellaneda` is what strategy_performance.json contained on 2026-07-29.

    If this ever stops resolving, the trim written from that data retires
    nothing and the strategy keeps trading while looking retired.
    """
    assert dop._expand_truncated("avellaneda", registry) == "avellaneda_stoikov_mm"


# ── ambiguity in the REAL registry ───────────────────────────────────────────

@pytest.mark.parametrize("prefix", ["supertrend", "commodity_"])
def test_known_ambiguous_prefixes_are_not_expanded(dop, registry, prefix):
    """`supertrend` is a real strategy AND a prefix; `commodity_` matches three.

    Expanding either would retire a strategy that was never judged.
    """
    assert dop._expand_truncated(prefix, registry) == prefix


def test_every_desk_strategy_round_trips_through_truncation(dop, registry):
    """The general property, across all 100+ desk strategies.

    For each one, truncate to the legacy 10 chars and expand back. It must
    return either the original name or — where the prefix is genuinely
    ambiguous — the untouched prefix. It must NEVER return a DIFFERENT
    strategy, which is the failure that would retire the wrong thing.
    """
    wrong = []
    for name in sorted(registry):
        got = dop._expand_truncated(name[:10], registry)
        if got != name and got != name[:10]:
            wrong.append(f"{name!r} truncates to {name[:10]!r} which expands to {got!r}")
    assert not wrong, "truncation resolves to the WRONG strategy:\n  " + "\n  ".join(wrong)


def test_the_ambiguous_set_is_small_and_known(dop, registry):
    """If this grows, more trims silently become phantoms — worth noticing."""
    unresolvable = sorted(
        name for name in registry
        if dop._expand_truncated(name[:10], registry) != name
    )
    assert unresolvable == ["commodity_momentum", "commodity_reversion",
                            "commodity_trend", "supertrend_rsi_tv"], unresolvable


# ── safety ───────────────────────────────────────────────────────────────────

def test_a_full_registry_name_is_never_altered(dop, registry):
    for name in sorted(registry):
        assert dop._expand_truncated(name, registry) == name


def test_an_unknown_key_is_returned_unchanged(dop, registry):
    assert dop._expand_truncated("definitely_not_a_strategy", registry) == \
        "definitely_not_a_strategy"
