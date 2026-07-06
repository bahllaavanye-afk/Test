"""Bot variant factory — the systematic search Options Alpha can't do."""
from app.bots.factory import MAX_VARIANTS, generate_variants
from app.schemas.bot import BotCreate


def test_variants_validate_and_are_bounded():
    v = generate_variants()
    assert 0 < len(v) <= MAX_VARIANTS
    for tid, t in v.items():
        BotCreate(**t)  # schema-valid → instantiable by the lifecycle manager


def test_generations_are_deterministic_and_disjoint():
    g0a, g0b, g1 = generate_variants(generation=0), generate_variants(generation=0), generate_variants(generation=1)
    assert g0a.keys() == g0b.keys()            # deterministic
    assert not (g0a.keys() & g1.keys())        # successive generations explore fresh grid


def test_research_variants_are_small_and_guarded():
    for t in generate_variants().values():
        assert t["action"]["size_pct"] == 1.0          # research budget, not a firehose
        assert {"type": "no_position"} in t["conditions"]
        kinds = {r["type"] for r in t["exit_rules"]}
        sells = any(l["side"] == "sell" for l in t["action"]["legs"])
        assert not sells or "stop_loss" in kinds       # premium sellers always have stops


def test_zero_dte_variants_never_hold_overnight():
    for t in generate_variants(generation=0).values():
        if any(l["dte"] == 0 for l in t["action"]["legs"]):
            assert any(r["type"] == "time_exit" for r in t["exit_rules"])
