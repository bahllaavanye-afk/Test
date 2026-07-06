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


def test_ml_generation_biases_toward_winners():
    from app.bots.factory import generate_variants_ml

    # iron_condor 16Δ 7DTE dominated; call spreads lost. The next generation
    # must lean toward condor/7DTE neighborhoods it hasn't tried yet.
    results = [
        {"name": "[gen] iron condor 16Δ 7DTE tp50", "total_pnl": 500.0, "trades": 10},
        {"name": "[gen] call credit spread 30Δ 30DTE tp25", "total_pnl": -300.0, "trades": 8},
    ]
    v = generate_variants_ml(results, max_variants=8)
    assert len(v) == 8
    assert "gen_iron_condor_16d_7dte_tp50" not in v          # never re-emit tried combos
    condorish = sum(1 for t in v.values() if "iron condor" in t["name"] or "7DTE" in t["name"])
    ccs = sum(1 for t in v.values() if "call credit" in t["name"])
    assert condorish > ccs                                    # exploit tilts to winners


def test_ml_generation_cold_start_falls_back_to_grid():
    from app.bots.factory import generate_variants, generate_variants_ml

    assert generate_variants_ml([]).keys() == generate_variants(generation=0).keys()
