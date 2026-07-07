"""The 11 Options Alpha clone bots (user's OA account, 2026-07-06 screenshot).

Every template must validate against BotCreate (so the UI, seeding, and the
lifecycle manager can instantiate them), run at 1-minute cadence with the
$2,500 (2.5%) allocation, and carry real exit rules — no fire-and-forget bots.
"""
from app.bots.templates import BOT_TEMPLATES
from app.schemas.bot import BotCreate

OA_IDS = [
    "oa_friday_14dte_bwb", "oa_ib_0945_10d", "oa_ncr_0dte_spx_a", "oa_vix_range",
    "oa_ncr_spx_0dte_b", "oa_iron_butter_clone", "oa_short_put_spread",
    "oa_ccs_friday_15d", "oa_long_calls_confirm", "oa_iron_boi",
    "oa_delta_adjusting_strangle", "oa_steamrolled_spy", "oa_steamrolled_qqq",
    "oa_td_npdt_s1_trend", "oa_trendy_short_put",
    "hybrid_regime_gamma", "hybrid_regime_theta", "hybrid_put_ladder_trend",
    "hybrid_wide_0dte_tp25", "hybrid_morning_momentum",
]


def test_all_11_oa_clones_exist():
    assert [i for i in OA_IDS if i in BOT_TEMPLATES] == OA_IDS


def test_oa_clones_validate_against_bot_schema():
    for tid in OA_IDS:
        BotCreate(**BOT_TEMPLATES[tid])  # raises on any schema violation


def test_oa_clones_one_minute_and_2500_allocation():
    for tid in OA_IDS:
        t = BOT_TEMPLATES[tid]
        assert t["trigger"]["interval"] == "1m", tid
        assert t["action"]["size_pct"] == 2.5, tid          # $2,500 of $100k paper
        assert t["market_type"] == "options", tid


def test_oa_clones_always_have_exits_and_entry_guard():
    for tid in OA_IDS:
        t = BOT_TEMPLATES[tid]
        assert t["exit_rules"], f"{tid} has no exit rules"
        # 1-minute cadence without no_position would stack positions every minute
        cond_types = {c["type"] for c in t["conditions"]}
        assert "no_position" in cond_types, f"{tid} missing no_position guard"


def test_short_premium_clones_have_stops():
    for tid in OA_IDS:
        t = BOT_TEMPLATES[tid]
        sells = any(l["side"] == "sell" for l in t["action"]["legs"])
        if sells:
            kinds = {r["type"] for r in t["exit_rules"]}
            assert "stop_loss" in kinds, f"{tid} sells premium without a stop"
