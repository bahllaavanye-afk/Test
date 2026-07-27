"""Additive bot seeding: new templates appear on existing installs.

The old seeder bailed if the demo user had ANY bots, freezing the live fleet
at the first-boot template count (29 bots while the repo grew to 57 templates
— every OA flagship invisible on the site). Now seeding is per-template:
missing ones are created, existing bots are never touched.
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_seed_is_additive_and_idempotent(client, monkeypatch):
    from app.bots import seed as seed_mod
    from app.bots.templates import BOT_TEMPLATES

    monkeypatch.setattr(seed_mod.settings, "demo_mode", True, raising=False)

    n_first = await seed_mod.seed_demo_bots()
    # Either this test seeded the templates (fresh worker DB) or another file
    # sharing this xdist worker already did — under `--dist loadfile` which
    # files share a worker is a scheduler detail, so requiring the full count
    # here made the test depend on running FIRST. It went red as `assert 0 ==
    # 61` when an unrelated speed-up reshuffled the packing.
    #
    # Deliberately no extra DB writes to make this order-independent: this file
    # shares its database with every other file on the worker, so a "fix" that
    # seeds or inserts to normalise state breaks *those* tests instead. (Tried
    # that first — it turned 1 failure into 21.)
    assert n_first in (0, len(BOT_TEMPLATES)), (
        f"seeding returned {n_first}: neither a full seed nor already-seeded"
    )
    assert await seed_mod.seed_demo_bots() == 0

    # a template added AFTER first boot must seed on the next boot
    fake_key = "test_late_template"
    monkeypatch.setitem(BOT_TEMPLATES, fake_key, {
        "name": "Late Arrival",
        "description": "added after initial seed",
        "symbol": "SPY",
        "market_type": "equity",
        "trigger": {"type": "schedule", "interval": "1d"},
        "conditions": [],
        "condition_logic": "ALL",
        "action": {"type": "open_long", "size_pct": 1.0},
        "exit_rules": [],
    })
    assert await seed_mod.seed_demo_bots() == 1
    assert await seed_mod.seed_demo_bots() == 0
