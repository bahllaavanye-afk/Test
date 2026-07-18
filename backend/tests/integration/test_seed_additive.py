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
    # fresh test DB → everything seeds; re-run → nothing new
    assert n_first == len(BOT_TEMPLATES)
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
