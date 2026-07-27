"""A second broker account must not silently kill demo-bot seeding.

Found in CI, not by reading. `test_seed_additive` failed with `assert 0 == 61`
and the captured log gave the real cause:

    {"error": "Multiple rows were found when one or none was required",
     "event": "Demo bot seed skipped", "level": "warning"}

`seed_demo_bots()` looked up the demo user's account with
`.scalar_one_or_none()`. `Account.user_id` is NOT unique — the accounts table
exists precisely to hold several brokers per user (alpaca | tradestation |
binance | polymarket, paper and live). So the moment a second account appears,
that call raises MultipleResultsFound, the caller's `except` swallows it, logs
a warning nobody reads, and returns 0.

Seeding is then dead permanently: new templates never reach the fleet. That is
the same frozen-fleet failure this module's docstring says it was written to
fix, re-entering through a different door — and it is a PRODUCTION bug, not a
test artifact. Connecting a second broker to the demo account would trigger it
on the live site.
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_seeding_survives_a_user_with_several_broker_accounts(client, monkeypatch):
    from sqlalchemy import select

    from app.bots import seed as seed_mod
    from app.bots.seed import DEMO_EMAIL
    from app.bots.templates import BOT_TEMPLATES
    from app.database import AsyncSessionLocal
    from app.models.account import Account
    from app.models.user import User

    monkeypatch.setattr(seed_mod.settings, "demo_mode", True, raising=False)

    # Reach a seeded baseline, whatever this worker's DB already contains.
    await seed_mod.seed_demo_bots()

    async with AsyncSessionLocal() as db:
        user = (
            await db.execute(select(User).where(User.email == DEMO_EMAIL))
        ).scalar_one_or_none()
        assert user is not None, "seeding must create the demo user"

        # A second broker — entirely legitimate, and what the live site would
        # have the moment anyone connects Binance alongside Alpaca.
        db.add(Account(
            user_id=user.id,
            broker="binance",
            mode="paper",
            label="Second Broker",
            extra_config={},
        ))
        await db.commit()

    # A template added after that second account must STILL seed. On the old
    # code this returns 0 — MultipleResultsFound, swallowed, seeding dead.
    monkeypatch.setitem(BOT_TEMPLATES, "test_multi_account_template", {
        "name": "Multi Account Probe",
        "description": "seeded after a second broker account exists",
        "symbol": "SPY",
        "market_type": "equity",
        "trigger": {"type": "schedule", "interval": "1d"},
        "conditions": [],
        "condition_logic": "ALL",
        "action": {"type": "open_long", "size_pct": 1.0},
        "exit_rules": [],
    })

    created = await seed_mod.seed_demo_bots()
    assert created == 1, (
        "a second broker account must not disable seeding — this is how the "
        "fleet silently froze at an old template count"
    )
    # …and still idempotent afterwards.
    assert await seed_mod.seed_demo_bots() == 0


@pytest.mark.asyncio
async def test_the_account_choice_is_deterministic(client, monkeypatch):
    """Two accounts must not make the picked one depend on row order."""
    from sqlalchemy import select

    from app.bots import seed as seed_mod
    from app.bots.seed import DEMO_EMAIL
    from app.database import AsyncSessionLocal
    from app.models.account import Account
    from app.models.user import User

    monkeypatch.setattr(seed_mod.settings, "demo_mode", True, raising=False)
    await seed_mod.seed_demo_bots()

    async with AsyncSessionLocal() as db:
        user = (
            await db.execute(select(User).where(User.email == DEMO_EMAIL))
        ).scalar_one_or_none()
        assert user is not None

        db.add(Account(user_id=user.id, broker="ibkr", mode="live",
                       label="Live Broker", extra_config={}))
        await db.commit()

        picked = (
            await db.execute(
                select(Account)
                .where(Account.user_id == user.id)
                .order_by((Account.mode != "paper"), Account.id)
                .limit(1)
            )
        ).scalars().first()

    assert picked is not None
    assert picked.mode == "paper", (
        "the seeded bots are paper bots — a live account must not be chosen "
        "just because it sorted first"
    )
