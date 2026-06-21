"""Seed the demo user with one enabled bot per template so every desk shows live
bots on a fresh deploy (the fix for "all desks empty"). Idempotent and DEMO_MODE-gated
— safe to run on every boot; it no-ops once the demo user already has bots.

Run standalone:  python -m app.bots.seed
"""
from __future__ import annotations

import asyncio
import secrets as _secrets
import uuid

from sqlalchemy import func, select

from app.bots.templates import BOT_TEMPLATES
from app.config import settings
from app.database import AsyncSessionLocal
from app.models.account import Account
from app.models.bot import Bot
from app.models.user import User
from app.utils.logging import logger
from app.utils.security import hash_password

DEMO_EMAIL = "demo@quantedge.app"


async def seed_demo_bots() -> int:
    """Create the demo user + paper account + a bot per template if none exist.
    Returns the number of bots created (0 when disabled or already seeded)."""
    if not settings.demo_mode:
        return 0
    try:
        async with AsyncSessionLocal() as db:
            user = (
                await db.execute(select(User).where(User.email == DEMO_EMAIL))
            ).scalar_one_or_none()
            if user is None:
                user = User(
                    id=str(uuid.uuid4()),
                    email=DEMO_EMAIL,
                    hashed_password=hash_password(_secrets.token_urlsafe(32)),
                    is_active=True,
                    is_superuser=False,
                )
                db.add(user)
                await db.commit()
                await db.refresh(user)

            existing = (
                await db.execute(
                    select(func.count()).select_from(Bot).where(Bot.user_id == user.id)
                )
            ).scalar() or 0
            if existing:
                return 0  # already seeded — idempotent

            account = (
                await db.execute(select(Account).where(Account.user_id == user.id))
            ).scalar_one_or_none()
            if account is None:
                account = Account(
                    user_id=user.id,
                    broker="alpaca",
                    mode="paper",
                    label="Paper Account",
                    extra_config={"equity": 100_000.0, "cash": 100_000.0},
                )
                db.add(account)
                await db.commit()
                await db.refresh(account)

            created = 0
            for template_id, t in BOT_TEMPLATES.items():
                db.add(
                    Bot(
                        id=str(uuid.uuid4()),
                        user_id=user.id,
                        account_id=str(account.id),
                        name=t["name"],
                        description=t.get("description", ""),
                        symbol=t["symbol"],
                        market_type=t.get("market_type", "equity"),
                        trigger=t["trigger"],
                        conditions=t.get("conditions", []),
                        condition_logic=t.get("condition_logic", "ALL"),
                        action=t["action"],
                        exit_rules=t.get("exit_rules", []),
                        is_enabled=True,
                        template_id=template_id,
                    )
                )
                created += 1
            await db.commit()
            logger.info("Seeded demo bots", count=created)
            return created
    except Exception as e:  # never let seeding break boot
        logger.warning("Demo bot seed skipped", error=str(e))
        return 0


if __name__ == "__main__":
    print("seeded bots:", asyncio.run(seed_demo_bots()))
