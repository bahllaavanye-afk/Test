"""Seed the demo user, paper Alpaca account, and a few default strategies.

Called from the GitHub Actions e2e-demo workflow before Playwright runs.
Uses the SQLAlchemy ORM directly so we don't need the API to be up yet.
"""

import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from app.models.user import User
from app.models.account import Account
from app.models.strategy import Strategy
from app.api.v1.auth import hash_password
from app.utils.security import encrypt_secret

DEMO_EMAIL = os.environ.get("DEMO_EMAIL", "demo@quantedge.local")
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "demo-pass-1234")
ALPACA_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")
DB_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./dev.db")

# Strategy metadata: (name, market_type, strategy_type, risk_bucket, symbols)
DEFAULT_STRATEGIES = [
    ("momentum",       "equity",  "manual",      "directional", ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]),
    ("mean_reversion", "equity",  "manual",      "directional", ["SPY", "AAPL", "MSFT"]),
    ("pairs_trading",  "equity",  "manual",      "arbitrage",   ["GLD", "SLV"]),
    ("kalman_pairs",   "equity",  "manual",      "arbitrage",   ["GLD", "SLV"]),
    ("vrp_systematic", "equity",  "manual",      "arbitrage",   ["SPY", "VIX"]),
]


async def main():
    print(f"Seeding {DEMO_EMAIL} in {DB_URL.split('@')[-1]}")
    engine = create_async_engine(DB_URL, echo=False)
    SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with SessionLocal() as db:
        # 1. User
        existing = (await db.execute(select(User).where(User.email == DEMO_EMAIL))).scalar_one_or_none()
        if existing:
            user_id = existing.id
            print(f"  user already exists: {user_id}")
        else:
            user = User(
                id=str(uuid.uuid4()),
                email=DEMO_EMAIL,
                hashed_password=hash_password(DEMO_PASSWORD),
                is_active=True,
            )
            db.add(user)
            await db.commit()
            user_id = user.id
            print(f"  + created user {user_id}")

        # 2. Paper Alpaca account
        existing_acct = (
            await db.execute(select(Account).where(Account.user_id == user_id, Account.broker == "alpaca"))
        ).scalar_one_or_none()
        if existing_acct:
            print(f"  account already exists: {existing_acct.id}")
        elif ALPACA_KEY and ALPACA_SECRET:
            acct = Account(
                id=str(uuid.uuid4()),
                user_id=user_id,
                broker="alpaca",
                mode="paper",
                encrypted_key=encrypt_secret(ALPACA_KEY),
                encrypted_secret=encrypt_secret(ALPACA_SECRET),
                is_active=True,
            )
            db.add(acct)
            await db.commit()
            print(f"  + created alpaca paper account {acct.id}")
        else:
            print("  no ALPACA_API_KEY/SECRET — skipping account")

        # 3. Enable a few default strategies (no-op if already enabled)
        try:
            for name, market_type, strategy_type, risk_bucket, symbols in DEFAULT_STRATEGIES:
                existing_strat = (
                    await db.execute(select(Strategy).where(Strategy.name == name))
                ).scalar_one_or_none()
                if not existing_strat:
                    strat = Strategy(
                        id=str(uuid.uuid4()),
                        name=name,
                        market_type=market_type,
                        strategy_type=strategy_type,
                        risk_bucket=risk_bucket,
                        symbols=symbols,
                        is_enabled=True,
                        params={},
                    )
                    db.add(strat)
            await db.commit()
            print(f"  + enabled {len(DEFAULT_STRATEGIES)} default strategies")
        except Exception as e:
            print(f"  strategy seed skipped: {e}")

    await engine.dispose()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
