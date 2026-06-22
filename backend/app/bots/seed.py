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
from app.models.risk import RiskRule
from app.models.strategy import Strategy
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


_STRATEGIES = [
    {"name": "momentum", "market_type": "equity", "strategy_type": "manual", "risk_bucket": "directional", "symbols": ["SPY", "QQQ"], "tick_interval_seconds": 3600},
    {"name": "mean_reversion", "market_type": "equity", "strategy_type": "manual", "risk_bucket": "directional", "symbols": ["AAPL", "MSFT"], "tick_interval_seconds": 3600},
    {"name": "rsi_macd", "market_type": "equity", "strategy_type": "manual", "risk_bucket": "directional", "symbols": ["TSLA", "NVDA"], "tick_interval_seconds": 1800},
    {"name": "breakout", "market_type": "equity", "strategy_type": "manual", "risk_bucket": "directional", "symbols": ["AMD", "AMZN"], "tick_interval_seconds": 1800},
    {"name": "supertrend", "market_type": "equity", "strategy_type": "manual", "risk_bucket": "directional", "symbols": ["GOOGL", "META"], "tick_interval_seconds": 1800},
    {"name": "low_volatility", "market_type": "equity", "strategy_type": "manual", "risk_bucket": "directional", "symbols": ["JNJ", "PG"], "tick_interval_seconds": 3600},
    {"name": "sector_rotation", "market_type": "equity", "strategy_type": "manual", "risk_bucket": "directional", "symbols": ["XLK", "XLY"], "tick_interval_seconds": 3600},
    {"name": "pairs_trading", "market_type": "equity", "strategy_type": "manual", "risk_bucket": "arbitrage", "symbols": ["KO", "PEP"], "tick_interval_seconds": 3600},
    {"name": "pca_stat_arb", "market_type": "equity", "strategy_type": "manual", "risk_bucket": "arbitrage", "symbols": ["XLF", "XLE"], "tick_interval_seconds": 3600},
    {"name": "triangular_arb", "market_type": "crypto", "strategy_type": "manual", "risk_bucket": "arbitrage", "symbols": ["BTC/USDT"], "tick_interval_seconds": 10},
    {"name": "funding_rate_arb", "market_type": "crypto", "strategy_type": "manual", "risk_bucket": "arbitrage", "symbols": ["BTC/USDT", "ETH/USDT"], "tick_interval_seconds": 60},
    {"name": "btc_eth_stat_arb", "market_type": "crypto", "strategy_type": "manual", "risk_bucket": "arbitrage", "symbols": ["BTC/USDT", "ETH/USDT"], "tick_interval_seconds": 60},
    {"name": "crypto_adaptive_trend", "market_type": "crypto", "strategy_type": "manual", "risk_bucket": "directional", "symbols": ["ETH/USDT", "SOL/USDT"], "tick_interval_seconds": 300},
]
_RISK_RULES = [
    {"rule_type": "max_drawdown", "threshold": 0.10, "action": "halt_all"},
    {"rule_type": "arb_drawdown", "threshold": 0.05, "action": "halt_bucket"},
    {"rule_type": "max_position", "threshold": 0.05, "action": "alert"},
]


async def seed_demo_strategies() -> int:
    """Seed default strategies if the table is empty (so the Strategies desk isn't blank)."""
    if not settings.demo_mode:
        return 0
    try:
        async with AsyncSessionLocal() as db:
            existing = (await db.execute(select(func.count()).select_from(Strategy))).scalar() or 0
            if existing:
                return 0
            for s in _STRATEGIES:
                db.add(Strategy(id=str(uuid.uuid4()), is_enabled=True, confidence_threshold=0.6, params={}, **s))
            await db.commit()
            logger.info("Seeded demo strategies", count=len(_STRATEGIES))
            return len(_STRATEGIES)
    except Exception as e:
        logger.warning("Demo strategy seed skipped", error=str(e))
        return 0


async def seed_demo_risk_rules() -> int:
    """Seed default risk rules if none exist (so the Risk page isn't blank)."""
    if not settings.demo_mode:
        return 0
    try:
        async with AsyncSessionLocal() as db:
            existing = (await db.execute(select(func.count()).select_from(RiskRule))).scalar() or 0
            if existing:
                return 0
            for r in _RISK_RULES:
                db.add(RiskRule(id=str(uuid.uuid4()), is_active=True, **r))
            await db.commit()
            logger.info("Seeded demo risk rules", count=len(_RISK_RULES))
            return len(_RISK_RULES)
    except Exception as e:
        logger.warning("Demo risk-rule seed skipped", error=str(e))
        return 0


async def seed_all() -> dict:
    """Seed bots + strategies + risk rules so every desk/page is populated on boot."""
    return {
        "bots": await seed_demo_bots(),
        "strategies": await seed_demo_strategies(),
        "risk_rules": await seed_demo_risk_rules(),
    }


if __name__ == "__main__":
    print("seeded:", asyncio.run(seed_all()))
