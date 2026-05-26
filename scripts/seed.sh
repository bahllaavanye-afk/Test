#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -c "
import asyncio
from backend.app.database import AsyncSessionLocal
from backend.app.models.strategy import Strategy
from backend.app.models.risk import RiskRule
import uuid

async def seed():
    async with AsyncSessionLocal() as db:
        strategies = [
            {'name': 'momentum', 'market_type': 'equity', 'strategy_type': 'manual', 'risk_bucket': 'directional', 'symbols': ['SPY', 'QQQ'], 'tick_interval_seconds': 3600},
            {'name': 'mean_reversion', 'market_type': 'equity', 'strategy_type': 'manual', 'risk_bucket': 'directional', 'symbols': ['AAPL', 'MSFT'], 'tick_interval_seconds': 3600},
            {'name': 'triangular_arb', 'market_type': 'crypto', 'strategy_type': 'manual', 'risk_bucket': 'arbitrage', 'symbols': ['BTC/USDT'], 'tick_interval_seconds': 10},
        ]
        for s in strategies:
            db.add(Strategy(id=str(uuid.uuid4()), is_active=False, confidence_threshold=0.6, params={}, **s))

        rules = [
            {'rule_type': 'max_drawdown', 'threshold': 0.10, 'action': 'halt_all'},
            {'rule_type': 'arb_drawdown', 'threshold': 0.05, 'action': 'halt_bucket'},
            {'rule_type': 'max_position', 'threshold': 0.05, 'action': 'alert'},
        ]
        for r in rules:
            db.add(RiskRule(id=str(uuid.uuid4()), is_active=True, **r))
        await db.commit()
    print('Seeded successfully.')

asyncio.run(seed())
"
