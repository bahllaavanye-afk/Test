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
            # ── Equity desk: directional ──────────────────────────────────────
            {'name': 'momentum', 'market_type': 'equity', 'strategy_type': 'manual', 'risk_bucket': 'directional', 'symbols': ['SPY', 'QQQ'], 'tick_interval_seconds': 3600},
            {'name': 'mean_reversion', 'market_type': 'equity', 'strategy_type': 'manual', 'risk_bucket': 'directional', 'symbols': ['AAPL', 'MSFT'], 'tick_interval_seconds': 3600},
            {'name': 'rsi_macd', 'market_type': 'equity', 'strategy_type': 'manual', 'risk_bucket': 'directional', 'symbols': ['TSLA', 'NVDA'], 'tick_interval_seconds': 1800},
            {'name': 'breakout', 'market_type': 'equity', 'strategy_type': 'manual', 'risk_bucket': 'directional', 'symbols': ['AMD', 'AMZN'], 'tick_interval_seconds': 1800},
            {'name': 'supertrend', 'market_type': 'equity', 'strategy_type': 'manual', 'risk_bucket': 'directional', 'symbols': ['GOOGL', 'META'], 'tick_interval_seconds': 1800},
            {'name': 'low_volatility', 'market_type': 'equity', 'strategy_type': 'manual', 'risk_bucket': 'directional', 'symbols': ['JNJ', 'PG'], 'tick_interval_seconds': 3600},
            {'name': 'sector_rotation', 'market_type': 'equity', 'strategy_type': 'manual', 'risk_bucket': 'directional', 'symbols': ['XLK', 'XLY'], 'tick_interval_seconds': 3600},
            {'name': 'yield_curve_momentum', 'market_type': 'equity', 'strategy_type': 'manual', 'risk_bucket': 'directional', 'symbols': ['TLT', 'IEF'], 'tick_interval_seconds': 3600},
            # ── Equity desk: arbitrage ─────────────────────────────────────────
            {'name': 'pairs_trading', 'market_type': 'equity', 'strategy_type': 'manual', 'risk_bucket': 'arbitrage', 'symbols': ['KO', 'PEP'], 'tick_interval_seconds': 3600},
            {'name': 'pca_stat_arb', 'market_type': 'equity', 'strategy_type': 'manual', 'risk_bucket': 'arbitrage', 'symbols': ['XLF', 'XLE'], 'tick_interval_seconds': 3600},
            {'name': 'skew_arb', 'market_type': 'equity', 'strategy_type': 'manual', 'risk_bucket': 'arbitrage', 'symbols': ['SPY'], 'tick_interval_seconds': 3600},
            # ── Crypto desk: arbitrage ─────────────────────────────────────────
            {'name': 'triangular_arb', 'market_type': 'crypto', 'strategy_type': 'manual', 'risk_bucket': 'arbitrage', 'symbols': ['BTC/USDT'], 'tick_interval_seconds': 10},
            {'name': 'funding_rate_arb', 'market_type': 'crypto', 'strategy_type': 'manual', 'risk_bucket': 'arbitrage', 'symbols': ['BTC/USDT', 'ETH/USDT'], 'tick_interval_seconds': 60},
            {'name': 'btc_eth_stat_arb', 'market_type': 'crypto', 'strategy_type': 'manual', 'risk_bucket': 'arbitrage', 'symbols': ['BTC/USDT', 'ETH/USDT'], 'tick_interval_seconds': 60},
            {'name': 'dex_cex_arb', 'market_type': 'crypto', 'strategy_type': 'manual', 'risk_bucket': 'arbitrage', 'symbols': ['BTC/USDT'], 'tick_interval_seconds': 60},
            # ── Crypto desk: directional ───────────────────────────────────────
            {'name': 'crypto_adaptive_trend', 'market_type': 'crypto', 'strategy_type': 'manual', 'risk_bucket': 'directional', 'symbols': ['ETH/USDT', 'SOL/USDT'], 'tick_interval_seconds': 300},
        ]
        for s in strategies:
            db.add(Strategy(id=str(uuid.uuid4()), is_enabled=True, confidence_threshold=0.6, params={}, **s))

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
