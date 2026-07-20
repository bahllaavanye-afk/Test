"""Env-keyed desk → Trades attribution — the fix for "the site shows no trades".

The desks place ~59 strategies' paper orders on the *env* ALPACA key (a GitHub
Actions secret relayed to Render), which normally has NO corresponding DB
``Account``. The old ``sync_desk_trades`` only fetched fills for accounts with a
stored ``encrypted_key``, so those ``qe-``-tagged desk fills never became
``Trade`` rows — the leaderboard and every per-user trade view were empty.

These tests pin the new env fallback: when the env credentials are present,
``sync_desk_trades`` fetches the desk account's fills directly and attributes the
reconstructed round trips to the system/demo (keyless) paper account, idempotently.
"""
from __future__ import annotations

import uuid


async def _make_keyless_demo_account() -> tuple[str, str]:
    """Create a demo user + a keyless Alpaca paper account (as seed.py does)."""
    from app.database import AsyncSessionLocal
    from app.models.account import Account
    from app.models.user import User

    uid = str(uuid.uuid4())
    async with AsyncSessionLocal() as s:
        s.add(User(id=uid, email=f"demo-{uid[:8]}@quantedge.app", hashed_password="x"))
        await s.commit()
        acct = Account(
            user_id=uid, broker="alpaca", mode="paper",
            label="Paper Account", encrypted_key=None,
        )
        s.add(acct)
        await s.commit()
        await s.refresh(acct)
    return uid, acct.id


def _order(coid, side, qty, price, ts, oid, symbol="SPY"):
    return {
        "id": oid, "client_order_id": coid, "symbol": symbol, "side": side,
        "filled_qty": str(qty), "filled_avg_price": str(price),
        "filled_at": ts, "status": "filled",
    }


async def test_env_desk_fills_attributed_to_keyless_account(_create_tables, monkeypatch):
    from sqlalchemy import select

    from app.database import AsyncSessionLocal
    from app.models.account import Account
    from app.models.trade import Trade
    from app.tasks import desk_trade_sync as dts

    await _make_keyless_demo_account()

    tok = uuid.uuid4().hex[:8]
    strat = f"deskx{tok}"  # unique per run so dedup/assertions don't collide across tests
    orders = [
        _order(f"qe-{strat}-SPY-1", "buy", 10, 100.0, "2026-07-06T14:00:00Z", f"o1-{tok}"),
        _order(f"qe-{strat}-SPY-2", "sell", 10, 110.0, "2026-07-06T15:00:00Z", f"o2-{tok}"),
    ]

    async def fake_env_fetch(creds, lookback_days=30):
        return orders

    async def no_keyed_orders(acct, lookback_days=30):  # hermetic: keyed path never hits network
        return []

    monkeypatch.setattr(dts, "_env_alpaca_creds", lambda: ("KID", "SEC"))
    monkeypatch.setattr(dts, "_fetch_closed_orders_env", fake_env_fetch)
    monkeypatch.setattr(dts, "_fetch_closed_orders", no_keyed_orders)

    written = await dts.sync_desk_trades(AsyncSessionLocal)
    assert written == 1

    async with AsyncSessionLocal() as s:
        rows = (await s.execute(
            select(Trade).where(Trade.strategy_name == strat)
        )).scalars().all()
        assert len(rows) == 1
        t = rows[0]
        assert t.side == "buy"
        assert float(t.realized_pnl) == 100.0        # (110-100)*10
        assert t.hold_seconds == 3600
        assert (t.raw_payload or {}).get("source") == "desk_alpaca_sync"

        # attributed to a keyless Alpaca paper account (the system/demo account)
        acct = (await s.execute(
            select(Account).where(Account.id == t.account_id)
        )).scalar_one()
        assert acct.broker == "alpaca" and acct.mode == "paper"
        assert acct.encrypted_key is None

    # idempotent: a second sync writes no new rows for this strategy
    written2 = await dts.sync_desk_trades(AsyncSessionLocal)
    assert written2 == 0
    async with AsyncSessionLocal() as s:
        again = (await s.execute(
            select(Trade).where(Trade.strategy_name == strat)
        )).scalars().all()
        assert len(again) == 1


async def test_no_env_key_and_no_keyed_accounts_is_noop(_create_tables, monkeypatch):
    from app.database import AsyncSessionLocal
    from app.tasks import desk_trade_sync as dts

    # No env credentials → the env fallback must not run (and must not raise).
    monkeypatch.setattr(dts, "_env_alpaca_creds", lambda: None)

    async def boom(*a, **k):  # would fire only if the fallback wrongly ran
        raise AssertionError("env fetch attempted despite no credentials")

    monkeypatch.setattr(dts, "_fetch_closed_orders_env", boom)

    written = await dts.sync_desk_trades(AsyncSessionLocal)
    assert written == 0
