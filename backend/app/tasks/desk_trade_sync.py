"""Desk → Trades ingestion: turn desk paper fills into closed Trade records.

The `.github/scripts/desk_order_placer.py` job places REAL paper orders on
Alpaca for ~59 strategies across 6 desks, tagging each with a client_order_id
of the form ``qe-{strategy[:10]}-{symbol[:4]}-{unix_ts}``. Those orders live
only on Alpaca — they never touch the backend ``Order`` table — so their round
trips never became ``Trade`` rows, and the P&L feedback loop
(`leaderboard.compute_live_strategy_performance`) only ever saw *bot* trades.

This module closes that gap. It pulls filled ``qe-``-tagged orders from Alpaca,
reconstructs closed round trips with FIFO lot accounting, and writes ``Trade``
rows attributed to the originating strategy. That makes EVERY strategy — not
just bots — feed the leaderboard and the self-scaling weighting, which is the
whole point of the P&L loop.

The reconstruction (`reconstruct_closed_trades`) is a pure function with no DB
or network dependency, so it is fully unit-testable on synthetic fills. The
scheduler job (`sync_desk_trades`) is the thin I/O wrapper around it.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from app.utils.logging import logger

# Every desk order is tagged with this prefix by desk_order_placer.py. Scoping
# strictly to it isolates desk fills from anything else on the Alpaca account
# (manual orders, bot orders) so we never double-count.
DESK_COID_PREFIX = "qe-"


def _sign(x: float) -> int:
    return 1 if x > 0 else -1


def _parse_ts(value: Any) -> datetime | None:
    """Parse an Alpaca ISO-8601 timestamp (…Z or +00:00) into an aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Fall back: strip fractional seconds Alpaca sometimes over-pads.
        try:
            head = s.split(".")[0]
            dt = datetime.fromisoformat(head + "+00:00" if "+" not in head else head)
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_strategy_from_coid(
    client_order_id: str | None, registry_names: Iterable[str] | None = None
) -> str | None:
    """Extract the strategy name from a ``qe-{strat}-{sym}-{ts}`` client_order_id.

    The strategy token is truncated to 10 chars at order time. When the full
    strategy registry is supplied, a truncated token is expanded back to its
    full name (so attribution matches the leaderboard's ``strategy_name``
    grouping) whenever exactly one registry entry shares those first 10 chars.
    """
    if not client_order_id or not client_order_id.startswith(DESK_COID_PREFIX):
        return None
    rest = client_order_id[len(DESK_COID_PREFIX):]
    # rsplit from the right: [strategy, symbol, unix_ts]. Strategy names use
    # underscores (never hyphens), so this is unambiguous.
    parts = rest.rsplit("-", 2)
    if len(parts) < 3:
        return None
    strat = parts[0].strip()
    if not strat:
        return None
    if registry_names:
        names = list(registry_names)
        if strat in names:
            return strat
        matches = [n for n in names if n[:10] == strat]
        if len(matches) == 1:
            return matches[0]
    return strat


def _close_untagged(lots: dict, trades: list, f: dict) -> None:
    """Apply an untagged fill against open lots for its symbol, oldest first.

    Attribution stays with whichever strategy OPENED each lot — the close was
    not originated by a strategy, so it cannot introduce one. Excess quantity
    beyond open inventory is dropped rather than opening an unattributed lot.
    """
    remaining = f["qty"] if f["side"] == "buy" else -f["qty"]
    want = -_sign(remaining)  # the lot direction this fill can close

    # Oldest open lot first, across every strategy holding this symbol.
    def _candidates():
        out = []
        for key, q in lots.items():
            if key[1] != f["symbol"]:
                continue
            if q and _sign(q[0]["signed"]) == want:
                out.append((q[0]["opened_at"], key, q))
        out.sort(key=lambda t: t[0])
        return out

    while abs(remaining) > 1e-12:
        cands = _candidates()
        if not cands:
            return  # nothing left to close; excess is deliberately discarded
        _, key, q = cands[0]
        lot = q[0]
        match = min(abs(lot["signed"]), abs(remaining))
        entry, exit_price = lot["price"], f["price"]
        if lot["signed"] > 0:
            pnl = (exit_price - entry) * match
            lot_side = "buy"
        else:
            pnl = (entry - exit_price) * match
            lot_side = "sell"
        trades.append({
            "strategy_name": key[0],
            "symbol": f["symbol"],
            "side": lot_side,
            "entry_price": entry,
            "exit_price": exit_price,
            "quantity": match,
            "realized_pnl": pnl,
            "opened_at": lot["opened_at"],
            "closed_at": f["ts"],
            "hold_seconds": int((f["ts"] - lot["opened_at"]).total_seconds()),
            "close_order_id": f["order_id"],
            "open_order_id": lot["order_id"],
        })
        lot_remaining = abs(lot["signed"]) - match
        if lot_remaining <= 1e-12:
            q.popleft()
        else:
            lot["signed"] = _sign(lot["signed"]) * lot_remaining
        remaining = _sign(remaining) * (abs(remaining) - match)


def reconstruct_closed_trades(
    orders: list[dict], registry_names: Iterable[str] | None = None
) -> list[dict]:
    """Reconstruct closed round trips from a list of Alpaca order dicts.

    Uses FIFO lot accounting per ``(strategy, symbol)``: a fill in the opposite
    direction to open inventory closes the oldest lot(s) first, emitting one
    closed-trade dict per matched slice (with the lot's opening side, entry/exit
    prices, quantity, realized P&L, hold time, and the opening/closing order
    ids). A fill that exceeds inventory flips the position: the excess opens a
    fresh lot. Fills that only open inventory produce no trade until they close.

    Pure function — no DB, no network. Each returned dict maps 1:1 onto the
    ``Trade`` model columns plus ``close_order_id``/``open_order_id`` for
    idempotent dedup by the caller.
    """
    fills: list[dict] = []
    for o in orders:
        if str(o.get("status", "")).lower() != "filled":
            continue
        # `None` means "not a desk-tagged order". Those are NOT skipped any
        # more — see the untagged-close handling in the FIFO loop below.
        strat = parse_strategy_from_coid(o.get("client_order_id"), registry_names)
        try:
            qty = float(o.get("filled_qty") or 0)
            price = float(o.get("filled_avg_price") or 0)
        except (TypeError, ValueError):
            continue
        if qty <= 0 or price <= 0:
            continue
        ts = _parse_ts(o.get("filled_at") or o.get("updated_at") or o.get("submitted_at"))
        if ts is None:
            continue
        side = str(o.get("side", "")).lower()
        if side not in ("buy", "sell"):
            continue
        fills.append({
            "strategy": strat,
            "symbol": o.get("symbol"),
            "side": side,
            "qty": qty,
            "price": price,
            "ts": ts,
            "order_id": o.get("id"),
        })

    # Chronological order is required for correct FIFO matching.
    fills.sort(key=lambda f: f["ts"])

    lots: dict[tuple, deque] = defaultdict(deque)
    trades: list[dict] = []

    for f in fills:
        if f["strategy"] is None:
            # UNTAGGED FILL — a close this system did not originate.
            #
            # `recover_negative_cash` flattens via DELETE /v2/positions, so
            # Alpaca generates those closing orders itself and they carry no
            # `qe-` client_order_id. The backend's PositionMonitor exits are
            # the same shape. Previously such fills were dropped outright,
            # which meant the opening `qe-` buy created a lot that could NEVER
            # close: no Trade row, ever.
            #
            # Measured 2026-07-27: 25 positions were flattened at 18:43 and
            # /api/v1/trades/ still returned []. Every one of those round trips
            # was invisible to the P&L loop and the leaderboard.
            #
            # An untagged fill closes open lots for its SYMBOL, oldest first
            # across whichever strategies hold them — that is what actually
            # happened at the broker. Any excess beyond open inventory is
            # discarded rather than opening a lot, because there is no strategy
            # to attribute it to and inventing one would corrupt attribution.
            _close_untagged(lots, trades, f)
            continue

        key = (f["strategy"], f["symbol"])
        q = lots[key]
        remaining = f["qty"] if f["side"] == "buy" else -f["qty"]

        # Close opposing lots FIFO.
        while abs(remaining) > 1e-12 and q and _sign(q[0]["signed"]) == -_sign(remaining):
            lot = q[0]
            match = min(abs(lot["signed"]), abs(remaining))
            entry = lot["price"]
            exit_price = f["price"]
            if lot["signed"] > 0:  # closing a long
                pnl = (exit_price - entry) * match
                lot_side = "buy"
            else:                  # closing a short
                pnl = (entry - exit_price) * match
                lot_side = "sell"
            hold_seconds = int((f["ts"] - lot["opened_at"]).total_seconds())
            trades.append({
                "strategy_name": f["strategy"],
                "symbol": f["symbol"],
                "side": lot_side,
                "entry_price": entry,
                "exit_price": exit_price,
                "quantity": match,
                "realized_pnl": pnl,
                "opened_at": lot["opened_at"],
                "closed_at": f["ts"],
                "hold_seconds": hold_seconds,
                "close_order_id": f["order_id"],
                "open_order_id": lot["order_id"],
            })
            lot_remaining = abs(lot["signed"]) - match
            if lot_remaining <= 1e-12:
                q.popleft()
            else:
                lot["signed"] = _sign(lot["signed"]) * lot_remaining
            remaining = _sign(remaining) * (abs(remaining) - match)

        # Leftover (same-direction add, or the excess of a position flip) opens a lot.
        if abs(remaining) > 1e-12:
            q.append({
                "signed": remaining,
                "price": f["price"],
                "opened_at": f["ts"],
                "order_id": f["order_id"],
            })

    return trades


async def _fetch_closed_orders_raw(
    headers: dict[str, str], base: str, lookback_days: int = 30
) -> list[dict]:
    """Fetch recently-closed Alpaca orders given ready headers + base URL (paginated, bounded)."""
    import httpx

    after = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    collected: list[dict] = []
    # Alpaca caps `limit` at 500 and paginates by the last seen submitted_at.
    async with httpx.AsyncClient(timeout=15) as client:
        for _ in range(10):  # hard cap: 5000 orders / sync
            params = {
                "status": "closed",
                "limit": 500,
                "direction": "asc",
                "after": after,
                "nested": "false",
            }
            resp = await client.get(f"{base}/v2/orders", headers=headers, params=params)
            if resp.status_code != 200:
                logger.warning("Desk trade sync: order fetch failed", status=resp.status_code)
                break
            page = resp.json()
            if not page:
                break
            collected.extend(page)
            if len(page) < 500:
                break
            # Advance the window past the last order to avoid re-fetching it.
            last_ts = page[-1].get("submitted_at") or page[-1].get("created_at")
            if not last_ts:
                break
            after = last_ts
    return collected


async def _fetch_closed_orders(acct, lookback_days: int = 30) -> list[dict]:
    """Fetch recently-closed orders for a DB Account (decrypts its stored keys)."""
    from app.brokers.alpaca_orders import _base_url, _headers

    headers = await _headers(acct)
    base = _base_url(acct)
    return await _fetch_closed_orders_raw(headers, base, lookback_days)


def _env_alpaca_creds() -> tuple[str, str] | None:
    """The desk trades on the *env* ALPACA key (GitHub Actions secret, relayed to Render),
    which usually has NO corresponding DB Account. Without this, `qe-`-tagged desk fills
    live only on Alpaca and never become Trades — the exact reason the site shows "no
    trades". Returns (key, secret) when both are set, else None.
    """
    import os

    from app.config import settings

    key = (getattr(settings, "alpaca_api_key", "") or os.environ.get("ALPACA_API_KEY", "")).strip()
    secret = (
        getattr(settings, "alpaca_secret_key", "") or os.environ.get("ALPACA_SECRET_KEY", "")
    ).strip()
    return (key, secret) if key and secret else None


async def _fetch_closed_orders_env(creds: tuple[str, str], lookback_days: int = 30) -> list[dict]:
    """Fetch the desk account's closed paper orders directly via the env credentials."""
    from app.brokers.alpaca_orders import ALPACA_PAPER

    key, secret = creds
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    return await _fetch_closed_orders_raw(headers, ALPACA_PAPER, lookback_days)


async def _persist_reconstructed(
    db_session_factory,
    account_id: str,
    reconstructed: list[dict],
    strat_id_by_name: dict[str, str],
    lookback_start: datetime,
) -> int:
    """Write newly-closed round trips as Trades under ``account_id`` (idempotent).

    Dedup key is the closing Alpaca order id stored in ``raw_payload.close_order_id``,
    scoped to the account + lookback window, so re-running never duplicates rows.
    Returns the number of new Trade rows written.
    """
    from sqlalchemy import select

    from app.models.trade import Trade

    async with db_session_factory() as db:
        existing = (await db.execute(
            select(Trade).where(
                Trade.account_id == account_id,
                Trade.closed_at >= lookback_start,
            )
        )).scalars().all()
        seen_close_ids = {
            (t.raw_payload or {}).get("close_order_id")
            for t in existing
            if t.raw_payload
        }

        new_rows = 0
        for tr in reconstructed:
            close_id = tr.get("close_order_id")
            if close_id and close_id in seen_close_ids:
                continue
            seen_close_ids.add(close_id)
            db.add(Trade(
                account_id=account_id,
                strategy_id=strat_id_by_name.get(tr["strategy_name"]),
                strategy_name=tr["strategy_name"],
                symbol=tr["symbol"],
                side=tr["side"],
                entry_price=tr["entry_price"],
                exit_price=tr["exit_price"],
                quantity=tr["quantity"],
                realized_pnl=tr["realized_pnl"],
                fees=0.0,
                opened_at=tr["opened_at"],
                closed_at=tr["closed_at"],
                hold_seconds=tr["hold_seconds"],
                raw_payload={
                    "source": "desk_alpaca_sync",
                    "close_order_id": tr["close_order_id"],
                    "open_order_id": tr["open_order_id"],
                    "strategy": tr["strategy_name"],
                },
            ))
            new_rows += 1

        if new_rows:
            await db.commit()
            logger.info("Desk trade sync wrote trades", account_id=account_id, count=new_rows)
        return new_rows


async def sync_desk_trades(db_session_factory=None, lookback_days: int = 30) -> int:
    """Pull desk fills from Alpaca and persist newly-closed round trips as Trades.

    Idempotent: a trade is keyed by its closing Alpaca order id
    (``raw_payload.close_order_id``); trades already recorded in the lookback
    window are skipped, so re-running never duplicates rows.

    Returns the number of new Trade rows written.
    """
    from sqlalchemy import select

    from app.models.account import Account
    from app.models.strategy import Strategy

    if db_session_factory is None:
        try:
            from app.database import AsyncSessionLocal as db_session_factory  # type: ignore
        except Exception as exc:  # noqa: BLE001
            logger.debug("Desk trade sync: no DB session factory", error=str(exc))
            return 0

    try:
        from app.strategies import STRATEGY_REGISTRY

        registry_names = set(STRATEGY_REGISTRY.keys())
    except Exception:
        registry_names = set()

    written = 0
    try:
        async with db_session_factory() as db:
            keyed_accounts = (await db.execute(
                select(Account).where(
                    Account.broker == "alpaca",
                    Account.mode == "paper",
                    Account.is_active == True,  # noqa: E712
                    Account.encrypted_key.isnot(None),
                )
            )).scalars().all()

            # The system/demo paper account owns the seeded bots but stores NO key
            # (the desk trades on the env secret). It's what the site & global
            # leaderboard read, so env-keyed desk fills are attributed here.
            fallback_account = (await db.execute(
                select(Account).where(
                    Account.broker == "alpaca",
                    Account.mode == "paper",
                    Account.is_active == True,  # noqa: E712
                    Account.encrypted_key.is_(None),
                ).order_by(Account.created_at.asc())
            )).scalars().first()

            # name → strategy_id, for FK attribution (best-effort).
            strat_rows = (await db.execute(select(Strategy.id, Strategy.name))).all()
            strat_id_by_name = {r.name: r.id for r in strat_rows}

        lookback_start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        processed_key_ids: set[str] = set()

        # ── Keyed DB accounts (real user-connected Alpaca accounts) ──────────
        for acct in keyed_accounts:
            try:
                from app.brokers.alpaca_orders import _headers
                kid = (await _headers(acct)).get("APCA-API-KEY-ID")
                if kid:
                    processed_key_ids.add(kid)
                orders = await _fetch_closed_orders(acct, lookback_days=lookback_days)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Desk trade sync: fetch error", account_id=acct.id, error=str(exc))
                continue
            if not orders:
                continue
            reconstructed = reconstruct_closed_trades(orders, registry_names)
            if not reconstructed:
                continue
            written += await _persist_reconstructed(
                db_session_factory, acct.id, reconstructed, strat_id_by_name, lookback_start
            )

        # ── Env-keyed desk account (the qe-* fills the site was missing) ─────
        # The desks place ~59 strategies' orders on the env ALPACA key, which
        # normally has no DB Account, so those fills never became Trades. Fetch
        # them directly and attribute to the system/demo account. Skip if a keyed
        # account already covered this exact key (avoids double-counting).
        creds = _env_alpaca_creds()
        if creds and creds[0] not in processed_key_ids:
            if fallback_account is None:
                logger.debug(
                    "Desk trade sync: env ALPACA key present but no system paper account to attribute to"
                )
            else:
                try:
                    orders = await _fetch_closed_orders_env(creds, lookback_days=lookback_days)
                    reconstructed = reconstruct_closed_trades(orders, registry_names)
                    if reconstructed:
                        written += await _persist_reconstructed(
                            db_session_factory, fallback_account.id, reconstructed,
                            strat_id_by_name, lookback_start,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Desk trade sync: env fetch error", error=str(exc))

        if not keyed_accounts and creds is None:
            logger.debug("Desk trade sync: no Alpaca paper accounts and no env key")

    except Exception as exc:  # noqa: BLE001
        logger.error("Desk trade sync failed", error=str(exc))

    return written
