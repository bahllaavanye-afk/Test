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
from typing import Any, Iterable, List, Dict

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
    client_order_id: str | None,
    registry_names: Iterable[str] | None = None,
) -> str | None:
    """Extract the strategy name from a ``qe-{strat}-{sym}-{ts}`` client_order_id.

    The strategy token is truncated to 10 chars at order time. When the full
    strategy registry is supplied, a truncated token is expanded back to its
    full name (so attribution matches the leaderboard's ``strategy_name``
    grouping) whenever exactly one registry entry shares those first 10 chars.
    """
    # ---- Input validation ----
    if client_order_id is not None and not isinstance(client_order_id, str):
        raise ValueError(
            f"client_order_id must be a string or None, got {type(client_order_id).__name__}"
        )
    if registry_names is not None:
        if not isinstance(registry_names, Iterable):
            raise ValueError(
                "registry_names must be an iterable of strings or None"
            )
        for name in registry_names:
            if not isinstance(name, str):
                raise ValueError(
                    f"All entries in registry_names must be strings, got {type(name).__name__}"
                )
    # ---- End validation ----

    if not client_order_id or not client_order_id.startswith(DESK_COID_PREFIX):
        return None
    rest = client_order_id[len(DESK_COID_PREFIX) :]
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


def reconstruct_closed_trades(
    orders: List[Dict],
    registry_names: Iterable[str] | None = None,
) -> List[Dict]:
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
    # ---- Input validation ----
    if not isinstance(orders, list):
        raise ValueError(f"orders must be a list of dicts, got {type(orders).__name__}")
    for i, o in enumerate(orders):
        if not isinstance(o, dict):
            raise ValueError(f"order at index {i} is not a dict, got {type(o).__name__}")

    if registry_names is not None:
        if not isinstance(registry_names, Iterable):
            raise ValueError("registry_names must be an iterable of strings or None")
        for name in registry_names:
            if not isinstance(name, str):
                raise ValueError(
                    f"All entries in registry_names must be strings, got {type(name).__name__}"
                )
    # ---- End validation ----

    fills: list[dict] = []
    for o in orders:
        if str(o.get("status", "")).lower() != "filled":
            continue
        strat = parse_strategy_from_coid(o.get("client_order_id"), registry_names)
        if strat is None:
            continue
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
        fills.append(
            {
                "strategy": strat,
                "symbol": o.get("symbol"),
                "side": side,
                "qty": qty,
                "price": price,
                "ts": ts,
                "order_id": o.get("id"),
            }
        )

    # Chronological order is required for correct FIFO matching.
    fills.sort(key=lambda f: f["ts"])

    lots: dict[tuple, deque] = defaultdict(deque)
    trades: list[dict] = []

    for f in fills:
        key = (f["strategy"], f["symbol"])
        q = lots[key]
        remaining = f["qty"] if f["side"] == "buy" else -f["qty"]

        # Close opposing lots FIFO.
        while abs(remaining) > 1e-12 and q and _sign(q[0]["signed"]) == -_sign(
            remaining
        ):
            lot = q[0]
            match = min(abs(lot["signed"]), abs(remaining))
            entry = lot["price"]
            exit_price = f["price"]
            if lot["signed"] > 0:  # closing a long
                pnl = (exit_price - entry) * match
                lot_side = "buy"
            else:  # closing a short
                pnl = (entry - exit_price) * match
                lot_side = "sell"
            hold_seconds = int((f["ts"] - lot["opened_at"]).total_seconds())
            trades.append(
                {
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
                }
            )
            lot_remaining = abs(lot["signed"]) - match
            if lot_remaining <= 1e-12:
                q.popleft()
            else:
                lot["signed"] = _sign(lot["signed"]) * lot_remaining
            remaining = _sign(remaining) * (abs(remaining) - match)

        # Leftover (same-direction add, or the excess of a position flip) opens a lot.
        if abs(remaining) > 1e-12:
            q.append(
                {
                    "signed": remaining,
                    "price": f["price"],
                    "opened_at": f["ts"],
                    "order_id": f["order_id"],
                }
            )

    return trades


async def _fetch_closed_orders(acct, lookback_days: int = 30) -> list[dict]:
    """Fetch recently-closed orders for an Alpaca account (paginated, bounded)."""
    import httpx

    from app.brokers.alpaca_orders import _base_url, _headers

    headers = await _headers(acct)
    base = _base_url(acct)
    after = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    collected: list
# ... (truncated for brevity)