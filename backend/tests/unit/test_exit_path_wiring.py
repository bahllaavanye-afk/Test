"""Stop-losses must actually be enforced, and the price they use must be real.

Found 2026-07-27, continuing the review that started with the risk gate. Two
independent faults on the exit path, both silent:

1. `PositionMonitor` was NEVER STARTED. `start_position_monitor()` claimed to be
   a "factory function called from scheduler.py"; scheduler.py had no such job,
   and nothing anywhere constructed a PositionMonitor. Meanwhile the strategy
   runner writes `pos_exit:<symbol>` on every fill — stop_loss, take_profit,
   peak_price — under the comment "Store exit config in Redis for
   position_monitor.py". The producer ran; the consumer did not exist. Every
   strategy stop-loss was recorded and none was enforced, and the entire
   CompositeExit engine was reachable only from that dead module.

2. EVERY Redis price read in the codebase used the wrong key. `set_price()`
   writes `price:<exchange>:<symbol>`. All three readers built
   `prices:<symbol>` — which is the WEBSOCKET TOPIC (ws/prices.py), never
   written to Redis. The miss is indistinguishable from a cold cache, so each
   silently took its fallback:
     * bots/engine._fetch_current_price → a yfinance DAILY bar, meaning the
       live `bot_exit_checker` job evaluated intraday TP/LP against a daily
       close
     * tasks/position_monitor          → a broker quote per position per tick
     * api/v1/positions                → pnl_pct always None

Together with the inert risk manager, these are why a paper account reached
-$8,287.81: nothing capped size, nothing halted on drawdown, and nothing
enforced a stop.
"""
from __future__ import annotations

import ast
import json
import pathlib

import pytest

from app.redis_client import exchange_for, price_key

# ── Input validation for public functions ─────────────────────────────────────

# Validate `price_key` inputs
_original_price_key = price_key


def _validated_price_key(exchange: str, symbol: str) -> str:
    if not isinstance(exchange, str) or not exchange:
        raise ValueError("price_key: 'exchange' must be a non‑empty string")
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("price_key: 'symbol' must be a non‑empty string")
    return _original_price_key(exchange, symbol)


# Validate `exchange_for` inputs
_original_exchange_for = exchange_for


def _validated_exchange_for(symbol: str) -> str:
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("exchange_for: 'symbol' must be a non‑empty string")
    return _original_exchange_for(symbol)


# Apply validated wrappers globally
price_key = _validated_price_key
exchange_for = _validated_exchange_for

APP = pathlib.Path(__file__).resolve().parents[2] / "app"
MAIN = APP / "main.py"


# ── the key that nothing wrote ───────────────────────────────────────────────

def test_writer_and_reader_agree_on_the_price_key():
    """The bug in one line: the readers did not use what the writer wrote."""
    assert price_key("alpaca", "SPY") == "price:alpaca:SPY"
    assert price_key("crypto", "BTC/USD") == "price:crypto:BTC/USD"
    assert price_key(exchange_for("SPY"), "SPY") != "prices:SPY", (
        "`prices:<symbol>` is the WebSocket topic, not a Redis key"
    )


def test_exchange_namespace_matches_the_price_feeds_own_split():
    """price_feed uses `"crypto" if "/" in symbol else "alpaca"` — must match."""
    assert exchange_for("BTC/USD") == "crypto"
    assert exchange_for("ETH/USD") == "crypto"
    assert exchange_for("SPY") == "alpaca"
    assert exchange_for("BRK.B") == "alpaca"


def test_set_price_uses_the_shared_key_builder():
    """Both sides must route through one function or they can drift again."""
    src = (APP / "redis_client.py").read_text(encoding="utf-8")
    assert src.count('f"price:{exchange}:{symbol}"') <= 1, (
        "the key format must exist in exactly one place (price_key); found "
        "inline copies that can drift from it"
    )


def test_no_redis_read_uses_the_websocket_topic_as_a_key():
    """Guards the actual regression across the whole package.

    Deliberately narrow: `prices:{symbol}` is *correct* as a WebSocket topic in
    ws/ and price_feed's broadcast call. What must never come back is passing
    it to a Redis get.

    AST-based, not text: the first draft of this test matched its own
    explanatory comment quoting the old code. Same trap as the reachability
    guard in test_risk_gate_wiring — a comment is not a call site.
    """
    offenders = []
    for path in APP.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and node.args):
                continue
            arg = node.args[0]
            if not isinstance(arg, ast.JoinedStr) or not arg.values:
                continue
            head = arg.values[0]
            if isinstance(head, ast.Constant) and str(head.value).startswith("prices:"):
                offenders.append(f"{path.relative_to(APP)}:{node.lineno}")
    assert not offenders, (
        f"these read a Redis key nothing writes, and the miss looks exactly "
        f"like a cold cache: {offenders}"
    )


# ── the monitor that never ran ───────────────────────────────────────────────

def test_the_position_exit_monitor_is_started():
    """Nothing constructed a PositionMonitor, so no strategy stop ever fired."""
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    names = {
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_position_exit_monitor" in names, (
        "main.py must run the exit monitor; without it the strategy runner "
        "writes stop-losses to Redis that nothing ever reads"
    )
    # …and it must actually be scheduled, not merely defined. That distinction
    # is the whole bug: start_position_monitor() was defined and never called.
    referenced = {
        n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    assert "_position_exit_monitor" in referenced, (
        "defined but never called is exactly how the original went unnoticed"
    )


def test_no_orphan_position_monitor_factory_remains():
    """The dead factory claimed scheduler.py called it. It did not."""
    src = (APP / "tasks" / "position_monitor.py").read_text(encoding="utf-8")
    assert "async def start_position_monitor" not in src, (
        "a factory nothing calls, whose docstring says something does, is the "
        "trap that hid this for the life of the module"
    )


@pytest.mark.asyncio
async def test_the_monitor_reads_the_price_the_feed_actually_wrote():
    """End to end on the key: feed writes, monitor reads, same string."""
    from app.tasks.position_monitor import PositionMonitor

    written: dict[str, str] = {}

    class _Redis:
        async def get(self, key):
            return written.get(key)

        async def set(self, *a, **kw):
            return None

        async def setex(self, *a, **kw):
            return None

    # What the price feed puts in Redis, via the real writer's key format.
    written[price_key(exchange_for("SPY"), "SPY")] = json.dumps(
        {"last": 123.45, "bid": 123.4, "ask": 123.5}
    )

    seen = {}

    class _Broker:
        async def get_positions(self):
            return [{"symbol": "SPY", "qty": 10.0, "avg_cost": 100.0, "side": "long"}]

        async def get_quote(self, symbol):
            seen["fell_back_to_broker"] = True
            raise AssertionError("must not need a broker quote — the cache has it")

    monitor = PositionMonitor(_Broker(), _Redis(), None)
    await monitor.start()   # no pos_exit config → returns after the price read

    assert "fell_back_to_broker" not in seen, (
        "the Redis price fast path missed, so every position hit the broker"
    )


@pytest.mark.asyncio
async def test_the_bot_exit_checker_no_longer_needs_a_daily_bar():
    """The live exit job priced intraday stops off a yfinance daily close."""
    import app.redis_client as rc
    from app.bots import engine

    class _Cache:
        async def get_price(self, exchange, symbol):
            assert exchange == "crypto" and symbol == "BTC/USD"
            return {"last": 61_234.0}

    original = rc.price_cache
    rc.price_cache = _Cache()
    try:
        price = await engine._fetch_current_price("BTC/USD", market_type="crypto")
    finally:
        rc.price_cache = original

    assert price == pytest.approx(61_234.0), (
        "the cached live price must be used; falling through to a daily bar "
        "cannot tell you whether an intraday stop was breached"
    )