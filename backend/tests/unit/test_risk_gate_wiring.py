"""The risk gate must be WIRED, not merely written.

Found 2026-07-27. Every documented risk control — position cap, drawdown
circuit breaker, correlation cluster limit — was inert in the running system:

  * `app/api/v1/orders.py` reads `getattr(request.app.state, "risk_manager", None)`
    and skips the gate when it is None. Nothing in the codebase ever assigned
    `app.state.risk_manager`.
  * `main.py` handed the strategy runner `risk_manager=None` outright.
  * The ONLY `RiskManager()` construction sat in
    `strategy_runner.start_strategy_runner()`, whose docstring says "registered
    as a supervised background task in main.py". main.py never called it — the
    function had exactly one textual occurrence in the whole package: its own
    `def`.

`tests/unit/test_security_invariants.py` passed throughout, because it asserts
the *string* "check_order" appears in orders.py twice. That proves the call is
typed, not that it runs. These tests target the properties that were actually
false.

Scope note: the wiring assertions below are static (AST) rather than a live
lifespan boot — the real lifespan opens brokers, DB and a price feed, which a
unit test cannot stand up. They are deliberately *semantic* rather than
textual: presence of an assignment, and reachability of the constructor. That
is the specific gap the string-matching test missed.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from app.brokers.base import OrderRequest
from app.risk.manager import RiskManager

APP = pathlib.Path(__file__).resolve().parents[2] / "app"
MAIN = APP / "main.py"


def _order(qty=10.0, price=100.0, bucket="directional", symbol="AAPL",
           order_type="limit") -> OrderRequest:
    return OrderRequest(
        symbol=symbol, side="buy", order_type=order_type,
        quantity=qty, limit_price=price, risk_bucket=bucket,
    )


# ── wiring: the gate exists in the running app ───────────────────────────────

def test_app_state_risk_manager_is_actually_assigned():
    """orders.py silently skips the gate when this is absent."""
    found = []
    for path in APP.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            targets = node.targets if isinstance(node, ast.Assign) else (
                [node.target] if isinstance(node, ast.AnnAssign) else []
            )
            for target in targets:
                # app.state.risk_manager = ...  (not self.risk_manager = ...)
                if (isinstance(target, ast.Attribute)
                        and target.attr == "risk_manager"
                        and isinstance(target.value, ast.Attribute)
                        and target.value.attr == "state"):
                    found.append(f"{path.name}:{node.lineno}")
    assert found, (
        "nothing assigns app.state.risk_manager, so orders.py's risk gate is "
        "dead code — every REST order reaches the broker unchecked"
    )


def test_strategy_runner_is_not_constructed_with_a_null_risk_manager():
    """main.py passed risk_manager=None literally."""
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "ContinuousStrategyRunner"):
            continue
        for kw in node.keywords:
            if kw.arg == "risk_manager":
                assert not (isinstance(kw.value, ast.Constant) and kw.value.value is None), (
                    "the always-running strategy runner was given no risk manager, "
                    "so strategy-generated orders bypassed every risk control"
                )


def test_every_risk_manager_construction_is_reachable():
    """A constructor inside a never-called function is not wiring.

    This is the shape of the original bug: the sole RiskManager() lived in
    start_strategy_runner(), which nothing invoked.
    """
    trees = {}
    for path in APP.rglob("*.py"):
        try:
            trees[path] = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

    # Every name REFERENCED anywhere (load context) or imported by name.
    # Deliberately AST-based, not str.count: a prose mention of a function in a
    # comment must not count as a call site. An earlier draft of this test used
    # str.count and was defeated by a comment in main.py naming the very
    # function it was meant to catch.
    referenced: set[str] = set()
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                referenced.add(node.attr)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    referenced.add(alias.asname or alias.name)

    unreachable = []
    for path, tree in trees.items():
        # A reference from inside the function's own body (recursion) is not a
        # call site, but the enclosing-scope check below is close enough here:
        # none of these functions recurse.
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            constructs = any(
                isinstance(n, ast.Call) and getattr(n.func, "id", "") == "RiskManager"
                for n in ast.walk(func)
            )
            if constructs and func.name not in referenced:
                unreachable.append(f"{path.name}:{func.name}")

    assert not unreachable, (
        f"these build a RiskManager but are never called, so the gate they "
        f"construct never runs: {unreachable}"
    )


# ── behaviour: the cap must not swallow the correlation check ────────────────

@pytest.mark.asyncio
async def test_a_capped_order_is_still_correlation_checked():
    """The size cap used to `return` early.

    Only oversized orders reach that branch — so the largest orders, the ones
    most able to breach a cluster limit, were the exact ones that skipped it.
    """
    rm = RiskManager(max_position_pct=0.05, max_cluster_pct=0.30, initial_equity=100_000)
    rm.update_equity(100_000)
    rm._clusters = {"tech": ["AAPL", "MSFT", "NVDA"]}
    rm.update_positions([{"symbol": "MSFT", "market_value": 28_000}])

    # $50k order → capped to $5k (5% of NAV). The tech cluster already holds
    # $28k, so +$5k = $33k > 30% of NAV → must be BLOCKED, not merely capped.
    decision = await rm.check_order(_order(qty=500, price=100, symbol="AAPL"))
    assert not decision.allowed, (
        "capping the size approved the order outright and skipped the cluster "
        "limit — concentration risk was unenforced for every oversized order"
    )


@pytest.mark.asyncio
async def test_capping_still_reports_the_capped_quantity():
    rm = RiskManager(max_position_pct=0.05, initial_equity=100_000)
    rm.update_equity(100_000)
    decision = await rm.check_order(_order(qty=100, price=100))
    assert decision.allowed and decision.reason == "size capped"
    assert decision.adjusted_quantity == pytest.approx(50.0, rel=0.01)


@pytest.mark.asyncio
async def test_an_uncapped_order_reports_its_own_quantity():
    rm = RiskManager(max_position_pct=0.05, initial_equity=100_000)
    rm.update_equity(100_000)
    decision = await rm.check_order(_order(qty=10, price=100))
    assert decision.allowed and decision.reason == "ok"
    assert decision.adjusted_quantity == pytest.approx(10.0)


# ── behaviour: no fabricated $100 price ──────────────────────────────────────

@pytest.mark.asyncio
async def test_market_orders_are_not_sized_against_a_hardcoded_100_dollars():
    """1 BTC is not a $100 position.

    check_order() used `limit_price if not None else 100.0`. Every market order
    carries no limit price, so 1 BTC at ~$60k was measured as $100 — 0.1% of a
    $100k account instead of 60%, and the cap never fired.
    """
    rm = RiskManager(max_position_pct=0.05, initial_equity=100_000)
    rm.update_equity(100_000)
    rm.update_prices({"BTC/USD": 60_000.0})

    decision = await rm.check_order(
        _order(qty=1, price=None, symbol="BTC/USD", order_type="market")
    )
    assert decision.reason == "size capped", (
        "a 1 BTC market order is $60k against a $100k account — it must be capped"
    )
    # 5% of $100k = $5,000 → 0.0833 BTC
    assert decision.adjusted_quantity == pytest.approx(5_000 / 60_000, rel=1e-6)


@pytest.mark.asyncio
async def test_a_sub_penny_symbol_is_not_measured_as_100_dollars_a_unit():
    """The same bug in the other direction: SHIB read 10-million-fold too big."""
    rm = RiskManager(max_position_pct=0.05, initial_equity=100_000)
    rm.update_equity(100_000)
    rm.update_prices({"SHIB/USD": 0.00001})

    # 10M SHIB = $100. Well inside the $5k cap; the old code scored it at $1bn.
    decision = await rm.check_order(
        _order(qty=10_000_000, price=None, symbol="SHIB/USD", order_type="market")
    )
    assert decision.allowed and decision.reason == "ok"


@pytest.mark.asyncio
async def test_an_unpriced_order_says_so_instead_of_inventing_a_price():
    rm = RiskManager(initial_equity=100_000)
    rm.update_equity(100_000)
    decision = await rm.check_order(
        _order(qty=1, price=None, symbol="NOPRICE", order_type="market")
    )
    assert decision.allowed, "must not halt trading on a cold price cache"
    assert "unpriced" in decision.reason
    assert rm.unpriced_orders == 1, "unpriced orders must be countable, not silent"


def test_bad_price_updates_cannot_disable_the_cap():
    """A NaN/zero/negative mark would silently un-cap that symbol."""
    rm = RiskManager(initial_equity=100_000)
    rm.update_prices({"A": float("nan"), "B": 0.0, "C": -5.0,
                      "D": float("inf"), "E": "junk", "F": 42.0})
    assert rm._prices == {"F": 42.0}


def test_update_prices_survives_garbage_input():
    """This sits in front of every order — it must never raise."""
    rm = RiskManager(initial_equity=100_000)
    rm.update_prices(None)          # type: ignore[arg-type]
    rm.update_prices("not a dict")  # type: ignore[arg-type]
    rm.update_prices({})
    assert rm._prices == {}


# ── the sync loop that feeds the gate ────────────────────────────────────────
# Runtime tests with a fake broker: a manager that is wired but never fed real
# equity still cannot trip a drawdown breaker (one data point is not a series)
# and still caps against a fabricated NAV.

class _FakeBroker:
    def __init__(self, equity, positions=None, fail_positions=False):
        self.equity = equity
        self.positions = positions or []
        self.fail_positions = fail_positions
        self.account_calls = 0

    async def get_account(self):
        self.account_calls += 1
        return {"equity": self.equity, "cash": 0.0}

    async def get_positions(self):
        if self.fail_positions:
            raise RuntimeError("broker down")
        return self.positions


async def _sync_once(risk_manager, broker):
    """Drive exactly one iteration of the loop, then stop."""
    import asyncio as _asyncio

    from app.main import _risk_state_sync

    task = _asyncio.create_task(_risk_state_sync(risk_manager, broker, interval_seconds=3600))
    for _ in range(200):                      # let it complete one pass
        await _asyncio.sleep(0)
        if broker.account_calls:
            break
    await _asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except _asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_sync_feeds_real_equity_to_the_manager():
    rm = RiskManager(initial_equity=100_000)
    await _sync_once(rm, _FakeBroker(equity=250_000.0))
    assert rm._equity == pytest.approx(250_000.0)
    assert rm._equity_confirmed, "the cap must size against broker NAV, not the seed"


@pytest.mark.asyncio
async def test_negative_equity_halts_instead_of_failing_open():
    """The live Alpaca paper account is negative — this path is not theoretical.

    update_equity() raises on a negative value. If the sync loop simply let that
    propagate into its except-and-log, the manager would sit on its seeded
    $100k forever and keep approving orders — failing OPEN on precisely the
    condition that must halt trading.
    """
    rm = RiskManager(initial_equity=100_000)
    await _sync_once(rm, _FakeBroker(equity=-8_287.81))
    assert rm._equity == 0.0
    decision = await rm.check_order(_order())
    assert not decision.allowed and "equity" in decision.reason.lower()


@pytest.mark.asyncio
async def test_a_positions_failure_is_reported_as_a_positions_failure(capsys):
    """Distinct failure modes must not share one log line.

    Equity and positions are fetched separately, so a triage reading the logs
    can tell "the drawdown breaker is blind" apart from "concentration limits
    are stale". A single combined "risk state sync failed" cannot.

    structlog does not propagate to stdlib logging, so this asserts on captured
    output rather than caplog.
    """
    rm = RiskManager(initial_equity=100_000)
    await _sync_once(rm, _FakeBroker(equity=250_000.0, fail_positions=True))

    out = capsys.readouterr().out + capsys.readouterr().err
    assert "positions sync failed" in out, (
        "a positions outage must name positions, not a generic combined failure"
    )
    assert rm._equity == pytest.approx(250_000.0), "equity must still be current"


# ── the mark cache needs a producer, or market orders stay uncapped ──────────

@pytest.mark.asyncio
async def test_the_price_feed_actually_feeds_the_risk_manager():
    """A mark cache nothing writes to is the same bug in a new place.

    Without this the price feed publishes to Redis and the WebSocket, and the
    risk manager — the one consumer that needs a mark to size a market order —
    is never told.
    """
    from app.tasks.price_feed import _fetch_and_publish

    class _Quote:
        bid, ask, last, volume = 59_900.0, 60_100.0, 60_000.0, 1.0

    class _Broker:
        async def get_quote(self, symbol):
            return _Quote()

    class _Cache:
        async def set_price(self, *a, **kw):
            return None

    rm = RiskManager(initial_equity=100_000)
    await _fetch_and_publish(
        _Broker(), "BTC/USD", _Cache(),
        on_mark=lambda sym, last: rm.update_prices({sym: last}),
    )
    assert rm._prices.get("BTC/USD") == pytest.approx(60_000.0)


@pytest.mark.asyncio
async def test_a_failing_mark_sink_does_not_break_the_price_feed():
    """Price publication must survive a bad consumer."""
    from app.tasks.price_feed import _fetch_and_publish

    class _Quote:
        bid, ask, last, volume = 1.0, 1.0, 1.0, 1.0

    class _Broker:
        async def get_quote(self, symbol):
            return _Quote()

    published = []

    class _Cache:
        async def set_price(self, *a, **kw):
            published.append(a)

    def _boom(symbol, last):
        raise RuntimeError("sink exploded")

    await _fetch_and_publish(_Broker(), "SPY", _Cache(), on_mark=_boom)
    assert published, "a broken risk sink must not stop the Redis price write"


@pytest.mark.asyncio
async def test_sync_survives_a_broker_that_is_entirely_down():
    class _Dead:
        account_calls = 0

        async def get_account(self):
            type(self).account_calls += 1
            raise RuntimeError("connection refused")

        async def get_positions(self):
            raise RuntimeError("connection refused")

    rm = RiskManager(initial_equity=100_000)
    await _sync_once(rm, _Dead())   # must not raise out of the loop
    assert rm._equity == pytest.approx(100_000.0)
