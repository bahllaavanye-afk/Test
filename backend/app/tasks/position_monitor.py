"""
Active position monitoring loop. Runs every 30 seconds, started as a supervised
background task by `main._position_exit_monitor`.

Until 2026-07-27 nothing started it. `start_position_monitor()` claimed to be a
"factory function called from scheduler.py"; scheduler.py had no such job and
nothing anywhere constructed a PositionMonitor. The strategy runner has been
writing `pos_exit:<symbol>` on every fill the whole time, so every strategy
stop-loss was recorded and none was enforced.

For each open position:
  1. Fetches current price from Redis
  2. Fetches stored exit config from Redis (key: pos_exit:<position_id>)
  3. Runs CompositeExit.should_exit()
  4. If exit triggered: submits close order via broker
  5. Updates peak_price tracking in Redis for trailing stops
  6. Broadcasts exit event via WebSocket manager

Redis keys used:
  price:<exchange>:<SYMBOL>  -> {last: float, bid, ask}  (build via
                                redis_client.price_key — this docstring used to
                                say `prices:<SYMBOL>`, which is the WEBSOCKET
                                TOPIC and is never written to Redis, and the
                                code matched the docstring rather than reality)
  pos_exit:<position_id>     -> JSON {exit_strategies, entry_price, peak_price,
                                      bars_held, atr_at_entry, zscore}
  market:regime              -> "0"|"1"|"2"
  market:vix                 -> float

Note position_id is the SYMBOL on the broker path (broker position dicts carry
no "id"), which is what the strategy runner writes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from app.utils.logging import logger

# Alpaca reports crypto POSITIONS slashless — "UNIUSD", "SHIBUSD", "AAVEUSD" —
# while every price path in this system keys on the slashed pair, "UNI/USD".
# Both lookups therefore missed on every crypto position, every tick:
#
#   Redis   exchange_for("UNIUSD") sees no "/" -> "alpaca" -> reads
#           price:alpaca:UNIUSD, but price_feed writes price:crypto:UNI/USD
#   broker  _is_crypto("UNIUSD") is False (no "/", and it does not end with
#           BTC/ETH/SOL/DOGE), so it went to the STOCK quote endpoint and
#           KeyError'd. Observed live 2026-07-28:
#             502: Alpaca quote failed for UNIUSD: 'UNIUSD'
#             502: Alpaca quote failed for SHIBUSD: 'SHIBUSD'
#
# With no price the exit check returns early, so crypto positions had NO
# stop-loss or take-profit enforcement at all — silently, because a missing
# price is indistinguishable from a cold cache.
#
# Deriving candidates instead of hardcoding a crypto universe: this needs no
# maintenance as the desks add pairs, and it fails safe — the original symbol
# is always tried, so an equity that happens to end in "USD" still resolves.
# The fix lives here rather than in brokers/alpaca.py because
# app/tasks/CLAUDE.md marks brokers/*.py do-not-modify.
_QUOTE_CURRENCIES = ("USDT", "USDC", "USD")


def _price_symbol_candidates(symbol: str) -> list[str]:
    """Symbol forms to try when pricing a position, most-likely first."""
    s = (symbol or "").strip().upper()
    if not s:
        return []
    if "/" in s:
        return [s]
    for quote in _QUOTE_CURRENCIES:
        if s.endswith(quote) and len(s) > len(quote):
            return [f"{s[:-len(quote)]}/{quote}", s]
    return [s]


class PositionMonitor:
    def __init__(self, broker, redis_client, db_session_factory):
        self.broker = broker
        self.redis = redis_client
        self.db_session_factory = db_session_factory
        self._running = False

    async def start(self) -> None:
        """Entry point called by scheduler every 30s."""
        self._running = True
        await self._check_all_positions()

    async def _check_all_positions(self) -> None:
        """Load open positions from broker + DB, check exits for each."""
        positions: list[dict] = []

        # Try broker first (live positions)
        if self.broker is not None:
            try:
                positions = await self.broker.get_positions()
            except Exception as exc:
                logger.warning("PositionMonitor: broker.get_positions failed", error=str(exc))

        # If broker unavailable, fall back to DB
        if not positions and self.db_session_factory is not None:
            try:
                from sqlalchemy import select
                from app.models.position import Position

                async with self.db_session_factory() as db:
                    result = await db.execute(
                        select(Position).where(Position.quantity != 0)
                    )
                    db_positions = result.scalars().all()

                positions = [
                    {
                        "id": p.id,
                        "symbol": p.symbol,
                        "side": p.side,
                        "qty": float(p.quantity),
                        "avg_cost": float(p.avg_cost),
                        "entry_price": float(p.avg_cost),
                    }
                    for p in db_positions
                ]
            except Exception as exc:
                logger.warning("PositionMonitor: DB positions fetch failed", error=str(exc))

        if not positions:
            return

        self._unmonitored: list[str] = []
        for position in positions:
            try:
                await self._check_position_exits(position)
            except Exception as exc:
                symbol = position.get("symbol", "?")
                logger.error(
                    "PositionMonitor: error checking position",
                    symbol=symbol,
                    error=str(exc),
                )

        # Say plainly how much of the book is NOT protected. Each skip was a
        # logger.debug, invisible in production, so an unmonitored position
        # looked exactly like a monitored one. Measured 2026-07-28: all 16 open
        # positions were skipped, because `pos_exit:` keys are written only by
        # strategy_runner and every live order is placed by the Actions desks.
        if self._unmonitored:
            logger.warning(
                "PositionMonitor: positions have NO exit config — not enforcing "
                "stop-loss or take-profit for them",
                unmonitored=len(self._unmonitored),
                of_total=len(positions),
                symbols=self._unmonitored[:20],
            )

    async def _check_position_exits(self, position: dict) -> None:
        """Run exit checks for a single position. Fire close order if triggered."""
        symbol = position.get("symbol", "")
        position_id = position.get("id") or symbol

        if not symbol:
            return

        candidates = _price_symbol_candidates(symbol)

        # 1. Fetch current price from Redis
        current_price: float | None = None
        if self.redis is not None:
            from app.redis_client import exchange_for, price_key

            for cand in candidates:
                try:
                    # `prices:{symbol}` is the WebSocket topic, not the Redis key —
                    # nothing writes it, so this always missed and fell through to a
                    # broker quote on every position, every tick.
                    raw_price = await self.redis.get(price_key(exchange_for(cand), cand))
                    if raw_price:
                        price_data = json.loads(raw_price)
                        current_price = float(price_data.get("last") or price_data.get("ask") or 0)
                        if current_price:
                            break
                except Exception as exc:
                    logger.warning(
                        "PositionMonitor: failed to read price from Redis",
                        symbol=cand,
                        error=str(exc),
                    )

        if not current_price:
            # Try broker quote as fallback
            if self.broker is not None:
                last_exc: Exception | None = None
                for cand in candidates:
                    try:
                        quote = await self.broker.get_quote(cand)
                        current_price = float(quote.last or quote.ask)
                        if current_price:
                            break
                    except Exception as exc:  # noqa: BLE001 — try the next form
                        last_exc = exc
                if not current_price and last_exc is not None:
                    logger.warning(
                        "PositionMonitor: broker quote failed, skipping",
                        symbol=symbol,
                        tried=candidates,
                        error=str(last_exc),
                    )
            if not current_price:
                return

        # 2. Fetch exit config from Redis
        exit_config: dict = {}
        if self.redis is not None:
            try:
                raw_exit = await self.redis.get(f"pos_exit:{position_id}")
                if raw_exit:
                    exit_config = json.loads(raw_exit)
            except Exception as exc:
                logger.warning(
                    "PositionMonitor: failed to read exit config from Redis",
                    position_id=position_id,
                    error=str(exc),
                )

        if not exit_config:
            # No exit config stored — skip monitoring for this position.
            #
            # This is NOT a rare edge case: `pos_exit:` keys are written only by
            # strategy_runner, on ITS own fills. The GitHub Actions desks
            # (.github/scripts/desk_order_placer.py) place the live paper orders
            # and write nothing — their workflow even sets REDIS_URL="". So
            # every desk-placed position lands here and is skipped, meaning it
            # has NO stop-loss or take-profit enforcement.
            #
            # Recording the symbol so the caller can report a count. This was
            # logger.debug, which is invisible in production — an unmonitored
            # position looked exactly like a monitored one.
            self._unmonitored.append(symbol)
            logger.debug(
                "PositionMonitor: no exit config found, skipping",
                position_id=position_id,
                symbol=symbol,
            )
            return

        # 3. Fetch market context from Redis
        regime: int | None = None
        vix: float | None = None
        if self.redis is not None:
            try:
                raw_regime = await self.redis.get("market:regime")
                if raw_regime is not None:
                    regime = int(raw_regime)
            except Exception as exc:  # noqa: BLE001 — exits run context-blind without regime
                logger.debug("position monitor: regime read failed", error=str(exc))
            try:
                raw_vix = await self.redis.get("market:vix")
                if raw_vix is not None:
                    vix = float(raw_vix)
            except Exception as exc:  # noqa: BLE001 — exits run context-blind without vix
                logger.debug("position monitor: vix read failed", error=str(exc))

        # 4. Build context dict for exit strategies
        context = {
            "peak_price": exit_config.get("peak_price", current_price),
            "bars_held": exit_config.get("bars_held", 0),
            "atr_at_entry": exit_config.get("atr_at_entry"),
            "zscore": exit_config.get("zscore"),
            "regime": regime,
            "vix": vix,
        }

        # 5. Build CompositeExit and check
        try:
            from app.execution.position_exit import build_exit_strategy

            strategy_type = exit_config.get("strategy_type", "directional")
            risk_bucket = exit_config.get("risk_bucket", "directional")
            exit_params = {
                "stop_loss": exit_config.get("stop_loss"),
                "take_profit": exit_config.get("take_profit"),
            }

            composite = build_exit_strategy(strategy_type, risk_bucket, exit_params)
            triggered, reason = composite.should_exit(position, current_price, context)
        except Exception as exc:
            logger.error(
                "PositionMonitor: exit strategy check failed",
                symbol=symbol,
                error=str(exc),
            )
            return

        # 6. Update peak price tracking for trailing stops
        await self._update_peak_price(position_id, current_price)

        # 7. Increment bars_held
        if self.redis is not None:
            try:
                exit_config["bars_held"] = context["bars_held"] + 1
                peak = float(context.get("peak_price") or current_price)
                side = position.get("side", "long")
                if side == "long" and current_price > peak:
                    exit_config["peak_price"] = current_price
                elif side == "short" and current_price < peak:
                    exit_config["peak_price"] = current_price
                else:
                    exit_config["peak_price"] = peak
                await self.redis.set(
                    f"pos_exit:{position_id}",
                    json.dumps(exit_config),
                    ex=86400,
                )
            except Exception as exc:
                logger.warning("PositionMonitor: failed to update exit config", error=str(exc))

        # 8. Fire close order if triggered
        if triggered:
            logger.info(
                "PositionMonitor: exit triggered",
                symbol=symbol,
                reason=reason,
                current_price=current_price,
            )
            await self._close_position(position, reason or "exit_triggered")

    async def _close_position(self, position: dict, reason: str) -> None:
        """Submit a market sell/buy order to fully close the position."""
        symbol = position.get("symbol", "")
        qty = float(position.get("qty", position.get("quantity", 0)))
        side = position.get("side", "long")

        if not symbol or qty <= 0:
            return

        close_side = "sell" if side == "long" else "buy"

        try:
            if self.broker is not None:
                from app.brokers.base import OrderRequest

                close_req = OrderRequest(
                    symbol=symbol,
                    side=close_side,
                    order_type="market",
                    quantity=qty,
                    time_in_force="GTC",
                )
                result = await self.broker.place_order(close_req)
                logger.info(
                    "PositionMonitor: close order submitted",
                    symbol=symbol,
                    reason=reason,
                    qty=qty,
                    order_id=getattr(result, "broker_order_id", "?"),
                )
            else:
                logger.warning(
                    "PositionMonitor: no broker — cannot close position",
                    symbol=symbol,
                    reason=reason,
                )
        except Exception as exc:
            logger.error(
                "PositionMonitor: close order failed",
                symbol=symbol,
                reason=reason,
                error=str(exc),
            )
            return

        # Broadcast exit event via WebSocket
        try:
            from app.ws.manager import manager

            await manager.broadcast(
                "alerts",
                {
                    "type": "position_exit",
                    "symbol": symbol,
                    "reason": reason,
                    "close_side": close_side,
                    "qty": qty,
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as exc:
            logger.debug("PositionMonitor: WebSocket broadcast failed", error=str(exc))

        # Clean up Redis exit config for this position
        if self.redis is not None:
            position_id = position.get("id") or symbol
            try:
                await self.redis.delete(f"pos_exit:{position_id}")
            except Exception as exc:  # noqa: BLE001 — stale exit config lingers
                logger.debug("position monitor: exit-config cleanup failed",
                             position_id=position_id, error=str(exc))

    async def _update_peak_price(self, position_id: str, current_price: float) -> None:
        """Update trailing stop peak price in Redis."""
        if self.redis is None:
            return
        try:
            raw_exit = await self.redis.get(f"pos_exit:{position_id}")
            if not raw_exit:
                return
            exit_config = json.loads(raw_exit)
            stored_peak = float(exit_config.get("peak_price", current_price))
            # For long positions, peak is the maximum price seen
            # For short positions, peak is the minimum price seen
            # We update conservatively here (just max) — direction is handled
            # in _check_position_exits which has side context
            new_peak = max(stored_peak, current_price)
            exit_config["peak_price"] = new_peak
            await self.redis.set(
                f"pos_exit:{position_id}",
                json.dumps(exit_config),
                ex=86400,
            )
        except Exception as exc:
            logger.warning(
                "PositionMonitor: peak price update failed",
                position_id=position_id,
                error=str(exc),
            )
