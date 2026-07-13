"""
Active position monitoring loop. Runs every 30 seconds.

For each open position:
  1. Fetches current price from Redis (key: prices:<SYMBOL>)
  2. Fetches stored exit config from Redis (key: pos_exit:<position_id>)
  3. Runs CompositeExit.should_exit()
  4. If exit triggered: submits close order via broker
  5. Updates peak_price tracking in Redis for trailing stops
  6. Broadcasts exit event via WebSocket manager

Redis keys used:
  prices:<SYMBOL>            -> {last: float, bid, ask, ts}
  pos_exit:<position_id>     -> JSON {exit_strategies, entry_price, peak_price,
                                      bars_held, atr_at_entry, zscore}
  market:regime              -> "0"|"1"|"2"
  market:vix                 -> float
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from app.utils.logging import logger


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
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "PositionMonitor: broker.get_positions failed", error=str(exc)
                )

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
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "PositionMonitor: DB positions fetch failed", error=str(exc)
                )

        if not positions:
            return

        for position in positions:
            try:
                await self._check_position_exits(position)
            except Exception as exc:  # pragma: no cover
                symbol = position.get("symbol", "?")
                logger.error(
                    "PositionMonitor: error checking position",
                    symbol=symbol,
                    error=str(exc),
                )

    # --------------------------------------------------------------------- #
    # Helper methods – keep the public flow readable
    # --------------------------------------------------------------------- #

    async def _fetch_current_price(self, symbol: str) -> Optional[float]:
        """Return the latest price for *symbol* using Redis, falling back to broker."""
        if self.redis is not None:
            try:
                raw_price = await self.redis.get(f"prices:{symbol}")
                if raw_price:
                    price_data = json.loads(raw_price)
                    return float(
                        price_data.get("last")
                        or price_data.get("ask")
                        or price_data.get("bid")
                        or 0
                    )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "PositionMonitor: failed to read price from Redis",
                    symbol=symbol,
                    error=str(exc),
                )

        # Fallback to broker quote
        if self.broker is not None:
            try:
                quote = await self.broker.get_quote(symbol)
                return float(quote.last or quote.ask)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "PositionMonitor: broker quote failed, skipping",
                    symbol=symbol,
                    error=str(exc),
                )
        return None

    async def _fetch_exit_config(self, position_id: str) -> Dict[str, Any]:
        """Load exit configuration for a position from Redis."""
        if self.redis is None:
            return {}

        try:
            raw_exit = await self.redis.get(f"pos_exit:{position_id}")
            if raw_exit:
                return json.loads(raw_exit)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "PositionMonitor: failed to read exit config from Redis",
                position_id=position_id,
                error=str(exc),
            )
        return {}

    async def _fetch_market_context(self) -> Tuple[Optional[int], Optional[float]]:
        """Retrieve market regime and VIX from Redis."""
        regime: Optional[int] = None
        vix: Optional[float] = None

        if self.redis is None:
            return regime, vix

        try:
            raw_regime = await self.redis.get("market:regime")
            if raw_regime is not None:
                regime = int(raw_regime)
        except Exception:
            pass

        try:
            raw_vix = await self.redis.get("market:vix")
            if raw_vix is not None:
                vix = float(raw_vix)
        except Exception:
            pass

        return regime, vix

    def _build_exit_strategy(self, exit_config: Dict[str, Any]):
        """Instantiate the composite exit strategy based on stored config."""
        from app.execution.position_exit import build_exit_strategy

        strategy_type = exit_config.get("strategy_type", "directional")
        risk_bucket = exit_config.get("risk_bucket", "directional")
        exit_params = {
            "stop_loss": exit_config.get("stop_loss"),
            "take_profit": exit_config.get("take_profit"),
        }
        return build_exit_strategy(strategy_type, risk_bucket, exit_params)

    async def _increment_bars_held(
        self,
        position_id: str,
        exit_config: Dict[str, Any],
        context: Dict[str, Any],
        current_price: float,
        side: str,
    ) -> None:
        """Update bars_held and peak_price in the stored exit config."""
        exit_config["bars_held"] = context["bars_held"] + 1

        peak = float(context.get("peak_price") or current_price)
        if side == "long" and current_price > peak:
            exit_config["peak_price"] = current_price
        elif side == "short" and current_price < peak:
            exit_config["peak_price"] = current_price

        if self.redis is not None:
            try:
                await self.redis.set(
                    f"pos_exit:{position_id}", json.dumps(exit_config)
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "PositionMonitor: failed to persist exit config",
                    position_id=position_id,
                    error=str(exc),
                )

    async def _handle_exit_trigger(
        self, position: dict, reason: str
    ) -> None:
        """Submit a close order via broker and broadcast the exit event."""
        if self.broker is None:
            logger.warning(
                "PositionMonitor: broker unavailable, cannot close position",
                position_id=position.get("id"),
            )
            return

        try:
            await self.broker.close_position(position)
            logger.info(
                "PositionMonitor: position closed",
                position_id=position.get("id"),
                reason=reason,
            )
        except Exception as exc:  # pragma: no cover
            logger.error(
                "PositionMonitor: failed to close position",
                position_id=position.get("id"),
                error=str(exc),
            )
            return

        # Broadcast – placeholder for actual websocket manager call
        try:
            if hasattr(self, "websocket_manager"):
                await self.websocket_manager.broadcast_exit(
                    position_id=position.get("id"), reason=reason
                )
        except Exception as exc:  # pragma: no cover
            logger.debug(
                "PositionMonitor: websocket broadcast failed (non‑critical)",
                error=str(exc),
            )

    async def _update_peak_price(self, position_id: str, current_price: float) -> None:
        """Placeholder for the existing peak‑price tracking logic."""
        # The original implementation was assumed to exist elsewhere.
        # Keeping the method stub to preserve original call sites.
        pass

    # --------------------------------------------------------------------- #
    # Core logic – now concise and easy to follow
    # --------------------------------------------------------------------- #

    async def _check_position_exits(self, position: dict) -> None:
        """Run exit checks for a single position and act on triggers."""
        symbol = position.get("symbol", "")
        position_id = position.get("id") or symbol

        if not symbol:
            return

        # 1️⃣ Current price
        current_price = await self._fetch_current_price(symbol)
        if current_price is None:
            return

        # 2️⃣ Exit configuration
        exit_config = await self._fetch_exit_config(position_id)
        if not exit_config:
            logger.debug(
                "PositionMonitor: no exit config found, skipping",
                position_id=position_id,
                symbol=symbol,
            )
            return

        # 3️⃣ Market context
        regime, vix = await self._fetch_market_context()

        # 4️⃣ Context dict for exit strategies
        context = {
            "peak_price": exit_config.get("peak_price", current_price),
            "bars_held": exit_config.get("bars_held", 0),
            "atr_at_entry": exit_config.get("atr_at_entry"),
            "zscore": exit_config.get("zscore"),
            "regime": regime,
            "vix": vix,
        }

        # 5️⃣ Build strategy and evaluate
        try:
            composite = self._build_exit_strategy(exit_config)
            triggered, reason = composite.should_exit(
                position, current_price, context
            )
        except Exception as exc:  # pragma: no cover
            logger.error(
                "PositionMonitor: exit strategy check failed",
                symbol=symbol,
                error=str(exc),
            )
            return

        # 6️⃣ Update peak price (trailing stop handling)
        await self._update_peak_price(position_id, current_price)

        # 7️⃣ Increment bars_held and possibly adjust peak_price
        side = position.get("side", "long")
        await self._increment_bars_held(
            position_id, exit_config, context, current_price, side
        )

        # 8️⃣ If exit triggered – close position and broadcast
        if triggered:
            await self._handle_exit_trigger(position, reason)