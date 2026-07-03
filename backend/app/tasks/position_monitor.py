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
from typing import Any, Dict, List, Optional

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
        positions: List[dict] = []

        # Try broker first (live positions)
        if self.broker is not None:
            try:
                positions = await self.broker.get_positions()
            except Exception as exc:
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
            except Exception as exc:
                logger.warning(
                    "PositionMonitor: DB positions fetch failed", error=str(exc)
                )

        if not positions:
            return

        # ------------------------------------------------------------------
        # Optimized bulk fetches from Redis
        # ------------------------------------------------------------------
        symbols = [p.get("symbol", "") for p in positions]
        position_ids = [
            p.get("id") or p.get("symbol", "") for p in positions
        ]  # fallback to symbol if id missing

        price_map: Dict[str, float] = {}
        exit_map: Dict[str, dict] = {}

        if self.redis is not None:
            # Bulk fetch prices
            price_keys = [f"prices:{sym}" for sym in symbols]
            try:
                raw_prices = await self.redis.mget(*price_keys)  # type: ignore[arg-type]
                for sym, raw in zip(symbols, raw_prices):
                    if raw:
                        try:
                            data = json.loads(raw)
                            price = float(
                                data.get("last") or data.get("ask") or 0
                            )
                            if price:
                                price_map[sym] = price
                        except Exception:
                            continue
            except Exception as exc:
                logger.warning(
                    "PositionMonitor: bulk price fetch failed", error=str(exc)
                )

            # Bulk fetch exit configs
            exit_keys = [f"pos_exit:{pid}" for pid in position_ids]
            try:
                raw_exits = await self.redis.mget(*exit_keys)  # type: ignore[arg-type]
                for pid, raw in zip(position_ids, raw_exits):
                    if raw:
                        try:
                            exit_map[pid] = json.loads(raw)
                        except Exception:
                            continue
            except Exception as exc:
                logger.warning(
                    "PositionMonitor: bulk exit config fetch failed", error=str(exc)
                )

            # Fetch market context once
            try:
                raw_regime = await self.redis.get("market:regime")
                regime = int(raw_regime) if raw_regime is not None else None
            except Exception:
                regime = None

            try:
                raw_vix = await self.redis.get("market:vix")
                vix = float(raw_vix) if raw_vix is not None else None
            except Exception:
                vix = None
        else:
            regime = None
            vix = None

        # ------------------------------------------------------------------
        # Process each position
        # ------------------------------------------------------------------
        for position in positions:
            try:
                await self._process_position(
                    position,
                    price_map.get(position.get("symbol", "")),
                    exit_map.get(position.get("id") or position.get("symbol", "")),
                    regime,
                    vix,
                )
            except Exception as exc:
                symbol = position.get("symbol", "?")
                logger.error(
                    "PositionMonitor: error checking position",
                    symbol=symbol,
                    error=str(exc),
                )

    async def _process_position(
        self,
        position: dict,
        cached_price: Optional[float],
        cached_exit: Optional[dict],
        regime: Optional[int],
        vix: Optional[float],
    ) -> None:
        """Handle exit logic for a single position using cached Redis data."""
        symbol = position.get("symbol", "")
        position_id = position.get("id") or symbol

        if not symbol:
            return

        # --------------------------------------------------------------
        # 1. Resolve current price (cached -> broker fallback)
        # --------------------------------------------------------------
        current_price: Optional[float] = cached_price
        if current_price is None:
            if self.broker is not None:
                try:
                    quote = await self.broker.get_quote(symbol)
                    current_price = float(quote.last or quote.ask)
                except Exception as exc:
                    logger.warning(
                        "PositionMonitor: broker quote failed, skipping",
                        symbol=symbol,
                        error=str(exc),
                    )
            if current_price is None:
                return

        # --------------------------------------------------------------
        # 2. Resolve exit config (cached)
        # --------------------------------------------------------------
        exit_config = cached_exit or {}
        if not exit_config:
            logger.debug(
                "PositionMonitor: no exit config found, skipping",
                position_id=position_id,
                symbol=symbol,
            )
            return

        # --------------------------------------------------------------
        # 3. Build context for exit strategies
        # --------------------------------------------------------------
        context = {
            "peak_price": exit_config.get("peak_price", current_price),
            "bars_held": exit_config.get("bars_held", 0),
            "atr_at_entry": exit_config.get("atr_at_entry"),
            "zscore": exit_config.get("zscore"),
            "regime": regime,
            "vix": vix,
        }

        # --------------------------------------------------------------
        # 4. Evaluate composite exit strategy
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # 5. Update peak price tracking (trailing stops)
        # --------------------------------------------------------------
        await self._update_peak_price(position_id, current_price)

        # --------------------------------------------------------------
        # 6. Increment bars_held and possibly adjust peak_price
        # --------------------------------------------------------------
        if self.redis is not None:
            try:
                # Increment bars_held
                exit_config["bars_held"] = context["bars_held"] + 1

                # Adjust peak_price based on side
                side = position.get("side", "long")
                peak = float(context.get("peak_price") or current_price)
                if side == "long" and current_price > peak:
                    exit_config["peak_price"] = current_price
                elif side == "short" and current_price < peak:
                    exit_config["peak_price"] = current_price

                await self.redis.set(
                    f"pos_exit:{position_id}", json.dumps(exit_config)
                )
            except Exception as exc:
                logger.warning(
                    "PositionMonitor: failed to persist exit config",
                    position_id=position_id,
                    error=str(exc),
                )

        # --------------------------------------------------------------
        # 7. If exit triggered, close position and broadcast event
        # --------------------------------------------------------------
        if triggered:
            if self.broker is not None:
                try:
                    await self.broker.close_position(position_id)
                except Exception as exc:
                    logger.error(
                        "PositionMonitor: broker close_position failed",
                        position_id=position_id,
                        error=str(exc),
                    )
            # Broadcast (placeholder – actual implementation may differ)
            try:
                if hasattr(self, "ws_manager"):
                    await self.ws_manager.broadcast(
                        {
                            "type": "position_exit",
                            "position_id": position_id,
                            "symbol": symbol,
                            "reason": reason,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
            except Exception as exc:
                logger.warning(
                    "PositionMonitor: failed to broadcast exit event",
                    position_id=position_id,
                    error=str(exc),
                )

    async def _update_peak_price(self, position_id: str, price: float) -> None:
        """Utility to persist the latest peak price for a position."""
        if self.redis is None:
            return
        try:
            raw = await self.redis.get(f"pos_exit:{position_id}")
            if not raw:
                return
            cfg = json.loads(raw)
            cfg["peak_price"] = price
            await self.redis.set(f"pos_exit:{position_id}", json.dumps(cfg))
        except Exception as exc:
            logger.warning(
                "PositionMonitor: failed to update peak price",
                position_id=position_id,
                error=str(exc),
            )