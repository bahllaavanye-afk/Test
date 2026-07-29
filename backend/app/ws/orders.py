"""Real-time order status WebSocket endpoint with enhanced signal validation and confirmation."""
import json
import logging
import asyncio
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration constants
CONFIRMATION_WINDOW_SECONDS = 5  # Time window to require a duplicate signal for confirmation
INACTIVITY_TIMEOUT_SECONDS = 60  # Close the WS if no messages are received within this period
REQUIRED_SIGNAL_FIELDS = {"signal_id", "symbol", "entry_price", "confidence"}

# In‑memory store for pending confirmations
_pending_confirmations: dict[str, dict] = {}


def _is_valid_signal(signal: dict) -> bool:
    """Validate basic structure and business rules of an incoming signal."""
    missing = REQUIRED_SIGNAL_FIELDS - signal.keys()
    if missing:
        logger.debug("Signal missing required fields: %s", missing)
        return False

    # Basic numeric checks
    try:
        price = float(signal["entry_price"])
        confidence = float(signal["confidence"])
    except (ValueError, TypeError):
        logger.debug("Signal fields have invalid types: %s", signal)
        return False

    if price <= 0:
        logger.debug("Invalid entry_price (<=0): %s", price)
        return False
    if not (0.0 <= confidence <= 1.0):
        logger.debug("Confidence out of bounds [0,1]: %s", confidence)
        return False
    if confidence < 0.6:  # tighter entry condition per strategy notes
        logger.debug("Confidence below threshold (0.6): %s", confidence)
        return False

    return True


def _check_confirmation(signal_id: str, payload: dict) -> bool:
    """Require two identical signals within CONFIRMATION_WINDOW_SECONDS."""
    now = datetime.now(timezone.utc)

    prev = _pending_confirmations.get(signal_id)
    if prev:
        # Compare payloads (order of keys doesn't matter)
        if prev["data"] == payload:
            elapsed = (now - prev["timestamp"]).total_seconds()
            if elapsed <= CONFIRMATION_WINDOW_SECONDS:
                # Confirmation satisfied; clear entry
                del _pending_confirmations[signal_id]
                logger.debug(
                    "Signal %s confirmed after %.2f seconds", signal_id, elapsed
                )
                return True
        # Either payload differs or window expired – replace with latest
        _pending_confirmations[signal_id] = {"data": payload, "timestamp": now}
        return False
    else:
        # First occurrence – store and wait for confirmation
        _pending_confirmations[signal_id] = {"data": payload, "timestamp": now}
        return False


@router.websocket("/ws/orders")
async def orders_ws(websocket: WebSocket):
    topic = "orders"
    await manager.connect(websocket, topic)
    try:
        while True:
            try:
                # Enforce inactivity timeout
                raw_msg = await asyncio.wait_for(
                    websocket.receive_text(), timeout=INACTIVITY_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                logger.info(
                    "WebSocket inactive for %s seconds, closing connection.",
                    INACTIVITY_TIMEOUT_SECONDS,
                )
                break
            except Exception as exc:  # includes WebSocketDisconnect
                logger.warning("orders_ws receive error: %s", exc)
                break

            try:
                signal = json.loads(raw_msg)
            except json.JSONDecodeError:
                logger.warning("Received non‑JSON message: %s", raw_msg)
                continue

            if not _is_valid_signal(signal):
                logger.info("Discarded invalid signal: %s", signal.get("signal_id"))
                continue

            signal_id = str(signal["signal_id"])
            if not _check_confirmation(signal_id, signal):
                # Awaiting confirmation – do not forward yet
                continue

            # At this point the signal is valid and confirmed.
            # Forwarding logic can be added here; for now we simply log.
            logger.info("Confirmed signal ready for processing: %s", signal_id)

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, topic)