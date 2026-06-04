"""Bot evaluation engine — checks triggers, conditions, and executes actions."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, time as dtime
from typing import Any

import pandas as pd
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.bot import Bot
from app.schemas.bot import ConditionConfig, ActionConfig, ExitRuleConfig

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Return the most recent RSI value."""
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def compute_sma(prices: pd.Series, period: int = 50) -> float:
    """Return the most recent SMA value."""
    if len(prices) < period:
        return float(prices.mean())
    return float(prices.iloc[-period:].mean())


def compute_ema(prices: pd.Series, period: int = 20) -> float:
    """Return the most recent EMA value."""
    if len(prices) < period:
        return float(prices.mean())
    return float(prices.ewm(span=period, adjust=False).mean().iloc[-1])


def compute_macd(prices: pd.Series) -> tuple[float, float]:
    """Return (macd_line, signal_line) most recent values."""
    if len(prices) < 26:
        return 0.0, 0.0
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])


def compute_bb(prices: pd.Series, period: int = 20) -> tuple[float, float, float]:
    """Return (upper, middle, lower) Bollinger Bands."""
    if len(prices) < period:
        mean = float(prices.mean())
        return mean, mean, mean
    window = prices.iloc[-period:]
    mid = float(window.mean())
    std = float(window.std())
    return mid + 2 * std, mid, mid - 2 * std


def _compare(a: float, op: str, b: float) -> bool:
    if op == "<":
        return a < b
    if op == ">":
        return a > b
    if op == "<=":
        return a <= b
    if op == ">=":
        return a >= b
    if op == "==":
        return abs(a - b) < 1e-9
    if op == "!=":
        return abs(a - b) >= 1e-9
    return False


# ---------------------------------------------------------------------------
# Condition evaluator
# ---------------------------------------------------------------------------

def evaluate_condition(cond: ConditionConfig, data: pd.DataFrame, current_price: float) -> bool:  # noqa: C901
    """Evaluate a single condition against the data and current price."""
    close = data["close"] if "close" in data.columns else pd.Series([current_price])

    ctype = cond.type

    if ctype == "indicator":
        ind = (cond.indicator or "").lower()
        if ind == "rsi":
            val = compute_rsi(close, cond.period)
        elif ind == "sma":
            val = compute_sma(close, cond.period)
        elif ind == "ema":
            val = compute_ema(close, cond.period)
        elif ind == "bb":
            upper, mid, lower = compute_bb(close)
            op = cond.operator
            if op == "price_below_lower":
                return current_price < lower
            if op == "price_above_upper":
                return current_price > upper
            return False
        elif ind == "macd":
            macd_val, signal_val = compute_macd(close)
            op = cond.operator
            if op == "bullish_cross":
                # Check if MACD crossed above signal in last two bars
                if len(close) < 28:
                    return False
                macd_prev, sig_prev = compute_macd(close.iloc[:-1])
                return (macd_prev <= sig_prev) and (macd_val > signal_val)
            if op == "bearish_cross":
                if len(close) < 28:
                    return False
                macd_prev, sig_prev = compute_macd(close.iloc[:-1])
                return (macd_prev >= sig_prev) and (macd_val < signal_val)
            # fall-through numeric compare using macd_val
            val = macd_val
        elif ind == "ema_cross":
            op = cond.operator
            if len(close) < 51:
                return False
            ema20_now = compute_ema(close, 20)
            ema50_now = compute_ema(close, 50)
            ema20_prev = compute_ema(close.iloc[:-1], 20)
            ema50_prev = compute_ema(close.iloc[:-1], 50)
            if op == "bullish_cross":
                return (ema20_prev <= ema50_prev) and (ema20_now > ema50_now)
            if op == "bearish_cross":
                return (ema20_prev >= ema50_prev) and (ema20_now < ema50_now)
            return False
        else:
            logger.warning("Unknown indicator", indicator=ind)
            return False

        # Numeric compare
        if cond.value is None:
            return False
        return _compare(val, cond.operator, cond.value)

    elif ctype == "price_vs_ma":
        period = cond.ma_period or 50
        ma = compute_sma(close, period)
        return _compare(current_price, cond.operator, ma)

    elif ctype == "pnl":
        # Compare day's return as pct vs threshold
        if len(close) < 2:
            return False
        day_pnl_pct = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
        threshold = cond.pnl_pct if cond.pnl_pct is not None else 0.0
        return _compare(day_pnl_pct, cond.operator, threshold)

    elif ctype == "time_window":
        # Check current UTC time converted to ET (UTC-5 approx, ignoring DST for simplicity)
        now_et = datetime.now(timezone.utc).replace(tzinfo=None)
        # Approximate ET = UTC - 5 hours
        from datetime import timedelta
        now_et = (datetime.now(timezone.utc) - timedelta(hours=5)).time()
        try:
            start = dtime(*[int(x) for x in (cond.start_time or "09:30").split(":")])
            end = dtime(*[int(x) for x in (cond.end_time or "16:00").split(":")])
            return start <= now_et <= end
        except Exception:
            return True

    elif ctype == "position_exists":
        # We don't track real positions in this simplified engine; return True
        return True

    elif ctype == "no_position":
        # Simplified: assume no position
        return True

    return False


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _map_crypto_symbol(symbol: str) -> str:
    """Map exchange crypto symbols to yfinance format."""
    mapping = {
        "BTCUSDT": "BTC-USD",
        "ETHUSDT": "ETH-USD",
        "SOLUSDT": "SOL-USD",
        "BNBUSDT": "BNB-USD",
        "ADAUSDT": "ADA-USD",
        "XRPUSDT": "XRP-USD",
        "DOTUSDT": "DOT-USD",
        "AVAXUSDT": "AVAX-USD",
        "MATICUSDT": "MATIC-USD",
        "LTCUSDT": "LTC-USD",
    }
    if symbol in mapping:
        return mapping[symbol]
    # Generic USDT pair
    if symbol.endswith("USDT"):
        return symbol[:-4] + "-USD"
    return symbol


async def _fetch_ohlcv(symbol: str, market_type: str) -> pd.DataFrame:
    """Fetch OHLCV data: try Redis cache first, then yfinance fallback."""
    # Try Redis
    try:
        from app.redis_client import price_cache
        raw = await price_cache.get(f"ohlcv:{symbol}:1d")
        if raw:
            rows = json.loads(raw)
            if rows and len(rows) >= 20:
                df = pd.DataFrame(rows)
                if "close" in df.columns:
                    return df
    except Exception as e:
        logger.debug("Redis OHLCV fetch failed", symbol=symbol, error=str(e))

    # yfinance fallback
    try:
        import yfinance as yf
        yf_symbol = symbol
        if market_type == "crypto":
            yf_symbol = _map_crypto_symbol(symbol)
        df = yf.download(yf_symbol, period="3mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.columns = [c.lower() for c in df.columns]
        df = df.reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning("yfinance fallback failed", symbol=symbol, error=str(e))
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


# ---------------------------------------------------------------------------
# BotResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class BotResult:
    fired: bool
    reason: str
    signal: str  # "buy" | "sell" | "hold" | "alert"
    orders_created: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BotEngine
# ---------------------------------------------------------------------------

class BotEngine:
    """Evaluates a bot's trigger/conditions/action."""

    async def evaluate(self, bot: Bot, db: AsyncSession) -> BotResult:
        """
        1. Fetch recent OHLCV data for bot.symbol
        2. Compute indicators needed by conditions
        3. Evaluate all conditions (AND/OR based on condition_logic)
        4. If conditions pass → execute action (paper order)
        5. Update bot stats in DB
        Returns BotResult.
        """
        try:
            return await self._evaluate_inner(bot, db)
        except Exception as exc:
            logger.error("BotEngine evaluation failed", bot_id=bot.id, error=str(exc))
            result = BotResult(fired=False, reason=f"Error: {exc}", signal="hold")
            await self._update_bot_stats(bot, db, result)
            return result

    async def _evaluate_inner(self, bot: Bot, db: AsyncSession) -> BotResult:
        # Fetch price data
        df = await _fetch_ohlcv(bot.symbol, bot.market_type)
        current_price = float(df["close"].iloc[-1]) if not df.empty and "close" in df.columns else 0.0

        # Parse conditions and action from JSON (they're stored as plain dicts)
        raw_conditions: list[dict] = bot.conditions or []
        conditions = [ConditionConfig(**c) for c in raw_conditions]

        # Evaluate conditions
        condition_results: list[bool] = []
        for cond in conditions:
            try:
                passed = evaluate_condition(cond, df, current_price)
                condition_results.append(passed)
            except Exception as exc:
                logger.warning("Condition evaluation error", bot_id=bot.id, error=str(exc))
                condition_results.append(False)

        # Apply condition logic
        logic = (bot.condition_logic or "ALL").upper()
        if not condition_results:
            conditions_passed = True  # no conditions = always fire
        elif logic == "ANY":
            conditions_passed = any(condition_results)
        else:  # ALL
            conditions_passed = all(condition_results)

        if not conditions_passed:
            result = BotResult(
                fired=False,
                reason=f"Conditions not met ({logic}: {condition_results})",
                signal="hold",
            )
            await self._update_bot_stats(bot, db, result)
            return result

        # Execute action
        action_dict: dict = bot.action or {}
        action = ActionConfig(**action_dict)
        orders_created: list[str] = []
        signal = "hold"

        if action.type in ("open_long", "open_short"):
            signal = "buy" if action.type == "open_long" else "sell"
            order_id = await self._create_paper_order(bot, action, current_price, signal, db)
            if order_id:
                orders_created.append(order_id)
            reason = f"Action fired: {action.type} {bot.symbol} @ {current_price:.4f}"
        elif action.type == "close_position":
            signal = "sell"
            reason = f"Close position: {bot.symbol} @ {current_price:.4f}"
        elif action.type == "send_alert":
            signal = "alert"
            msg = action.alert_message or "Bot alert triggered"
            logger.info("Bot alert", bot_id=bot.id, bot_name=bot.name, message=msg, symbol=bot.symbol)
            reason = f"Alert sent: {msg}"
        elif action.type == "reduce_position":
            signal = "sell"
            reason = f"Reduce position by {action.reduce_by_pct}%"
        else:
            signal = "hold"
            reason = f"Unknown action: {action.type}"

        result = BotResult(
            fired=True,
            reason=reason,
            signal=signal,
            orders_created=orders_created,
            details={
                "price": current_price,
                "conditions": condition_results,
                "logic": logic,
            },
        )
        await self._update_bot_stats(bot, db, result)
        return result

    async def _create_paper_order(
        self,
        bot: Bot,
        action: ActionConfig,
        current_price: float,
        side: str,
        db: AsyncSession,
    ) -> str | None:
        """Create a paper Order record in the DB."""
        try:
            from app.models.order import Order
            from sqlalchemy import select

            # Find or use the bot's account_id; fall back to no account
            account_id = bot.account_id

            order = Order(
                id=str(uuid.uuid4()),
                account_id=account_id or "paper",
                broker_order_id="paper",
                symbol=bot.symbol,
                side=side,
                order_type="market",
                quantity=None,  # size_pct-based, not fixed qty
                status="paper",
                raw_payload={
                    "bot_id": bot.id,
                    "bot_name": bot.name,
                    "size_pct": action.size_pct,
                    "stop_loss_pct": action.stop_loss_pct,
                    "take_profit_pct": action.take_profit_pct,
                    "trailing_stop_pct": action.trailing_stop_pct,
                    "entry_price": current_price,
                },
                take_profit_price=(
                    current_price * (1 + action.take_profit_pct / 100)
                    if action.take_profit_pct and side == "buy"
                    else current_price * (1 - action.take_profit_pct / 100)
                    if action.take_profit_pct and side == "sell"
                    else None
                ),
                stop_loss_price=(
                    current_price * (1 - action.stop_loss_pct / 100)
                    if action.stop_loss_pct and side == "buy"
                    else current_price * (1 + action.stop_loss_pct / 100)
                    if action.stop_loss_pct and side == "sell"
                    else None
                ),
                trailing_stop_pct=action.trailing_stop_pct,
                notional=None,
            )

            # Only add to DB if we have a real account_id
            if account_id:
                db.add(order)
                await db.flush()
                logger.info(
                    "Paper order created",
                    bot_id=bot.id,
                    order_id=order.id,
                    symbol=bot.symbol,
                    side=side,
                    price=current_price,
                )
            else:
                logger.info(
                    "Paper order (no account, not persisted)",
                    bot_id=bot.id,
                    symbol=bot.symbol,
                    side=side,
                    price=current_price,
                )
            return order.id
        except Exception as exc:
            logger.error("Failed to create paper order", bot_id=bot.id, error=str(exc))
            return None

    async def _update_bot_stats(self, bot: Bot, db: AsyncSession, result: BotResult) -> None:
        """Persist run stats back to the Bot row."""
        try:
            from sqlalchemy import select

            bot.run_count = (bot.run_count or 0) + 1
            bot.last_run_at = datetime.now(timezone.utc)
            bot.last_signal = result.signal
            bot.last_result = {
                "fired": result.fired,
                "reason": result.reason,
                "orders": result.orders_created,
                "details": result.details,
            }
            await db.commit()
        except Exception as exc:
            logger.error("Failed to update bot stats", bot_id=bot.id, error=str(exc))
            try:
                await db.rollback()
            except Exception:
                pass
