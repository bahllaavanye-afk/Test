"""
Continuous strategy runner: one asyncio task per (strategy, symbol) pair.
Scales to hundreds of concurrent strategy+symbol combinations.
"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pandas as pd

from app.redis_client import get_redis, price_cache
from app.risk.vol_targeting import vol_targeter
from app.services.agent_logger import agent_logger
from app.strategies import STRATEGY_REGISTRY
from app.utils.logging import logger
from app.ws.manager import manager

# Default paper-trading strategy configuration used when no active strategies are
# found in the database (e.g. fresh install, no DB yet).  Covers the major
# regime buckets so the runner produces signals immediately after boot.
DEFAULT_ACTIVE_STRATEGIES: list[dict] = [
    {
        "name": "momentum",
        "symbols": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"],
        "params": {},
        "tick_interval_seconds": 3600,
        "confidence_threshold": 0.60,
    },
    {
        "name": "mean_reversion",
        "symbols": ["SPY", "QQQ", "AAPL"],
        "params": {},
        "tick_interval_seconds": 3600,
        "confidence_threshold": 0.60,
    },
    {
        "name": "rsi_macd",
        "symbols": ["SPY", "TSLA", "GOOGL"],
        "params": {},
        "tick_interval_seconds": 1800,
        "confidence_threshold": 0.55,
    },
    {
        "name": "btc_eth_stat_arb",
        "symbols": ["BTC/USD"],
        "params": {},
        "tick_interval_seconds": 3600,
        "confidence_threshold": 0.60,
    },
]


# ── Item 4: Regime-Conditional Strategy Activation Matrix ───────────────────
# Regimes: 0=bear, 1=sideways, 2=bull (from HMM regime detector stored in Redis)
# Source: research into 2 Sigma / Renaissance regime-conditional allocation
STRATEGY_REGIME_MAP: dict[str, list[int]] = {
    "momentum":                 [2],        # bull only
    "cross_sectional_momentum": [2],        # bull only
    "mean_reversion":           [1],        # sideways only
    "vwap_reversion":           [1],        # sideways only
    "rsi_macd":                 [1, 2],     # sideways + bull
    "breakout":                 [2],        # bull only
    "supertrend":               [2],        # bull only
    "pairs_trading":            [0, 1, 2],  # all regimes (market-neutral)
    "btc_eth_stat_arb":         [0, 1, 2],  # all regimes
    "triangular_arb":           [0, 1, 2],  # all regimes
    "poly_binary_arb":          [0, 1, 2],  # all regimes
    "funding_rate_arb":         [0, 1, 2],  # all regimes
    "basis_carry":              [0, 1, 2],  # all regimes
    "vix_mean_reversion":       [0, 1],     # bear + sideways
    "liquidation_cascade_fade": [0],        # bear only
    "hmm_regime":               [0, 1, 2],  # always active (meta-strategy)
    # ── Research strategies ──────────────────────────────────────────────────
    "realized_vol_asymmetry":    [0, 1, 2],  # all regimes (skew predictor)
    "analyst_revision_momentum": [1, 2],     # sideways + bull (momentum factor)
    "on_chain_exchange_netflow": [0, 1, 2],  # all regimes (crypto OI signal)
    "vol_of_vol_timing":         [0, 1, 2],  # all regimes (vvix regime signal)
    # ── Equities intraday ────────────────────────────────────────────────────
    "opening_range_breakout":    [1, 2],
    "residual_momentum":         [1, 2],
    "idio_vol_anomaly":          [0, 1, 2],
    # ── Crypto ───────────────────────────────────────────────────────────────
    "crypto_adaptive_trend":     [1, 2],
    "mvrv_zscore_timing":        [0, 1, 2],
    "intraday_seasonality":      [0, 1, 2],
    # ── Options / vol ────────────────────────────────────────────────────────
    "gamma_exposure":            [0, 1, 2],
    "skew_arb":                  [0, 1, 2],
    "vrp_systematic":            [0, 1, 2],
    "dispersion_trading":        [0, 1, 2],
    "vol_term_structure":        [0, 1, 2],
    # ── Polymarket ───────────────────────────────────────────────────────────
    "polymarket_sentiment_momentum": [1, 2],
    "poly_calibration_arb":      [0, 1, 2],
    "poly_late_resolution":      [0, 1, 2],
    # ── Macro / FX ───────────────────────────────────────────────────────────
    "cross_asset_carry":         [0, 1, 2],
    "sector_rotation":           [1, 2],
    "time_series_momentum":      [1, 2],
    "intraday_fomc_momentum":    [0, 1, 2],
    "pead_sue":                  [1, 2],
    "multi_factor_equity":       [1, 2],
    # ── StatArb ──────────────────────────────────────────────────────────────
    "pca_stat_arb":              [0, 1, 2],
    "kalman_pairs":              [0, 1, 2],
    "stablecoin_depeg_arb":      [0, 1, 2],
}
DEFAULT_REGIMES = [0, 1, 2]


_regime_cache: dict[str, object] = {"value": None, "ts": 0.0}
_REGIME_CACHE_TTL = 30.0  # seconds — avoids Redis round-trip on every strategy tick


def _is_market_open(market_type: str) -> bool:
    """
    Returns True if the given market is currently open.
    - equity: NYSE hours 9:30-16:00 ET, Mon-Fri only
    - crypto/polymarket: always open (24/7)
    """
    if market_type in ("crypto", "polymarket"):
        return True
    # Equity market hours: 9:30-16:00 US/Eastern, Mon-Fri
    try:
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return open_time <= now_et <= close_time
    except Exception:
        # If timezone lookup fails, default to allowing execution
        return True


async def get_current_regime(redis_client) -> int | None:
    """Read current market regime (0=bear, 1=sideways, 2=bull) from Redis key 'market:regime'.
    Cached for 30 s to reduce Redis round-trips across concurrent strategy loops.
    """
    import time
    now = time.monotonic()
    if now - _regime_cache["ts"] < _REGIME_CACHE_TTL:
        return _regime_cache["value"]  # type: ignore[return-value]
    try:
        raw = await redis_client.get("market:regime")
        val = int(raw) if raw is not None else None
        _regime_cache["value"] = val
        _regime_cache["ts"] = now
        return val
    except Exception as exc:
        logger.debug("Failed to read market regime from Redis", error=str(exc))
    return _regime_cache["value"]  # type: ignore[return-value]


def _broker_interval(strategy_cls) -> str:
    """Map a strategy's tick_interval_seconds to an Alpaca timeframe string."""
    secs = getattr(strategy_cls, "tick_interval_seconds", 3600)
    if secs <= 300:
        return "5m"
    if secs <= 3600:
        return "1h"
    return "1d"


class ContinuousStrategyRunner:
    def __init__(self, broker, risk_manager=None):
        self.broker = broker
        self.risk_manager = risk_manager
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self, active_strategies: list[dict]) -> None:
        """
        active_strategies: list of {name, symbols, params, tick_interval_seconds, confidence_threshold}
        """
        self._running = True
        self._tasks = [
            asyncio.create_task(
                self._run_loop(s["name"], symbol, s.get("params", {}),
                               s.get("tick_interval_seconds", 60),
                               s.get("confidence_threshold", 0.6))
            )
            for s in active_strategies
            for symbol in s.get("symbols", [])
        ]
        logger.info("Strategy runner started", tasks=len(self._tasks))
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()

    async def _get_ohlcv(self, symbol: str, strategy_cls) -> pd.DataFrame | None:
        """Fetch OHLCV: try Redis cache first, fall back to broker REST call."""
        interval = _broker_interval(strategy_cls)
        exchange = "crypto" if "/" in symbol else "alpaca"

        # Try Redis cache (written by price_feed.py)
        cached = await price_cache.get_ohlcv(exchange, symbol, interval)
        if cached and len(cached) >= 30:
            df = pd.DataFrame(cached)
            df.rename(columns={"ts": "timestamp"}, inplace=True)
            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    df[col] = df[col].astype(float)
            return df

        # Fall back to broker REST
        if self.broker is None:
            return None
        try:
            end = datetime.now(UTC)
            limit = 500
            bars = await self.broker.get_historical(symbol, interval, limit=limit)
            if bars:
                return pd.DataFrame(bars)
        except Exception as e:
            logger.warning("OHLCV fetch failed", symbol=symbol, error=str(e))
        return None

    async def _run_loop(self, strategy_name: str, symbol: str, params: dict,
                        tick_interval: int, confidence_threshold: float) -> None:
        strategy_cls = STRATEGY_REGISTRY.get(strategy_name)
        if not strategy_cls:
            logger.error("Unknown strategy", name=strategy_name)
            return

        strategy = strategy_cls(params=params) if params else strategy_cls()
        logger.info("Strategy loop started", strategy=strategy_name, symbol=symbol)

        # Shared Redis client for regime checks
        _redis = get_redis()

        while self._running:
            try:
                # ── Item 4: Skip strategies in hostile regimes ──────────────
                current_regime = await get_current_regime(_redis)
                allowed_regimes = STRATEGY_REGIME_MAP.get(strategy_name, DEFAULT_REGIMES)
                if current_regime is not None and current_regime not in allowed_regimes:
                    logger.debug(
                        "Strategy skipped: hostile regime",
                        strategy=strategy_name,
                        regime=current_regime,
                        allowed=allowed_regimes,
                    )
                    await asyncio.sleep(tick_interval)
                    continue

                # ── Market hours enforcement ─────────────────────────────────
                market_type = getattr(strategy_cls, "market_type", "equity")
                if not _is_market_open(market_type):
                    logger.debug(
                        "Skipping %s — market closed for type=%s",
                        strategy_name,
                        market_type,
                    )
                    await asyncio.sleep(tick_interval)
                    continue

                df = await self._get_ohlcv(symbol, strategy_cls)

                if df is None or len(df) < 30:
                    if df is None:
                        logger.warning("No OHLCV data — price feed may be offline",
                                       strategy=strategy_name, symbol=symbol)
                    await asyncio.sleep(tick_interval)
                    continue

                # Standardize column name for close price
                if "close" not in df.columns and "c" in df.columns:
                    df = df.rename(columns={"c": "close", "o": "open", "h": "high", "l": "low", "v": "volume"})

                import time as _time
                _t0 = _time.monotonic()
                signal = await strategy.analyze(df, symbol)
                _dur_ms = int((_time.monotonic() - _t0) * 1000)
                if signal and signal.confidence >= confidence_threshold:
                    agent_logger.log_action_fire_and_forget(
                        action="run_strategy",
                        employee_id="strategy_runner",
                        agent_type="strategy",
                        input_summary=f"{strategy_name} on {symbol}, {len(df)} bars",
                        output_summary=f"signal={signal.side} conf={signal.confidence:.3f}",
                        duration_ms=_dur_ms,
                        status="ok",
                        strategy_name=strategy_name,
                        symbol=symbol,
                    )
                    alert = {
                        "type": "signal",
                        "strategy": strategy_name,
                        "symbol": symbol,
                        "side": signal.side,
                        "confidence": round(signal.confidence, 4),
                        "target_price": signal.target_price,
                        "stop_loss": signal.stop_loss,
                    }
                    await manager.broadcast("alerts", alert)
                    logger.info("Signal generated", **alert)

                    try:
                        from app.notifications.slack import slack
                        from app.notifications.tracker import tracker
                        tracker.record("signal_fired", "signal",
                                        f"{strategy_name} → {symbol} {signal.side} (conf={signal.confidence:.2f})")
                        await slack.notify_signal(strategy_name, symbol, signal.side,
                                                    signal.confidence, signal.target_price)
                    except Exception as notify_err:
                        logger.debug("Notification failed", error=str(notify_err))

                    # ── Submit order through risk-gated smart router ──────────
                    if self.broker is not None:
                        from app.brokers.base import OrderRequest
                        from app.execution.smart_router import SmartOrderRouter
                        router = SmartOrderRouter(
                            broker=self.broker,
                            risk_manager=self.risk_manager,
                        )
                        _vol_scalar = vol_targeter.get_scalar(f"{strategy_name}_{symbol}")
                        _base_qty = signal.metadata.get("quantity", 1)
                        order_req = OrderRequest(
                            symbol=symbol,
                            quantity=max(1, round(_base_qty * _vol_scalar)),
                            side=signal.side,
                            order_type=signal.metadata.get("order_type", "market"),
                            limit_price=signal.target_price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            risk_bucket=strategy.risk_bucket,
                        )
                        result = await router.execute(order_req, signal_price=signal.target_price)
                        if result:
                            logger.info("Order submitted", strategy=strategy_name, symbol=symbol,
                                        order_id=getattr(result, "order_id", "?"),
                                        side=signal.side, qty=order_req.quantity)

                            # Store exit config in Redis for position_monitor.py
                            try:
                                redis_client = get_redis()
                                if redis_client is not None:
                                    exit_config = {
                                        "strategy_name": strategy_name,
                                        "strategy_type": getattr(strategy_cls, "strategy_type", "manual"),
                                        "risk_bucket": getattr(strategy_cls, "risk_bucket", "directional"),
                                        "entry_price": signal.target_price,
                                        "stop_loss": signal.stop_loss,
                                        "take_profit": signal.take_profit,
                                        "peak_price": signal.target_price,
                                        "bars_held": 0,
                                        "stored_at": datetime.now(UTC).isoformat(),
                                    }
                                    await redis_client.set(
                                        f"pos_exit:{symbol}",
                                        json.dumps(exit_config),
                                        ex=86400,
                                    )
                            except Exception as _exit_cfg_exc:
                                logger.debug(
                                    "Failed to store exit config in Redis",
                                    symbol=symbol,
                                    error=str(_exit_cfg_exc),
                                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Catch all per-strategy exceptions so one broken strategy
                # does NOT kill the runner loop for other strategies.
                logger.error("Strategy loop error", strategy=strategy_name, symbol=symbol, error=str(e))
                agent_logger.log_action_fire_and_forget(
                    action="run_strategy",
                    employee_id="strategy_runner",
                    agent_type="strategy",
                    input_summary=f"{strategy_name} on {symbol}",
                    status="error",
                    error_message=str(e)[:200],
                    strategy_name=strategy_name,
                    symbol=symbol,
                )

            # Record daily return for volatility targeting scalar update
            try:
                if df is not None and len(df) >= 2 and "close" in df.columns:
                    closes = df["close"].values
                    if closes[-2] != 0:
                        daily_ret = float((closes[-1] - closes[-2]) / closes[-2])
                        vol_targeter.record_return(f"{strategy_name}_{symbol}", daily_ret)
            except Exception:
                pass

            await asyncio.sleep(tick_interval)


async def start_strategy_runner() -> None:
    """
    Factory coroutine registered as a supervised background task in main.py.

    1. Creates the Alpaca broker from settings when API keys are present.
       If ALPACA_API_KEY is missing, broker is None and the runner still ticks
       (strategies will skip the order submission path but keep generating signals
       whenever OHLCV data is available from Redis).
    2. Loads active strategies from the DB.  Falls back to DEFAULT_ACTIVE_STRATEGIES
       when the DB is empty or unavailable (graceful cold-start).
    3. Passes everything to ContinuousStrategyRunner.start() which spawns one
       asyncio Task per (strategy, symbol) pair.
    """
    from app.config import settings

    # ── Build broker (best-effort) ────────────────────────────────────────────
    broker = None
    risk_manager = None

    if settings.alpaca_api_key and settings.alpaca_secret_key:
        try:
            from app.brokers.alpaca import AlpacaBroker
            broker = AlpacaBroker(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
                paper=(settings.trading_mode != "live"),
            )
            logger.info(
                "Strategy runner: Alpaca broker connected",
                paper=(settings.trading_mode != "live"),
            )
        except Exception as exc:
            logger.warning(
                "Strategy runner: failed to create Alpaca broker — signals only, no orders",
                error=str(exc),
            )
    else:
        logger.warning(
            "Strategy runner: ALPACA_API_KEY not set — running in signal-only mode. "
            "Paper orders will be skipped until credentials are configured."
        )

    # ── Build risk manager (best-effort) ─────────────────────────────────────
    try:
        from app.risk.manager import RiskManager
        risk_manager = RiskManager()
    except Exception as exc:
        logger.warning("Strategy runner: RiskManager unavailable", error=str(exc))

    # ── Load active strategies from DB ────────────────────────────────────────
    active_strategies: list[dict] = []
    try:
        from sqlalchemy import select

        from app.database import AsyncSessionLocal
        from app.models.strategy import Strategy

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Strategy).where(Strategy.is_enabled == True)  # noqa: E712
            )
            db_strategies = result.scalars().all()

        if db_strategies:
            active_strategies = [
                {
                    "name": s.name,
                    "symbols": s.symbols if s.symbols else [],
                    "params": {},
                    "tick_interval_seconds": int(s.tick_interval_seconds),
                    "confidence_threshold": float(s.confidence_threshold),
                }
                for s in db_strategies
                if s.name in STRATEGY_REGISTRY
            ]
            logger.info(
                "Strategy runner: loaded strategies from DB",
                count=len(active_strategies),
            )
    except Exception as exc:
        logger.warning(
            "Strategy runner: could not load strategies from DB — using defaults",
            error=str(exc),
        )

    if not active_strategies:
        logger.info(
            "Strategy runner: no DB strategies found — using default paper-trading set",
            count=len(DEFAULT_ACTIVE_STRATEGIES),
        )
        active_strategies = DEFAULT_ACTIVE_STRATEGIES

    # ── Run ───────────────────────────────────────────────────────────────────
    runner = ContinuousStrategyRunner(broker=broker, risk_manager=risk_manager)
    await runner.start(active_strategies)
