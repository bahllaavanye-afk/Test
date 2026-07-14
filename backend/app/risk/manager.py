"""
Real-time risk manager: Kelly sizing, correlation limits, circuit breakers.
All order requests pass through here before reaching the broker.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.brokers.base import OrderRequest
from app.risk.kelly import size_from_kelly
from app.risk.correlation import compute_correlation_clusters, check_cluster_limits
from app.risk.circuit_breaker import CircuitBreaker, BreakerState
from app.utils.logging import logger


# ----------------------------------------------------------------------
# Strategy entry/exit filter thresholds – can be tuned per regime
# ----------------------------------------------------------------------
ENTRY_SIGNAL_THRESHOLD: float = 0.70  # Minimum signal strength to accept a new position
MIN_VOLUME: int = 1_000               # Minimum daily volume required for entry
CONFIRMATION_REQUIRED: bool = True   # Whether a secondary confirmation flag is required
EXIT_SIGNAL_REQUIRED: bool = True    # Whether an explicit exit signal must be present for closing trades


class RiskManagerError(Exception):
    """Base exception for RiskManager related errors."""


class EquityUpdateError(RiskManagerError):
    """Raised when updating equity fails."""


class PositionsUpdateError(RiskManagerError):
    """Raised when updating positions fails."""


class ReturnsUpdateError(RiskManagerError):
    """Raised when updating returns history fails."""


class OrderCheckError(RiskManagerError):
    """Raised when order risk checking encounters an unexpected error."""


class KellySizingError(RiskManagerError):
    """Raised when Kelly sizing calculation fails."""


@dataclass
class RiskDecision:
    allowed: bool
    reason: str
    adjusted_quantity: float | None = None


class RiskManager:
    def __init__(
        self,
        max_position_pct: float = 0.05,
        max_drawdown_pct: float = 0.10,
        arb_drawdown_pct: float = 0.05,
        max_cluster_pct: float = 0.30,
        initial_equity: float = 100_000.0,
    ):
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_cluster_pct = max_cluster_pct

        # Seed with a conservative default so orders are not blocked during broker
        # cold-start. update_equity() replaces this with the real broker value.
        self._equity: float = initial_equity
        self._equity_confirmed: bool = False   # True once a real broker snapshot arrives
        self._positions: dict[str, float] = {}   # symbol → market value USD
        self._returns_history: pd.DataFrame = pd.DataFrame()
        self._clusters: dict[str, list[str]] = {}

        self.global_breaker = CircuitBreaker(
            name="global", max_drawdown_pct=max_drawdown_pct
        )
        self.arb_breaker = CircuitBreaker(
            name="arb", max_drawdown_pct=arb_drawdown_pct
        )

    # ------------------------------------------------------------------
    # Equity / position / returns updates
    # ------------------------------------------------------------------
    def update_equity(self, equity: float) -> None:
        try:
            if not isinstance(equity, (int, float)):
                raise TypeError(f"Equity must be numeric, got {type(equity)}")
            if equity < 0:
                raise ValueError("Equity cannot be negative")
            self._equity = float(equity)
            self._equity_confirmed = True
            self.global_breaker.update(self._equity)
        except Exception as exc:
            logger.error(
                "Failed to update equity",
                equity=equity,
                exc_type=type(exc).__name__,
                exc_msg=str(exc),
                exc_info=True,
            )
            raise EquityUpdateError("Error updating equity") from exc

    def update_positions(self, positions: list[dict]) -> None:
        try:
            if not isinstance(positions, list):
                raise TypeError("Positions must be a list of dicts")
            self._positions = {
                p["symbol"]: float(p.get("market_value", 0))
                for p in positions
                if "symbol" in p
            }
        except Exception as exc:
            logger.error(
                "Failed to update positions",
                positions=positions,
                exc_type=type(exc).__name__,
                exc_msg=str(exc),
                exc_info=True,
            )
            raise PositionsUpdateError("Error updating positions") from exc

    def update_returns(self, returns_df: pd.DataFrame) -> None:
        try:
            if not isinstance(returns_df, pd.DataFrame):
                raise TypeError("returns_df must be a pandas DataFrame")
            self._returns_history = returns_df
            if not returns_df.empty and len(returns_df) >= 20:
                self._clusters = compute_correlation_clusters(returns_df, threshold=0.70)
        except Exception as exc:
            logger.error(
                "Failed to update returns history",
                exc_type=type(exc).__name__,
                exc_msg=str(exc),
                exc_info=True,
            )
            raise ReturnsUpdateError("Error updating returns history") from exc

    # ------------------------------------------------------------------
    # Internal helper: entry filters
    # ------------------------------------------------------------------
    def _entry_filters_pass(self, request: OrderRequest) -> tuple[bool, str]:
        """
        Apply tightened entry conditions.
        Returns (True, "ok") if all filters pass, otherwise (False, reason).
        """
        # 1. Signal strength filter
        signal_strength = getattr(request, "signal_strength", None)
        if signal_strength is None:
            return False, "missing signal_strength attribute"
        if signal_strength < ENTRY_SIGNAL_THRESHOLD:
            return False, f"signal_strength {signal_strength:.2f} below threshold {ENTRY_SIGNAL_THRESHOLD:.2f}"

        # 2. Volume filter
        daily_volume = getattr(request, "daily_volume", None)
        if daily_volume is None:
            return False, "missing daily_volume attribute"
        if daily_volume < MIN_VOLUME:
            return False, f"daily_volume {daily_volume} below minimum {MIN_VOLUME}"

        # 3. Confirmation filter (optional secondary flag)
        if CONFIRMATION_REQUIRED:
            confirmation = getattr(request, "confirmation", None)
            if not confirmation:
                return False, "required confirmation flag not set"

        return True, "ok"

    # ------------------------------------------------------------------
    # Internal helper: exit validation
    # ------------------------------------------------------------------
    def _exit_validation(self, request: OrderRequest) -> tuple[bool, str]:
        """
        For closing orders, ensure an explicit exit signal is present.
        """
        if request.side.lower() == "sell" or request.side.lower() == "close":
            exit_signal = getattr(request, "exit_signal", None)
            if EXIT_SIGNAL_REQUIRED and not exit_signal:
                return False, "exit_signal required for closing position"
        return True, "ok"

    # ------------------------------------------------------------------
    # Core order risk checks
    # ------------------------------------------------------------------
    async def check_order(self, request: OrderRequest) -> RiskDecision:
        """Gate every order through risk checks. Returns RiskDecision."""
        try:
            if self.global_breaker.is_halted:
                reason = (
                    self.global_breaker.halt_reasons[-1]
                    if self.global_breaker.halt_reasons
                    else "unknown"
                )
                return RiskDecision(False, f"Global circuit breaker halted: {reason}")

            if request.risk_bucket == "arbitrage" and self.arb_breaker.is_halted:
                reason = (
                    self.arb_breaker.halt_reasons[-1]
                    if self.arb_breaker.halt_reasons
                    else "unknown"
                )
                return RiskDecision(False, f"Arb circuit breaker halted: {reason}")

            if not self._equity_confirmed:
                logger.warning(
                    "risk.manager: using estimated equity — broker snapshot not yet received",
                    estimated_equity=self._equity,
                )
            if self._equity <= 0:
                return RiskDecision(False, "equity is zero or negative — orders halted")

            # ----------------------------------------------------------------
            # Entry / exit validation
            # ----------------------------------------------------------------
            # Exit validation may reject a close order lacking proper signal.
            exit_ok, exit_reason = self._exit_validation(request)
            if not exit_ok:
                return RiskDecision(False, f"Exit validation failed: {exit_reason}")

            # Entry filters are only applied to opening orders.
            if request.side.lower() in ("buy", "long"):
                entry_ok, entry_reason = self._entry_filters_pass(request)
                if not entry_ok:
                    return RiskDecision(False, f"Entry filter failed: {entry_reason}")

            # ----------------------------------------------------------------
            # Position size cap
            # ----------------------------------------------------------------
            price = request.limit_price if request.limit_price is not None else 100.0
            if price == 0:
                raise ZeroDivisionError("limit_price is zero, cannot compute position size")
            estimated_value = request.quantity * price
            max_allowed = self._equity * self.max_position_pct
            if estimated_value > max_allowed:
                adj_qty = max_allowed / price
                logger.warning(
                    "Position size capped",
                    symbol=request.symbol,
                    original=request.quantity,
                    adjusted=adj_qty,
                )
                return RiskDecision(True, "size capped", adj_qty)

            # ----------------------------------------------------------------
            # Correlation cluster check
            # ----------------------------------------------------------------
            if self._clusters:
                allowed, reason = check_cluster_limits(
                    request.symbol,
                    estimated_value,
                    self._positions,
                    self._clusters,
                    self.max_cluster_pct,
                    self._equity,
                )
                if not allowed:
                    return RiskDecision(False, reason)

            return RiskDecision(True, "ok", request.quantity)
        except RiskManagerError:
            # Propagate known risk manager errors without extra logging
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error during order risk check",
                request_id=getattr(request, "id", None),
                symbol=getattr(request, "symbol", None),
                exc_type=type(exc).__name__,
                exc_msg=str(exc),
                exc_info=True,
            )
            raise OrderCheckError("Error checking order risk") from exc

    # ------------------------------------------------------------------
    # Kelly sizing helper
    # ------------------------------------------------------------------
    def kelly_size(
        self,
        symbol: str,
        price: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
    ) -> int:
        try:
            return size_from_kelly(
                equity=self._equity,
                win_rate=win_rate,
                avg_win_pct=avg_win_pct,
                avg_loss_pct=avg_loss_pct,
                price=price,
                symbol=symbol,
            )
        except Exception as exc:
            logger.error(
                "Kelly sizing calculation failed",
                symbol=symbol,
                price=price,
                win_rate=win_rate,
                avg_win_pct=avg_win_pct,
                avg_loss_pct=avg_loss_pct,
                exc_type=type(exc).__name__,
                exc_msg=str(exc),
                exc_info=True,
            )
            raise KellySizingError("Kelly sizing error") from exc