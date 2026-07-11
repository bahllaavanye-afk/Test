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
            raise ReturnsUpdateError("Error updating returns") from exc

    async def check_order(self, request: OrderRequest) -> RiskDecision:
        """Gate every order through risk checks. Returns RiskDecision."""
        try:
            # 1. Global circuit breaker
            breaker_decision = self._check_global_breaker()
            if breaker_decision:
                return breaker_decision

            # 2. Arbitrage circuit breaker (if applicable)
            arb_decision = self._check_arb_breaker(request)
            if arb_decision:
                return arb_decision

            # 3. Equity verification
            equity_decision = self._verify_equity()
            if equity_decision:
                return equity_decision

            # 4. Position size cap
            size_decision = self._cap_position_size(request)
            if size_decision:
                return size_decision

            # 5. Correlation cluster limits
            cluster_decision = self._check_correlation_cluster(request)
            if cluster_decision:
                return cluster_decision

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

    def _check_global_breaker(self) -> RiskDecision | None:
        """Return a RiskDecision if the global breaker halts trading, otherwise None."""
        if self.global_breaker.is_halted:
            reason = (
                self.global_breaker.halt_reasons[-1]
                if self.global_breaker.halt_reasons
                else "unknown"
            )
            return RiskDecision(False, f"Global circuit breaker halted: {reason}")
        return None

    def _check_arb_breaker(self, request: OrderRequest) -> RiskDecision | None:
        """Return a RiskDecision if the arbitrage breaker halts trading for the request."""
        if request.risk_bucket == "arbitrage" and self.arb_breaker.is_halted:
            reason = (
                self.arb_breaker.halt_reasons[-1]
                if self.arb_breaker.halt_reasons
                else "unknown"
            )
            return RiskDecision(False, f"Arb circuit breaker halted: {reason}")
        return None

    def _verify_equity(self) -> RiskDecision | None:
        """Check equity readiness and positivity."""
        if not self._equity_confirmed:
            logger.warning(
                "risk.manager: using estimated equity — broker snapshot not yet received",
                estimated_equity=self._equity,
            )
        if self._equity <= 0:
            return RiskDecision(False, "equity is zero or negative — orders halted")
        return None

    def _cap_position_size(self, request: OrderRequest) -> RiskDecision | None:
        """Enforce max position percentage; return adjusted decision if capped."""
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
        return None

    def _check_correlation_cluster(self, request: OrderRequest) -> RiskDecision | None:
        """Validate that the order does not exceed cluster exposure limits."""
        if not self._clusters:
            return None
        price = request.limit_price if request.limit_price is not None else 100.0
        estimated_value = request.quantity * price
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
        return None

    def kelly_size(
        self,
        symbol: str,
        price: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
    ) -> int:
        """Calculate position size using the Kelly criterion."""
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
                "Kelly sizing failed",
                symbol=symbol,
                price=price,
                win_rate=win_rate,
                avg_win_pct=avg_win_pct,
                avg_loss_pct=avg_loss_pct,
                exc_type=type(exc).__name__,
                exc_msg=str(exc),
                exc_info=True,
            )
            raise KellySizingError("Error calculating Kelly size") from exc

    # Additional methods and logic may follow...