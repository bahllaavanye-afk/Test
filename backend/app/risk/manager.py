"""
Real-time risk manager: Kelly sizing, correlation limits, circuit breakers.
All order requests pass through here before reaching the broker.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

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
    """
    Result of a risk check for an order.

    Attributes
    ----------
    allowed: bool
        Whether the order is permitted to proceed.
    reason: str
        Human‑readable explanation for the decision.
    adjusted_quantity: float | None
        If the order quantity was modified (e.g., size capped), the new quantity.
        ``None`` indicates no adjustment.
    """
    allowed: bool
    reason: str
    adjusted_quantity: float | None = None


class RiskManager:
    """
    Centralised risk manager handling equity, position limits, correlation clusters,
    circuit breakers, and Kelly‑criterion sizing.

    Parameters
    ----------
    max_position_pct: float, default 0.05
        Maximum allowed position size as a fraction of equity.
    max_drawdown_pct: float, default 0.10
        Maximum allowed drawdown for the global circuit breaker.
    arb_drawdown_pct: float, default 0.05
        Maximum allowed drawdown for the arbitrage circuit breaker.
    max_cluster_pct: float, default 0.30
        Maximum exposure to any correlation cluster as a fraction of equity.
    initial_equity: float, default 100_000.0
        Starting equity used until a broker snapshot arrives.
    """

    def __init__(
        self,
        max_position_pct: float = 0.05,
        max_drawdown_pct: float = 0.10,
        arb_drawdown_pct: float = 0.05,
        max_cluster_pct: float = 0.30,
        initial_equity: float = 100_000.0,
    ) -> None:
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_cluster_pct = max_cluster_pct

        # Seed with a conservative default so orders are not blocked during broker
        # cold-start. update_equity() replaces this with the real broker value.
        self._equity: float = initial_equity
        self._equity_confirmed: bool = False  # True once a real broker snapshot arrives
        self._positions: Dict[str, float] = {}  # symbol → market value USD
        self._returns_history: pd.DataFrame = pd.DataFrame()
        self._clusters: Dict[str, List[str]] = {}

        self.global_breaker = CircuitBreaker(name="global", max_drawdown_pct=max_drawdown_pct)
        self.arb_breaker = CircuitBreaker(name="arb", max_drawdown_pct=arb_drawdown_pct)

    def update_equity(self, equity: float) -> None:
        """
        Update the manager's view of total equity.

        Parameters
        ----------
        equity: float
            New equity value; must be non‑negative.

        Raises
        ------
        EquityUpdateError
            If the provided value is invalid or an unexpected error occurs.
        """
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

    def update_positions(self, positions: List[Dict[str, Any]]) -> None:
        """
        Refresh the internal position map.

        Parameters
        ----------
        positions: list[dict]
            Each dict should contain a ``symbol`` key and optionally a
            ``market_value`` key (defaults to 0).

        Raises
        ------
        PositionsUpdateError
            If the input is malformed or an unexpected error occurs.
        """
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
        """
        Store a DataFrame of historical returns and recompute correlation clusters.

        Parameters
        ----------
        returns_df: pandas.DataFrame
            DataFrame where rows are timestamps and columns are symbols.

        Raises
        ------
        ReturnsUpdateError
            If the input is not a DataFrame or an unexpected error occurs.
        """
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
        """
        Evaluate an order against all active risk controls.

        Parameters
        ----------
        request: OrderRequest
            The incoming order to be validated.

        Returns
        -------
        RiskDecision
            Decision object indicating whether the order may proceed and
            any quantity adjustments.

        Raises
        ------
        OrderCheckError
            For unexpected failures not covered by specific risk exceptions.
        """
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

            # Position size cap
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

            # Correlation cluster check
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

    def kelly_size(
        self,
        symbol: str,
        price: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
    ) -> int:
        """
        Compute an order quantity using the Kelly criterion.

        Parameters
        ----------
        symbol: str
            Ticker symbol for which the sizing is performed.
        price: float
            Current price of the instrument.
        win_rate: float
            Historical probability of a winning trade (0‑1).
        avg_win_pct: float
            Average profit as a fraction of the trade size.
        avg_loss_pct: float
            Average loss as a fraction of the trade size.

        Returns
        -------
        int
            Position size (number of shares/contracts) suggested by Kelly sizing.

        Raises
        ------
        KellySizingError
            If the Kelly calculation fails.
        """
        try:
            size = size_from_kelly(
                equity=self._equity,
                win_rate=win_rate,
                avg_win_pct=avg_win_pct,
                avg_loss_pct=avg_loss_pct,
            )
            # Convert to integer number of units; rounding down is conservative.
            return int(size)
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
            raise KellySizingError("Error calculating Kelly size") from exc