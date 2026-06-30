"""
Real-time risk manager: Kelly sizing, correlation limits, circuit breakers.
All order requests pass through here before reaching the broker.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import pandas as pd

from app.brokers.base import OrderRequest
from app.risk.kelly import size_from_kelly
from app.risk.correlation import compute_correlation_clusters, check_cluster_limits
from app.risk.circuit_breaker import CircuitBreaker, BreakerState
from app.utils.logging import logger


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
        # Validation of initialization parameters
        if not (0 < max_position_pct <= 1):
            raise ValueError("max_position_pct must be between 0 (exclusive) and 1 (inclusive).")
        if not (0 < max_drawdown_pct <= 1):
            raise ValueError("max_drawdown_pct must be between 0 (exclusive) and 1 (inclusive).")
        if not (0 < arb_drawdown_pct <= 1):
            raise ValueError("arb_drawdown_pct must be between 0 (exclusive) and 1 (inclusive).")
        if not (0 < max_cluster_pct <= 1):
            raise ValueError("max_cluster_pct must be between 0 (exclusive) and 1 (inclusive).")
        if initial_equity <= 0:
            raise ValueError("initial_equity must be a positive number.")

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
        """Update the manager's view of total equity."""
        if not isinstance(equity, (int, float)):
            raise ValueError("equity must be a numeric type.")
        if equity <= 0:
            raise ValueError("equity must be a positive number.")
        self._equity = float(equity)
        self._equity_confirmed = True
        self.global_breaker.update(self._equity)

    def update_positions(self, positions: Sequence[Mapping[str, Any]]) -> None:
        """Refresh the current position dictionary from a list of position dicts."""
        if not isinstance(positions, (list, tuple)):
            raise ValueError("positions must be a list or tuple of dictionaries.")
        new_positions: dict[str, float] = {}
        for idx, p in enumerate(positions):
            if not isinstance(p, Mapping):
                raise ValueError(f"position at index {idx} is not a mapping.")
            symbol = p.get("symbol")
            if not isinstance(symbol, str) or not symbol:
                raise ValueError(f"position at index {idx} missing a valid 'symbol'.")
            market_value = float(p.get("market_value", 0))
            if market_value < 0:
                raise ValueError(f"market_value for symbol '{symbol}' cannot be negative.")
            new_positions[symbol] = market_value
        self._positions = new_positions

    def update_returns(self, returns_df: pd.DataFrame) -> None:
        """Update the historical returns DataFrame used for correlation clustering."""
        if not isinstance(returns_df, pd.DataFrame):
            raise ValueError("returns_df must be a pandas DataFrame.")
        self._returns_history = returns_df
        if not returns_df.empty and len(returns_df) >= 20:
            self._clusters = compute_correlation_clusters(returns_df, threshold=0.70)

    async def check_order(self, request: OrderRequest) -> RiskDecision:
        """Gate every order through risk checks. Returns RiskDecision."""
        if not isinstance(request, OrderRequest):
            raise ValueError("request must be an instance of OrderRequest.")
        if not isinstance(request.symbol, str) or not request.symbol:
            raise ValueError("request.symbol must be a non-empty string.")
        if request.quantity <= 0:
            raise ValueError("request.quantity must be a positive number.")
        if request.limit_price is not None and request.limit_price <= 0:
            raise ValueError("request.limit_price must be positive if provided.")

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
        estimated_value = request.quantity * (request.limit_price or 100)
        max_allowed = self._equity * self.max_position_pct
        if estimated_value > max_allowed:
            adj_qty = max_allowed / (request.limit_price or 100)
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

    def kelly_size(
        self,
        symbol: str,
        price: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
    ) -> int:
        """Calculate position size based on Kelly criterion."""
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non-empty string.")
        if price <= 0:
            raise ValueError("price must be a positive number.")
        if not (0 <= win_rate <= 1):
            raise ValueError("win_rate must be between 0 and 1 inclusive.")
        if avg_win_pct < 0:
            raise ValueError("avg_win_pct cannot be negative.")
        if avg_loss_pct < 0:
            raise ValueError("avg_loss_pct cannot be negative.")

        return size_from_kelly(
            equity=self._equity,
            win_rate=win_rate,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            price=price,
            max_pct=self.max_position_pct,
        )