"""
Real-time risk manager: Kelly sizing, correlation limits, circuit breakers.
All order requests pass through here before reaching the broker.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

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
    ):
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_cluster_pct = max_cluster_pct

        self._equity: float = 0.0
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
        self._equity = equity
        self.global_breaker.update(equity)

    def update_positions(self, positions: list[dict]) -> None:
        self._positions = {p["symbol"]: float(p.get("market_value", 0)) for p in positions}

    def update_returns(self, returns_df: pd.DataFrame) -> None:
        self._returns_history = returns_df
        if not returns_df.empty and len(returns_df) >= 20:
            self._clusters = compute_correlation_clusters(returns_df, threshold=0.70)

    async def check_order(self, request: OrderRequest) -> RiskDecision:
        """Gate every order through risk checks. Returns RiskDecision."""
        if self.global_breaker.is_halted:
            reason = self.global_breaker.halt_reasons[-1] if self.global_breaker.halt_reasons else "circuit breaker tripped"
            return RiskDecision(False, f"Global circuit breaker halted: {reason}")

        if request.risk_bucket == "arbitrage" and self.arb_breaker.is_halted:
            reason = self.arb_breaker.halt_reasons[-1] if self.arb_breaker.halt_reasons else "arb circuit breaker tripped"
            return RiskDecision(False, f"Arb circuit breaker halted: {reason}")

        if self._equity <= 0:
            return RiskDecision(False, "equity not yet initialized — orders halted until account snapshot loaded")

        # Position size cap
        estimated_value = request.quantity * (request.limit_price or 100)
        max_allowed = self._equity * self.max_position_pct
        if estimated_value > max_allowed:
            adj_qty = max_allowed / (request.limit_price or 100)
            logger.warning("Position size capped", symbol=request.symbol, original=request.quantity, adjusted=adj_qty)
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
        return size_from_kelly(
            equity=self._equity,
            win_rate=win_rate,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            price=price,
            max_pct=self.max_position_pct,
        )
