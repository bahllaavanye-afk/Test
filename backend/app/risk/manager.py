"""
Real-time risk manager: Kelly sizing, correlation limits, circuit breakers.
All order requests pass through here before reaching the broker.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.brokers.base import OrderRequest
from app.risk.circuit_breaker import CircuitBreaker
from app.risk.correlation import check_cluster_limits, compute_correlation_clusters
from app.risk.hrp import HRPOptimizer
from app.risk.kelly import size_from_kelly
from app.risk.portfolio_optimizer import CVaROptimizer
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
        self._hrp_weights: pd.Series | None = None    # HRP per-symbol allocation weights
        self._cvar_weights: pd.Series | None = None   # CVaR tail-risk overlay weights

        self.global_breaker = CircuitBreaker(
            name="global", max_drawdown_pct=max_drawdown_pct
        )
        self.arb_breaker = CircuitBreaker(
            name="arb", max_drawdown_pct=arb_drawdown_pct
        )

    def update_equity(self, equity: float) -> None:
        self._equity = equity
        self._equity_confirmed = True
        was_halted = self.global_breaker.is_halted
        self.global_breaker.update(equity)
        # Fire a Slack alert on the NORMAL → HALTED transition (edge-triggered,
        # so it notifies once per trip rather than on every snapshot).
        if not was_halted and self.global_breaker.is_halted:
            self._notify_breaker_trip(self.global_breaker)

    def _notify_breaker_trip(self, breaker: CircuitBreaker) -> None:
        """Fire-and-forget Slack alert when a circuit breaker trips."""
        import asyncio

        async def _send() -> None:
            try:
                from app.notifications.slack import slack
                from app.notifications.tracker import tracker
                tracker.record(
                    "circuit_breaker",
                    "risk",
                    f"{breaker.name} breaker tripped at {breaker.current_drawdown:.2%}",
                )
                await slack.notify_circuit_breaker(
                    breaker.name, breaker.current_drawdown, breaker.max_drawdown_pct
                )
            except Exception:
                pass

        try:
            asyncio.get_running_loop().create_task(_send())
        except RuntimeError:
            # No running loop (sync context, e.g. tests) — skip notification.
            pass

    def update_positions(self, positions: list[dict]) -> None:
        self._positions = {
            p["symbol"]: float(p.get("market_value", 0)) for p in positions
        }

    def update_returns(self, returns_df: pd.DataFrame) -> None:
        self._returns_history = returns_df
        if not returns_df.empty and len(returns_df) >= 20:
            self._clusters = compute_correlation_clusters(returns_df, threshold=0.70)

        # Recompute HRP and CVaR weights whenever returns data is refreshed.
        # Multi-asset returns DataFrame (columns = symbols) required.
        if not returns_df.empty and returns_df.shape[1] >= 2 and len(returns_df) >= 10:
            try:
                self._hrp_weights = HRPOptimizer().compute_weights(returns_df)
            except Exception as _hrp_err:
                logger.debug("HRP weight computation failed", error=str(_hrp_err))
                self._hrp_weights = None
            try:
                self._cvar_weights = CVaROptimizer(confidence=0.95).compute_weights(
                    returns_df
                )
            except Exception as _cvar_err:
                logger.debug("CVaR weight computation failed", error=str(_cvar_err))
                self._cvar_weights = None

    async def check_order(self, request: OrderRequest) -> RiskDecision:
        """Gate every order through risk checks. Returns RiskDecision."""
        # Global circuit breaker
        if self._is_global_halted():
            reason = self.global_breaker.halt_reasons[-1] if self.global_breaker.halt_reasons else "unknown"
            return RiskDecision(False, f"Global circuit breaker halted: {reason}")

        # Arbitrage circuit breaker
        if request.risk_bucket == "arbitrage" and self._is_arb_halted():
            reason = self.arb_breaker.halt_reasons[-1] if self.arb_breaker.halt_reasons else "unknown"
            return RiskDecision(False, f"Arb circuit breaker halted: {reason}")

        # Equity sanity checks
        self._log_equity_warning()
        if self._equity <= 0:
            return RiskDecision(False, "equity is zero or negative — orders halted")

        # Position size cap logic
        capped_decision = self._apply_position_cap(request)
        if capped_decision is not None:
            return capped_decision

        # Correlation cluster limit check
        if self._clusters:
            allowed, reason = check_cluster_limits(
                request.symbol,
                request.quantity * (request.limit_price or 100),
                self._positions,
                self._clusters,
                self.max_cluster_pct,
                self._equity,
            )
            if not allowed:
                return RiskDecision(False, reason)

        return RiskDecision(True, "ok", request.quantity)

    def _is_global_halted(self) -> bool:
        return self.global_breaker.is_halted

    def _is_arb_halted(self) -> bool:
        return self.arb_breaker.is_halted

    def _log_equity_warning(self) -> None:
        if not self._equity_confirmed:
            logger.warning(
                "risk.manager: using estimated equity — broker snapshot not yet received",
                estimated_equity=self._equity,
            )

    def _effective_pct_for_symbol(self, symbol: str) -> float:
        """Calculate the most restrictive pct cap for a given symbol."""
        effective_pct = self.max_position_pct
        if self._hrp_weights is not None and symbol in self._hrp_weights.index:
            hrp_cap = float(self._hrp_weights[symbol])
            effective_pct = min(effective_pct, hrp_cap)
        if self._cvar_weights is not None and symbol in self._cvar_weights.index:
            cvar_cap = float(self._cvar_weights[symbol])
            effective_pct = min(effective_pct, cvar_cap)
        return effective_pct

    def _apply_position_cap(self, request: OrderRequest) -> RiskDecision | None:
        """Enforce position size limits; returns a RiskDecision if capped, else None."""
        estimated_value = request.quantity * (request.limit_price or 100)
        effective_pct = self._effective_pct_for_symbol(request.symbol)

        max_allowed = self._equity * effective_pct
        if estimated_value > max_allowed:
            adj_qty = max_allowed / (request.limit_price or 100)
            logger.warning(
                "Position size capped",
                symbol=request.symbol,
                original=request.quantity,
                adjusted=adj_qty,
                effective_pct=round(effective_pct, 4),
            )
            return RiskDecision(True, "size capped", adj_qty)
        return None

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