"""
Real-time risk manager: Kelly sizing, correlation limits, circuit breakers.
All order requests pass through here before reaching the broker.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.brokers.base import OrderRequest
from app.risk.kelly import size_from_kelly
from app.risk.correlation import compute_correlation_clusters, check_cluster_limits
from app.risk.circuit_breaker import CircuitBreaker, BreakerState
from app.risk.var import historical_var
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
        sample_interval_seconds: float = 300.0,
        cluster_refresh_seconds: float = 900.0,
        max_var_pct: float = 0.02,
        var_sample_interval_seconds: float = 3600.0,
        min_samples_for_var: int = 20,
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
        # symbol → last known mark price, fed by update_prices() from the price
        # feed. Market orders carry no limit_price, so without this there is no
        # way to convert quantity into notional (see _reference_price).
        self._prices: dict[str, float] = {}
        # Orders that passed without a notional check because no price was known.
        # Surfaced by /risk so "the cap never fires" is visible instead of silent.
        self.unpriced_orders: int = 0

        # ── Correlation inputs ────────────────────────────────────────────────
        # _clusters was permanently empty because update_returns() had no caller
        # anywhere in app/, and check_order() guards the cluster limit with
        # `if self._clusters:`. Wiring the manager into the app did not fix that
        # on its own — the limit needs a returns series to exist at all.
        #
        # Marks arrive from the price feed every ~2s. Correlating 2-second ticks
        # would measure microstructure noise, not the co-movement that
        # compute_correlation_clusters(threshold=0.70) is about, so marks are
        # DOWNSAMPLED to one observation per sample_interval_seconds and the
        # clustering (which is O(symbols²)) is throttled separately.
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.cluster_refresh_seconds = float(cluster_refresh_seconds)
        self._price_samples: dict[str, deque[float]] = {}
        self._last_sample_at: dict[str, float] = {}
        self._last_cluster_refresh: float = 0.0
        # Enough points for a correlation to mean anything; also the minimum
        # compute_correlation_clusters needs before it stops returning {}.
        self.min_samples_for_clusters: int = 21

        # ── VaR gate ──────────────────────────────────────────────────────────
        # risk/CLAUDE.md diagrams "var.py → block if 1-day 99% VaR > 2% of NAV".
        # var.py was implemented and never called from check_order().
        #
        # UNITS ARE THE WHOLE PROBLEM HERE. _risk_state_sync polls equity every
        # 60s, so returns built from every update are 1-MINUTE returns; a 99%
        # VaR off those is a 1-minute VaR, roughly 30x too small against a
        # 1-day limit. Equity is therefore downsampled to
        # var_sample_interval_seconds (default hourly, matching the existing
        # snapshot job) and the resulting VaR is scaled to one day by the
        # square-root-of-time rule.
        #
        # sqrt-time assumes i.i.d. returns. It UNDERSTATES tail risk when
        # returns are positively autocorrelated (trending drawdowns), which is
        # the honest caveat on this number — it is a floor, not a ceiling.
        self.max_var_pct = max_var_pct
        self.var_sample_interval_seconds = float(var_sample_interval_seconds)
        self.min_samples_for_var = int(min_samples_for_var)
        self._equity_samples: deque[float] = deque(maxlen=max(min_samples_for_var * 6, 120))
        self._last_equity_sample_at: float | None = None
        self.last_var: dict[str, Any] = {}   # surfaced by /risk for visibility

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
            self._sample_equity(self._equity)
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

    def _sample_equity(self, equity: float) -> None:
        """Downsample the NAV stream for VaR. See the __init__ note on units."""
        if equity <= 0:
            # A clamped-to-zero equity (broker reporting negative NAV) is a halt
            # signal, not an observation — a -100% return would poison the VaR
            # series for as long as it stays in the window.
            return
        now = time.monotonic()
        last = self._last_equity_sample_at
        if last is not None and (now - last) < self.var_sample_interval_seconds:
            return
        self._last_equity_sample_at = now
        self._equity_samples.append(float(equity))

    def _var_decision(self) -> RiskDecision | None:
        """1-day 99% VaR gate. Returns None when it must not express an opinion.

        risk/CLAUDE.md: "block if 1-day 99% VaR > 2% of NAV".

        FAILS OPEN ON THIN DATA, deliberately. historical_var() returns a
        DEFAULT var_99 of 0.03 when it has fewer than 10 observations — wired
        naively against a 2% limit, a cold start would block EVERY order until
        enough samples accumulate, which is a fleet-wide halt dressed as a risk
        control. Both the sample count and the returned `method` are checked.
        """
        if len(self._equity_samples) < self.min_samples_for_var:
            return None

        equity = list(self._equity_samples)
        returns = [
            (equity[i] - equity[i - 1]) / equity[i - 1]
            for i in range(1, len(equity))
            if equity[i - 1] > 0
        ]
        if len(returns) < self.min_samples_for_var - 1:
            return None

        try:
            result = historical_var(returns, portfolio_value=self._equity)
        except Exception as exc:
            logger.warning("risk.manager: VaR computation failed", error=str(exc))
            return None

        if result.method == "default_insufficient_data":
            return None

        # Scale the per-sample VaR to a 1-day horizon (square-root-of-time).
        periods_per_day = max(86_400.0 / max(self.var_sample_interval_seconds, 1.0), 1.0)
        var_99_daily = float(result.var_99) * (periods_per_day ** 0.5)

        self.last_var = {
            "var_99_per_sample": round(float(result.var_99), 6),
            "var_99_daily": round(var_99_daily, 6),
            "limit": self.max_var_pct,
            "n_observations": result.n_observations,
            "sample_interval_seconds": self.var_sample_interval_seconds,
        }

        if var_99_daily > self.max_var_pct:
            return RiskDecision(
                False,
                f"1-day 99% VaR {var_99_daily:.2%} exceeds limit {self.max_var_pct:.2%}",
            )
        return None

    def update_prices(self, prices: dict[str, float]) -> None:
        """Feed last-known marks so market orders can be sized.

        Best-effort and non-raising: a bad price update must never take down the
        gate that every order passes through. Non-positive and non-finite values
        are dropped rather than stored, because they would silently disable the
        position cap for that symbol.
        """
        if not isinstance(prices, dict):
            return
        now = time.monotonic()
        for symbol, price in prices.items():
            try:
                value = float(price)
            except (TypeError, ValueError):
                continue
            if value > 0 and value == value and value != float("inf"):
                key = str(symbol)
                self._prices[key] = value
                self._sample(key, value, now)
        self._maybe_refresh_clusters(now)

    def _sample(self, symbol: str, price: float, now: float) -> None:
        """Downsample the mark stream into a per-symbol price history."""
        last = self._last_sample_at.get(symbol)
        if last is not None and (now - last) < self.sample_interval_seconds:
            return
        self._last_sample_at[symbol] = now
        series = self._price_samples.get(symbol)
        if series is None:
            # Bounded: this runs for the life of the process. 4x the minimum
            # keeps roughly the last few hours at the default cadence, matching
            # the tail(60) that compute_correlation_clusters actually uses.
            series = deque(maxlen=self.min_samples_for_clusters * 4)
            self._price_samples[symbol] = series
        series.append(price)

    def _maybe_refresh_clusters(self, now: float) -> None:
        """Rebuild correlation clusters from sampled marks, throttled.

        Non-raising by construction: this is reached from the price feed, and a
        clustering failure must not stop marks reaching the position cap.
        """
        if (now - self._last_cluster_refresh) < self.cluster_refresh_seconds:
            return
        ready = {
            sym: list(series)
            for sym, series in self._price_samples.items()
            if len(series) >= self.min_samples_for_clusters
        }
        if len(ready) < 2:
            return
        self._last_cluster_refresh = now
        try:
            # Align by position: every series is sampled on the same cadence, so
            # the last N points cover the same window. Symbols that joined late
            # simply contribute fewer rows, hence the common truncation.
            depth = min(len(v) for v in ready.values())
            frame = pd.DataFrame({sym: vals[-depth:] for sym, vals in ready.items()})
            returns = frame.pct_change().dropna()
            if len(returns) < 3:
                return
            self.update_returns(returns)
            logger.info(
                "risk.manager: correlation clusters refreshed",
                symbols=len(ready),
                observations=len(returns),
                clusters=len(self._clusters),
            )
        except Exception as exc:
            logger.warning("risk.manager: cluster refresh failed", error=str(exc))

    def _reference_price(self, request: OrderRequest) -> float | None:
        """Price used to turn quantity into notional, or None if genuinely unknown.

        This previously defaulted to a hardcoded 100.0 for any order without a
        limit price — i.e. every market order. That silently made the position
        cap meaningless: 1 BTC priced at $100 instead of ~$60k reads as 0.1% of a
        $100k account rather than 60%, and SHIB at $0.00001 reads 10-million-fold
        too large. Returning None instead lets the caller skip the notional
        checks *visibly* rather than enforce them against a fabricated number.
        """
        if request.limit_price is not None and request.limit_price > 0:
            return float(request.limit_price)
        mark = self._prices.get(request.symbol)
        if mark is not None and mark > 0:
            return float(mark)
        return None

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

            price = self._reference_price(request)
            if price is None:
                # No limit price and no mark. Both notional checks below are
                # meaningless without one, so skip them and say so — do NOT
                # invent a price and pretend the cap was enforced.
                self.unpriced_orders += 1
                logger.warning(
                    "risk.manager: order not size-checked — no limit price and no known mark",
                    symbol=request.symbol,
                    quantity=request.quantity,
                )
                return RiskDecision(True, "allowed unpriced — no notional check", request.quantity)

            # Position size cap. Capping ADJUSTS the order, it does not approve
            # it: the correlation check below must still run against the capped
            # notional. It previously returned here, so the largest orders — the
            # only ones that reach this branch — were the ones that skipped the
            # concentration limit entirely.
            quantity = request.quantity
            reason = "ok"
            max_allowed = self._equity * self.max_position_pct
            if quantity * price > max_allowed:
                quantity = max_allowed / price
                reason = "size capped"
                logger.warning(
                    "Position size capped",
                    symbol=request.symbol,
                    original=request.quantity,
                    adjusted=quantity,
                )

            estimated_value = quantity * price

            # Correlation cluster check
            if self._clusters:
                allowed, cluster_reason = check_cluster_limits(
                    request.symbol,
                    estimated_value,
                    self._positions,
                    self._clusters,
                    self.max_cluster_pct,
                    self._equity,
                )
                if not allowed:
                    return RiskDecision(False, cluster_reason)

            # Portfolio VaR gate — risk/CLAUDE.md's fifth documented check, and
            # the last one that was diagrammed but never called. Returns None
            # (no opinion) rather than blocking whenever it lacks the data to
            # have one; see _var_decision.
            var_block = self._var_decision()
            if var_block is not None:
                return var_block

            return RiskDecision(True, reason, quantity)
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
        try:
            return size_from_kelly(
                equity=self._equity,
                win_rate=win_rate,
                avg_win_pct=avg_win_pct,
                avg_loss_pct=avg_loss_pct,
                price=price,
                max_pct=self.max_position_pct,
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
            raise KellySizingError("Error computing Kelly size") from exc