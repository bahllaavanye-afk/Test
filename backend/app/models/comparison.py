import uuid
from datetime import datetime, date
from typing import Optional, Dict, Any

from sqlalchemy import String, Numeric, DateTime, Date, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ComparisonResult(Base):
    """
    ORM model that stores back‑test comparison results between a manual strategy
    and its ML‑enhanced counterpart.  In addition to the persisted columns, the
    class provides helpers that encapsulate tighter entry/exit logic used by
    the trading engine.

    The entry logic requires:
    * Statistical significance (|t| > 2 and p < 0.05)
    * Both manual and ML Sharpe ratios above configurable thresholds
    * Win‑rate above a configurable minimum
    * Consistent positive return over the test period

    The exit logic is based on maximum draw‑down and a trailing‑stop style
    check on the equity curve.
    """
    __tablename__ = "comparison_results"

    # --------------------------------------------------------------------- #
    # Primary / identifying fields
    # --------------------------------------------------------------------- #
    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)

    # --------------------------------------------------------------------- #
    # Manual strategy metrics
    # --------------------------------------------------------------------- #
    manual_sharpe: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    manual_sortino: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    manual_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    manual_max_dd: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    manual_win_rate: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))

    # --------------------------------------------------------------------- #
    # ML‑enhanced strategy metrics
    # --------------------------------------------------------------------- #
    ml_sharpe: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    ml_sortino: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    ml_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    ml_max_dd: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    ml_win_rate: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))

    # --------------------------------------------------------------------- #
    # Benchmark metrics
    # --------------------------------------------------------------------- #
    spy_sharpe: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    spy_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))

    # --------------------------------------------------------------------- #
    # Statistical significance
    # --------------------------------------------------------------------- #
    t_statistic: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    p_value: Mapped[Optional[float]] = mapped_column(Numeric(8, 6))
    is_significant: Mapped[Optional[bool]] = mapped_column(Boolean)
    winner: Mapped[Optional[str]] = mapped_column(String(8))  # manual|ml|neither

    # --------------------------------------------------------------------- #
    # Equity curves for chart rendering
    # --------------------------------------------------------------------- #
    equity_curves: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # --------------------------------------------------------------------- #
    # Configurable thresholds – can be overridden per‑instance if needed
    # --------------------------------------------------------------------- #
    MIN_SHARPE: float = 0.5
    MIN_WIN_RATE: float = 0.55
    MAX_DRAWDOWN: float = 0.20  # 20 %
    MIN_T_STAT: float = 2.0
    MAX_P_VALUE: float = 0.05

    # --------------------------------------------------------------------- #
    # Helper methods implementing tighter entry / exit criteria
    # --------------------------------------------------------------------- #
    def _has_statistical_edge(self) -> bool:
        """Return True if the back‑test shows a statistically significant edge."""
        if self.t_statistic is None or self.p_value is None:
            return False
        return abs(self.t_statistic) >= self.MIN_T_STAT and self.p_value <= self.MAX_P_VALUE

    def _meets_performance_thresholds(self, sharpe: Optional[float], win_rate: Optional[float]) -> bool:
        """Check that Sharpe and win‑rate exceed the configured minima."""
        if sharpe is None or win_rate is None:
            return False
        return sharpe >= self.MIN_SHARPE and win_rate >= self.MIN_WIN_RATE

    def _drawdown_within_limit(self, max_dd: Optional[float]) -> bool:
        """Validate that the maximum draw‑down does not breach the allowed limit."""
        if max_dd is None:
            return False
        return max_dd <= self.MAX_DRAWDOWN

    def should_enter(self) -> bool:
        """
        Determine whether the strategy should be entered based on tightened
        entry conditions:

        * Statistical edge exists.
        * Both manual and ML Sharpe ratios exceed ``MIN_SHARPE``.
        * Both win‑rates exceed ``MIN_WIN_RATE``.
        * Returns are positive.
        * Draw‑down stays within the allowed limit.
        """
        if not self._has_statistical_edge():
            return False

        # Manual side checks
        manual_ok = (
            self._meets_performance_thresholds(self.manual_sharpe, self.manual_win_rate)
            and (self.manual_return or 0) > 0
            and self._drawdown_within_limit(self.manual_max_dd)
        )

        # ML side checks
        ml_ok = (
            self._meets_performance_thresholds(self.ml_sharpe, self.ml_win_rate)
            and (self.ml_return or 0) > 0
            and self._drawdown_within_limit(self.ml_max_dd)
        )

        return manual_ok and ml_ok

    def _trailing_drawdown_exceeded(self, equity_curve: Optional[Dict[str, Any]]) -> bool:
        """
        Simple trailing‑stop check: if the equity curve falls more than
        ``MAX_DRAWDOWN`` from its peak, signal an exit.
        """
        if not equity_curve:
            return False

        equity_series = equity_curve.get("equity")
        if not equity_series or not isinstance(equity_series, list):
            return False

        peak = equity_series[0]
        for value in equity_series:
            if value > peak:
                peak = value
            elif (peak - value) / peak > self.MAX_DRAWDOWN:
                return True
        return False

    def should_exit(self) -> bool:
        """
        Determine whether the strategy should be exited.  Exit is triggered
        when any of the following conditions are met:

        * The observed maximum draw‑down exceeds ``MAX_DRAWDOWN``.
        * The trailing draw‑down from the equity curve exceeds the same limit.
        * The statistical edge is no longer significant (t‑stat falls below
          ``MIN_T_STAT`` or p‑value rises above ``MAX_P_VALUE``).
        """
        # Draw‑down breach on either side
        if not self._drawdown_within_limit(self.manual_max_dd):
            return True
        if not self._drawdown_within_limit(self.ml_max_dd):
            return True

        # Trailing draw‑down on the stored equity curve
        if self._trailing_drawdown_exceeded(self.equity_curves):
            return True

        # Edge erosion
        if self.t_statistic is not None and abs(self.t_statistic) < self.MIN_T_STAT:
            return True
        if self.p_value is not None and self.p_value > self.MAX_P_VALUE:
            return True

        return False

    # --------------------------------------------------------------------- #
    # Representation helpers
    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:
        return (
            f"<ComparisonResult id={self.id} strategy={self.strategy_name} "
            f"symbol={self.symbol} interval={self.interval}>"
        )