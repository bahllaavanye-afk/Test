import uuid
from datetime import datetime, date
from typing import Optional, Dict

from sqlalchemy import String, Numeric, DateTime, Date, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class ComparisonResult(Base):
    """
    SQLAlchemy model that stores back‑test comparison results between a manual
    strategy and its ML‑enhanced counterpart.  In addition to the persisted
    columns, helper methods are provided to evaluate the quality of a signal
    according to tightened entry criteria, confirmation filters and improved
    exit logic.
    """
    __tablename__ = "comparison_results"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)

    # Manual strategy metrics
    manual_sharpe: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    manual_sortino: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    manual_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    manual_max_dd: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    manual_win_rate: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))

    # ML‑enhanced strategy metrics
    ml_sharpe: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    ml_sortino: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    ml_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    ml_max_dd: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    ml_win_rate: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))

    # Benchmark metrics
    spy_sharpe: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    spy_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))

    # Statistical significance
    t_statistic: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    p_value: Mapped[Optional[float]] = mapped_column(Numeric(8, 6))
    is_significant: Mapped[Optional[bool]] = mapped_column(Boolean)
    winner: Mapped[Optional[str]] = mapped_column(String(8))  # manual|ml|neither

    # Full equity curves for chart rendering
    equity_curves: Mapped[Optional[Dict]] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # --------------------------------------------------------------------- #
    # Signal evaluation helpers
    # --------------------------------------------------------------------- #

    # Thresholds used to tighten entry conditions
    ENTRY_SHARPE_THRESHOLD: float = 0.6
    ENTRY_WIN_RATE_THRESHOLD: float = 0.55
    ENTRY_TSTAT_THRESHOLD: float = 2.0
    ENTRY_PVALUE_THRESHOLD: float = 0.05
    ENTRY_MAX_DD_THRESHOLD: float = 0.15  # 15 % drawdown limit

    def _has_sufficient_metrics(self) -> bool:
        """
        Ensure that the essential metrics needed for signal evaluation are
        present.  Missing data automatically disqualifies the signal.
        """
        required = [
            self.ml_sharpe,
            self.ml_win_rate,
            self.t_statistic,
            self.p_value,
            self.ml_max_dd,
        ]
        return all(metric is not None for metric in required)

    def is_strong_entry(self) -> bool:
        """
        Determine whether the back‑test result qualifies as a strong entry
        signal.  The logic tightens the original criteria by requiring:

        * ML Sharpe ratio above ``ENTRY_SHARPE_THRESHOLD``.
        * ML win rate above ``ENTRY_WIN_RATE_THRESHOLD``.
        * Statistically significant t‑statistic and p‑value.
        * Max drawdown below ``ENTRY_MAX_DD_THRESHOLD``.
        * Both ML and manual Sharpe ratios are positive (to avoid pathological
          cases where one side is negative).

        Returns
        -------
        bool
            ``True`` if all conditions are satisfied, otherwise ``False``.
        """
        if not self._has_sufficient_metrics():
            return False

        if (
            (self.ml_sharpe or 0) < self.ENTRY_SHARPE_THRESHOLD
            or (self.ml_win_rate or 0) < self.ENTRY_WIN_RATE_THRESHOLD
            or (self.t_statistic or 0) < self.ENTRY_TSTAT_THRESHOLD
            or (self.p_value or 1) > self.ENTRY_PVALUE_THRESHOLD
            or (self.ml_max_dd or 1) > self.ENTRY_MAX_DD_THRESHOLD
        ):
            return False

        # Confirmation filter: manual Sharpe must be non‑negative
        if (self.manual_sharpe or 0) < 0:
            return False

        return True

    def is_exit_signal(self, recent_dd: Optional[float] = None) -> bool:
        """
        Evaluate whether an exit condition is met.  Two complementary checks are
        applied:

        1. **Absolute drawdown** – if the ML max drawdown exceeds the absolute
           threshold (15 % of capital), an exit is triggered.
        2. **Recent drawdown** – if a caller provides a recent drawdown figure
           (e.g., from the last N bars) that exceeds half of the absolute
           threshold, the position is also exited.  This acts as a dynamic
           confirmation filter.

        Parameters
        ----------
        recent_dd : float, optional
            Recent drawdown expressed as a decimal (e.g., 0.07 for 7 %).  If
            ``None`` only the absolute drawdown check is performed.

        Returns
        -------
        bool
            ``True`` when an exit condition is satisfied.
        """
        if self.ml_max_dd is None:
            # Without a max drawdown we cannot safely decide – assume no exit.
            return False

        if self.ml_max_dd > self.ENTRY_MAX_DD_THRESHOLD:
            return True

        if recent_dd is not None and recent_dd > (self.ENTRY_MAX_DD_THRESHOLD / 2):
            return True

        return False

    def determine_winner(self) -> str:
        """
        Resolve the ``winner`` field based on a hierarchy of performance
        metrics.  The method prefers the ML‑enhanced strategy when it outperforms
        the manual version on Sharpe, win rate and return, while also being
        statistically significant.  If neither side meets the criteria, the
        result is ``'neither'``.

        The method updates the ``winner`` column in‑place and returns the
        resolved value.
        """
        # Guard against missing data
        if not self._has_sufficient_metrics():
            self.winner = "neither"
            return self.winner

        ml_better = (
            (self.ml_sharpe or 0) > (self.manual_sharpe or 0)
            and (self.ml_win_rate or 0) > (self.manual_win_rate or 0)
            and (self.ml_return or 0) > (self.manual_return or 0)
        )

        if ml_better and self.is_significant:
            self.winner = "ml"
        elif not ml_better and self.is_significant:
            self.winner = "manual"
        else:
            self.winner = "neither"

        return self.winner

    def __repr__(self) -> str:
        return (
            f"ComparisonResult(id={self.id!r}, strategy={self.strategy_name!r}, "
            f"symbol={self.symbol!r}, interval={self.interval!r}, "
            f"winner={self.winner!r})"
        )