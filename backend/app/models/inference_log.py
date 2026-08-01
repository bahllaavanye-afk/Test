"""InferenceLog ORM — records every prediction made by a serving model."""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class InferenceLog(Base):
    """
    Immutable record of a single model inference.

    The model's raw output (``prediction``) is stored together with a
    discretised ``signal`` (buy|sell|hold) and a ``confidence`` metric.
    After the market closes the actual return is recorded via
    ``POST /releases/{id}/record-outcome`` which allows live accuracy
    monitoring.

    This class also provides lightweight strategy‑logic helpers that
    tighten entry conditions, add confirmation filters and improve exit
    decisions.  The helpers are deliberately simple – they operate solely
    on the data available in the row and avoid any external state so
    that they remain safe for bulk inserts and background jobs.
    """

    __tablename__ = "inference_logs"
    __table_args__ = (
        Index("ix_inf_release_ts", "release_id", "ts"),
        Index("ix_inf_model_symbol", "model_name", "symbol"),
    )

    # --------------------------------------------------------------------- #
    # Columns
    # --------------------------------------------------------------------- #
    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    release_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("model_releases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    # Raw model output in [0, 1]
    prediction: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)
    # Discretised trading signal
    signal: Mapped[str] = mapped_column(String(8), nullable=False)  # buy|sell|hold
    # Calibration metric: abs(pred - 0.5) * 2
    confidence: Mapped[float] = mapped_column(Numeric(6, 4), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Numeric(8, 3), nullable=False)
    # Which branch of the A/B test served this request
    ab_group: Mapped[str] = mapped_column(String(16), nullable=False)  # champion|challenger|shadow
    # Filled in ex-post when actual market return is known
    actual_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))
    is_correct: Mapped[Optional[bool]] = mapped_column(Boolean)

    # --------------------------------------------------------------------- #
    # Configuration constants – these can be tuned centrally
    # --------------------------------------------------------------------- #
    #: Minimum confidence required for a signal to be considered actionable.
    MIN_CONFIDENCE: float = 0.60
    #: Prediction thresholds that define a strong directional bias.
    BUY_THRESHOLD: float = 0.55
    SELL_THRESHOLD: float = 0.45
    #: Default profit‑target and stop‑loss expressed as fractional returns.
    DEFAULT_PROFIT_TARGET: float = 0.02   # 2 %
    DEFAULT_STOP_LOSS: float = -0.01      # -1 %

    # --------------------------------------------------------------------- #
    # Helper methods – pure, side‑effect‑free logic
    # --------------------------------------------------------------------- #
    def _valid_signal(self) -> bool:
        """Return ``True`` if ``signal`` is one of the allowed values."""
        return self.signal in {"buy", "sell", "hold"}

    @property
    def is_strong_signal(self) -> bool:
        """
        Determine whether the inference qualifies as a strong entry signal.

        Conditions:
        * ``signal`` must be ``buy`` or ``sell`` (``hold`` is ignored).
        * ``confidence`` must meet or exceed :attr:`MIN_CONFIDENCE`.
        * ``prediction`` must be beyond the directional threshold
          (:attr:`BUY_THRESHOLD` for ``buy`` or :attr:`SELL_THRESHOLD` for
          ``sell``).

        Returns:
            bool: ``True`` if all conditions are satisfied.
        """
        if not self._valid_signal() or self.signal == "hold":
            return False
        if self.confidence < self.MIN_CONFIDENCE:
            return False
        if self.signal == "buy" and self.prediction >= self.BUY_THRESHOLD:
            return True
        if self.signal == "sell" and self.prediction <= self.SELL_THRESHOLD:
            return True
        return False

    def passes_confirmation(self, recent_same_signal: int = 2) -> bool:
        """
        Simple confirmation filter based on recent identical signals.

        In a live system the caller can supply the count of consecutive
        signals that match the current one.  If the count meets the
        ``recent_same_signal`` requirement the signal is considered
        confirmed.

        Args:
            recent_same_signal: Minimum number of consecutive identical
                signals required for confirmation.  Defaults to 2.

        Returns:
            bool: ``True`` if the signal is confirmed.
        """
        # The ORM model does not have access to historical rows, so the
        # method expects the caller to provide the count.  A value of
        # ``None`` or ``0`` means no confirmation.
        return recent_same_signal >= 2 and self.is_strong_signal

    def should_exit(
        self,
        profit_target: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> bool:
        """
        Evaluate whether an open position associated with this inference
        should be closed.

        The decision is based on the realised ``actual_return`` compared
        against a profit‑target and a stop‑loss.  If ``actual_return`` is
        not yet recorded the method returns ``False``.

        Args:
            profit_target: Desired profit level as a fractional return.
                If ``None``, :attr:`DEFAULT_PROFIT_TARGET` is used.
            stop_loss: Maximum tolerated loss as a fractional return.
                If ``None``, :attr:`DEFAULT_STOP_LOSS` is used.

        Returns:
            bool: ``True`` if the position meets exit criteria.
        """
        if self.actual_return is None:
            return False

        pt = profit_target if profit_target is not None else self.DEFAULT_PROFIT_TARGET
        sl = stop_loss if stop_loss is not None else self.DEFAULT_STOP_LOSS

        return self.actual_return >= pt or self.actual_return <= sl

    # --------------------------------------------------------------------- #
    # Representation helpers
    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:
        return (
            f"InferenceLog(id={self.id!r}, symbol={self.symbol!r}, "
            f"signal={self.signal!r}, confidence={self.confidence:.4f})"
        )