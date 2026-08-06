import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import String, ForeignKey, Boolean, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.base import TimestampMixin


class Strategy(Base, TimestampMixin):
    """
    ORM model representing a trading strategy configuration.

    Attributes
    ----------
    id : str
        Primary key, generated as a UUID string.
    account_id : Optional[str]
        Foreign key reference to the owning account; may be ``None`` for
        strategies not yet attached to an account.
    name : str
        Internal identifier for the strategy (e.g. ``'pairs_trading'``).
    display_name : Optional[str]
        Human‑readable name shown in UI; optional.
    market_type : str
        Market classification, such as ``'equity'``, ``'crypto'`` or ``'polymarket'``.
    strategy_type : str
        Type of strategy implementation, e.g. ``'manual'`` or ``'ml_enhanced'``.
    risk_bucket : str
        Risk category, e.g. ``'arbitrage'`` or ``'directional'``.
    params : Dict[str, Any]
        JSON‑serialisable dictionary of strategy‑specific parameters.
    symbols : List[str]
        List of symbols that the strategy monitors or trades.
    tick_interval_seconds : float
        Desired tick interval for the strategy execution loop.
    confidence_threshold : float
        Minimum confidence level required to activate the strategy.
    is_enabled : bool
        Flag indicating whether the strategy is active.
    account : "Account"
        SQLAlchemy relationship to the owning ``Account`` model.
    """

    __tablename__ = "strategies"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    account_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("accounts.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. 'pairs_trading'
    display_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    market_type: Mapped[str] = mapped_column(String(16), nullable=False)  # equity|crypto|polymarket
    strategy_type: Mapped[str] = mapped_column(String(16), nullable=False)  # manual|ml_enhanced
    risk_bucket: Mapped[str] = mapped_column(String(16), nullable=False)  # arbitrage|directional
    params: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    symbols: Mapped[List[str]] = mapped_column(JSON, default=list)  # tracked symbols
    tick_interval_seconds: Mapped[float] = mapped_column(Float, default=60.0)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.60)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=False)

    account: Mapped["Account"] = relationship("Account", back_populates="strategies")