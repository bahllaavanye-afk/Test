import uuid
from sqlalchemy import String, ForeignKey, Boolean, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
from app.models.base import TimestampMixin


class Strategy(Base, TimestampMixin):
    __tablename__ = "strategies"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    account_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("accounts.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. 'pairs_trading'
    display_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    market_type: Mapped[str] = mapped_column(String(16), nullable=False)  # equity|crypto|polymarket
    strategy_type: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # manual|ml_enhanced
    risk_bucket: Mapped[str] = mapped_column(String(16), nullable=False)  # arbitrage|directional
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    symbols: Mapped[list] = mapped_column(JSON, default=list)  # tracked symbols
    tick_interval_seconds: Mapped[float] = mapped_column(Float, default=60.0)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.60)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=False)

    account: Mapped["Account"] = relationship(
        "Account", back_populates="strategies"
    )

    def __init__(
        self,
        *,
        name: str,
        market_type: str,
        strategy_type: str,
        risk_bucket: str,
        account_id: str | None = None,
        display_name: str | None = None,
        params: dict | None = None,
        symbols: list | None = None,
        tick_interval_seconds: float | None = None,
        confidence_threshold: float | None = None,
        is_enabled: bool = False,
        **kwargs,
    ):
        """
        Initialise a Strategy with robust handling of edge cases.

        - None values for mutable fields (`params`, `symbols`) are replaced with empty containers.
        - Empty collections are accepted but stored as empty list/dict to avoid NULL in JSON columns.
        - Numerical fields are validated for sensible ranges (e.g., tick_interval_seconds > 0,
          confidence_threshold within [0, 1]).
        """
        super().__init__(**kwargs)

        self.account_id = account_id
        self.name = name
        self.display_name = display_name
        self.market_type = market_type
        self.strategy_type = strategy_type
        self.risk_bucket = risk_bucket

        # Ensure mutable defaults are never None
        self.params = params if params is not None else {}
        self.symbols = symbols if symbols is not None else []

        # Validate numeric edge cases
        self.tick_interval_seconds = (
            tick_interval_seconds
            if tick_interval_seconds is not None and tick_interval_seconds > 0
            else 60.0
        )
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            and 0.0 <= confidence_threshold <= 1.0
            else 0.60
        )
        self.is_enabled = bool(is_enabled)

    # Property setters to guard against future direct assignment of invalid values
    @property
    def tick_interval_seconds(self) -> float:
        return self.__dict__["tick_interval_seconds"]

    @tick_interval_seconds.setter
    def tick_interval_seconds(self, value: float) -> None:
        if value is None or value <= 0:
            raise ValueError("tick_interval_seconds must be a positive number")
        self.__dict__["tick_interval_seconds"] = float(value)

    @property
    def confidence_threshold(self) -> float:
        return self.__dict__["confidence_threshold"]

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        if value is None or not (0.0 <= value <= 1.0):
            raise ValueError(
                "confidence_threshold must be between 0.0 and 1.0 inclusive"
            )
        self.__dict__["confidence_threshold"] = float(value)

    @property
    def params(self) -> dict:
        return self.__dict__["params"]

    @params.setter
    def params(self, value: dict | None) -> None:
        if value is None:
            self.__dict__["params"] = {}
        elif not isinstance(value, dict):
            raise TypeError("params must be a dict")
        else:
            self.__dict__["params"] = value

    @property
    def symbols(self) -> list:
        return self.__dict__["symbols"]

    @symbols.setter
    def symbols(self, value: list | None) -> None:
        if value is None:
            self.__dict__["symbols"] = []
        elif not isinstance(value, list):
            raise TypeError("symbols must be a list")
        else:
            self.__dict__["symbols"] = value