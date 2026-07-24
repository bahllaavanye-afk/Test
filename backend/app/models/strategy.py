import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import String, ForeignKey, Boolean, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.database import Base
from app.models.base import TimestampMixin


class Strategy(Base, TimestampMixin):
    __tablename__ = "strategies"

    # Allowed enumerations
    ALLOWED_MARKET_TYPES = {"equity", "crypto", "polymarket"}
    ALLOWED_STRATEGY_TYPES = {"manual", "ml_enhanced"}
    ALLOWED_RISK_BUCKETS = {"arbitrage", "directional"}

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
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

    def __init__(
        self,
        *,
        account_id: Optional[str] = None,
        name: str,
        display_name: Optional[str] = None,
        market_type: str,
        strategy_type: str,
        risk_bucket: str,
        params: Optional[Dict[str, Any]] = None,
        symbols: Optional[List[str]] = None,
        tick_interval_seconds: float = 60.0,
        confidence_threshold: float = 0.60,
        is_enabled: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialise a Strategy instance with validation.

        Raises:
            ValueError: If any supplied argument fails validation.
        """
        # Perform validation before assigning attributes
        self._validate_name(name)
        if display_name is not None:
            self._validate_display_name(display_name)
        self._validate_market_type(market_type)
        self._validate_strategy_type(strategy_type)
        self._validate_risk_bucket(risk_bucket)
        self._validate_params(params or {})
        self._validate_symbols(symbols or [])
        self._validate_tick_interval_seconds(tick_interval_seconds)
        self._validate_confidence_threshold(confidence_threshold)
        self._validate_is_enabled(is_enabled)

        super().__init__(
            account_id=account_id,
            name=name,
            display_name=display_name,
            market_type=market_type,
            strategy_type=strategy_type,
            risk_bucket=risk_bucket,
            params=params or {},
            symbols=symbols or [],
            tick_interval_seconds=tick_interval_seconds,
            confidence_threshold=confidence_threshold,
            is_enabled=is_enabled,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Validation helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _validate_name(value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Strategy name must be a non‑empty string.")
        if len(value) > 64:
            raise ValueError("Strategy name must not exceed 64 characters.")

    @staticmethod
    def _validate_display_name(value: str) -> None:
        if not isinstance(value, str):
            raise ValueError("Display name must be a string.")
        if len(value) > 128:
            raise ValueError("Display name must not exceed 128 characters.")

    @classmethod
    def _validate_market_type(cls, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError("Market type must be a string.")
        if value not in cls.ALLOWED_MARKET_TYPES:
            raise ValueError(
                f"Market type '{value}' is invalid; allowed values are {sorted(cls.ALLOWED_MARKET_TYPES)}."
            )

    @classmethod
    def _validate_strategy_type(cls, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError("Strategy type must be a string.")
        if value not in cls.ALLOWED_STRATEGY_TYPES:
            raise ValueError(
                f"Strategy type '{value}' is invalid; allowed values are {sorted(cls.ALLOWED_STRATEGY_TYPES)}."
            )

    @classmethod
    def _validate_risk_bucket(cls, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError("Risk bucket must be a string.")
        if value not in cls.ALLOWED_RISK_BUCKETS:
            raise ValueError(
                f"Risk bucket '{value}' is invalid; allowed values are {sorted(cls.ALLOWED_RISK_BUCKETS)}."
            )

    @staticmethod
    def _validate_params(value: Dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise ValueError("Params must be a dictionary.")

    @staticmethod
    def _validate_symbols(value: List[Any]) -> None:
        if not isinstance(value, list):
            raise ValueError("Symbols must be a list.")
        for sym in value:
            if not isinstance(sym, str) or not sym.strip():
                raise ValueError("Each symbol must be a non‑empty string.")

    @staticmethod
    def _validate_tick_interval_seconds(value: float) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError("Tick interval seconds must be a numeric type.")
        if value <= 0:
            raise ValueError("Tick interval seconds must be positive.")

    @staticmethod
    def _validate_confidence_threshold(value: float) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError("Confidence threshold must be a numeric type.")
        if not (0.0 <= value <= 1.0):
            raise ValueError("Confidence threshold must be between 0 and 1 inclusive.")

    @staticmethod
    def _validate_is_enabled(value: Any) -> None:
        if not isinstance(value, bool):
            raise ValueError("is_enabled must be a boolean value.")

    # -------------------------------------------------------------------------
    # SQLAlchemy attribute validation (ensures updates via ORM also respect rules)
    # -------------------------------------------------------------------------

    @validates("name")
    def validate_name(self, key: str, value: str) -> str:  # pragma: no cover
        self._validate_name(value)
        return value

    @validates("display_name")
    def validate_display_name(self, key: str, value: Optional[str]) -> Optional[str]:  # pragma: no cover
        if value is not None:
            self._validate_display_name(value)
        return value

    @validates("market_type")
    def validate_market_type(self, key: str, value: str) -> str:  # pragma: no cover
        self._validate_market_type(value)
        return value

    @validates("strategy_type")
    def validate_strategy_type(self, key: str, value: str) -> str:  # pragma: no cover
        self._validate_strategy_type(value)
        return value

    @validates("risk_bucket")
    def validate_risk_bucket(self, key: str, value: str) -> str:  # pragma: no cover
        self._validate_risk_bucket(value)
        return value

    @validates("params")
    def validate_params(self, key: str, value: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
        self._validate_params(value)
        return value

    @validates("symbols")
    def validate_symbols(self, key: str, value: List[Any]) -> List[Any]:  # pragma: no cover
        self._validate_symbols(value)
        return value

    @validates("tick_interval_seconds")
    def validate_tick_interval_seconds(self, key: str, value: float) -> float:  # pragma: no cover
        self._validate_tick_interval_seconds(value)
        return value

    @validates("confidence_threshold")
    def validate_confidence_threshold(self, key: str, value: float) -> float:  # pragma: no cover
        self._validate_confidence_threshold(value)
        return value

    @validates("is_enabled")
    def validate_is_enabled(self, key: str, value: bool) -> bool:  # pragma: no cover
        self._validate_is_enabled(value)
        return value