import uuid
from typing import List

from sqlalchemy import String, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.base import TimestampMixin


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    accounts: Mapped[list["Account"]] = relationship("Account", back_populates="user", lazy="select")

    def _moving_average(self, data: List[float], window: int) -> float:
        """Calculate simple moving average for the given window."""
        if window <= 0:
            raise ValueError("Window size must be positive")
        if len(data) < window:
            raise ValueError("Not enough data points for the requested window")
        return sum(data[-window:]) / window

    def generate_signal(self, price_series: List[float]) -> str:
        """
        Generate a trading signal based on recent price data.

        Entry condition:
            - Latest price > 20‑period moving average
            - 20‑period MA > 50‑period MA (trend confirmation)

        Exit condition:
            - Latest price < 20‑period moving average
            - OR 20‑period MA < 50‑period MA (trend weakening)

        Returns:
            "enter" – signal to open a position
            "exit"  – signal to close an existing position
            "hold"  – no action
        """
        # Require a minimum of 20 data points for reliable calculation
        if len(price_series) < 20:
            return "hold"

        try:
            ma20 = self._moving_average(price_series, 20)
        except ValueError:
            return "hold"

        # Use 50‑period MA when enough data is available; otherwise fall back to the overall average
        if len(price_series) >= 50:
            ma50 = self._moving_average(price_series, 50)
        else:
            ma50 = sum(price_series) / len(price_series)

        latest_price = price_series[-1]

        # Entry logic with confirmation filter
        if latest_price > ma20 and ma20 > ma50:
            return "enter"

        # Exit logic with trend weakening detection
        if latest_price < ma20 or ma20 < ma50:
            return "exit"

        return "hold"