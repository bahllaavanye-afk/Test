from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class OHLCV(Base):
    __tablename__ = "ohlcv"
    __table_args__ = (
        Index("ix_ohlcv_symbol_exchange_interval_ts", "symbol", "exchange", "interval", "ts", unique=True),
        Index("ix_ohlcv_ts", "ts"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)  # alpaca|tradestation|binance|polymarket
    interval: Mapped[str] = mapped_column(String(8), nullable=False)   # 1m|5m|15m|1h|4h|1d
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    high: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    low: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    close: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    volume: Mapped[float] = mapped_column(Numeric(24, 8), nullable=False)
