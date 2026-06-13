from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd


@dataclass
class Signal:
    symbol: str
    side: str                  # 'buy' | 'sell'
    confidence: float          # 0.0 to 1.0
    strategy_name: str
    strategy_type: str         # 'manual' | 'ml_enhanced'
    risk_bucket: str           # 'arbitrage' | 'directional'
    target_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    metadata: dict = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class BacktestSignals:
    """Container for vectorized backtest signal series."""
    entries: pd.Series    # boolean: True = enter long
    exits: pd.Series      # boolean: True = exit long
    short_entries: pd.Series | None = None
    short_exits: pd.Series | None = None


class AbstractStrategy(ABC):
    """
    Base class for all QuantEdge trading strategies.

    Every strategy must implement:
    - analyze(): generate a real-time signal from current market data
    - backtest_signals(): return vectorized entry/exit signals for VectorBT
    - execute(): forward a signal through the smart order router

    Strategies are stateless: all state lives in Redis or the DB.
    """
    name: str = "base"
    display_name: str = "Base Strategy"
    market_type: str = "equity"         # equity|crypto|polymarket
    strategy_type: str = "manual"       # manual|ml_enhanced
    risk_bucket: str = "directional"    # arbitrage|directional
    tick_interval_seconds: float = 60.0
    confidence_threshold: float = 0.60

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    @abstractmethod
    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Analyze current market data and return a Signal if conditions met, else None.
        Must not place orders — only produce signals.
        """

    @abstractmethod
    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Pure function: given OHLCV DataFrame, return entry/exit boolean Series.
        No side effects. Used by VectorBT backtesting engine.
        Apply .shift(1) to all indicators to prevent lookahead bias.
        """

    def description(self) -> str:
        return f"{self.display_name} ({self.strategy_type})"
