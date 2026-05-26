from app.strategies.base import AbstractStrategy, Signal, BacktestSignals

# Registry: maps strategy name → class
from app.strategies.manual.pairs_trading import PairsTradingStrategy
from app.strategies.manual.momentum import MomentumStrategy
from app.strategies.manual.mean_reversion import MeanReversionStrategy
from app.strategies.manual.rsi_macd import RSIMACDStrategy
from app.strategies.manual.breakout import BreakoutStrategy
from app.strategies.manual.supertrend import SupertrendStrategy
from app.strategies.manual.low_volatility import LowVolatilityStrategy
from app.strategies.manual.triangular_arb import TriangularArbStrategy
from app.strategies.manual.poly_binary_arb import PolyBinaryArbStrategy
from app.strategies.ml_enhanced.ml_momentum import MLMomentumStrategy
from app.strategies.ml_enhanced.ml_mean_reversion import MLMeanReversionStrategy
from app.strategies.ml_enhanced.ml_breakout import MLBreakoutStrategy
from app.strategies.ml_enhanced.lorentzian_knn import LorentzianStrategy
from app.strategies.ml_enhanced.ensemble import EnsembleStrategy

STRATEGY_REGISTRY: dict[str, type[AbstractStrategy]] = {
    "pairs_trading": PairsTradingStrategy,
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "rsi_macd": RSIMACDStrategy,
    "breakout": BreakoutStrategy,
    "supertrend": SupertrendStrategy,
    "low_volatility": LowVolatilityStrategy,
    "triangular_arb": TriangularArbStrategy,
    "poly_binary_arb": PolyBinaryArbStrategy,
    "ml_momentum": MLMomentumStrategy,
    "ml_mean_reversion": MLMeanReversionStrategy,
    "ml_breakout": MLBreakoutStrategy,
    "lorentzian_knn": LorentzianStrategy,
    "ensemble": EnsembleStrategy,
}


def get_strategy(name: str, params: dict | None = None) -> AbstractStrategy:
    cls = STRATEGY_REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY)}")
    return cls(params=params)
