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
from app.strategies.manual.pca_stat_arb import PCAStatArbStrategy
from app.strategies.manual.news_momentum import NewsMomentumStrategy
from app.strategies.manual.vix_mean_reversion import VIXMeanReversionStrategy
from app.strategies.manual.sector_rotation import SectorRotationStrategy
from app.strategies.ml_enhanced.ml_momentum import MLMomentumStrategy
from app.strategies.ml_enhanced.ml_pca_arb import MLPCAStatArbStrategy
from app.strategies.ml_enhanced.ml_mean_reversion import MLMeanReversionStrategy
from app.strategies.ml_enhanced.ml_breakout import MLBreakoutStrategy
from app.strategies.ml_enhanced.lorentzian_knn import LorentzianStrategy
from app.strategies.ml_enhanced.ensemble import EnsembleStrategy
from app.strategies.ml_enhanced.rl_trader import RLTraderStrategy

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
    "pca_stat_arb": PCAStatArbStrategy,
    "news_momentum": NewsMomentumStrategy,
    "vix_mean_reversion": VIXMeanReversionStrategy,
    "sector_rotation": SectorRotationStrategy,
    "ml_momentum": MLMomentumStrategy,
    "ml_pca_arb": MLPCAStatArbStrategy,
    "ml_mean_reversion": MLMeanReversionStrategy,
    "ml_breakout": MLBreakoutStrategy,
    "lorentzian_knn": LorentzianStrategy,
    "ensemble": EnsembleStrategy,
    "rl_trader": RLTraderStrategy,
}


def get_strategy(name: str, params: dict | None = None) -> AbstractStrategy:
    cls = STRATEGY_REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY)}")
    return cls(params=params)
