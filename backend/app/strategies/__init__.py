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
from app.strategies.manual.dispersion_trading import DispersionTradingStrategy
from app.strategies.manual.pead_sue import PEADStrategy
from app.strategies.manual.skew_arb import SkewArbitrageStrategy
from app.strategies.manual.gamma_exposure import GammaExposureStrategy
from app.strategies.manual.kalman_pairs import KalmanPairsStrategy
from app.strategies.manual.funding_rate_arb import FundingRateArbStrategy
from app.strategies.manual.dex_cex_arb import DexCexArbStrategy
from app.strategies.manual.liquidation_cascade_fade import LiquidationCascadeFadeStrategy
from app.strategies.manual.vrp_systematic import VRPSystematicStrategy
from app.strategies.manual.hmm_regime import HMMRegimeStrategy
from app.strategies.manual.opening_range_breakout import OpeningRangeBreakoutStrategy
from app.strategies.manual.overnight_return import OvernightReturnStrategy
from app.strategies.manual.order_flow_imbalance import OrderFlowImbalanceStrategy
from app.strategies.manual.earnings_accruals import EarningsAccrualsStrategy
from app.strategies.manual.cross_asset_carry import CrossAssetCarryStrategy
from app.strategies.manual.vol_term_structure import VolTermStructureStrategy
from app.strategies.manual.triple_barrier_momentum import TripleBarrierMomentumStrategy
from app.strategies.manual.residual_momentum import ResidualMomentumStrategy
from app.strategies.manual.idio_vol_anomaly import IdiosyncraticVolAnomalyStrategy
from app.strategies.manual.fifty_two_week_high import FiftyTwoWeekHighStrategy
from app.strategies.manual.open_close_revert import OpenCloseRevertStrategy
from app.strategies.manual.polymarket_sentiment_momentum import PolymarketSentimentMomentumStrategy
from app.strategies.manual.intraday_fomc_momentum import IntradayFOMCMomentumStrategy
from app.strategies.manual.crypto_adaptive_trend import CryptoAdaptiveTrendStrategy
from app.strategies.manual.stablecoin_depeg_arb import StablecoinDepegArbStrategy
from app.strategies.manual.moc_auction_imbalance import MOCAuctionImbalanceStrategy
from app.strategies.manual.options_pcr_reversal import OptionsPCRReversalStrategy
from app.strategies.manual.time_series_momentum import TimeSeriesMomentumStrategy
from app.strategies.manual.cross_sectional_momentum import CrossSectionalMomentumStrategy
from app.strategies.manual.vwap_reversion import VWAPReversionStrategy
from app.strategies.manual.basis_carry import BasisCarryStrategy
from app.strategies.manual.btc_eth_stat_arb import BTCETHStatArb
from app.strategies.manual.intraday_seasonality import IntradaySeasonality
from app.strategies.manual.avellaneda_stoikov_mm import AvellanedaStoikovMM
from app.strategies.manual.funding_settlement_timer import FundingSettlementTimer
from app.strategies.manual.mvrv_zscore_timing import MVRVZScoreTimingStrategy
from app.strategies.manual.token_unlock_fade import TokenUnlockFade
from app.strategies.manual.poly_late_resolution import PolymarketLateResolution
from app.strategies.manual.poly_market_maker import PolymarketMarketMaker
from app.strategies.manual.poly_calibration_arb import PolymarketCalibrationArb
from app.strategies.manual.poly_time_value_fade import PolyTimeValueFadeStrategy
from app.strategies.manual.poly_cross_market_hedge import PolyCrossMarketHedgeStrategy
from app.strategies.manual.poly_liquidity_provision import PolyLiquidityProvisionStrategy
from app.strategies.manual.yield_curve_momentum import YieldCurveMomentumStrategy
from app.strategies.manual.macro_risk_barometer import MacroRiskBarometerStrategy
from app.strategies.manual.dollar_carry import DollarCarryStrategy
from app.strategies.manual.pmi_sector_rotation import PMISectorRotationStrategy
from app.strategies.manual.central_bank_window import CentralBankWindowStrategy
from app.strategies.manual.yield_spread_reversion import YieldSpreadReversionStrategy
from app.strategies.manual.tlt_spy_rotation import TLTSPYRotationStrategy
from app.strategies.manual.duration_momentum import DurationMomentumStrategy
from app.strategies.manual.breakeven_inflation import BreakevenInflationStrategy
from app.strategies.manual.multi_factor_equity import MultiFactorEquity
from app.strategies.manual.realized_vol_asymmetry import RealizedVolAsymmetryStrategy
from app.strategies.manual.analyst_revision_momentum import AnalystRevisionMomentumStrategy
from app.strategies.manual.on_chain_exchange_netflow import OnChainExchangeNetflowStrategy
from app.strategies.manual.vol_of_vol_timing import VolOfVolTimingStrategy
from app.strategies.manual.credit_spread_income import CreditSpreadIncomeStrategy
from app.strategies.manual.options_strategies import (
    CoveredCallStrategy,
    CashSecuredPutStrategy,
    IronCondorStrategy,
    LongCallMomentum,
    EarningsIVCrushStrategy,
    WheelStrategy,
)
from app.strategies.manual.stat_arb_etf import StatArbETFStrategy
from app.strategies.manual.bond_equity_rotation import BondEquityRotationStrategy
from app.strategies.manual.put_call_ratio_contrarian import PutCallRatioContrarianStrategy
from app.strategies.manual.crypto_basis_roll import CryptoBasisRollStrategy
from app.strategies.manual.micro_cap_momentum import MicroCapMomentumStrategy
from app.strategies.manual.event_driven_gap import EventDrivenGapStrategy
from app.strategies.manual.vol_carry_short import VolCarryShortStrategy
from app.strategies.manual.crypto_whale_momentum import CryptoWhaleMomentumStrategy
from app.strategies.manual.interest_rate_differential import InterestRateDifferentialStrategy
from app.strategies.manual.options_gamma_scalp import OptionsGammaScalpStrategy
from app.strategies.manual.tv_indicators import (
    EMAStackStrategy,
    SqueezeProStrategy,
    WaveTrendStrategy,
    HullSuiteStrategy,
    SupertrendRsiComboStrategy,
    KamaRocStrategy,
    VwapBandsStrategy,
    IchimokuCloudStrategy,
    MacdDivergenceStrategy,
    AdxDmiStrategy,
    StochRsiMacdStrategy,
    ElliottWaveProxyStrategy,
)

# ML strategies depend on optional heavy libs (torch, stable_baselines3, gymnasium,
# xgboost, lightgbm, optuna, shap, vectorbt). In environments where these aren't
# installed (CI, lightweight deploys), we skip the strategy gracefully instead
# of failing the whole import chain.
_OPTIONAL_ML_STRATEGIES: list[tuple[str, str, str]] = [
    ("ml_momentum",       "app.strategies.ml_enhanced.ml_momentum",       "MLMomentumStrategy"),
    ("ml_pca_arb",        "app.strategies.ml_enhanced.ml_pca_arb",        "MLPCAStatArbStrategy"),
    ("ml_mean_reversion", "app.strategies.ml_enhanced.ml_mean_reversion", "MLMeanReversionStrategy"),
    ("ml_breakout",       "app.strategies.ml_enhanced.ml_breakout",       "MLBreakoutStrategy"),
    ("lorentzian_knn",    "app.strategies.ml_enhanced.lorentzian_knn",    "LorentzianStrategy"),
    ("ensemble",          "app.strategies.ml_enhanced.ensemble",          "EnsembleStrategy"),
    ("rl_trader",         "app.strategies.ml_enhanced.rl_trader",         "RLTraderStrategy"),
]


def _try_import_ml(module_path: str, class_name: str):
    """Best-effort import of an ML strategy. Returns the class or None.

    Catches ImportError (missing optional dep like torch) and AttributeError
    (e.g. `class X(nn.Module)` where nn is None because torch wasn't installed).
    Either way, the strategy is skipped gracefully instead of breaking the
    whole registry import on lightweight deploys (Render free tier, CI).
    """
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        import logging
        logging.getLogger(__name__).info(
            "ML strategy %s skipped (optional dep missing: %s)", class_name, e
        )
        return None

# ── Options strategy group ────────────────────────────────────────────────────
# Convenience list for enabling/disabling all options strategies as a group.
# Mirrors the "Options" desk in desk_order_placer.py.
OPTIONS_STRATEGIES: list[str] = [
    "skew_arb",
    "vrp_systematic",
    "gamma_exposure",
    "options_pcr_reversal",
    "dispersion_trading",
    "vol_term_structure",
    "credit_spread_income",
]

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
    "dispersion_trading": DispersionTradingStrategy,
    "pead_sue": PEADStrategy,
    "skew_arb": SkewArbitrageStrategy,
    "gamma_exposure": GammaExposureStrategy,
    "kalman_pairs": KalmanPairsStrategy,
    "funding_rate_arb": FundingRateArbStrategy,
    "dex_cex_arb": DexCexArbStrategy,
    "liquidation_cascade_fade": LiquidationCascadeFadeStrategy,
    "vrp_systematic": VRPSystematicStrategy,
    "hmm_regime": HMMRegimeStrategy,
    "opening_range_breakout": OpeningRangeBreakoutStrategy,
    "overnight_return": OvernightReturnStrategy,
    "order_flow_imbalance": OrderFlowImbalanceStrategy,
    "earnings_accruals": EarningsAccrualsStrategy,
    "cross_asset_carry": CrossAssetCarryStrategy,
    "vol_term_structure": VolTermStructureStrategy,
    "triple_barrier_momentum": TripleBarrierMomentumStrategy,
    "residual_momentum": ResidualMomentumStrategy,
    "idio_vol_anomaly": IdiosyncraticVolAnomalyStrategy,
    "fifty_two_week_high": FiftyTwoWeekHighStrategy,
    "open_close_revert": OpenCloseRevertStrategy,
    "polymarket_sentiment_momentum": PolymarketSentimentMomentumStrategy,
    "intraday_fomc_momentum": IntradayFOMCMomentumStrategy,
    "crypto_adaptive_trend": CryptoAdaptiveTrendStrategy,
    "stablecoin_depeg_arb": StablecoinDepegArbStrategy,
    "moc_auction_imbalance": MOCAuctionImbalanceStrategy,
    "options_pcr_reversal": OptionsPCRReversalStrategy,
    "time_series_momentum": TimeSeriesMomentumStrategy,
    "cross_sectional_momentum": CrossSectionalMomentumStrategy,
    "vwap_reversion": VWAPReversionStrategy,
    "basis_carry": BasisCarryStrategy,
    "btc_eth_stat_arb": BTCETHStatArb,
    "intraday_seasonality": IntradaySeasonality,
    "avellaneda_stoikov_mm": AvellanedaStoikovMM,
    "funding_settlement_timer": FundingSettlementTimer,
    "mvrv_zscore_timing": MVRVZScoreTimingStrategy,
    "token_unlock_fade": TokenUnlockFade,
    "poly_late_resolution": PolymarketLateResolution,
    "poly_market_maker": PolymarketMarketMaker,
    "poly_calibration_arb": PolymarketCalibrationArb,
    "multi_factor_equity": MultiFactorEquity,
    "credit_spread_income": CreditSpreadIncomeStrategy,
    "covered_call": CoveredCallStrategy,
    "cash_secured_put": CashSecuredPutStrategy,
    "iron_condor": IronCondorStrategy,
    "long_call_momentum": LongCallMomentum,
    "earnings_iv_crush": EarningsIVCrushStrategy,
    "wheel": WheelStrategy,
    # ── New strategies ────────────────────────────────────────────────────────
    "stat_arb_etf": StatArbETFStrategy,
    "bond_equity_rotation": BondEquityRotationStrategy,
    "put_call_ratio_contrarian": PutCallRatioContrarianStrategy,
    "crypto_basis_roll": CryptoBasisRollStrategy,
    "micro_cap_momentum": MicroCapMomentumStrategy,
    "event_driven_gap": EventDrivenGapStrategy,
    "vol_carry_short": VolCarryShortStrategy,
    "crypto_whale_momentum": CryptoWhaleMomentumStrategy,
    "interest_rate_differential": InterestRateDifferentialStrategy,
    "options_gamma_scalp": OptionsGammaScalpStrategy,
    # ── Polymarket expanded desk ──────────────────────────────────────────────
    "poly_time_value_fade": PolyTimeValueFadeStrategy,
    "poly_cross_market_hedge": PolyCrossMarketHedgeStrategy,
    "poly_liquidity_provision": PolyLiquidityProvisionStrategy,
    # ── Macro desk ────────────────────────────────────────────────────────────
    "yield_curve_momentum": YieldCurveMomentumStrategy,
    "macro_risk_barometer": MacroRiskBarometerStrategy,
    "dollar_carry": DollarCarryStrategy,
    "pmi_sector_rotation": PMISectorRotationStrategy,
    "central_bank_window": CentralBankWindowStrategy,
    # ── Rates desk ────────────────────────────────────────────────────────────
    "yield_spread_reversion": YieldSpreadReversionStrategy,
    "tlt_spy_rotation": TLTSPYRotationStrategy,
    "duration_momentum": DurationMomentumStrategy,
    "breakeven_inflation": BreakevenInflationStrategy,
    # ── TradingView Indicator Desk ────────────────────────────────────────────
    "ema_stack_tv": EMAStackStrategy,
    "squeeze_pro_tv": SqueezeProStrategy,
    "wave_trend_tv": WaveTrendStrategy,
    "hull_suite_tv": HullSuiteStrategy,
    "supertrend_rsi_tv": SupertrendRsiComboStrategy,
    "kama_roc_tv": KamaRocStrategy,
    "vwap_bands_tv": VwapBandsStrategy,
    "ichimoku_cloud_tv": IchimokuCloudStrategy,
    "macd_divergence_tv": MacdDivergenceStrategy,
    "adx_dmi_tv": AdxDmiStrategy,
    "stoch_rsi_macd_tv": StochRsiMacdStrategy,
    "elliott_wave_proxy_tv": ElliottWaveProxyStrategy,
}

# Best-effort load ML strategies; missing optional deps don't break the registry
for _name, _path, _cls in _OPTIONAL_ML_STRATEGIES:
    _strategy_cls = _try_import_ml(_path, _cls)
    if _strategy_cls is not None:
        STRATEGY_REGISTRY[_name] = _strategy_cls


def get_strategy(name: str, params: dict | None = None) -> AbstractStrategy:
    cls = STRATEGY_REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY)}")
    return cls(params=params)
