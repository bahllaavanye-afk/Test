"""
Sector Rotation Strategy — Tactical Asset Allocation.

Academic basis: Faber (2007), "A Quantitative Approach to Tactical Asset Allocation".
Monthly rebalance: rank 11 SPDR sector ETFs by 3-month (63-day) price momentum,
buy the top 3 sectors, avoid/short the bottom 3.

Sharpe target: ~0.8 (documented in academic literature)
Risk bucket: directional
"""
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class SectorRotationStrategy(AbstractStrategy):
    name = "sector_rotation"
    display_name = "Sector Rotation (Faber TAA)"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0  # Daily check — rebalance monthly

    SECTOR_ETFS = [
        "XLK",   # Technology
        "XLF",   # Financials
        "XLV",   # Health Care
        "XLE",   # Energy
        "XLI",   # Industrials
        "XLY",   # Consumer Discretionary
        "XLP",   # Consumer Staples
        "XLU",   # Utilities
        "XLB",   # Materials
        "XLRE",  # Real Estate
        "XLC",   # Communication Services
    ]

    TOP_N = 3    # Buy top N sectors
    BOTTOM_N = 3  # Sell/short bottom N sectors
    MOMENTUM_PERIOD = 63  # ~3 months of trading days

    DEFAULT_PARAMS = {
        "lookback_days": 63,
        "top_n_sectors": 3,
        "rebalance_freq": 21,
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        effective = {**self.DEFAULT_PARAMS, **(params or {})}
        self.momentum_period = int(effective["lookback_days"])
        self.top_n = int(effective["top_n_sectors"])
        self.rebalance_freq = int(effective["rebalance_freq"])
        p = params or {}
        self.bottom_n = int(p.get("bottom_n", self.BOTTOM_N))

    def _compute_momentum(self, close: pd.Series, period: int) -> float | None:
        """3-month price return for momentum ranking."""
        if len(close) < period + 1:
            return None
        past_price = float(close.iloc[-period])
        current_price = float(close.iloc[-1])
        if past_price <= 0:
            return None
        return (current_price - past_price) / past_price

    def _get_sector_scores(self, data: pd.DataFrame) -> dict[str, float]:
        """
        Compute 3-month momentum for each sector ETF found in data columns.
        Accepts column names directly matching the ticker or using
        ``close_{ticker}`` / ``{ticker}_close`` conventions.
        Returns a mapping of ticker → momentum score.
        """
        scores: dict[str, float] = {}
        for etf in self.SECTOR_ETFS:
            col = None
            if etf in data.columns:
                col = etf
            elif f"close_{etf}" in data.columns:
                col = f"close_{etf}"
            elif f"{etf}_close" in data.columns:
                col = f"{etf}_close"

            if col is not None:
                score = self._compute_momentum(data[col], self.momentum_period)
                if score is not None:
                    scores[etf] = score
        return scores

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Generate signals based on sector momentum.

        * Multi‑sector mode: when ``data`` contains all sector columns, rank the
          ETFs and emit BUY for the top ``top_n`` and SELL for the bottom ``bottom_n``.
        * Single‑symbol mode: when only a ``close`` column is present, treat the
          symbol as a standalone ETF and base the signal on its 3‑month return.
        """
        sector_scores = self._get_sector_scores(data)

        if sector_scores:
            ranked = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
            top_symbols = [s for s, _ in ranked[:self.top_n]]
            bottom_symbols = [s for s, _ in ranked[-self.bottom_n:]]

            sym_upper = symbol.upper()
            if sym_upper in top_symbols:
                rank_idx = top_symbols.index(sym_upper)
                confidence = round(0.70 + (self.top_n - rank_idx) / self.top_n * 0.20, 4)
                score = sector_scores.get(sym_upper, 0.0)
                return Signal(
                    symbol=sym_upper,
                    side="buy",
                    confidence=confidence,
                    strategy_name=self.name,
                    strategy_type=self.strategy_type,
                    risk_bucket=self.risk_bucket,
                    metadata={
                        "rank": rank_idx + 1,
                        "momentum_3m": round(score * 100, 2),
                        "top_sectors": top_symbols,
                        "method": "sector_rotation_top3",
                    },
                )
            if sym_upper in bottom_symbols:
                rank_idx = bottom_symbols.index(sym_upper)
                confidence = round(0.65 + rank_idx / self.bottom_n * 0.15, 4)
                score = sector_scores.get(sym_upper, 0.0)
                return Signal(
                    symbol=sym_upper,
                    side="sell",
                    confidence=confidence,
                    strategy_name=self.name,
                    strategy_type=self.strategy_type,
                    risk_bucket=self.risk_bucket,
                    metadata={
                        "rank": len(ranked) - self.bottom_n + rank_idx + 1,
                        "momentum_3m": round(score * 100, 2),
                        "bottom_sectors": bottom_symbols,
                        "method": "sector_rotation_bottom3",
                    },
                )
            return None

        # Single‑symbol fallback
        if "close" not in data.columns or len(data) < self.momentum_period + 1:
            return None

        mom = self._compute_momentum(data["close"], self.momentum_period)
        if mom is None:
            return None

        if mom > 0.05:  # > 5% 3‑month return
            confidence = min(0.85, 0.65 + mom * 0.80)
            return Signal(
                symbol=symbol.upper(),
                side="buy",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "momentum_3m": round(mom * 100, 2),
                    "method": "sector_rotation_single_symbol",
                },
            )
        if mom < -0.05:  # < -5% 3‑month return
            confidence = min(0.80, 0.65 + abs(mom) * 0.60)
            return Signal(
                symbol=symbol.upper(),
                side="sell",
                confidence=round(confidence, 4),
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "momentum_3m": round(mom * 100, 2),
                    "method": "sector_rotation_single_symbol",
                },
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Vectorized backtest for a single sector ETF.

        * Entry long: 63‑day return > 5% (shifted 1 bar to emulate execution on next day).
        * Exit long: 63‑day return ≤ 0.
        * Entry short: 63‑day return < -5% (shifted 1 bar).
        * Exit short: 63‑day return ≥ 0.
        """
        if "close" not in df.columns:
            false_series = pd.Series(False, index=df.index)
            return BacktestSignals(
                entry_long=false_series,
                exit_long=false_series,
                entry_short=false_series,
                exit_short=false_series,
            )

        # 63‑day percentage change
        momentum = df["close"].pct_change(self.momentum_period)

        # Long side signals
        entry_long = (momentum > 0.05).shift(1).fillna(False)
        exit_long = (momentum <= 0).fillna(False)

        # Short side signals
        entry_short = (momentum < -0.05).shift(1).fillna(False)
        exit_short = (momentum >= 0).fillna(False)

        return BacktestSignals(
            entry_long=entry_long,
            exit_long=exit_long,
            entry_short=entry_short,
            exit_short=exit_short,
        )