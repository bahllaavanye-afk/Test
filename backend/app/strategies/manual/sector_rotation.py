"""
Sector Rotation Strategy — Tactical Asset Allocation.

Academic basis: Faber (2007), "A Quantitative Approach to Tactical Asset Allocation".
Monthly rebalance: rank 11 SPDR sector ETFs by 3-month (63-day) price momentum,
buy the top 3 sectors, avoid/short the bottom 3.

Sharpe target: ~0.8 (documented in academic literature)
Risk bucket: directional
"""
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


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
        Expects columns named after the sector ETF tickers (e.g., 'XLK', 'XLF', ...).
        Also accepts close_{symbol} column naming convention.
        Returns dict of {symbol: momentum_score}.
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
        When called with multi-sector data (columns = ETF tickers),
        generates BUY signals for top-3 momentum sectors and SELL for bottom-3.

        When called for a single symbol, returns BUY/SELL based on whether
        that symbol is in the top/bottom ranked cohort (requires rankings in metadata column).

        For single-symbol mode: use data['close'] and check if the 63-day return
        exceeds a reasonable threshold (acting as if we've already ranked it).
        """
        sector_scores = self._get_sector_scores(data)

        if sector_scores:
            # Multi-sector data available — full ranking
            ranked = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
            top_symbols = [s for s, _ in ranked[:self.top_n]]
            bottom_symbols = [s for s, _ in ranked[-self.bottom_n:]]

            # Generate signal for the requested symbol (if it's in scope)
            sym_upper = symbol.upper()
            if sym_upper in top_symbols:
                rank_idx = top_symbols.index(sym_upper)
                # Higher rank → higher confidence
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
            elif sym_upper in bottom_symbols:
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

        # Single-symbol fallback: use close column for standalone momentum check
        if "close" not in data.columns or len(data) < self.momentum_period + 1:
            return None

        mom = self._compute_momentum(data["close"], self.momentum_period)
        if mom is None:
            return None

        if mom > 0.05:  # > 5% 3-month return — treat as top-tier
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
        elif mom < -0.05:  # < -5% 3-month return — bottom-tier
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
        Entry: 63-day return > 5% (top-tier momentum), shifted 1 bar.
        Exit: 63-day return falls below 0 (rotate out).
        Short: 63-day return < -5% (bottom-tier).

        For multi-sector cross-sectional ranking, use a portfolio-level backtest runner.
        """
        if "close" not in df.columns:
            false_series = pd.Series(False, index=df.index)
            return BacktestSignals(
                entries=false_series,
                exits=false_series,
                short_entries=false_series,
                short_exits=false_series,
            )

        close = df["close"]

        # 63-day momentum (shifted to prevent lookahead)
        momentum = (close / close.shift(self.momentum_period) - 1).shift(1)

        entries = (momentum > 0.05).fillna(False)
        exits = (momentum <= 0.0).fillna(False)
        short_entries = (momentum < -0.05).fillna(False)
        short_exits = (momentum >= 0.0).fillna(False)

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )
