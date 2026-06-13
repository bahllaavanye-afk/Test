"""
Crypto Triangular Arbitrage on Binance.

Scans all 3-leg cycles (A→B→C→A) in real time.
Executes when profit > min_profit_pct after estimated fees.

Example: BTC→ETH→USDT→BTC
  - Buy ETH with BTC: ETH_BTC ask
  - Buy USDT with ETH: ETH_USDT bid  (or sell ETH for USDT)
  - Buy BTC with USDT: BTC_USDT ask
  Profit = product of all conversion rates - 1
"""
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

# Most liquid Binance triangles
TRIANGLE_UNIVERSE = [
    ("BTC", "ETH", "USDT"),
    ("BTC", "BNB", "USDT"),
    ("ETH", "BNB", "USDT"),
    ("BTC", "SOL", "USDT"),
    ("ETH", "SOL", "USDT"),
    ("BTC", "XRP", "USDT"),
]


class TriangularArbStrategy(AbstractStrategy):
    name = "triangular_arb"
    display_name = "Triangular Arbitrage (Binance)"
    market_type = "crypto"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 0.5   # scan every 500ms

    DEFAULT_PARAMS = {
        "min_profit_bps": 15,
        "max_position_usd": 10000,
        "slippage_bps": 5,
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        effective = {**self.DEFAULT_PARAMS, **(params or {})}
        self.min_profit_pct = effective["min_profit_bps"] / 100.0  # convert bps to pct
        self.max_position_usd = effective["max_position_usd"]
        self.slippage_bps = effective["slippage_bps"]
        self.fee_pct = params.get("fee_pct", 0.10) if params else 0.10                # 10 bps per leg (Binance)

    def compute_triangle_profit(self, prices: dict, a: str, b: str, c: str) -> float | None:
        """
        Given price dict {pair: {bid, ask}}, compute profit of A→B→C→A cycle.
        Returns profit pct (e.g. 0.002 = 0.2%) or None if prices missing.
        """
        try:
            # Leg 1: sell A, buy B
            pair1 = f"{b}/{a}"
            rate1 = prices.get(pair1, {}).get("ask", 0)
            if not rate1:
                pair1 = f"{a}/{b}"
                rate1 = 1.0 / prices.get(pair1, {}).get("bid", 0) if prices.get(pair1, {}).get("bid") else 0
            if not rate1:
                return None

            # Leg 2: sell B, buy C
            pair2 = f"{c}/{b}"
            rate2 = prices.get(pair2, {}).get("bid", 0)
            if not rate2:
                pair2 = f"{b}/{c}"
                rate2 = prices.get(pair2, {}).get("ask", 0)
                if rate2:
                    rate2 = 1.0 / rate2
            if not rate2:
                return None

            # Leg 3: sell C, buy A (back to start)
            pair3 = f"{a}/{c}"
            rate3 = prices.get(pair3, {}).get("bid", 0)
            if not rate3:
                pair3 = f"{c}/{a}"
                rate3 = prices.get(pair3, {}).get("ask", 0)
                if rate3:
                    rate3 = 1.0 / rate3
            if not rate3:
                return None

            # Profit = product of rates - 1 - fees (3 legs × fee_pct)
            gross = rate1 * rate2 * rate3
            fees = 3 * self.fee_pct / 100
            return gross - 1 - fees
        except Exception:
            return None

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        # data is not used here — prices come from price_cache in task runner
        # This method is called by the runner with live orderbook data
        return None  # signals produced directly by TriangularArbTask

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        # True triangular arb cannot be backtested with OHLCV — need tick data
        empty = pd.Series(False, index=df.index)
        return BacktestSignals(entries=empty, exits=empty)
