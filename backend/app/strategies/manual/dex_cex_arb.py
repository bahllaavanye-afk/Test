"""
DEX-CEX Arbitrage Strategy
==========================
Monitors Uniswap v3 vs Binance spot price for WBTC/USDC and WETH/USDC.
Enters when price divergence exceeds break-even threshold (gas + fees).

Academic basis:
  Makarov & Schoar (2020) "Trading and Arbitrage in Cryptocurrency Markets"
  Cao, Chen, Jiang & Russell (2021) "How to Talk When a Machine is Listening"

Price discovery:
  - DEX price: Uniswap v3 via The Graph public API (no auth required)
    Endpoint: https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3
    Pool addresses:
      WBTC/USDC — 0x99ac8cA7087fA4A2A1FB6357269965A2014ABc35
      WETH/USDC — 0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D5
  - CEX price: Binance public REST, no auth
    GET https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT

Break-even spread = gas_cost_usd / position_usd * 10_000
                  + cex_fee_bps + dex_fee_bps

Signal is generated when observed spread > break-even + min_spread_bps buffer.

Backtest proxy (standard OHLCV, no tick DEX data available):
  Returns empty signals unless 'dex_price' column is present in the DataFrame.
  When dex_price is present, uses (close - dex_price) / dex_price spread series
  with rolling z-score to proxy divergence opportunities.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

logger = logging.getLogger(__name__)

# ── Pool configuration ────────────────────────────────────────────────────────

# WETH/USDC (0.05 % fee tier pool, most liquid)
_WETH_USDC_POOL = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d5"
# WBTC/USDC (0.3 % fee tier pool)
_WBTC_USDC_POOL = "0x99ac8ca7087fa4a2a1fb6357269965a2014abc35"

_POOL_ADDRESSES: dict[str, str] = {
    "ETHUSDC": _WETH_USDC_POOL,
    "BTCUSDC": _WBTC_USDC_POOL,
}

# Binance ticker symbols corresponding to each pair
_BINANCE_SYMBOLS: dict[str, str] = {
    "ETHUSDC": "ETHUSDT",
    "BTCUSDC": "BTCUSDT",
}

# The Graph public subgraph endpoint — no API key required
_GRAPH_URL = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"

# Binance public REST — no auth required
_BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"

# GraphQL query: fetch pool sqrtPrice and token decimals in one call
_POOL_QUERY = """
{{
  pools(where: {{id: "{pool_id}"}}) {{
    sqrtPrice
    token0 {{ symbol decimals }}
    token1 {{ symbol decimals }}
  }}
}}
"""


class DexCexArbStrategy(AbstractStrategy):
    """
    DEX-CEX Arbitrage strategy.

    Fetches Uniswap v3 and Binance prices concurrently and generates a signal
    when the spread exceeds the break-even threshold.

    Direction:
      spread > 0  → CEX price > DEX price → buy on DEX, sell on CEX
      spread < 0  → DEX price > CEX price → buy on CEX, sell on DEX
      (signal side reflects the CEX leg direction for order routing)
    """

    name = "dex_cex_arb"
    display_name = "DEX-CEX Arbitrage"
    market_type = "crypto"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 15.0       # check every 15 seconds
    confidence_threshold = 0.60

    DEFAULT_PARAMS: dict[str, Any] = {
        "min_spread_bps": 15,          # minimum spread buffer after gas/fees
        "gas_cost_usd": 50.0,          # estimated L1 gas per round-trip (USDC)
        "cex_fee_bps": 10,             # Binance taker fee (10 bps = 0.10 %)
        "dex_fee_bps": 30,             # Uniswap v3 0.3 % pool fee
        "max_position_usd": 10_000.0,  # max notional per trade
        "pairs": ["BTCUSDC", "ETHUSDC"],
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        effective: dict[str, Any] = {**self.DEFAULT_PARAMS, **(params or {})}
        self.min_spread_bps: int = int(effective["min_spread_bps"])
        self.gas_cost_usd: float = float(effective["gas_cost_usd"])
        self.cex_fee_bps: int = int(effective["cex_fee_bps"])
        self.dex_fee_bps: int = int(effective["dex_fee_bps"])
        self.max_position_usd: float = float(effective["max_position_usd"])
        self.pairs: list[str] = list(effective["pairs"])

    def description(self) -> str:
        return (
            "DEX-CEX Arbitrage (manual) — monitors Uniswap v3 vs Binance spot "
            "for WBTC/USDC and WETH/USDC. Enters when price divergence exceeds "
            "break-even (gas + DEX fee + CEX fee) plus a configurable buffer. "
            "Backtest mode is a no-op without live 'dex_price' column data."
        )

    # ── Price fetching helpers ────────────────────────────────────────────────

    @staticmethod
    def _sqrt_price_x96_to_price(
        sqrt_price_x96: int,
        token0_decimals: int,
        token1_decimals: int,
    ) -> float:
        """
        Convert Uniswap v3 sqrtPriceX96 to human-readable token1/token0 price.

        Formula:
          price = (sqrtPriceX96 / 2^96)^2 * (10^token0_decimals / 10^token1_decimals)

        For WETH(18)/USDC(6): price = USDC per WETH
        For WBTC(8)/USDC(6):  price = USDC per WBTC
        """
        raw = (sqrt_price_x96 / (2**96)) ** 2
        decimal_adjustment = 10 ** token0_decimals / 10 ** token1_decimals
        return raw * decimal_adjustment

    async def _fetch_uniswap_price(self, pair: str) -> float | None:
        """
        Query The Graph for the given pair's Uniswap v3 pool price.
        Returns price in USDC per token, or None on any error.
        """
        pool_id = _POOL_ADDRESSES.get(pair)
        if not pool_id:
            logger.warning("dex_cex_arb: no pool address configured for %s", pair)
            return None

        query = _POOL_QUERY.format(pool_id=pool_id)
        try:
            import httpx  # already in pyproject.toml

            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(
                    _GRAPH_URL,
                    json={"query": query},
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()

            pools = data.get("data", {}).get("pools", [])
            if not pools:
                logger.warning("dex_cex_arb: no pool data returned for %s", pair)
                return None

            pool = pools[0]
            sqrt_price_x96 = int(pool["sqrtPrice"])
            token0_decimals = int(pool["token0"]["decimals"])
            token1_decimals = int(pool["token1"]["decimals"])

            price = self._sqrt_price_x96_to_price(
                sqrt_price_x96, token0_decimals, token1_decimals
            )
            return price

        except Exception as exc:  # noqa: BLE001
            logger.warning("dex_cex_arb: Uniswap fetch failed for %s: %s", pair, exc)
            return None

    async def _fetch_binance_price(self, pair: str) -> float | None:
        """
        Fetch spot price from Binance public REST API (no auth required).
        Returns price in USDT (treated as ≈ USDC 1:1), or None on error.
        """
        symbol = _BINANCE_SYMBOLS.get(pair)
        if not symbol:
            logger.warning("dex_cex_arb: no Binance symbol configured for %s", pair)
            return None

        try:
            import httpx

            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(
                    _BINANCE_PRICE_URL, params={"symbol": symbol}
                )
                resp.raise_for_status()
                data = resp.json()

            return float(data["price"])

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "dex_cex_arb: Binance fetch failed for %s: %s", pair, exc
            )
            return None

    # ── Break-even calculation ────────────────────────────────────────────────

    def _break_even_bps(self, reference_price: float) -> float:
        """
        Total cost in basis points for one round-trip arbitrage.

          gas_bps    = gas_cost_usd / max_position_usd * 10_000
          total_bps  = gas_bps + cex_fee_bps + dex_fee_bps

        The result is the minimum spread (in bps) before the trade is profitable.
        """
        if self.max_position_usd <= 0 or reference_price <= 0:
            return float("inf")
        gas_bps = (self.gas_cost_usd / self.max_position_usd) * 10_000.0
        return gas_bps + self.cex_fee_bps + self.dex_fee_bps

    # ── Core signal logic ─────────────────────────────────────────────────────

    async def _signal_for_pair(
        self, pair: str
    ) -> Signal | None:
        """
        Fetch both legs concurrently and return a Signal if arb opportunity exists.
        """
        import asyncio

        uniswap_task = asyncio.create_task(self._fetch_uniswap_price(pair))
        binance_task = asyncio.create_task(self._fetch_binance_price(pair))

        dex_price, cex_price = await asyncio.gather(
            uniswap_task, binance_task, return_exceptions=False
        )

        if dex_price is None or cex_price is None:
            logger.info(
                "dex_cex_arb: skipping %s — price fetch incomplete "
                "(dex=%s, cex=%s)",
                pair,
                dex_price,
                cex_price,
            )
            return None

        if dex_price <= 0 or cex_price <= 0:
            return None

        # Spread in basis points (CEX relative to DEX)
        spread_bps = (cex_price - dex_price) / dex_price * 10_000.0

        breakeven_bps = self._break_even_bps(dex_price)
        required_bps = breakeven_bps + self.min_spread_bps
        spread_above_breakeven = abs(spread_bps) - breakeven_bps

        if abs(spread_bps) <= required_bps:
            logger.debug(
                "dex_cex_arb: %s spread %.2f bps below required %.2f bps",
                pair,
                abs(spread_bps),
                required_bps,
            )
            return None

        # Direction: positive spread → CEX is expensive → sell CEX, buy DEX
        # Signal side is expressed as CEX-leg direction
        if spread_bps > 0:
            side = "sell"   # sell on CEX (high), buy on DEX (low)
        else:
            side = "buy"    # buy on CEX (low), sell on DEX (high)

        # Confidence grows with the spread excess, capped at 0.95
        confidence = min(0.95, 0.70 + spread_above_breakeven / 100.0)

        # Use token name as symbol for the signal
        token = "BTC" if "BTC" in pair else "ETH"
        symbol = f"{token}USDT"

        return Signal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            target_price=cex_price,
            metadata={
                "pair": pair,
                "dex_price": dex_price,
                "cex_price": cex_price,
                "spread_bps": round(spread_bps, 4),
                "breakeven_bps": round(breakeven_bps, 4),
                "spread_above_breakeven_bps": round(spread_above_breakeven, 4),
                "max_position_usd": self.max_position_usd,
            },
        )

    # ── AbstractStrategy interface ────────────────────────────────────────────

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Live signal: fetch Uniswap v3 and Binance prices concurrently for all
        configured pairs and return the first signal found.

        Returns None if:
          - Either exchange is unreachable
          - Spread does not exceed break-even + buffer
        """
        # Determine which pair(s) to evaluate based on the requested symbol
        symbol_upper = symbol.upper().replace("-", "").replace("/", "").replace("_", "")

        candidates: list[str] = []
        for pair in self.pairs:
            token = "BTC" if "BTC" in pair else "ETH"
            if token in symbol_upper or pair.replace("USDC", "") in symbol_upper:
                candidates.append(pair)

        # If symbol matches nothing, evaluate all configured pairs
        if not candidates:
            candidates = list(self.pairs)

        for pair in candidates:
            sig = await self._signal_for_pair(pair)
            if sig is not None:
                return sig

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Vectorized backtest for DEX-CEX arbitrage.

        Real DEX-CEX arb requires tick-by-tick data with simultaneous DEX and
        CEX quotes — this is not available in standard OHLCV datasets.

        If the DataFrame contains a 'dex_price' column (sourced from historical
        Uniswap event logs), this method computes the rolling z-score of the
        (close - dex_price) / dex_price spread and generates signals when |z| > 2.

        Without 'dex_price', returns empty (all-False) signals to avoid spurious
        backtest results from an invalid proxy.
        """
        false_series = pd.Series(False, index=df.index, dtype=bool)

        if "dex_price" not in df.columns or "close" not in df.columns:
            # Standard OHLCV — no valid proxy for DEX-CEX spread; return empty
            return BacktestSignals(
                entries=false_series,
                exits=false_series,
                short_entries=false_series,
                short_exits=false_series,
            )

        close = df["close"].astype(float)
        dex_price = df["dex_price"].astype(float)

        # Spread in bps (CEX close ≈ spot, DEX = Uniswap price)
        spread_bps = (close - dex_price) / dex_price.replace(0, float("nan")) * 10_000.0

        # Rolling z-score of spread (90-bar lookback)
        lookback = 90
        roll_mean = spread_bps.rolling(lookback, min_periods=lookback // 2).mean()
        roll_std = spread_bps.rolling(lookback, min_periods=lookback // 2).std()
        z = (spread_bps - roll_mean) / roll_std.clip(lower=1e-8)

        # Shift by 1 bar to prevent lookahead bias
        z_lag = z.shift(1)

        entry_z = 2.0
        exit_z = 0.5

        # Long entry: CEX significantly cheaper than DEX → buy on CEX
        entries = (z_lag < -entry_z).fillna(False)
        exits = (z_lag > -exit_z).fillna(False)

        # Short entry: CEX significantly more expensive than DEX → sell on CEX
        short_entries = (z_lag > entry_z).fillna(False)
        short_exits = (z_lag < exit_z).fillna(False)

        return BacktestSignals(
            entries=entries.astype(bool),
            exits=exits.astype(bool),
            short_entries=short_entries.astype(bool),
            short_exits=short_exits.astype(bool),
        )
