"""
Polymarket Cross-Market Hedge.

Find two Polymarket markets that are logically correlated
(same event, different framing) and trade the mismatch.
E.g., "Will X happen" at 0.70 vs "Will X NOT happen" at 0.40 → sum = 1.10 → arb.
Extended version of binary arb for non-binary correlated markets.
Uses Polymarket Gamma API.
"""
from __future__ import annotations

import pandas as pd

try:
    import httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

GAMMA_BASE = "https://gamma-api.polymarket.com"


class PolyCrossMarketHedgeStrategy(AbstractStrategy):
    """
    Find two Polymarket markets that are logically correlated
    (same event, different framing) and trade the mismatch.
    E.g., "Will X happen" at 0.70 vs "Will X NOT happen" at 0.40 → sum = 1.10 → arb.
    Extended version of binary arb for non-binary correlated markets.
    Uses Polymarket Gamma API.
    """

    name = "poly_cross_market_hedge"
    display_name = "Polymarket Cross-Market Hedge"
    market_type = "polymarket"
    strategy_type = "manual"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 15.0   # poll every 15 seconds

    DEFAULT_PARAMS = {
        "min_edge_pct": 5.0,         # min % profit after fees for the pair
        "max_position_usd": 400,
        "kelly_fraction": 0.25,
        "min_liquidity": 500,        # $500 min depth on each leg
    }

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        self.min_edge_pct: float = float(p["min_edge_pct"])
        self.max_position_usd: float = float(p["max_position_usd"])
        self.kelly_fraction: float = float(p["kelly_fraction"])
        self.min_liquidity: float = float(p["min_liquidity"])
        # Mismatch threshold: sum > 1 + edge means both legs can be bought for arb
        self.max_sum = 1.0 - self.min_edge_pct / 100.0

    def description(self) -> str:
        return (
            f"Exploit YES/NO price mismatch across correlated Polymarket events. "
            f"Min edge: {self.min_edge_pct}%. Source: Gamma API."
        )

    async def _fetch_markets(self) -> list[dict]:
        if not _HTTPX:
            return []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    f"{GAMMA_BASE}/markets",
                    params={"active": "true", "closed": "false", "limit": 500},
                )
                r.raise_for_status()
                data = r.json()
                return data if isinstance(data, list) else data.get("markets", [])
        except Exception:
            return []

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Scan Gamma API for correlated market pairs where the sum of
        complementary probabilities exceeds 1 + min_edge_pct.
        """
        markets = await self._fetch_markets()
        if not markets:
            return None

        # Build a map of event group → markets for correlation matching
        # Gamma API groups markets by event/slug
        event_groups: dict[str, list[dict]] = {}
        for m in markets:
            event_key = (
                m.get("groupItemTitle")
                or m.get("slug", "").rsplit("-", 1)[0]  # strip YES/NO suffix
                or m.get("question", "")[:60]
            )
            event_groups.setdefault(event_key, []).append(m)

        for event_key, group in event_groups.items():
            if len(group) < 2:
                continue

            # Try all pairs within the same event group
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    m1, m2 = group[i], group[j]

                    tokens1 = m1.get("tokens", [])
                    tokens2 = m2.get("tokens", [])

                    # Extract YES prices for each market
                    def yes_price(tokens: list[dict], market: dict) -> float:
                        if tokens:
                            yes_tok = next(
                                (t for t in tokens if t.get("outcome", "").upper() == "YES"), None
                            )
                            if yes_tok:
                                return float(yes_tok.get("price", 0.5) or 0.5)
                        return float(
                            market.get("bestAsk", market.get("lastTradePrice", 0.5)) or 0.5
                        )

                    p1 = yes_price(tokens1, m1)
                    p2 = yes_price(tokens2, m2)

                    # Cross-market arb: "Will X happen" + "Will X NOT happen" should sum to 1
                    # If p1 + (1 - p2) > 1 + edge, buy YES on m1 and NO on m2
                    combo_sum = p1 + (1.0 - p2)
                    if combo_sum < (1.0 + self.min_edge_pct / 100.0):
                        # Try the other combination
                        combo_sum = (1.0 - p1) + p2
                        if combo_sum < (1.0 + self.min_edge_pct / 100.0):
                            continue
                        # Swap so we always describe: buy m1_side + m2_side
                        m1_side, m2_side = "NO", "YES"
                        m1_price, m2_price = 1.0 - p1, p2
                    else:
                        m1_side, m2_side = "YES", "NO"
                        m1_price, m2_price = p1, 1.0 - p2

                    profit_pct = (combo_sum - 1.0) / 1.0 * 100
                    confidence = min(0.95, 0.75 + profit_pct / 40.0)

                    return Signal(
                        symbol=f"{m1.get('slug', 'MKT1')}|{m2.get('slug', 'MKT2')}",
                        side="buy",
                        confidence=confidence,
                        strategy_name=self.name,
                        strategy_type=self.strategy_type,
                        risk_bucket=self.risk_bucket,
                        metadata={
                            "market_1_id": m1.get("id", m1.get("conditionId", "")),
                            "market_1_question": m1.get("question", "")[:80],
                            "market_1_side": m1_side,
                            "market_1_price": round(m1_price, 4),
                            "market_2_id": m2.get("id", m2.get("conditionId", "")),
                            "market_2_question": m2.get("question", "")[:80],
                            "market_2_side": m2_side,
                            "market_2_price": round(m2_price, 4),
                            "combo_sum": round(combo_sum, 4),
                            "profit_pct": round(profit_pct, 2),
                            "arb_type": "cross_market_hedge",
                        },
                    )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Proxy backtest using yes_price + no_price columns.
        Signals when sum of correlated pair diverges past min_edge threshold.
        Uses shift(1) to prevent lookahead bias.
        """
        false_series = pd.Series(False, index=df.index)
        if "yes_price" not in df.columns or "no_price" not in df.columns:
            return BacktestSignals(entries=false_series, exits=false_series)

        yes = df["yes_price"].shift(1)
        no = df["no_price"].shift(1)
        price_sum = yes + no

        entries = (price_sum < self.max_sum).fillna(False)
        exits = (price_sum >= 0.99).fillna(False)

        return BacktestSignals(entries=entries.astype(bool), exits=exits.astype(bool))
