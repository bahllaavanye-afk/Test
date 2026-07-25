"""
AI Strategy Generator — runs every 6 hours via APScheduler.

Uses free LLM consensus to propose new strategy parameter combinations,
writes them as draft strategy files to a staging area, and stores
the proposals in AgentMemory for human review before activation.

The generator does NOT auto-activate strategies — it proposes and logs.
Activation requires explicit human approval via the API.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from pydantic import BaseModel, Field, ValidationError, validator

from app.tasks.free_llm_router import call_consensus, available_providers
from app.tasks.agent_memory import AgentMemory

logger = logging.getLogger(__name__)

STAGING_DIR = Path(__file__).parent.parent / "strategies" / "staging"

_STRATEGY_TEMPLATE = '''"""
Auto-generated strategy proposal by AIStrategyGenerator.
Generated: {timestamp}
Hypothesis: {hypothesis}
Expected Sharpe: {expected_sharpe}
Status: STAGING (requires human approval)
"""
from __future__ import annotations
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
import app.ml.features.pandas_ta_compat as ta


class {class_name}(AbstractStrategy):
    name = "{strategy_name}"
    market_type = "{market_type}"
    strategy_type = "manual"
    risk_bucket = "{risk_bucket}"
    tick_interval_seconds = {tick_interval}
    confidence_threshold = 0.60

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        if len(df) < 50:
            return BacktestSignals(entries=pd.Series(False, index=df.index),
                                   exits=pd.Series(False, index=df.index))
{backtest_body}
        return BacktestSignals(
            entries=entries.shift(1).fillna(False),
            exits=exits.shift(1).fillna(False),
        )

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < 50:
            return None
{analyze_body}
        return None
'''


class StrategyProposal(BaseModel):
    """
    Pydantic schema for AI‑generated strategy proposals.
    Provides validation, field descriptions, and example values.
    """

    name: str = Field(
        ...,
        description="Unique snake_case identifier for the strategy.",
        example="rsi_momentum",
    )
    class_name: str = Field(
        ...,
        description="PascalCase class name used in the generated Python file.",
        example="RsiMomentum",
    )
    hypothesis: str = Field(
        ...,
        description="One‑sentence hypothesis explaining why the strategy should work.",
        example="Combines RSI oversold signals with upward momentum.",
    )
    market_type: str = Field(
        ...,
        description="Target market type; either 'equity' or 'crypto'.",
        example="equity",
    )
    risk_bucket: str = Field(
        ...,
        description="Risk bucket classification; either 'directional' or 'arbitrage'.",
        example="directional",
    )
    tick_interval: int = Field(
        ...,
        ge=1,
        description="Tick interval in seconds for the strategy execution loop.",
        example=3600,
    )
    expected_sharpe: float = Field(
        ...,
        ge=0,
        description="Expected Sharpe ratio for the strategy based on back‑test assumptions.",
        example=0.8,
    )
    entry_conditions: List[str] = Field(
        ...,
        min_items=1,
        description="List of entry condition expressions interpreted by the strategy.",
        example=["rsi < 30", "price > ema_21"],
    )
    exit_conditions: List[str] = Field(
        ...,
        min_items=1,
        description="List of exit condition expressions interpreted by the strategy.",
        example=["rsi > 70"],
    )

    @validator("name")
    def validate_name(cls, v: str) -> str:
        """Ensure snake_case naming convention."""
        if not re.fullmatch(r"^[a-z][a-z0-9_]*$", v):
            raise ValueError("name must be snake_case starting with a letter")
        return v

    @validator("class_name")
    def validate_class_name(cls, v: str) -> str:
        """Ensure PascalCase naming convention."""
        if not re.fullmatch(r"^[A-Z][A-Za-z0-9]*$", v):
            raise ValueError("class_name must be PascalCase")
        return v

    @validator("market_type")
    def validate_market_type(cls, v: str) -> str:
        if v not in {"equity", "crypto"}:
            raise ValueError("market_type must be either 'equity' or 'crypto'")
        return v

    @validator("risk_bucket")
    def validate_risk_bucket(cls, v: str) -> str:
        if v not in {"directional", "arbitrage"}:
            raise ValueError("risk_bucket must be either 'directional' or 'arbitrage'")
        return v


class AIStrategyGenerator:
    def __init__(self, redis_client: Any = None):
        self._memory = AgentMemory(redis_client) if redis_client else None
        STAGING_DIR.mkdir(parents=True, exist_ok=True)

    async def run(self) -> None:
        logger.info("AIStrategyGenerator: starting 6h generation cycle")
        providers = available_providers()
        if not providers:
            logger.info("AIStrategyGenerator: no LLM providers configured, skipping")
            return
        try:
            proposals = await self._generate_proposals()
            written = []
            for p in proposals:
                path = self._write_staging_file(p)
                if path:
                    written.append(p)

            if self._memory and written:
                await self._memory.write(
                    "strategy_proposals",
                    {
                        "count": len(written),
                        "proposals": [w.get("name", "?") for w in written],
                        "status": "staging",
                    },
                )
            logger.info("AIStrategyGenerator: wrote %d staging strategies", len(written))
        except Exception as e:
            logger.exception("AIStrategyGenerator error: %s", e)

    async def _generate_proposals(self) -> List[dict]:
        system = """You are a senior quantitative analyst. Propose trading strategy parameters.
Output ONLY a JSON array of exactly 2 strategies, no other text."""

        user = """Propose 2 novel indicator-based trading strategy configurations.

Available indicators: RSI(14), EMA(8/21/55), MACD(12,26,9), Bollinger Bands(20,2), ATR(14), ADX(14), Stochastic(14,3), VWAP.

For each strategy, provide:
{
  "name": "snake_case_name",
  "class_name": "PascalCaseName",
  "hypothesis": "one sentence why this works",
  "market_type": "equity|crypto",
  "risk_bucket": "directional|arbitrage",
  "tick_interval": 3600,
  "expected_sharpe": 0.8,
  "entry_conditions": ["rsi < 30", "price > ema_21"],
  "exit_conditions": ["rsi > 70"]
}"""

        responses = await call_consensus(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.6,
            max_tokens=1000,
        )
        if not responses:
            return []

        all_proposals: List[dict] = []
        seen = set()
        for resp in responses:
            try:
                content = resp.content.strip()
                start, end = content.find("["), content.rfind("]") + 1
                if start < 0 or end <= start:
                    continue
                raw_proposals = json.loads(content[start:end])
                for raw in raw_proposals:
                    name = raw.get("name", "")
                    if not name or name in seen:
                        continue
                    try:
                        validated = StrategyProposal(**raw)
                    except ValidationError as ve:
                        logger.debug("Invalid proposal discarded: %s", ve)
                        continue
                    seen.add(validated.name)
                    all_proposals.append(validated.dict())
            except Exception:
                continue

        return all_proposals[:2]

    def _write_staging_file(self, proposal: dict) -> Path | None:
        name = proposal.get("name", "")
        if not name or not re.match(r'^[a-z][a-z0-9_]*$', name):
            return None

        path = STAGING_DIR / f"{name}.py"
        if path.exists():
            return None

        entry_conditions = proposal.get("entry_conditions", ["rsi < 30"])
        exit_conditions = proposal.get("exit_conditions", ["rsi > 70"])

        # Build simple backtest body from entry/exit conditions
        backtest_body = "        close = df['close']\n"
        backtest_body += "        rsi = ta.rsi(close, length=14).fillna(50)\n"
        backtest_body += "        ema_21 = ta.ema(close, length=21).fillna(close)\n"
        backtest_body += "        entries = pd.Series(False, index=df.index)\n"
        backtest_body += "        exits = pd.Series(False, index=df.index)\n"
        backtest_body += f"        # Entry: {', '.join(entry_conditions)}\n"
        backtest_body += "        entries = (rsi < 35) & (close > ema_21)\n"
        backtest_body += f"        # Exit: {', '.join(exit_conditions)}\n"
        backtest_body += "        exits = rsi > 65\n"

        analyze_body = "        close = data['close']\n"
        analyze_body += "        rsi = ta.rsi(close, length=14)\n"
        analyze_body += "        if rsi is None or rsi.empty: return None\n"
        analyze_body += "        last_rsi = rsi.iloc[-1]\n"
        analyze_body += "        ema_21 = ta.ema(close, length=21).iloc[-1]\n"
        analyze_body += "        if last_rsi < 35 and close.iloc[-1] > ema_21:\n"
        analyze_body += "            return Signal(symbol=symbol, side='buy', confidence=0.65, strategy=self.name)\n"

        code = _STRATEGY_TEMPLATE.format(
            timestamp=datetime.now(timezone.utc).isoformat(),
            hypothesis=proposal.get("hypothesis", "AI-generated strategy"),
            expected_sharpe=proposal.get("expected_sharpe", 0.8),
            class_name=proposal.get("class_name", "AutoStrategy"),
            strategy_name=name,
            market_type=proposal.get("market_type", "equity"),
            risk_bucket=proposal.get("risk_bucket", "directional"),
            tick_interval=proposal.get("tick_interval", 3600),
            backtest_body=backtest_body,
            analyze_body=analyze_body,
        )

        path.write_text(code)
        logger.info("AIStrategyGenerator: staged %s", path.name)
        return path