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
from typing import Any, Dict, List

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


class AIStrategyGenerator:
    def __init__(self, redis_client: Any = None):
        """
        Initialise the generator.

        Args:
            redis_client: Optional Redis client used by AgentMemory.
                If provided, it must implement a ``write`` coroutine.
        Raises:
            ValueError: If ``redis_client`` is not ``None`` and does not provide
                the required interface.
        """
        if redis_client is not None and not hasattr(redis_client, "write"):
            raise ValueError("redis_client must have a 'write' method or be None")
        self._memory = AgentMemory(redis_client) if redis_client else None
        STAGING_DIR.mkdir(parents=True, exist_ok=True)

    async def run(self) -> None:
        """Execute a generation cycle."""
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

    async def _generate_proposals(self) -> List[Dict]:
        """Request the LLM to produce up to two strategy proposals."""
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

        all_proposals: List[Dict] = []
        seen = set()
        for resp in responses:
            try:
                content = resp.content.strip()
                start, end = content.find("["), content.rfind("]") + 1
                if start < 0 or end <= start:
                    continue
                proposals = json.loads(content[start:end])
                for p in proposals:
                    name = p.get("name", "")
                    if name and name not in seen:
                        seen.add(name)
                        all_proposals.append(p)
            except Exception:
                continue

        return all_proposals[:2]

    def _write_staging_file(self, proposal: Dict) -> Path | None:
        """
        Validate a proposal and write it to the staging directory.

        Args:
            proposal: Dictionary containing strategy specifications.

        Returns:
            Path to the written file, or ``None`` if the file could not be written.

        Raises:
            ValueError: If required fields are missing or have invalid types/values.
        """
        if not isinstance(proposal, dict):
            raise ValueError("Proposal must be a dictionary")

        # Required string fields
        required_str_fields = ["name", "class_name", "hypothesis", "market_type", "risk_bucket"]
        for field in required_str_fields:
            value = proposal.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Proposal field '{field}' must be a non‑empty string")

        # Validate name pattern
        name = proposal["name"]
        if not re.match(r"^[a-z][a-z0-9_]*$", name):
            raise ValueError(
                f"Proposal name '{name}' is invalid; must match '^[a-z][a-z0-9_]*$'"
            )

        # Validate enumerated fields
        if proposal["market_type"] not in {"equity", "crypto"}:
            raise ValueError(
                f"market_type must be 'equity' or 'crypto', got '{proposal['market_type']}'"
            )
        if proposal["risk_bucket"] not in {"directional", "arbitrage"}:
            raise ValueError(
                f"risk_bucket must be 'directional' or 'arbitrage', got '{proposal['risk_bucket']}'"
            )

        # Validate numeric fields
        tick_interval = proposal.get("tick_interval")
        if not isinstance(tick_interval, int) or tick_interval <= 0:
            raise ValueError("tick_interval must be a positive integer")

        expected_sharpe = proposal.get("expected_sharpe")
        if not isinstance(expected_sharpe, (int, float)):
            raise ValueError("expected_sharpe must be a number")

        # Validate condition lists
        entry_conditions = proposal.get("entry_conditions")
        exit_conditions = proposal.get("exit_conditions")
        if not isinstance(entry_conditions, list) or not all(isinstance(c, str) for c in entry_conditions):
            raise ValueError("entry_conditions must be a list of strings")
        if not isinstance(exit_conditions, list) or not all(isinstance(c, str) for c in exit_conditions):
            raise ValueError("exit_conditions must be a list of strings")

        path = STAGING_DIR / f"{name}.py"
        if path.exists():
            return None

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