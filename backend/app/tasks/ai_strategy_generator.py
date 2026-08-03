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
                await self._memory.write("strategy_proposals", {
                    "count": len(written),
                    "proposals": [w.get("name", "?") for w in written],
                    "status": "staging",
                })
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
                proposals = json.loads(content[start:end])
                for p in proposals:
                    name = p.get("name", "")
                    if name and name not in seen:
                        seen.add(name)
                        all_proposals.append(p)
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

        # Helper to generate indicator calculation snippets and condition expressions
        def build_body(conditions: List[str], prefix: str) -> (str, str):
            """Returns (indicator_setup_code, combined_condition_code)."""
            indicator_code = ""
            locals_map = {}
            for cond in conditions:
                tokens = cond.replace(">", " > ").replace("<", " < ").replace(">=", " >= ").replace("<=", " <= ").split()
                if len(tokens) != 3:
                    continue
                indicator, op, value = tokens
                # Normalise indicator names to variables
                var_name = indicator.replace(".", "_")
                if var_name not in locals_map:
                    if indicator.startswith("rsi"):
                        indicator_code += f"        {var_name} = ta.rsi(close, length=14).fillna(50)\n"
                    elif indicator.startswith("ema_"):
                        length = int(indicator.split("_")[1])
                        indicator_code += f"        {var_name} = ta.ema(close, length={length}).fillna(close)\n"
                    elif indicator == "price":
                        indicator_code += f"        {var_name} = close\n"
                    else:
                        # Fallback: treat as raw series (e.g., bollinger bands)
                        indicator_code += f"        {var_name} = ta.{indicator}(close).fillna(0)\n"
                    locals_map[indicator] = var_name
                # Build comparison string
                locals_map.setdefault(indicator, var_name)
            # Combine conditions using logical AND
            condition_expr = " & ".join(
                f"({cond.replace(ind, locals_map.get(ind, ind))})" for cond in conditions
            )
            return indicator_code, condition_expr

        # Build backtest body
        backtest_body = "        close = df['close']\n"
        entry_setup, entry_expr = build_body(entry_conditions, "entry")
        backtest_body += entry_setup
        backtest_body += "        entries = pd.Series(False, index=df.index)\n"
        # Confirmation: require entry condition to hold for 2 consecutive bars
        backtest_body += f"        entries = ({entry_expr}).rolling(2).sum() == 2\n"
        exit_setup, exit_expr = build_body(exit_conditions, "exit")
        backtest_body += exit_setup
        backtest_body += "        exits = pd.Series(False, index=df.index)\n"
        backtest_body += f"        exits = {exit_expr}\n"

        # Build analyze body
        analyze_body = "        close = data['close']\n"
        entry_setup_an, entry_expr_an = build_body(entry_conditions, "entry")
        analyze_body += entry_setup_an
        exit_setup_an, exit_expr_an = build_body(exit_conditions, "exit")
        analyze_body += exit_setup_an
        # Confirmation in live analysis: require current bar meets entry_expr and previous bar also met it
        analyze_body += "        if (" + entry_expr_an + ").iloc[-1] and (" + entry_expr_an + ").iloc[-2]:\n"
        analyze_body += "            return Signal(symbol=symbol, side='buy', confidence=0.70, strategy=self.name)\n"
        analyze_body += "        if " + exit_expr_an + ".iloc[-1]:\n"
        analyze_body += "            return Signal(symbol=symbol, side='sell', confidence=0.70, strategy=self.name)\n"

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