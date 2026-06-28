"""
Research Pipeline — runs every 4 hours via APScheduler.

Pipeline:
  1. Fetch recent market data summary (prices, volumes, regimes)
  2. Call free LLM to identify SOTA research opportunities
  3. Generate experiment configs for promising ideas
  4. Queue experiments to run_experiment.py async subprocess
  5. Store findings in AgentMemory for strategy_generator to act on
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, List

from app.tasks.agent_memory import AgentMemory
from app.tasks.free_llm_router import call_race

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(__file__).parent.parent.parent.parent / "experiments"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
RESULTS_DIR = EXPERIMENTS_DIR / "results"


class ResearchPipeline:
    def __init__(self, redis_client: Any = None):
        self._memory = AgentMemory(redis_client) if redis_client else None

    async def run(self) -> None:
        logger.info("ResearchPipeline: starting 4h research cycle")
        try:
            context = await self._build_market_context()
            ideas = await self._generate_research_ideas(context)
            # Guard against None returns
            ideas = ideas or []
            configs = await self._ideas_to_experiment_configs(ideas)
            # Guard against None returns
            configs = configs or []
            await self._queue_experiments(configs)
            if self._memory:
                await self._memory.write(
                    "research_findings",
                    {
                        "ideas_count": len(ideas),
                        "configs_queued": len(configs),
                        "ideas": ideas[:3],
                    },
                )
            logger.info("ResearchPipeline: queued %d experiments", len(configs))
        except Exception as e:
            logger.exception("ResearchPipeline error: %s", e)

    async def _build_market_context(self) -> str:
        """Build a short market context string from memory (if available)."""
        if not self._memory:
            return "Unknown market regime. Assume neutral conditions."

        regime_data = await self._memory.get_latest("market_regime")
        platform_data = await self._memory.get_latest("platform_health")
        recent_suggestions = await self._memory.read_recent("llm_suggestions", n=3) or []

        regime = regime_data.get("regime", "unknown") if isinstance(regime_data, dict) else "unknown"
        health = platform_data.get("health_ratio", 0.5) if isinstance(platform_data, dict) else 0.5

        prev_ideas = [
            s.get("suggestion", "")[:100] for s in recent_suggestions if isinstance(s, dict)
        ]

        return (
            f"Current market regime: {regime}. "
            f"Platform health (% profitable strategies): {health:.0%}. "
            f"Recent LLM suggestions: {'; '.join(prev_ideas[:2])}"
        )

    async def _generate_research_ideas(self, context: str | None) -> List[dict]:
        """Ask the LLM for research ideas based on the provided market context."""
        context = context or ""
        prompt = f"""You are a quantitative trading researcher.

Market context: {context}

Generate 3 experiment ideas to improve trading performance. Each idea must:
- Be implementable with existing indicators (RSI, MACD, EMA, BB, ATR, ADX, VWAP)
- Have a clear hypothesis
- Specify: model type (lstm/xgboost/manual), symbol (BTC/USDT or SPY), interval (1h/1d)

Respond as JSON array:
[{{"name": "idea_name", "hypothesis": "...", "model": "lstm|xgboost|manual", "symbol": "BTC/USDT|SPY", "interval": "1h|1d", "features": ["rsi_14", ...]}}]"""

        response = await call_race(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=800,
        )
        if not response or not getattr(response, "content", None):
            return []

        try:
            content = response.content.strip()
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
                if isinstance(parsed, list):
                    return parsed
        except Exception as e:
            logger.warning("ResearchPipeline: failed to parse LLM ideas: %s", e)
        return []

    async def _ideas_to_experiment_configs(self, ideas: List[dict]) -> List[Path]:
        """Convert LLM ideas to YAML experiment configs."""
        if not ideas:
            return []

        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        configs: List[Path] = []

        for idea in ideas[:2]:  # limit to 2 per cycle
            if not isinstance(idea, dict):
                continue

            name = idea.get("name") or f"auto_{int(time.time())}"
            # Ensure filename safety: replace path separators
            safe_name = name.replace("/", "_").replace("\\", "_")
            model = idea.get("model", "lstm")
            symbol = idea.get("symbol", "BTC/USDT")
            interval = idea.get("interval", "1h")
            features = idea.get("features", ["rsi_14", "macd", "bb_width"])
            if not isinstance(features, list):
                features = ["rsi_14", "macd", "bb_width"]

            config_path = CONFIGS_DIR / f"{safe_name}.yaml"
            if config_path.exists():
                continue

            yaml_content = f"""# Auto-generated by ResearchPipeline at {datetime.now(UTC).isoformat()}
# Hypothesis: {idea.get('hypothesis', 'N/A')}
experiment:
  name: "{safe_name}"
  model: "{model}"
  symbol: "{symbol}"
  exchange: "{'binance' if '/' in symbol else 'alpaca'}"
  interval: "{interval}"

data:
  train_start: "2022-01-01"
  train_end: "2023-12-31"
  val_start: "2024-01-01"
  val_end: "2024-06-30"
  test_start: "2024-07-01"
  test_end: "2024-12-31"

features:
  technical: {json.dumps(features)}
  lookback: 60

model_params:
  hidden_size: 128
  num_layers: 2
  dropout: 0.3
  bidirectional: true

training:
  epochs: 50
  batch_size: 256
  lr: 0.001
  optimizer: "adamw"
  scheduler: "cosine"
  early_stopping_patience: 8

strategy:
  name: "ml_momentum"
  confidence_threshold: 0.60
"""
            config_path.write_text(yaml_content)
            configs.append(config_path)
            logger.info("ResearchPipeline: created config %s", config_path.name)

        return configs

    async def _queue_experiments(self, configs: List[Path]) -> None:
        """Fire-and-forget experiment runs as background subprocesses."""
        if not configs:
            logger.info("ResearchPipeline: no configs to queue")
            return

        script = EXPERIMENTS_DIR / "run_experiment.py"
        if not script.exists():
            logger.warning("ResearchPipeline: run_experiment.py not found at %s", script)
            return

        for config in configs:
            try:
                subprocess.Popen(
                    ["python", str(script), "--config", config.name],
                    cwd=str(EXPERIMENTS_DIR),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info("ResearchPipeline: queued experiment %s", config.name)
            except Exception as e:
                logger.warning("ResearchPipeline: failed to queue %s: %s", config.name, e)