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

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.tasks.free_llm_router import call_race, call_consensus
from app.tasks.agent_memory import AgentMemory

logger = logging.getLogger(__name__)

# Directory constants
EXPERIMENTS_DIR = Path(__file__).parent.parent.parent.parent / "experiments"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

# Default configuration constants
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 800
DEFAULT_IDEA_LIMIT = 2
DEFAULT_MODEL = "lstm"
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_INTERVAL = "1h"
DEFAULT_FEATURES = ["rsi_14", "macd", "bb_width"]
DEFAULT_TRAIN_START = "2022-01-01"
DEFAULT_TRAIN_END = "2023-12-31"
DEFAULT_VAL_START = "2024-01-01"
DEFAULT_VAL_END = "2024-06-30"
DEFAULT_TEST_START = "2024-07-01"
DEFAULT_TEST_END = "2024-12-31"
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.3
DEFAULT_BIDIRECTIONAL = True
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 0.001
DEFAULT_EARLY_STOPPING_PATIENCE = 8
DEFAULT_CONFIDENCE_THRESHOLD = 0.60
DEFAULT_HEALTH_RATIO = 0.5
DEFAULT_REGIME = "unknown"
DEFAULT_CONTEXT_PLACEHOLDER = "Unknown market regime. Assume neutral conditions."
DEFAULT_EXPERIMENT_NAME_PREFIX = "auto_"


class ResearchPipeline:
    def __init__(self, redis_client: Any = None):
        self._memory = AgentMemory(redis_client) if redis_client else None

    async def run(self) -> None:
        logger.info("ResearchPipeline: starting 4h research cycle")
        try:
            context = await self._build_market_context()
            ideas = await self._generate_research_ideas(context)
            configs = await self._ideas_to_experiment_configs(ideas)
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
            return DEFAULT_CONTEXT_PLACEHOLDER
        regime_data = await self._memory.get_latest("market_regime")
        platform_data = await self._memory.get_latest("platform_health")
        recent_suggestions = await self._memory.read_recent("llm_suggestions", n=3)

        regime = regime_data.get("regime", DEFAULT_REGIME) if regime_data else DEFAULT_REGIME
        health = platform_data.get("health_ratio", DEFAULT_HEALTH_RATIO) if platform_data else DEFAULT_HEALTH_RATIO
        prev_ideas = [s.get("suggestion", "")[:100] for s in recent_suggestions]

        return (
            f"Current market regime: {regime}. "
            f"Platform health (% profitable strategies): {health:.0%}. "
            f"Recent LLM suggestions: {'; '.join(prev_ideas[:2])}"
        )

    async def _generate_research_ideas(self, context: str) -> list[dict]:
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
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        if not response:
            return []

        try:
            content = response.content.strip()
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except Exception as e:
            logger.warning("ResearchPipeline: failed to parse LLM ideas: %s", e)
        return []

    async def _ideas_to_experiment_configs(self, ideas: list[dict]) -> list[Path]:
        """Convert LLM ideas to YAML experiment configs."""
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        configs = []

        for idea in ideas[:DEFAULT_IDEA_LIMIT]:
            name = idea.get("name", f"{DEFAULT_EXPERIMENT_NAME_PREFIX}{int(time.time())}")
            model = idea.get("model", DEFAULT_MODEL)
            symbol = idea.get("symbol", DEFAULT_SYMBOL)
            interval = idea.get("interval", DEFAULT_INTERVAL)
            features = idea.get("features", DEFAULT_FEATURES)

            config_path = CONFIGS_DIR / f"{name}.yaml"
            if config_path.exists():
                continue

            yaml_content = f"""# Auto-generated by ResearchPipeline at {datetime.now(timezone.utc).isoformat()}
# Hypothesis: {idea.get('hypothesis', 'N/A')}
experiment:
  name: "{name}"
  model: "{model}"
  symbol: "{symbol}"
  exchange: "{'binance' if '/' in symbol else 'alpaca'}"
  interval: "{interval}"

data:
  train_start: "{DEFAULT_TRAIN_START}"
  train_end: "{DEFAULT_TRAIN_END}"
  val_start: "{DEFAULT_VAL_START}"
  val_end: "{DEFAULT_VAL_END}"
  test_start: "{DEFAULT_TEST_START}"
  test_end: "{DEFAULT_TEST_END}"

features:
  technical: {json.dumps(features)}
  lookback: 60

model_params:
  hidden_size: {DEFAULT_HIDDEN_SIZE}
  num_layers: {DEFAULT_NUM_LAYERS}
  dropout: {DEFAULT_DROPOUT}
  bidirectional: {str(DEFAULT_BIDIRECTIONAL).lower()}

training:
  epochs: {DEFAULT_EPOCHS}
  batch_size: {DEFAULT_BATCH_SIZE}
  lr: {DEFAULT_LR}
  optimizer: "adamw"
  scheduler: "cosine"
  early_stopping_patience: {DEFAULT_EARLY_STOPPING_PATIENCE}

strategy:
  name: "ml_momentum"
  confidence_threshold: {DEFAULT_CONFIDENCE_THRESHOLD}
"""
            config_path.write_text(yaml_content)
            configs.append(config_path)
            logger.info("ResearchPipeline: created config %s", config_path.name)

        return configs

    async def _queue_experiments(self, configs: list[Path]) -> None:
        """Fire-and-forget experiment runs as background subprocesses."""
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