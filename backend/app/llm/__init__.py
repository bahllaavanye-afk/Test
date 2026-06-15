"""Shared free-LLM gateway for QuantEdge autonomous agents.

Every agent (API-triggered or autonomous) reasons through one gateway so that:
  * there is a single free-provider chain (Groq -> DeepSeek -> Gemini),
  * per-agent token spend is tracked centrally,
  * no paid API is ever required for the platform to run.

If no free provider is configured/reachable the gateway returns ``None`` and
callers degrade honestly (no fabricated analysis).
"""
from app.llm.gateway import (
    complete,
    complete_sync,
    providers_configured,
)

__all__ = ["complete", "complete_sync", "providers_configured"]
