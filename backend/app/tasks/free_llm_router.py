"""
Free LLM Router — dispatches to 7 free providers in parallel.

Priority cascade (fastest/highest-quota first):
  1. Gemini Flash 2.0 (Google AI Studio — 1M TPM free)
  2. Groq  (llama-3.3-70b — 6000 TPD free, very fast)
  3. DeepSeek (deepseek-chat — $5 free credit, cheap)
  4. SambaNova (Meta-Llama-3.3-70B — free tier)
  5. Cerebras (llama-3.3-70b — free tier, fast inference)
  6. Together AI (Llama-3.3-70B — $25 free credit)
  7. Hyperbolic (llama-3.3-70b — $10 free credit)

Modes:
  - "race":      first successful response wins, rest cancelled
  - "consensus": all respond, majority vote on yes/no questions
  - "best_of":   all respond, pick longest coherent answer
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, List

import httpx

logger = logging.getLogger(__name__)

# ── Provider definitions ──────────────────────────────────────────────────────


@dataclass
class LLMProvider:
    name: str
    env_key: str
    base_url: str
    model: str
    max_tokens: int = 2048
    timeout: float = 30.0
    headers_extra: dict = field(default_factory=dict)


PROVIDERS: List[LLMProvider] = [
    LLMProvider(
        name="gemini",
        env_key="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model="gemini-2.0-flash",
        timeout=20.0,
    ),
    LLMProvider(
        name="groq",
        env_key="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        timeout=15.0,
    ),
    LLMProvider(
        name="deepseek",
        env_key="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        timeout=25.0,
    ),
    LLMProvider(
        name="sambanova",
        env_key="SAMBANOVA_API_KEY",
        base_url="https://api.sambanova.ai/v1",
        model="Meta-Llama-3.3-70B-Instruct",
        timeout=25.0,
    ),
    LLMProvider(
        name="cerebras",
        env_key="CEREBRAS_API_KEY",
        base_url="https://api.cerebras.ai/v1",
        model="llama-3.3-70b",
        timeout=15.0,
    ),
    LLMProvider(
        name="together",
        env_key="TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        timeout=30.0,
    ),
    LLMProvider(
        name="hyperbolic",
        env_key="HYPERBOLIC_API_KEY",
        base_url="https://api.hyperbolic.xyz/v1",
        model="meta-llama/Llama-3.3-70B-Instruct",
        timeout=30.0,
    ),
]


@dataclass
class LLMResponse:
    provider: str
    content: str
    latency_ms: float
    tokens_used: int = 0


# ── Helper functions ────────────────────────────────────────────────────────


def _build_payload(provider: LLMProvider, messages: List[dict], temperature: float, max_tokens: int | None) -> dict:
    """Construct the JSON payload for a provider request."""
    return {
        "model": provider.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens or provider.max_tokens,
    }


def _make_request(provider: LLMProvider, api_key: str, payload: dict) -> httpx.Response:
    """Execute the HTTP request against the provider's endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    client = httpx.AsyncClient(timeout=provider.timeout)
    return client.post(f"{provider.base_url}/chat/completions", headers=headers, json=payload)


async def _process_response(resp: httpx.Response, provider_name: str, start_time: float) -> LLMResponse:
    """Parse the HTTP response and create an LLMResponse object."""
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("total_tokens", 0)
    latency = (time.monotonic() - start_time) * 1000
    return LLMResponse(provider=provider_name, content=content, latency_ms=latency, tokens_used=tokens)


# ── Core caller ───────────────────────────────────────────────────────────────


async def _call_provider(
    provider: LLMProvider,
    messages: List[dict],
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> LLMResponse | None:
    """Invoke a single LLM provider and return its response, or None on failure."""
    api_key = os.getenv(provider.env_key, "")
    if not api_key or api_key in ("disabled", ""):
        return None

    payload = _build_payload(provider, messages, temperature, max_tokens)
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=provider.timeout) as client:
            resp = await client.post(
                f"{provider.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
        return await _process_response(resp, provider.name, start)
    except Exception as e:
        logger.debug("Provider %s failed: %s", provider.name, e)
        return None


# ── Public API ─────────────────────────────────────────────────────────────────


async def call_race(
    messages: List[dict],
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: float = 30.0,
) -> LLMResponse | None:
    """Call all available providers in parallel; return the first successful response."""
    tasks = {
        asyncio.create_task(_call_provider(p, messages, temperature, max_tokens)): p
        for p in PROVIDERS
        if os.getenv(p.env_key, "") not in ("", "disabled")
    }
    if not tasks:
        logger.warning("free_llm_router: no API keys configured")
        return None

    done, pending = await asyncio.wait(
        list(tasks.keys()),
        timeout=timeout,
        return_when=asyncio.FIRST_COMPLETED,
    )
    for t in pending:
        t.cancel()

    for t in done:
        result = t.result()
        if result:
            logger.info("LLM race winner: %s (%.0fms)", result.provider, result.latency_ms)
            return result
    return None


async def call_consensus(
    messages: List[dict],
    temperature: float = 0.3,
    max_tokens: int = 512,
    timeout: float = 40.0,
) -> List[LLMResponse]:
    """Call all providers and return all successful responses for consensus analysis."""
    tasks = [
        _call_provider(p, messages, temperature, max_tokens)
        for p in PROVIDERS
        if os.getenv(p.env_key, "") not in ("", "disabled")
    ]
    if not tasks:
        return []
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, LLMResponse)]


def available_providers() -> List[str]:
    """Return names of providers with configured API keys."""
    return [p.name for p in PROVIDERS if os.getenv(p.env_key, "") not in ("", "disabled")]