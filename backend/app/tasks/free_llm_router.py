"""
Free LLM Router — dispatches to 7 free providers in parallel.

The router attempts to contact each configured free LLM provider according to a
priority cascade (fastest/highest‑quota first).  Three operating modes are
available:

* ``race`` – the first successful response wins and the remaining requests are
  cancelled.
* ``consensus`` – all providers are queried; the caller can perform a majority
  vote on the returned answers.
* ``best_of`` – not implemented here, but could be added to select the longest
  coherent answer.

Only providers with a valid API key (environment variable) are contacted.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Dict

import httpx

logger = logging.getLogger(__name__)

# ── Provider definitions ──────────────────────────────────────────────────────


@dataclass
class LLMProvider:
    """
    Configuration for a single LLM provider.

    Attributes
    ----------
    name: str
        Human‑readable identifier used in logs and responses.
    env_key: str
        Name of the environment variable that stores the provider's API key.
    base_url: str
        Base URL for the provider's OpenAI‑compatible endpoint.
    model: str
        Model identifier to request from the provider.
    max_tokens: int, default 2048
        Upper bound for the number of tokens the provider may generate.
    timeout: float, default 30.0
        HTTP client timeout (seconds) for the request.
    headers_extra: dict
        Additional HTTP headers that should be sent with each request.
    """

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
    """
    Normalised response from a provider.

    Attributes
    ----------
    provider: str
        Name of the provider that generated the response.
    content: str
        Generated text content.
    latency_ms: float
        Round‑trip latency measured in milliseconds.
    tokens_used: int, default 0
        Number of tokens reported by the provider for the request.
    """

    provider: str
    content: str
    latency_ms: float
    tokens_used: int = 0


# ── Core caller ───────────────────────────────────────────────────────────────


async def _call_provider(
    provider: LLMProvider,
    messages: List[Dict[str, Any]],
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> LLMResponse | None:
    """
    Send a chat completion request to a single provider.

    Parameters
    ----------
    provider: LLMProvider
        The provider configuration to use.
    messages: list[dict]
        Message history in OpenAI chat format.
    temperature: float, default 0.3
        Sampling temperature for the model.
    max_tokens: int | None, default None
        Maximum tokens to generate; falls back to the provider's default.

    Returns
    -------
    LLMResponse | None
        A populated ``LLMResponse`` on success, or ``None`` if the request
        fails or the API key is missing/disabled.
    """
    api_key = os.getenv(provider.env_key, "")
    if not api_key or api_key in ("disabled", ""):
        return None

    payload = {
        "model": provider.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens or provider.max_tokens,
    }

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=provider.timeout) as client:
            resp = await client.post(
                f"{provider.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            latency = (time.monotonic() - t0) * 1000
            return LLMResponse(provider=provider.name, content=content, latency_ms=latency, tokens_used=tokens)
    except Exception as e:
        logger.debug("Provider %s failed: %s", provider.name, e)
        return None


# ── Public API ─────────────────────────────────────────────────────────────────


async def call_race(
    messages: List[Dict[str, Any]],
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: float = 30.0,
) -> LLMResponse | None:
    """
    Query all configured providers in parallel and return the first successful response.

    Parameters
    ----------
    messages: list[dict]
        Chat history to send to each provider.
    temperature: float, default 0.3
        Sampling temperature for the model.
    max_tokens: int, default 2048
        Upper bound for generated tokens.
    timeout: float, default 30.0
        Overall timeout for the race operation (seconds).

    Returns
    -------
    LLMResponse | None
        The winning provider's response, or ``None`` if no provider succeeded.
    """
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
    messages: List[Dict[str, Any]],
    temperature: float = 0.3,
    max_tokens: int = 512,
    timeout: float = 40.0,
) -> List[LLMResponse]:
    """
    Query all configured providers and collect their successful responses.

    This mode is intended for downstream consensus logic (e.g., majority voting).

    Parameters
    ----------
    messages: list[dict]
        Chat history to send to each provider.
    temperature: float, default 0.3
        Sampling temperature for the model.
    max_tokens: int, default 512
        Upper bound for generated tokens.
    timeout: float, default 40.0
        Overall timeout for the operation (seconds).

    Returns
    -------
    list[LLMResponse]
        A list containing the successful responses from each provider.
    """
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
    """
    Retrieve the names of providers that have a usable API key configured.

    Returns
    -------
    list[str]
        Provider identifiers that can be used by the router.
    """
    return [p.name for p in PROVIDERS if os.getenv(p.env_key, "") not in ("", "disabled")]