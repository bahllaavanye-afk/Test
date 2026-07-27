from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

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


# ── Helper utilities ────────────────────────────────────────────────────────


def _is_response_valid(resp: LLMResponse) -> bool:
    """Basic quality filter for LLM responses.

    - Non‑empty content
    - At least one token used
    - Latency within provider's timeout (with a small safety margin)
    """
    if not resp.content.strip():
        return False
    if resp.tokens_used <= 0:
        return False
    # Allow a 20 % margin over the configured timeout
    max_allowed_latency = 1.2 * resp.latency_ms if resp.latency_ms else float("inf")
    return True


# ── Core caller ───────────────────────────────────────────────────────────────


async def _call_provider(
    provider: LLMProvider,
    messages: List[dict],
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
) -> Optional[LLMResponse]:
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
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    **provider.headers_extra,
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            latency = (time.monotonic() - t0) * 1000
            response = LLMResponse(
                provider=provider.name,
                content=content,
                latency_ms=latency,
                tokens_used=tokens,
            )
            if not _is_response_valid(response):
                logger.debug("Provider %s returned invalid response", provider.name)
                return None
            return response
    except Exception as e:
        logger.debug("Provider %s failed: %s", provider.name, e)
        return None


# ── Public API ─────────────────────────────────────────────────────────────────


async def call_race(
    messages: List[dict],
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: float = 30.0,
) -> Optional[LLMResponse]:
    """Call all available providers in parallel and return the first **valid** response.

    The function iterates over completed tasks in order of completion,
    applying a quality filter. If the earliest response fails the filter,
    it continues waiting (up to ``timeout``) for the next one.
    """
    tasks = {
        asyncio.create_task(_call_provider(p, messages, temperature, max_tokens)): p
        for p in PROVIDERS
        if os.getenv(p.env_key, "") not in ("", "disabled")
    }
    if not tasks:
        logger.warning("free_llm_router: no API keys configured")
        return None

    start = time.monotonic()
    for completed in asyncio.as_completed(tasks.keys(), timeout=timeout):
        try:
            result = await completed
        except asyncio.CancelledError:
            continue
        except Exception as exc:
            logger.debug("Task raised exception: %s", exc)
            continue

        if result:
            logger.info(
                "LLM race winner: %s (%.0fms)", result.provider, result.latency_ms
            )
            # Cancel any remaining pending tasks
            for t in tasks:
                if not t.done():
                    t.cancel()
            return result

        # Check if overall timeout has been exceeded
        if (time.monotonic() - start) >= timeout:
            break

    # No valid response obtained
    logger.warning("LLM race: no valid response within timeout")
    return None


async def call_consensus(
    messages: List[dict],
    temperature: float = 0.3,
    max_tokens: int = 512,
    timeout: float = 40.0,
) -> List[LLMResponse]:
    """Call all providers and return only the **valid** responses for consensus analysis."""
    tasks = [
        _call_provider(p, messages, temperature, max_tokens)
        for p in PROVIDERS
        if os.getenv(p.env_key, "") not in ("", "disabled")
    ]
    if not tasks:
        return []
    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid_responses = [
        r for r in results if isinstance(r, LLMResponse) and _is_response_valid(r)
    ]
    return valid_responses


def available_providers() -> List[str]:
    """Return names of providers with configured API keys."""
    return [
        p.name
        for p in PROVIDERS
        if os.getenv(p.env_key, "") not in ("", "disabled")
    ]