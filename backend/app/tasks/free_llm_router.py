from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_TEMPERATURE: float = 0.3
RACE_MAX_TOKENS: int = 2048
RACE_TIMEOUT: float = 30.0
CONSENSUS_MAX_TOKENS: int = 512
CONSENSUS_TIMEOUT: float = 40.0

HEADER_AUTH_PREFIX: str = "Bearer "
HEADER_CONTENT_TYPE: str = "application/json"

LOG_NO_API_KEYS: str = "free_llm_router: no API keys configured"
LOG_RACE_WINNER: str = "LLM race winner: %s (%.0fms)"

DISABLED_API_KEY_VALUES: tuple[str, ...] = ("", "disabled")

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


PROVIDERS: list[LLMProvider] = [
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


# ── Core caller ───────────────────────────────────────────────────────────────

async def _call_provider(
    provider: LLMProvider,
    messages: list[dict],
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int | None = None,
) -> LLMResponse | None:
    api_key = os.getenv(provider.env_key, "")
    if not api_key or api_key in DISABLED_API_KEY_VALUES:
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
                    "Authorization": f"{HEADER_AUTH_PREFIX}{api_key}",
                    "Content-Type": HEADER_CONTENT_TYPE,
                },
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
    messages: list[dict],
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = RACE_MAX_TOKENS,
    timeout: float = RACE_TIMEOUT,
) -> LLMResponse | None:
    """Call all available providers in parallel; return the first successful response."""
    tasks = {
        asyncio.create_task(_call_provider(p, messages, temperature, max_tokens)): p
        for p in PROVIDERS
        if os.getenv(p.env_key, "") not in DISABLED_API_KEY_VALUES
    }
    if not tasks:
        logger.warning(LOG_NO_API_KEYS)
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
            logger.info(LOG_RACE_WINNER, result.provider, result.latency_ms)
            return result
    return None


async def call_consensus(
    messages: list[dict],
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = CONSENSUS_MAX_TOKENS,
    timeout: float = CONSENSUS_TIMEOUT,
) -> list[LLMResponse]:
    """Call all providers and return all successful responses for consensus analysis."""
    tasks = [
        _call_provider(p, messages, temperature, max_tokens)
        for p in PROVIDERS
        if os.getenv(p.env_key, "") not in DISABLED_API_KEY_VALUES
    ]
    if not tasks:
        return []
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, LLMResponse)]


def available_providers() -> list[str]:
    """Return names of providers with configured API keys."""
    return [p.name for p in PROVIDERS if os.getenv(p.env_key, "") not in DISABLED_API_KEY_VALUES]