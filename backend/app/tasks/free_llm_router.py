from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS_RACE = 2048
DEFAULT_TIMEOUT_RACE = 30.0

DEFAULT_MAX_TOKENS_CONSENSUS = 512
DEFAULT_TIMEOUT_CONSENSUS = 40.0

DEFAULT_PROVIDER_MAX_TOKENS = 2048
DEFAULT_PROVIDER_TIMEOUT = 30.0

AUTH_HEADER = "Authorization"
CONTENT_TYPE_HEADER = "Content-Type"
JSON_CONTENT_TYPE = "application/json"
CHAT_COMPLETIONS_ENDPOINT = "/chat/completions"

NO_API_KEYS_MSG = "free_llm_router: no API keys configured"
DISABLED_KEYS = ("", "disabled")

logger = logging.getLogger(__name__)

# ── Provider definitions ──────────────────────────────────────────────────────


@dataclass
class LLMProvider:
    """
    Configuration for an LLM provider.

    Attributes
    ----------
    name: str
        Human‑readable name of the provider.
    env_key: str
        Environment variable name that holds the API key.
    base_url: str
        Base endpoint URL for the provider's API.
    model: str
        Model identifier to be used for completions.
    max_tokens: int
        Maximum number of tokens the provider allows per request.
    timeout: float
        HTTP timeout in seconds for the provider.
    headers_extra: dict
        Any additional headers required by the provider.
    """

    name: str
    env_key: str
    base_url: str
    model: str
    max_tokens: int = DEFAULT_PROVIDER_MAX_TOKENS
    timeout: float = DEFAULT_PROVIDER_TIMEOUT
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
    Normalized response from an LLM provider.

    Attributes
    ----------
    provider: str
        Name of the provider that generated the response.
    content: str
        The textual content returned by the LLM.
    latency_ms: float
        Request latency in milliseconds.
    tokens_used: int
        Number of tokens reported as used for the request.
    """

    provider: str
    content: str
    latency_ms: float
    tokens_used: int = 0


# ── Core caller ───────────────────────────────────────────────────────────────


async def _call_provider(
    provider: LLMProvider,
    messages: List[Dict[str, Any]],
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = None,
) -> Optional[LLMResponse]:
    """
    Perform a single request to the given LLM provider.

    Parameters
    ----------
    provider : LLMProvider
        The provider to query.
    messages : List[Dict[str, Any]]
        List of message dictionaries following the OpenAI chat format.
    temperature : float, optional
        Sampling temperature for the generation; defaults to ``DEFAULT_TEMPERATURE``.
    max_tokens : int | None, optional
        Maximum tokens to generate; if ``None`` the provider's default is used.

    Returns
    -------
    LLMResponse | None
        Normalized response on success, or ``None`` if the request fails or the API key is missing/disabled.
    """
    api_key = os.getenv(provider.env_key, "")
    if not api_key or api_key in DISABLED_KEYS:
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
                f"{provider.base_url}{CHAT_COMPLETIONS_ENDPOINT}",
                headers={AUTH_HEADER: f"Bearer {api_key}", CONTENT_TYPE_HEADER: JSON_CONTENT_TYPE},
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
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS_RACE,
    timeout: float = DEFAULT_TIMEOUT_RACE,
) -> Optional[LLMResponse]:
    """
    Query all configured providers in parallel and return the first successful response.

    Parameters
    ----------
    messages : List[Dict[str, Any]]
        Chat messages to send to each provider.
    temperature : float, optional
        Sampling temperature; defaults to ``DEFAULT_TEMPERATURE``.
    max_tokens : int, optional
        Maximum tokens to generate; defaults to ``DEFAULT_MAX_TOKENS_RACE``.
    timeout : float, optional
        Overall timeout for the race; defaults to ``DEFAULT_TIMEOUT_RACE`` seconds.

    Returns
    -------
    LLMResponse | None
        The first successful response, or ``None`` if no providers are available or all fail.
    """
    tasks = {
        asyncio.create_task(_call_provider(p, messages, temperature, max_tokens)): p
        for p in PROVIDERS
        if os.getenv(p.env_key, "") not in DISABLED_KEYS
    }
    if not tasks:
        logger.warning(NO_API_KEYS_MSG)
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
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS_CONSENSUS,
    timeout: float = DEFAULT_TIMEOUT_CONSENSUS,
) -> List[LLMResponse]:
    """
    Query all configured providers and collect all successful responses.

    Parameters
    ----------
    messages : List[Dict[str, Any]]
        Chat messages to send to each provider.
    temperature : float, optional
        Sampling temperature; defaults to ``DEFAULT_TEMPERATURE``.
    max_tokens : int, optional
        Maximum tokens to generate; defaults to ``DEFAULT_MAX_TOKENS_CONSENSUS``.
    timeout : float, optional
        Overall timeout for the consensus operation; defaults to ``DEFAULT_TIMEOUT_CONSENSUS`` seconds.

    Returns
    -------
    List[LLMResponse]
        List of successful responses; empty if none succeed.
    """
    tasks = [
        _call_provider(p, messages, temperature, max_tokens)
        for p in PROVIDERS
        if os.getenv(p.env_key, "") not in DISABLED_KEYS
    ]
    if not tasks:
        return []
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, LLMResponse)]


def available_providers() -> List[str]:
    """
    Retrieve the names of providers that have a valid API key configured.

    Returns
    -------
    List[str]
        Provider names with usable credentials.
    """
    return [p.name for p in PROVIDERS if os.getenv(p.env_key, "") not in DISABLED_KEYS]