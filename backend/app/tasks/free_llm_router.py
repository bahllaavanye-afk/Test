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
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

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
    LLMProvider(
        name="nvidia_nim",
        env_key="NVIDIA_NIM_API_KEY",
        base_url="https://integrate.api.nvidia.com/v1",
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        timeout=35.0,
    ),
    LLMProvider(
        name="openrouter",
        env_key="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        model="meta-llama/llama-3.3-70b-instruct:free",
        timeout=30.0,
    ),
    LLMProvider(
        name="gemini_thinking",
        env_key="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model="gemini-2.0-flash-thinking-exp",
        timeout=40.0,
    ),
]


@dataclass
class LLMResponse:
    provider: str
    content: str
    latency_ms: float
    tokens_used: int = 0
    key_label: str = ""


# ── Multi-key discovery, rotation, and usage tracking ──────────────────────────
#
# A provider may have several free keys: e.g. GROQ_API_KEY, GROQ_API_KEY_1,
# GROQ_API_KEY_2, ... Reading only the singular name (the old behaviour) left
# every numbered key idle. We discover all of them and round-robin across them so
# free quota is spread instead of hammering one key.

_MAX_KEY_SUFFIX = 12
_RR_INDEX: dict[str, int] = {}           # env_key -> next rotation index
_USAGE: dict[str, dict] = {}             # key_label -> {"calls", "tokens"}


def _keys_for(env_key: str) -> list[tuple[str, str]]:
    """All configured (label, value) keys for a provider, base + numbered."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    labels = [env_key] + [f"{env_key}_{i}" for i in range(1, _MAX_KEY_SUFFIX + 1)]
    for label in labels:
        val = os.getenv(label, "")
        if val and val != "disabled" and val not in seen:
            seen.add(val)
            out.append((label, val))
    return out


def _next_key(env_key: str) -> tuple[str, str] | None:
    """Round-robin the next (label, value) key for this provider, or None."""
    keys = _keys_for(env_key)
    if not keys:
        return None
    idx = _RR_INDEX.get(env_key, 0) % len(keys)
    _RR_INDEX[env_key] = idx + 1
    return keys[idx]


def _record_usage(key_label: str, tokens: int) -> None:
    u = _USAGE.setdefault(key_label, {"calls": 0, "tokens": 0})
    u["calls"] += 1
    u["tokens"] += int(tokens or 0)


def get_throughput_report() -> list[dict]:
    """Per-key call/token counts observed by the router (source of truth)."""
    return [{"key": k, **v} for k, v in sorted(_USAGE.items())]


def reset_usage() -> None:
    _USAGE.clear()


# ── Core caller ───────────────────────────────────────────────────────────────

async def _call_provider(
    provider: LLMProvider,
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> LLMResponse | None:
    selected = _next_key(provider.env_key)
    if selected is None:
        return None
    key_label, api_key = selected

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
            _record_usage(key_label, tokens)
            return LLMResponse(provider=provider.name, content=content, latency_ms=latency,
                               tokens_used=tokens, key_label=key_label)
    except Exception as e:
        logger.debug("Provider %s (%s) failed: %s", provider.name, key_label, e)
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

async def call_race(
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: float = 30.0,
) -> LLMResponse | None:
    """Call all available providers in parallel; return the first successful response."""
    tasks = {
        asyncio.create_task(_call_provider(p, messages, temperature, max_tokens)): p
        for p in PROVIDERS
        if _keys_for(p.env_key)
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
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    for t in done:
        result = t.result()
        if result:
            logger.info("LLM race winner: %s (%.0fms)", result.provider, result.latency_ms)
            return result
    return None


async def call_consensus(
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 512,
    timeout: float = 40.0,
) -> list[LLMResponse]:
    """Call all providers and return all successful responses for consensus analysis."""
    tasks = [
        _call_provider(p, messages, temperature, max_tokens)
        for p in PROVIDERS
        if _keys_for(p.env_key)
    ]
    if not tasks:
        return []
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, LLMResponse)]


def available_providers() -> list[str]:
    """Return names of providers with at least one configured API key."""
    return [p.name for p in PROVIDERS if _keys_for(p.env_key)]


def available_keys() -> list[str]:
    """Return labels of every configured key across all providers."""
    labels: list[str] = []
    for p in PROVIDERS:
        labels.extend(label for label, _ in _keys_for(p.env_key))
    return sorted(set(labels))


import hashlib
import functools

# ── Response cache ────────────────────────────────────────────────────────────

async def _cache_get(redis_client, messages: list[dict]) -> Optional[str]:
    key = "llm:cache:" + hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    if redis_client:
        try:
            return await redis_client.get(key)
        except Exception:
            pass
    return None

async def _cache_set(redis_client, messages: list[dict], response: str) -> None:
    key = "llm:cache:" + hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    if redis_client:
        try:
            await redis_client.set(key, response, ex=3600)  # 1h TTL
        except Exception:
            pass

# ── Routed call by task type ──────────────────────────────────────────────────

async def call_routed(
    messages: list[dict],
    task_type: str = "analysis",  # "code" | "analysis" | "fast"
    max_tokens: int = 2048,
    redis_client=None,
) -> Optional[str]:
    """
    Route to best free provider for the task type.
    Checks cache first. Returns response text or None.
    """
    # Check cache first
    cached = await _cache_get(redis_client, messages)
    if cached:
        return cached

    # Provider preference by task type. Names MUST match PROVIDERS[].name.
    if task_type == "fast":
        # Cerebras/Groq are fastest; Gemini Flash close behind.
        preferred = ["cerebras", "groq", "gemini"]
    elif task_type == "code":
        # Long-context models for code.
        preferred = ["gemini", "gemini_thinking", "together", "groq"]
    else:  # analysis
        # Llama 70B for reasoning quality, spread across providers.
        preferred = ["groq", "together", "cerebras", "openrouter", "gemini", "nvidia_nim"]

    for provider_name in preferred:
        provider = next((p for p in PROVIDERS if p.name == provider_name), None)
        if provider is None:
            continue
        if not _keys_for(provider.env_key):
            continue
        result = await _call_provider(provider, messages, temperature=0.3, max_tokens=max_tokens)
        if result:
            await _cache_set(redis_client, messages, result.content)
            return result.content

    # Fallback: race
    result = await call_race(messages, max_tokens=max_tokens)
    if result:
        await _cache_set(redis_client, messages, result.content)
        return result.content
    return None


async def call_batch(
    prompts: list[str],
    system: str = "",
    max_tokens: int = 512,
    concurrency: int = 5,
) -> list[Optional[str]]:
    """Run up to `concurrency` prompts in parallel. Returns list of responses."""
    import asyncio

    async def _one(prompt: str) -> Optional[str]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        result = await call_race(msgs, max_tokens=max_tokens, timeout=20.0)
        return result.content if result else None

    sem = asyncio.Semaphore(concurrency)
    async def _bounded(prompt):
        async with sem:
            return await _one(prompt)

    return list(await asyncio.gather(*[_bounded(p) for p in prompts], return_exceptions=False))
