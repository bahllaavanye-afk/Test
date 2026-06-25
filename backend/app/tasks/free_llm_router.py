"""
Free LLM Router — dispatches to many free providers in parallel.

Priority cascade (fastest/highest-quota first):
  1. Gemini Flash 2.0 (Google AI Studio — 1M TPM free)
  2. Groq  (llama-3.3-70b — 6000 TPD free, very fast)
  3. DeepSeek (deepseek-chat — $5 free credit, cheap)
  4. SambaNova (Meta-Llama-3.3-70B — free tier)
  5. Cerebras (llama-3.3-70b — free tier, fast inference)
  6. Together AI (Llama-3.3-70B — $25 free credit)
  7. Hyperbolic (llama-3.3-70b — $10 free credit)
  8. NVIDIA NIM (nemotron-70b — free tier)
  9. OpenRouter (llama-3.3-70b:free — free tier)

  Free OpenAI models (OpenAI-compatible, no paid tier required):
  10. GitHub Models — GPT-4o-mini   (free with any GitHub token)
  11. GitHub Models — GPT-4o        (free with any GitHub token)
  12. GitHub Models — o4-mini       (free reasoning model)

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
from typing import Any, List, Tuple, Optional

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
    # ── Free OpenAI models via GitHub Models ──────────────────────────────────
    LLMProvider(
        name="github_gpt4o_mini",
        env_key="GITHUB_MODELS_TOKEN",
        base_url="https://models.github.ai/inference",
        model="openai/gpt-4o-mini",
        timeout=30.0,
    ),
    LLMProvider(
        name="github_gpt4o",
        env_key="GITHUB_MODELS_TOKEN",
        base_url="https://models.github.ai/inference",
        model="openai/gpt-4o",
        timeout=40.0,
    ),
    LLMProvider(
        name="github_o4_mini",
        env_key="GITHUB_MODELS_TOKEN",
        base_url="https://models.github.ai/inference",
        model="openai/o4-mini",
        timeout=60.0,
    ),
    # ── Perplexity Sonar (web-grounded answers) ───────────────────────────────
    LLMProvider(
        name="perplexity_sonar",
        env_key="PERPLEXITY_API_KEY",
        base_url="https://api.perplexity.ai",
        model="sonar",
        timeout=40.0,
    ),
    LLMProvider(
        name="perplexity_sonar_reasoning",
        env_key="PERPLEXITY_API_KEY",
        base_url="https://api.perplexity.ai",
        model="sonar-reasoning",
        timeout=60.0,
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

_MAX_KEY_SUFFIX = 12
_RR_INDEX: dict[str, int] = {}           # env_key -> next rotation index
_USAGE: dict[str, dict] = {}             # key_label -> {"calls", "tokens"}

_ENV_KEY_FALLBACKS: dict[str, List[str]] = {
    "GITHUB_MODELS_TOKEN": ["GITHUB_TOKEN", "GH_TOKEN"],
}


def _keys_for(env_key: str) -> List[Tuple[str, str]]:
    """Return all distinct (label, value) pairs for a given environment key."""
    out: List[Tuple[str, str]] = []
    seen: set[str] = set()
    base_labels = [env_key] + [f"{env_key}_{i}" for i in range(1, _MAX_KEY_SUFFIX + 1)]
    labels = base_labels + _ENV_KEY_FALLBACKS.get(env_key, [])
    for label in labels:
        val = os.getenv(label, "")
        if val and val != "disabled" and val not in seen:
            seen.add(val)
            out.append((label, val))
    return out


def _next_key(env_key: str) -> Optional[Tuple[str, str]]:
    """Round‑robin select the next (label, value) pair for the given env_key."""
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


def get_throughput_report() -> List[dict]:
    """Return per‑key call and token counts observed by the router."""
    return [{"key": k, **v} for k, v in sorted(_USAGE.items())]


def reset_usage() -> None:
    _USAGE.clear()


# ── Helper functions for provider calls ────────────────────────────────────────


def _build_auth_header(api_key: str) -> dict:
    """Create the Authorization header expected by most providers."""
    return {"Authorization": f"Bearer {api_key}"}


def _build_payload(
    provider: LLMProvider,
    messages: List[dict],
    temperature: float,
    max_tokens: Optional[int],
) -> dict:
    """Construct the JSON body for a chat‑completion request."""
    payload = {
        "model": provider.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens if max_tokens is not None else provider.max_tokens,
    }
    return payload


def _parse_response(resp: httpx.Response) -> Tuple[str, int]:
    """
    Extract the generated text and token usage from a provider response.

    The function is tolerant to slight variations in response shape:
      * OpenAI‑compatible: {'choices': [{'message': {'content': ...}}]}
      * Legacy OpenAI: {'choices': [{'text': ...}]}
      * Generic: any top‑level 'content' field.
    Token count is read from ``usage.total_tokens`` when present.
    """
    try:
        data = resp.json()
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON from provider response")
        return "", 0

    # Content extraction
    content = ""
    if isinstance(data, dict):
        # OpenAI‑compatible path
        choices = data.get("choices", [])
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content", "")
                else:
                    content = first.get("text", "")
        # Fallback generic field
        if not content:
            content = data.get("content", "")

    # Token usage extraction
    usage = data.get("usage", {})
    tokens = int(usage.get("total_tokens", 0))

    return content, tokens


async def _call_provider(
    provider: LLMProvider,
    messages: List[dict],
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> Optional[LLMResponse]:
    """
    Send a chat request to a single LLM provider.

    The function selects an API key using round‑robin rotation, builds the request,
    measures latency, records usage, and returns a structured ``LLMResponse``.
    """
    selected = _next_key(provider.env_key)
    if selected is None:
        logger.warning("No API key available for provider %s (env key %s)", provider.name, provider.env_key)
        return None

    key_label, api_key = selected
    headers = {"Content-Type": "application/json"}
    headers.update(_build_auth_header(api_key))
    headers.update(provider.headers_extra)

    payload = _build_payload(provider, messages, temperature, max_tokens)

    url = f"{provider.base_url.rstrip('/')}/chat/completions"
    start = time.time()
    async with httpx.AsyncClient(timeout=provider.timeout) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Provider %s request failed: %s", provider.name, exc)
            return None

    latency_ms = (time.time() - start) * 1000.0
    content, tokens = _parse_response(resp)

    _record_usage(key_label, tokens)

    return LLMResponse(
        provider=provider.name,
        content=content,
        latency_ms=latency_ms,
        tokens_used=tokens,
        key_label=key_label,
    )