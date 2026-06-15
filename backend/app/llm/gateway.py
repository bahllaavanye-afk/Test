"""Free-LLM gateway: Groq -> DeepSeek -> Gemini, all free tiers.

This is the single chokepoint every agent calls. It contains no paid providers
(Anthropic/OpenAI are intentionally excluded to honour the 0-paid-API rule).

Egress hosts that must be reachable for this to work:
  * api.groq.com
  * api.deepseek.com
  * generativelanguage.googleapis.com
"""
from __future__ import annotations

import asyncio
import os

import httpx

from app.utils.logging import logger

# (provider, host) — surfaced by /agents/llm-status so the egress allowlist is discoverable.
PROVIDER_HOSTS: dict[str, str] = {
    "groq": "api.groq.com",
    "deepseek": "api.deepseek.com",
    "gemini": "generativelanguage.googleapis.com",
}

GROQ_MODEL = "llama-3.1-8b-instant"
DEEPSEEK_MODEL = "deepseek-chat"
GEMINI_MODEL = "gemini-2.0-flash"


def _resolve_key(*names: str) -> str:
    """Return the first non-empty env var, also trying a ``_1`` suffix.

    Supports rotating numbered keys (GROQ_API_KEY_1, GROQ_API_KEY_2, ...).
    """
    for name in names:
        v = os.environ.get(name, "")
        if v:
            return v
        if not name[-1].isdigit():
            v = os.environ.get(name + "_1", "")
            if v:
                return v
    return ""


def providers_configured() -> list[str]:
    """Free providers that have at least one API key set (in priority order)."""
    out: list[str] = []
    if _resolve_key("GROQ_API_KEY"):
        out.append("groq")
    if _resolve_key("DEEPSEEK_API_KEY"):
        out.append("deepseek")
    if _resolve_key("GEMINI_API_KEY"):
        out.append("gemini")
    return out


async def complete(
    messages: list[dict],
    max_tokens: int = 600,
    agent: str | None = None,
) -> str | None:
    """Chat completion via the free-provider chain.

    Returns the assistant text, or ``None`` if no provider is configured or all
    providers fail. Callers MUST treat ``None`` as "no LLM available" and degrade
    honestly — never fabricate output.
    """
    # Groq (fastest free tier — preferred for the always-on autonomous loop)
    groq_key = _resolve_key("GROQ_API_KEY")
    if groq_key:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}"},
                    json={"model": GROQ_MODEL, "messages": messages, "max_tokens": max_tokens},
                )
                if r.status_code == 200:
                    text = r.json()["choices"][0]["message"]["content"].strip()
                    await _record_spend(agent, text, messages)
                    return text
                logger.debug("llm.groq non-200", status=r.status_code)
        except Exception as e:
            logger.debug("llm.groq failed", error=str(e))

    # DeepSeek
    for key in (
        _resolve_key("DEEPSEEK_API_KEY"),
        os.environ.get("DEEPSEEK_API_KEY_2", ""),
        os.environ.get("DEEPSEEK_API_KEY_3", ""),
    ):
        if not key:
            continue
        try:
            async with httpx.AsyncClient(timeout=25) as client:
                r = await client.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": DEEPSEEK_MODEL, "messages": messages, "max_tokens": max_tokens},
                )
                if r.status_code == 200:
                    text = r.json()["choices"][0]["message"]["content"].strip()
                    await _record_spend(agent, text, messages)
                    return text
        except Exception as e:
            logger.debug("llm.deepseek failed", error=str(e))

    # Gemini
    gemini_key = _resolve_key("GEMINI_API_KEY")
    if gemini_key:
        try:
            prompt = "\n".join(m.get("content", "") for m in messages)
            async with httpx.AsyncClient(timeout=25) as client:
                r = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{GEMINI_MODEL}:generateContent?key={gemini_key}",
                    json={
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"maxOutputTokens": max_tokens},
                    },
                )
                if r.status_code == 200:
                    text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    await _record_spend(agent, text, messages)
                    return text
        except Exception as e:
            logger.debug("llm.gemini failed", error=str(e))

    return None


async def _record_spend(agent: str | None, response: str, messages: list[dict]) -> None:
    """Best-effort token-spend accounting against the agent's daily budget."""
    if not agent:
        return
    try:
        from app.tasks.token_budget import get_token_budget

        # ~4 chars per token, prompt + completion.
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        approx_tokens = (prompt_chars + len(response)) // 4
        await get_token_budget().record_spend(agent, approx_tokens)
    except Exception:
        pass


def complete_sync(messages: list[dict], max_tokens: int = 600, agent: str | None = None) -> str | None:
    """Synchronous wrapper for callers running outside an event loop.

    Used by the alpha miner, which runs in a worker thread. Safe to call from a
    thread that has no running loop; raises nothing — returns ``None`` on error.
    """
    try:
        return asyncio.run(complete(messages, max_tokens=max_tokens, agent=agent))
    except RuntimeError:
        # Already inside a running loop — fall back to a fresh loop in this thread.
        try:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(complete(messages, max_tokens=max_tokens, agent=agent))
            finally:
                loop.close()
        except Exception:
            return None
    except Exception:
        return None
