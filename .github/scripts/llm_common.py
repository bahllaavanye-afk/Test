"""
Shared LLM infrastructure for ALL .github/scripts/*.py agents.

This replaces 14 copies of the same 7-provider cascade function.
Every script should: from llm_common import llm, llm_chat, memory_write, memory_read

HOW CONTEXT IS SHARED ACROSS LLMs:
  Model weights are NOT shared — Gemini, Groq, DeepSeek, Cerebras are separate companies.
  What IS shared is external context injected into every prompt:

  1. company_brain.json — single shared JSON file, read by every agent before calling any LLM.
     Contains: regime, top strategies, recent lessons, Slack insights, trade outcomes.
     Built by company_brain.py every 15 minutes from all sources.

  2. ConversationStore (in memory_manager.py) — persists conversation history as OpenAI
     messages arrays. Any provider can load and continue a conversation started by another,
     because all providers accept the same {"role":..., "content":...} format.

  3. SemanticRetriever (in memory_manager.py) — TF-IDF search over company_brain.json
     to inject RELEVANT past context, not just the 5 most recent entries.

  4. llm_chat() — the multi-turn version of llm(). Pass a ConversationStore and it:
     - Loads the full conversation history
     - Adds your message
     - Calls the best available provider
     - Saves the reply back to the store
     So the next call (by any provider) continues seamlessly.

Token reduction:
  - 24h response cache keyed by prompt hash
  - Auto-compression for prompts >8000 tokens
  - Semantic retrieval: only relevant context injected, not all memory
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_manager import ConversationStore

logger = logging.getLogger(__name__)

# Lazy import — memory_manager lives in the same directory.
# Falls back gracefully if not found (e.g. running outside .github/scripts/).
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from memory_manager import build_context as _build_context, ConversationStore as _ConversationStore
    _MEMORY_MANAGER_OK = True
except Exception:  # noqa: BLE001
    _MEMORY_MANAGER_OK = False
    _ConversationStore = None  # type: ignore[assignment,misc]

_STATE_DIR = Path(os.environ.get("GITHUB_WORKSPACE", ".")) / ".github" / "state"
_STATE_DIR.mkdir(parents=True, exist_ok=True)
_BRAIN_FILE = _STATE_DIR / "company_brain.json"
_CACHE_FILE = _STATE_DIR / "llm_cache.json"
_METRICS_FILE = _STATE_DIR / "llm_metrics.jsonl"


def _record_metric(provider: str, ok: bool, ms: int, error: str = "") -> None:
    """Append one LLM-call metric so the brain is observable (provider, ok, latency).

    Best-effort and never raises; rolls the file to the last ~500 lines. This is the
    minimal observability that was missing when the whole cascade silently died.
    """
    try:
        import time as _t
        with _METRICS_FILE.open("a") as f:
            f.write(json.dumps({
                "ts": _t.time(), "provider": provider, "ok": ok, "ms": ms,
                "error": (error or "")[:160],
            }) + "\n")
        if _METRICS_FILE.stat().st_size > 200_000:
            tail = _METRICS_FILE.read_text().splitlines()[-500:]
            _METRICS_FILE.write_text("\n".join(tail) + "\n")
    except Exception:
        pass


def cascade_status(probe: bool = True) -> dict:
    """Health of the free-LLM cascade: which providers have keys and (if probe)
    actually answer right now. The brain-health canary uses this so a dead cascade
    screams instead of silently degrading to green.
    """
    import time as _t
    out: dict = {"checked_at": _t.time(), "providers": {}, "working": [], "healthy": False}
    for p in _PROVIDERS:
        name = p["name"]
        if not _has_key(p):
            out["providers"][name] = {"has_key": False, "ok": False}
            continue
        entry: dict = {"has_key": True, "keys": len(_provider_keys(p))}
        if probe:
            t0 = _t.time()
            try:
                r = _call_provider(p, "ping", "Reply with: OK", 8, 0.0)
                entry["ok"] = bool(r and r.strip())
                entry["ms"] = int((_t.time() - t0) * 1000)
            except Exception as e:  # noqa: BLE001
                entry["ok"] = False
                entry["error"] = str(e)[:140]
        out["providers"][name] = entry
    out["working"] = [n for n, v in out["providers"].items() if v.get("ok")]
    out["healthy"] = bool(out["working"])
    return out

# ── Provider config ───────────────────────────────────────────────────────────

_PROVIDERS = [
    {
        "name": "gemini",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        "key_env": "GEMINI_API_KEY",
        "fmt": "gemini",
        "rpm_free": 15,
    },
    {
        "name": "sambanova",
        "url": "https://api.sambanova.ai/v1/chat/completions",
        "key_env": "SAMBANOVA_API_KEY",
        "fmt": "openai",
        "model": "Meta-Llama-3.3-70B-Instruct",
        "rpm_free": 60,
    },
    {
        "name": "cerebras",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "key_env": "CEREBRAS_API_KEY",
        "fmt": "openai",
        "model": "llama-3.3-70b",
        "rpm_free": 30,
    },
    {
        "name": "groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "fmt": "openai",
        "model": "llama-3.3-70b-versatile",
        "rpm_free": 30,
    },
    {
        "name": "deepseek",
        "url": "https://api.deepseek.com/v1/chat/completions",
        "key_env": "DEEPSEEK_API_KEY",
        "fmt": "openai",
        "model": "deepseek-chat",
        "rpm_free": 60,
    },
    {
        "name": "together",
        "url": "https://api.together.xyz/v1/chat/completions",
        "key_env": "TOGETHER_API_KEY",
        "fmt": "openai",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "rpm_free": 60,
    },
    {
        "name": "hyperbolic",
        "url": "https://api.hyperbolic.xyz/v1/chat/completions",
        "key_env": "HYPERBOLIC_API_KEY",
        "fmt": "openai",
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "rpm_free": 60,
    },
    {
        "name": "nvidia_nim",
        "url": "https://integrate.api.nvidia.com/v1/chat/completions",
        # Key stored in GitHub as NVIDIA_AGENTS_API_KEYS
        # Models available free: meta/llama-3.3-70b-instruct, nvidia/llama-3.1-nemotron-70b-instruct,
        #   mistralai/mixtral-8x22b-instruct-v0.1, deepseek-ai/deepseek-r1, qwen/qwen2.5-72b-instruct
        # Using Nemotron-70B: NVIDIA's best instruction-following model, optimized for agents
        "key_env": "NVIDIA_AGENTS_API_KEYS",
        "key_env_alt": "NVIDIA_NIM_API_KEY",
        "fmt": "openai",
        "model": "nvidia/llama-3.1-nemotron-70b-instruct",
        "rpm_free": 40,
    },
]

# Providers to race in parallel (first N by index). Others are sequential fallbacks.
_PARALLEL_RACE_N = 3

# ── Escalation tiers (NOT part of the free race) ──────────────────────────────
# Cost ladder: FREE cascade → OPEN-MID (OpenRouter) → CLAUDE (rare backstop).
# Open models have closed the gap with frontier on agentic benchmarks at 10–50× lower
# cost (DeepSeek V4 / Qwen3 / Kimi K2 / GLM / MiniMax), so the open-mid tier handles the
# bulk of "hard" work and Claude is reserved for only the hardest tasks. All slugs/models
# are env-overridable — nothing is hard-coded to a model that may rotate.
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_MODELS = [
    m.strip() for m in os.environ.get(
        "OPENROUTER_MODELS",
        "deepseek/deepseek-chat,qwen/qwen-2.5-72b-instruct,"
        "moonshotai/kimi-k2,z-ai/glm-4.6,minimax/minimax-m2",
    ).split(",") if m.strip()
]
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
# Default backstop is a strong-but-not-Opus model; override to claude-haiku-4-5 (cheaper)
# or claude-opus-4-8 (hardest) via env. Backstop fires only on tier="hard" or last resort.
_CLAUDE_BACKSTOP_MODEL = os.environ.get("CLAUDE_BACKSTOP_MODEL", "claude-sonnet-4-6")

# ── Response cache ────────────────────────────────────────────────────────────

_CACHE_TTL = 86400  # 24 hours
_cache_mem: dict[str, dict] = {}
_cache_loaded = False


def _load_cache() -> None:
    global _cache_mem, _cache_loaded
    if _cache_loaded:
        return
    try:
        if _CACHE_FILE.exists():
            _cache_mem = json.loads(_CACHE_FILE.read_text())
    except Exception:
        _cache_mem = {}
    _cache_loaded = True


def _save_cache() -> None:
    try:
        # Evict expired
        now = time.time()
        _cache_mem.update({k: v for k, v in _cache_mem.items() if now - v.get("ts", 0) < _CACHE_TTL})
        _CACHE_FILE.write_text(json.dumps(_cache_mem))
    except Exception:
        pass


def _cache_key(prompt: str, system: str, max_tokens: int) -> str:
    content = f"{system}|||{prompt}|||{max_tokens}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── Core LLM caller ───────────────────────────────────────────────────────────

def llm(
    prompt: str,
    system: str = "You are a helpful AI agent at QuantEdge, a quantitative trading firm.",
    max_tokens: int = 400,
    temperature: float = 0.7,
    use_cache: bool = True,
    inject_company_context: bool = True,
) -> str:
    """
    Call the best available free LLM. Cascade through providers until one succeeds.

    Args:
        prompt: The user message.
        system: System prompt (keep short — company context is auto-injected).
        max_tokens: Response length cap.
        use_cache: Return cached response if same prompt was seen in last 24h.
        inject_company_context: Prepend shared company brain context to prompt.
    """
    _load_cache()

    # Cache key uses the BASE prompt (before context injection) so that the same
    # query deduplicates regardless of what's currently in the brain.
    ck = _cache_key(prompt, system, max_tokens)
    if use_cache and ck in _cache_mem:
        entry = _cache_mem[ck]
        if time.time() - entry.get("ts", 0) < _CACHE_TTL:
            return entry["text"]

    # Optionally inject shared company context (semantic retrieval if available,
    # otherwise fall back to recency-based snapshot from TTL-cached brain).
    if inject_company_context:
        if _MEMORY_MANAGER_OK:
            ctx = _build_context(prompt)
        else:
            ctx = get_company_context(max_tokens=600)
        if ctx:
            prompt = f"{ctx}\n\n---\n\n{prompt}"

    # Compress if too long (>8000 tokens estimated)
    if len(prompt) > 32000:
        prompt = prompt[:28000] + "\n\n[...truncated for token efficiency...]"

    # Race the first N providers in parallel; fall back sequentially for the rest.
    _t0 = time.time()
    result, provider_name = _call_parallel_race(system, prompt, max_tokens, temperature)
    _record_metric(provider_name or "none", bool(result), int((time.time() - _t0) * 1000),
                   "" if result else "all providers failed")

    # If the entire FREE cascade is down (e.g. only Groq was configured and it's rate
    # limited), escalate to the paid ladder — OpenRouter open-mid, then Claude — so a
    # single-provider outage can't silently blind every agent. Tiers without a key are
    # skipped automatically, so this needs no key work to stay safe.
    if not result:
        result = _call_openrouter(system, prompt, max_tokens, temperature)
        if result:
            provider_name = "openrouter"
        else:
            result = _call_claude(system, prompt, max_tokens, temperature)
            if result:
                provider_name = "claude"

    if result:
        _cache_mem[ck] = {"text": result, "ts": time.time(), "provider": provider_name}
        _save_cache()
        return result

    return "[LLM unavailable — all providers failed]"


def llm_with_provider(
    prompt: str,
    provider_name: str,
    system: str = "You are a helpful AI agent at QuantEdge, a quantitative trading firm.",
    max_tokens: int = 400,
    temperature: float = 0.7,
    inject_company_context: bool = False,
) -> str:
    """
    Call a SPECIFIC named provider (e.g. 'gemini', 'groq', 'nvidia_nim').
    Used when you want each agent pinned to an independent LLM so reviews are truly independent.
    Falls back to the cascade if the named provider is unavailable.
    Returns (response_text, actual_provider_name) tuple.
    """
    provider = next((p for p in _PROVIDERS if p["name"] == provider_name), None)

    if inject_company_context:
        ctx = get_company_context(max_tokens=400)
        if ctx:
            prompt = f"{ctx}\n\n---\n\n{prompt}"

    if len(prompt) > 32000:
        prompt = prompt[:28000] + "\n\n[...truncated...]"

    if provider and _has_key(provider):
        try:
            result = _call_provider(provider, system, prompt, max_tokens, temperature)
            if result:
                return result, provider_name
        except Exception as e:
            logger.warning("Provider %s failed in llm_with_provider: %s", provider_name, e)

    # Fall back to cascade
    result, used = _call_parallel_race(system, prompt, max_tokens, temperature)
    return (result or "[LLM unavailable]", used or "cascade")


def _env_keys(*names: str) -> list[str]:
    """Collect non-empty, de-duplicated keys from the given env var names, in order.

    Lets the paid tiers rotate across numbered variants (OPENROUTER_API_KEY,
    OPENROUTER_API_KEY_2, ...) the way the free providers do — more keys = more headroom.
    """
    keys: list[str] = []
    for n in names:
        v = os.environ.get(n, "").strip()
        if v and v != "disabled" and v not in keys:
            keys.append(v)
    return keys


def _call_openrouter(system: str, prompt: str, max_tokens: int, temperature: float) -> str | None:
    """Open-weight mid-tier via OpenRouter. Tries each configured model in order.

    Returns the text, or None if there's no key / every model fails. OpenRouter speaks
    the OpenAI schema, so this mirrors `_call_provider`'s OpenAI path.
    """
    keys = _env_keys("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2", "OPENROUTER_API_KEY_3")
    if not keys or not _OPENROUTER_MODELS:
        return None
    last_exc: Exception | None = None
    for key, model in ((k, m) for k in keys for m in _OPENROUTER_MODELS):
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": _UA,
            "Authorization": f"Bearer {key}",
            # Optional attribution headers OpenRouter recommends; harmless if unused.
            "HTTP-Referer": "https://github.com/quantedge",
            "X-Title": "QuantEdge agents",
        }
        req = urllib.request.Request(_OPENROUTER_URL, data=body, headers=headers, method="POST")
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                result = json.loads(resp.read())
            text = result["choices"][0]["message"]["content"].strip()
            _record_metric(f"openrouter:{model}", True, int((time.time() - t0) * 1000))
            return text
        except Exception as exc:  # noqa: BLE001 — fall through to the next model
            _record_metric(f"openrouter:{model}", False, int((time.time() - t0) * 1000), str(exc))
            last_exc = exc
    logger.debug("OpenRouter open-mid tier exhausted: %s", last_exc)
    return None


def _call_claude(system: str, prompt: str, max_tokens: int, temperature: float) -> str | None:
    """Claude backstop — the rare top of the ladder for only the hardest tasks.

    Uses the Anthropic Messages API. Returns None when no ANTHROPIC_API_KEY is set so
    the caller degrades gracefully instead of crashing.
    """
    keys = _env_keys("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_2")
    if not keys:
        return None
    body = json.dumps({
        "model": _CLAUDE_BACKSTOP_MODEL,
        "system": system,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    last_exc: Exception | None = None
    for key in keys:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": _UA,
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }
        req = urllib.request.Request(_ANTHROPIC_URL, data=body, headers=headers, method="POST")
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
            text = "".join(blk.get("text", "") for blk in result.get("content", [])).strip()
            _record_metric(f"claude:{_CLAUDE_BACKSTOP_MODEL}", bool(text), int((time.time() - t0) * 1000))
            if text:
                return text
        except Exception as exc:  # noqa: BLE001 — try the next key
            _record_metric(f"claude:{_CLAUDE_BACKSTOP_MODEL}", False, int((time.time() - t0) * 1000), str(exc))
            last_exc = exc
    if last_exc:
        logger.warning("Claude backstop failed: %s", last_exc)
    return None


def llm_routed(
    prompt: str,
    system: str = "You are a helpful AI agent at QuantEdge, a quantitative trading firm.",
    max_tokens: int = 400,
    temperature: float = 0.7,
    tier: str = "auto",
    use_cache: bool = True,
    inject_company_context: bool = True,
) -> str:
    """Cost-tiered routing ladder — escalate only on failure so the cheapest capable
    tier wins and Claude stays a rare backstop.

        FREE cascade  →  OPEN-MID (OpenRouter)  →  CLAUDE backstop

    `tier` sets the escalation *ceiling*:
      - "cheap": free cascade only (bulk classify/extract/summarize — the 80% case).
      - "mid":   free → OpenRouter open-mid.
      - "auto"  (default): free → OpenRouter open-mid. Claude only if BOTH fail.
      - "hard":  free → OpenRouter → Claude (reserve for the genuinely hardest tasks).

    Drop-in compatible with `llm()`'s caching + company-context injection.
    """
    _load_cache()
    ck = _cache_key(prompt, f"{system}|tier={tier}", max_tokens)
    if use_cache and ck in _cache_mem:
        entry = _cache_mem[ck]
        if time.time() - entry.get("ts", 0) < _CACHE_TTL:
            return entry["text"]

    if inject_company_context:
        ctx = _build_context(prompt) if _MEMORY_MANAGER_OK else get_company_context(max_tokens=600)
        if ctx:
            prompt = f"{ctx}\n\n---\n\n{prompt}"
    if len(prompt) > 32000:
        prompt = prompt[:28000] + "\n\n[...truncated for token efficiency...]"

    result: str | None = None
    used = "none"

    # Tier 1 — FREE cascade (always tried first; it's free).
    _t0 = time.time()
    result, won = _call_parallel_race(system, prompt, max_tokens, temperature)
    if result:
        used = won or "free"
    _record_metric(used if result else "free:none", bool(result), int((time.time() - _t0) * 1000),
                   "" if result else "free cascade failed")

    # Tier 2 — OPEN-MID via OpenRouter.
    if not result and tier in ("mid", "auto", "hard"):
        result = _call_openrouter(system, prompt, max_tokens, temperature)
        if result:
            used = "openrouter"

    # Tier 3 — CLAUDE backstop (rare): explicit "hard", or last resort when all else failed.
    if not result and (tier == "hard" or tier == "auto"):
        result = _call_claude(system, prompt, max_tokens, temperature)
        if result:
            used = "claude"

    if result:
        _cache_mem[ck] = {"text": result, "ts": time.time(), "provider": used}
        _save_cache()
        return result
    return "[LLM unavailable — all tiers failed]"


def _has_key(p: dict) -> bool:
    """Check if a provider has an API key configured (supports primary + alt env var)."""
    v = os.environ.get(p["key_env"], "")
    if v and v != "disabled":
        return True
    alt = p.get("key_env_alt", "")
    if alt:
        v2 = os.environ.get(alt, "")
        return bool(v2) and v2 != "disabled"
    return False


def _call_parallel_race(
    system: str, prompt: str, max_tokens: int, temperature: float
) -> tuple[str | None, str | None]:
    """
    Race the first _PARALLEL_RACE_N available providers in parallel threads.
    Returns (response_text, provider_name) for the first successful response.
    Falls back to sequential for remaining providers if the race fails.
    """
    available = [p for p in _PROVIDERS if _has_key(p)]
    if not available:
        return None, None

    race_pool = available[:_PARALLEL_RACE_N]
    sequential_tail = available[_PARALLEL_RACE_N:]

    # Phase 1: parallel race
    _result: list[str | None] = [None]
    _winner: list[str | None] = [None]
    _done = threading.Event()

    def _try(provider: dict) -> None:
        if _done.is_set():
            return
        try:
            r = _call_provider(provider, system, prompt, max_tokens, temperature)
            if r and not _done.is_set():
                _done.set()
                _result[0] = r
                _winner[0] = provider["name"]
        except Exception as e:
            logger.debug("Provider %s failed (race): %s", provider["name"], e)

    with ThreadPoolExecutor(max_workers=len(race_pool)) as ex:
        futs = [ex.submit(_try, p) for p in race_pool]
        _done.wait(timeout=32)
        # Cancel pending futures — we already have a winner
        for f in futs:
            f.cancel()

    if _result[0]:
        return _result[0], _winner[0]

    # Phase 2: sequential fallback for remaining providers
    for provider in sequential_tail:
        try:
            r = _call_provider(provider, system, prompt, max_tokens, temperature)
            if r:
                return r, provider["name"]
        except Exception as e:
            logger.debug("Provider %s failed (sequential): %s", provider["name"], e)

    return None, None


def _provider_key(provider: dict) -> str:
    """Return API key for a provider, checking primary env var then optional alt."""
    key = os.environ.get(provider["key_env"], "")
    if not key or key == "disabled":
        alt = provider.get("key_env_alt", "")
        key = os.environ.get(alt, "") if alt else ""
    if not key or key == "disabled":
        raise KeyError(f"No API key for provider {provider['name']}")
    return key


def _provider_keys(provider: dict) -> list[str]:
    """All configured keys for a provider: primary + numbered variants (_1.._3) + alt.

    Rotating across multiple free keys for the same provider multiplies rate-limit
    headroom — e.g. 4 working Groq keys behave like ~4x the per-key quota.
    """
    base = provider["key_env"]
    names = [base]
    if not base[-1:].isdigit():
        names += [f"{base}_{i}" for i in (1, 2, 3)]
    alt = provider.get("key_env_alt")
    if alt:
        names.append(alt)
    keys: list[str] = []
    for n in names:
        v = os.environ.get(n, "").strip()
        if v and v != "disabled" and v not in keys:
            keys.append(v)
    return keys


# Cloudflare (which fronts Groq, Cerebras and several others) blocks the default
# urllib User-Agent with "error code: 1010". A normal browser UA gets through, so
# every provider request must send one.
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def _call_provider(provider: dict, system: str, prompt: str, max_tokens: int, temperature: float) -> str:
    keys = _provider_keys(provider)
    if not keys:
        raise KeyError(f"No API key for provider {provider['name']}")
    url = provider["url"]

    if provider["fmt"] == "gemini":
        body = {
            "contents": [{"role": "user", "parts": [{"text": f"{system}\n\n{prompt}"}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        }
    else:
        body = {
            "model": provider.get("model", "llama-3.3-70b-versatile"),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    data = json.dumps(body).encode()

    # Rotate across all keys for this provider — a rate-limited/forbidden key
    # falls through to the next, multiplying effective free capacity.
    last_exc: Exception | None = None
    for key in keys:
        if provider["fmt"] == "gemini":
            url_with_key = f"{url}?key={key}"
            headers = {"Content-Type": "application/json", "User-Agent": _UA}
        else:
            url_with_key = url
            headers = {
                "Content-Type": "application/json",
                "User-Agent": _UA,
                "Authorization": f"Bearer {key}",
            }
        req = urllib.request.Request(url_with_key, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            if provider["fmt"] == "gemini":
                return result["candidates"][0]["content"]["parts"][0]["text"].strip()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001 — try the next key on any failure
            last_exc = exc
    raise last_exc if last_exc else RuntimeError(f"{provider['name']}: all keys failed")


def _call_provider_messages(
    provider: dict,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> str:
    """
    Call a provider with a full OpenAI-style messages array.
    Used by llm_chat() so that multi-turn history is preserved verbatim.
    All 7 providers accept this format; Gemini needs a small shape translation.
    """
    keys = _provider_keys(provider)
    if not keys:
        raise KeyError(f"No API key for provider {provider['name']}")
    url = provider["url"]

    if provider["fmt"] == "gemini":
        # Gemini uses {role: "user"|"model", parts: [{text: ...}]}
        # System message is prepended to the first user turn.
        system_content = next((m["content"] for m in messages if m["role"] == "system"), "")
        gemini_contents: list[dict] = []
        first_user = True
        for m in messages:
            if m["role"] == "system":
                continue
            role = "model" if m["role"] == "assistant" else "user"
            content = m["content"]
            if first_user and role == "user" and system_content:
                content = f"{system_content}\n\n{content}"
                first_user = False
            gemini_contents.append({"role": role, "parts": [{"text": content}]})
        body: dict = {
            "contents": gemini_contents,
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        }
    else:
        body = {
            "model": provider.get("model", "llama-3.3-70b-versatile"),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    data = json.dumps(body).encode()

    last_exc: Exception | None = None
    for key in keys:
        if provider["fmt"] == "gemini":
            url_with_key = f"{url}?key={key}"
            headers: dict = {"Content-Type": "application/json", "User-Agent": _UA}
        else:
            url_with_key = url
            headers = {
                "Content-Type": "application/json",
                "User-Agent": _UA,
                "Authorization": f"Bearer {key}",
            }
        req = urllib.request.Request(url_with_key, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            if provider["fmt"] == "gemini":
                return result["candidates"][0]["content"]["parts"][0]["text"].strip()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001 — try the next key on any failure
            last_exc = exc
    raise last_exc if last_exc else RuntimeError(f"{provider['name']}: all keys failed")


def llm_chat(
    store: "ConversationStore",
    user_message: str,
    system: str = "You are a helpful AI agent at QuantEdge, a quantitative trading firm.",
    max_tokens: int = 400,
    temperature: float = 0.7,
) -> str:
    """
    Multi-turn conversation that works across ALL free LLM providers.

    How cross-provider context sharing works:
      1. ConversationStore saves history in the universal OpenAI messages format.
      2. This function appends user_message, builds the full messages array,
         and sends it to whichever provider is currently available.
      3. The reply is saved back. Next call — even to a DIFFERENT provider — sees
         the complete history and continues naturally.

    Example:
        store = ConversationStore("regime_analysis")
        reply1 = llm_chat(store, "What regime are we in?")    # Gemini answers
        reply2 = llm_chat(store, "How should I size positions?")  # Groq continues seamlessly

    Args:
        store: A ConversationStore instance (persisted to .github/state/conversations/).
        user_message: The new user turn to add.
        system: System prompt (stable across all turns).
        max_tokens: Response length cap.
        temperature: Sampling temperature.

    Returns:
        The assistant's reply text, also saved into store.
    """
    store.add("user", user_message)
    messages = store.build_messages(system)

    for provider in _PROVIDERS:
        if not _has_key(provider):
            continue
        try:
            result = _call_provider_messages(provider, messages, max_tokens, temperature)
            if result:
                store.add("assistant", result)
                return result
        except Exception as e:
            logger.debug("Provider %s failed in llm_chat: %s", provider["name"], e)
            continue

    reply = "[LLM unavailable — all providers failed]"
    store.add("assistant", reply)
    return reply


# ── Shared memory (company brain) ─────────────────────────────────────────────

_DEFAULT_BRAIN: dict = {
    "core": {
        "market_regime": "unknown",
        "top_strategies": [],
        "best_model": None,
        "risk_status": "normal",
        "last_updated": 0,
    },
    "episodic": [],          # last 200 events with lessons
    "skills": [],            # reusable solutions
    "slack_insights": [],    # lessons from Slack threads
    "github_insights": [],   # lessons from PR reviews + issues
    "trade_outcomes": [],    # recent P&L + what worked
    "experiment_results": [], # ML experiment outcomes
}


_brain_cache: dict = {}
_brain_cache_ts: float = 0.0
_brain_cache_lock = threading.Lock()
_BRAIN_CACHE_TTL = 60.0  # seconds — avoids disk read on every context injection


def _load_brain() -> dict:
    global _brain_cache, _brain_cache_ts
    now = time.time()
    with _brain_cache_lock:
        if _brain_cache and now - _brain_cache_ts < _BRAIN_CACHE_TTL:
            return json.loads(json.dumps(_brain_cache))
        try:
            if _BRAIN_FILE.exists():
                loaded = json.loads(_BRAIN_FILE.read_text())
                _brain_cache = loaded
                _brain_cache_ts = now
                return json.loads(json.dumps(loaded))
        except Exception:
            pass
        fallback = json.loads(json.dumps(_DEFAULT_BRAIN))
        _brain_cache = fallback
        _brain_cache_ts = now
        return json.loads(json.dumps(fallback))


def _locked_json_write(path: Path, data: dict) -> None:
    """Atomic JSON write with exclusive flock — prevents concurrent-write corruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, path)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _save_brain(brain: dict) -> None:
    global _brain_cache, _brain_cache_ts
    try:
        _locked_json_write(_BRAIN_FILE, brain)
        with _brain_cache_lock:
            _brain_cache = json.loads(json.dumps(brain))
            _brain_cache_ts = time.time()
    except Exception as e:
        logger.debug("_save_brain failed: %s", e)


def get_company_context(max_tokens: int = 600) -> str:
    """
    Build a token-efficient company context string to inject into every prompt.
    Under 600 tokens — enough to inform without dominating.
    """
    brain = _load_brain()
    core = brain.get("core", {})
    recent_lessons = brain.get("episodic", [])[-5:]
    top_skills = brain.get("skills", [])[-3:]
    slack_insights = brain.get("slack_insights", [])[-2:]
    trade_outcomes = brain.get("trade_outcomes", [])[-2:]

    parts = []

    regime = core.get("market_regime", "unknown")
    if regime != "unknown":
        parts.append(f"Market regime: {regime}")

    top_strats = core.get("top_strategies", [])
    if top_strats:
        parts.append(f"Top strategies: {', '.join(top_strats[:3])}")

    risk = core.get("risk_status", "normal")
    if risk != "normal":
        parts.append(f"Risk status: {risk}")

    if trade_outcomes:
        outcomes_str = " | ".join(f"{o.get('strategy','?')}: {o.get('outcome','?')}" for o in trade_outcomes)
        parts.append(f"Recent trades: {outcomes_str}")

    if recent_lessons:
        lessons = [e.get("lesson", "") for e in recent_lessons if e.get("lesson")]
        if lessons:
            parts.append("Recent lessons: " + "; ".join(lessons[:3]))

    if slack_insights:
        parts.append("Slack: " + " | ".join(i.get("summary", "") for i in slack_insights if i.get("summary")))

    if top_skills:
        parts.append("Known solutions: " + " | ".join(s.get("name", s) if isinstance(s, dict) else str(s) for s in top_skills))

    if not parts:
        return ""

    return "[COMPANY CONTEXT]\n" + "\n".join(parts) + "\n[/COMPANY CONTEXT]"


def memory_write(category: str, entry: dict) -> None:
    """Write an entry to shared company brain. Category: episodic|skills|slack_insights|github_insights|trade_outcomes|experiment_results"""
    brain = _load_brain()
    entry["ts"] = time.time()
    lst = brain.setdefault(category, [])
    lst.append(entry)
    # Rolling window caps
    caps = {"episodic": 200, "skills": 100, "slack_insights": 100, "github_insights": 100,
            "trade_outcomes": 200, "experiment_results": 100}
    if len(lst) > caps.get(category, 200):
        brain[category] = lst[-caps.get(category, 200):]
    _save_brain(brain)


def memory_read(category: str, n: int = 10) -> list[dict]:
    """Read recent entries from a category."""
    brain = _load_brain()
    return brain.get(category, [])[-n:]


def core_update(key: str, value: Any) -> None:
    """Update a CORE memory slot (stable, overwritten)."""
    brain = _load_brain()
    brain.setdefault("core", {})[key] = value
    brain["core"]["last_updated"] = time.time()
    _save_brain(brain)


def core_get(key: str, default: Any = None) -> Any:
    brain = _load_brain()
    return brain.get("core", {}).get(key, default)


# ── Slack helpers ─────────────────────────────────────────────────────────────

def slack_post(channel: str, text: str, thread_ts: str | None = None) -> dict:
    """Post to Slack. Returns the message object (ts, channel)."""
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        return {}
    body: dict = {"channel": channel, "text": text}
    if thread_ts:
        body["thread_ts"] = thread_ts
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=data,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.debug("slack_post failed: %s", e)
        return {}


def slack_read_thread(channel: str, thread_ts: str, limit: int = 20) -> list[dict]:
    """Read full thread history — so agents can see replies directed at them."""
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        return []
    url = f"https://slack.com/api/conversations.replies?channel={channel}&ts={thread_ts}&limit={limit}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("messages", [])
    except Exception:
        return []


def slack_read_channel(channel: str, limit: int = 50, oldest: float = 0) -> list[dict]:
    """Read recent messages from a channel."""
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        return []
    params = f"channel={channel}&limit={limit}"
    if oldest:
        params += f"&oldest={oldest}"
    url = f"https://slack.com/api/conversations.history?{params}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("messages", [])
    except Exception:
        return []
