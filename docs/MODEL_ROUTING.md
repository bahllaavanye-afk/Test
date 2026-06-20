# Model Routing — the cost ladder agents follow

> How QuantEdge agents pick a model. Implemented in `.github/scripts/llm_common.py`
> (`llm_routed()`). Goal: spend the cheapest tokens that still clear the task's quality bar,
> and keep **Claude a rare backstop**. Full rationale: `docs/research/LLM_COST_OPTIMIZATION.md`.

## The ladder
```
tier="cheap"   FREE cascade only
tier="mid"     FREE → OpenRouter open-mid
tier="auto"    FREE → OpenRouter open-mid → Claude (only if BOTH fail)   ← default
tier="hard"    FREE → OpenRouter open-mid → Claude (reserved for the hardest tasks)
```
Escalation happens **only on failure**, so the cheapest capable tier wins and Claude is hit
rarely.

| Tier | Models | Cost | Use for |
|------|--------|------|---------|
| FREE | Groq, Gemini, Cerebras, SambaNova, DeepSeek, Together, Hyperbolic, NVIDIA NIM (raced) | $0 | classify, extract, route, summarize, triage — the 80% case |
| OPEN-MID | OpenRouter: DeepSeek V4, Qwen3, Kimi K2, GLM, MiniMax M2 | 10–50× cheaper than frontier | code edits, multi-step reasoning, reviews |
| CLAUDE | `claude-sonnet-4-6` (default), or Haiku 4.5 / Opus 4.8 | premium | hardest reasoning / architecture / tricky bugs only |

## Usage
```python
from llm_common import llm_routed

# Bulk, easy — never leaves the free tier:
label = llm_routed(prompt, tier="cheap", max_tokens=20)

# Default — free, escalate to open-mid, Claude only as last resort:
answer = llm_routed(prompt)                # tier="auto"

# Hard task — allow the Claude backstop:
plan = llm_routed(prompt, tier="hard", max_tokens=1200)
```
`llm_routed()` keeps `llm()`'s 24h cache + company-context injection, and records every
call (incl. tier) to `.github/state/llm_metrics.jsonl`.

## Configuration (env / Doppler — no hard-coded model lock-in)
| Var | Default | Purpose |
|-----|---------|---------|
| `OPENROUTER_API_KEY` | — | enables the open-mid tier (already provisioned) |
| `OPENROUTER_MODELS` | `deepseek/deepseek-chat,qwen/qwen-2.5-72b-instruct,moonshotai/kimi-k2,z-ai/glm-4.6,minimax/minimax-m2` | ordered open-mid fallback list; set to the exact current SOTA slugs |
| `ANTHROPIC_API_KEY` | — | enables the Claude backstop; absent ⇒ tier silently degrades |
| `CLAUDE_BACKSTOP_MODEL` | `claude-sonnet-4-6` | `claude-haiku-4-5-20251001` (cheaper) or `claude-opus-4-8` (hardest) |

If a tier has no key it is skipped, never an error — the ladder always degrades gracefully
down to the free cascade.

## Cost levers when a call does reach Claude
- **Prompt caching** — stable prefix (system + tool defs + long context) is ~90% cheaper.
- **Batch API** — non-interactive bulk jobs are ~50% cheaper.
- **`max_tokens` discipline** — cap output; long generations are the silent cost driver.

See `docs/research/LLM_COST_OPTIMIZATION.md` for the full playbook.
