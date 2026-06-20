# SOTA: Saving Claude Tokens & Offloading to Free/Open LLMs

> Durable research record for an AI-first agentic company that must run 24/7 cheaply.
> Captured so work survives session/token resets — see `CONTINUITY.md`.
> _Last updated: 2026-06-20._

## Principle
**Spend the cheapest tokens that still clear the quality bar for the task.** Most agentic
traffic (classify, extract, summarize, route, lint, triage) does **not** need a frontier
model. Reserve premium Claude calls for the genuinely hard reasoning/coding tasks.

Expected blended savings when applied together: **70–95%** versus "send everything to a
premium model," with no measurable quality loss on the easy majority.

---

## 1. Cost-tiered routing ladder (biggest lever)
Route by task difficulty, escalating **only on failure** so the cheapest capable tier wins:

```
FREE cascade  →  OPEN-MID (OpenRouter)  →  CLAUDE backstop
(Groq/Gemini/…)   (DeepSeek/Qwen/Kimi/…)   (rare, hardest tasks only)
```

- **Free cascade** — 7 free providers raced in parallel (`llm()`), already live.
- **Open-mid** — open-weight models at 10–50× lower cost than frontier, via OpenRouter.
- **Claude backstop** — only `tier="hard"` or when both lower tiers fail.

Implemented in `llm_common.llm_routed()`. See `docs/MODEL_ROUTING.md`.
FrugalGPT reports 50–98% cost reduction from cascading; RouteLLM shows ~85% cost cuts at
~95% of GPT-4 quality via a learned router. **Adopt FrugalGPT-style cascade now; add a
learned router later.**

## 2. Claude-native cost levers (when you *do* call Claude)
- **Prompt caching** — cache the stable prefix (system prompt, tool defs, long context).
  Cached input is **~90% cheaper** and faster. Put everything static at the top, the
  variable user turn at the bottom; mark the prefix as cacheable.
- **Batch API** — for non-interactive bulk jobs (nightly digests, backfills, evals), the
  Message Batches API is **~50% cheaper**. Most of our cron work is batchable.
- **Model ladder within Claude** — default to **Haiku 4.5**; escalate to **Sonnet 4.6**,
  and only the hardest reasoning to **Opus 4.8**. Haiku handles far more than people expect.
- **`max_tokens` discipline** — cap output length per task; long generations are the
  silent cost driver. Ask for structured/short outputs.
- **Stop sequences + JSON mode** — avoid paying for rambling; get parseable output.

## 3. Semantic + exact caching (free repeats)
- **Exact cache** — already live: 24h response cache keyed by prompt hash (`llm_cache.json`).
- **Semantic cache** — embed the prompt; if a past prompt is ≥ threshold cosine-similar,
  return its answer. Reported **50–80%** hit rates on repetitive agent traffic. Back it with
  the **Supabase pgvector** we already run. *(Phase-3 candidate.)*

## 4. Context compression (pay for fewer input tokens)
- **Retrieve, don't dump** — inject only task-relevant context (we do TF-IDF today; upgrade
  to embeddings). Smaller prompts = fewer input tokens on *every* call.
- **Prompt compression** — LLMLingua-style compression of long contexts (2–5× fewer tokens
  at similar quality) for the big-context calls.
- **Summarize-then-act** — distill long Slack/PR threads to a short brief with a *free*
  model, then reason over the brief with the premium model.

## 5. Offload structurally to free/open (do the work off Claude entirely)
- **Embeddings / classification / extraction** → free or open models, never Claude.
- **Bulk summarization & triage** → free cascade (`llm_routed(tier="cheap")`).
- **Draft → critique** → free model drafts, premium model only critiques/finalizes.
- **Key rotation across free providers** (live, #145) multiplies free rate-limit headroom
  so the free tier absorbs more traffic before any paid call is needed.

## 6. Measure it (you can't cut what you can't see)
- `llm_metrics.jsonl` records provider + latency per call (live). Extend with token counts
  and a cost estimate per tier so a weekly digest shows **% served free vs open vs Claude**
  and **$/task**. Tracing (Langfuse) gives this for free at the call boundary.

---

## Quick decision table
| Task | Tier | Why |
|------|------|-----|
| classify / extract / route / lint | `cheap` (free) | trivial; free models nail it |
| summarize threads, draft text, triage | `cheap` → `mid` | quality fine; escalate if free fails |
| code edits, multi-step reasoning, reviews | `mid` (open) | open mid-tier matches frontier on agentic |
| hardest reasoning / architecture / tricky bugs | `hard` (Claude) | rare backstop; worth premium |
| bulk / nightly / backfill jobs | Batch API + cheap | 50% off + cheapest tier |
| anything with a stable long prefix | prompt caching | ~90% off cached input |

## Primary sources / further reading
- FrugalGPT (Chen, Zaharia, Zou) — LLM cascades & cost reduction.
- RouteLLM (LMSYS) — learned weak/strong routing.
- Anthropic docs — prompt caching (~90% off cached input), Message Batches (~50% off).
- LLMLingua (Microsoft) — prompt compression.
- GPTCache — semantic caching for LLM apps.
- MindStudio / Epoch AI — open-model capability & cost tracking (DeepSeek/Qwen/Kimi/GLM/MiniMax).
