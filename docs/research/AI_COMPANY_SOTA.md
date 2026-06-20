# SOTA: How to Make QuantEdge a Top-Tier AI-First Company

> Durable research record. This is AI-first and must run 24/7. Captured so work
> survives session/token resets — see `CONTINUITY.md` for how to resume.
> _Last updated: 2026-06-20._

## TL;DR — the one thing that matters
The company already runs 24/7 (78 workflows, 63 scheduled crons, ~87% workflow
success). The failure mode is **silent degradation**: when the free-LLM cascade (the
agents' "brain") dies, workflows still go green but produce empty output. Everything
below is about making the system **observably alive, cheap, and self-correcting** rather
than just "running."

The maturity ladder we are climbing:

```
Phase 0  Brain alive          ✅ User-Agent fix + key rotation + hourly canary (#144/#145/#146)
Phase 1  Brain observable     ✅ llm_metrics.jsonl + cascade_status() + brain-health.yml
Phase 2  Cost-tiered routing  ▶  free → open-mid (OpenRouter) → Claude backstop  (this PR)
Phase 3  Real memory          ▢  pgvector (Mem0/Letta/Zep) instead of flat JSON state
Phase 4  Verifiable self-improvement  ▢  reward = CI-green + coverage Δ + paper Sharpe Δ
Phase 5  Durable orchestration ▢  Temporal/LangGraph instead of fire-and-forget cron
Phase 6  A2A protocol          ▢  typed agent-to-agent; Slack demoted to human digest
```

---

## 1. Observability + model routing (Phases 1–2)
**Problem:** no traces, no per-call metrics, no idea which provider answers or how much
anything costs. A dead cascade was invisible for days.

**SOTA practice:**
- **Per-call metrics** (provider, ok, latency) — shipped in `llm_metrics.jsonl`.
- **Active health canary** — `brain_health.py` probes every keyed provider hourly and
  pages Slack `#infra-alerts` + goes red when the brain is down (`brain-health.yml`).
- **Tracing** — next: Langfuse or OpenTelemetry GenAI spans on `llm_common` for latency,
  token, cost, and prompt/response capture (one decorator at the `llm()` boundary).
- **Task-tier model routing** — route by difficulty, not by habit. See
  `docs/MODEL_ROUTING.md` and `llm_routed()`. FrugalGPT / RouteLLM show 50–98% cost cuts
  at matched quality by sending the 80% easy traffic to the cheapest capable model.

## 2. A real memory layer (Phase 3)
**Problem:** "memory" is flat `.github/state/*.json` with recency windows + TF-IDF. No
episodic/semantic recall, no cross-run learning that compounds.

**SOTA practice:** an agent-memory service backed by the **Supabase pgvector** we already
have. Candidates: **Mem0** (extraction + vector recall, simple API), **Letta/MemGPT**
(tiered memory + self-editing), **Zep** (temporal knowledge graph). Store episodic events,
distilled "lessons," and reusable skills; retrieve by semantic relevance per task instead
of "last 5." This is the difference between an agent that *logs* and one that *learns*.

## 3. Outcome-driven self-improvement (Phase 4)
**Problem:** the self-improver edits code with no *verifiable* reward signal, so it can
"improve" things that regress quality.

**SOTA practice (DeepSWE / Darwin–Gödel-Machine pattern):** give every agent PR a
**verifiable reward** before it can merge:
- CI green (already gated) **+** coverage Δ ≥ 0 **+** paper-backtest Sharpe Δ ≥ 0
- An **LLM-judge eval** gate (a stronger model reviews the diff against a rubric).
Only PRs that clear the reward gate auto-merge; the rest open for human review. This turns
"agents writing code" into "agents writing *measurably better* code."

## 4. Durable, event-driven orchestration (Phase 5)
**Problem:** the core loop (lead → engineer → reviewer) is fire-and-forget cron. A crashed
step is just lost; there are no retries, no resumable state, no backpressure.

**SOTA practice:** move the loop onto **durable execution** — Temporal, Inngest, or
LangGraph. Each step becomes a retriable, checkpointed activity with typed I/O; a failure
resumes from the last checkpoint instead of silently vanishing. Cron stays only as the
*trigger*, not the *runtime*.

## 5. Typed agent-to-agent coordination (Phase 6)
**Problem:** agents coordinate by posting to Slack, which produces repeated-message noise
and is lossy/unstructured.

**SOTA practice:** an **A2A** (agent-to-agent) protocol — typed task hand-offs, capability
discovery, structured results — with **MCP** for tool/data access. Slack is then demoted to
a *human* digest (one concise summary per channel), killing the repeated-message problem.

## 6. Open-weight mid-tier so Claude is the rare backstop (shipping in Phase 2)
Open models have closed the gap with frontier on agentic benchmarks at **10–50× lower
cost**: DeepSeek V4, Qwen3, Kimi K2, GLM, MiniMax M2. Route **free → open-mid (via
OpenRouter) → Claude only for the hardest tasks.** Implemented in `llm_routed()`; models
are env-configurable (`OPENROUTER_MODELS`). See `docs/MODEL_ROUTING.md`.

---

## Where tasks live (so nothing is lost between sessions)
- **Canonical queue:** GitHub Issues labeled `agent-fix-needed` (agents create/work these).
- **Human board:** Notion, mirrored from GitHub Issues.
- **Cross-session continuity:** `CONTINUITY.md` + `IMPROVEMENTS.md` + `HANDOFF.md`, committed
  to the repo (chat sessions are ephemeral — only committed state survives).
- **Slack:** notifications / visibility only — never the source of truth.

## Primary sources / further reading
- FrugalGPT (Chen et al.) — LLM cascades, 50–98% cost reduction at matched quality.
- RouteLLM / xRouter — learned query routing between weak/strong models.
- Mem0, Letta (MemGPT), Zep — agent memory layers.
- DeepSWE, Darwin-Gödel Machine — verifiable-reward self-improving agents.
- Temporal, Inngest, LangGraph — durable/event-driven agent execution.
- MCP + A2A — open protocols for tools and agent-to-agent coordination.
- MindStudio "best open agentic models 2026"; Epoch AI "open–closed gap" tracking.
