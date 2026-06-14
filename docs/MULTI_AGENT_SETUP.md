# Multi-Agent & Cross-Provider Setup

QuantEdge is built to be developed by a fleet of AI agents in parallel. This
guide covers two layers:

1. **Cross-provider plugins** inside Claude Code (Codex, Gemini, Grok) — get a
   second/third model's opinion without leaving the terminal.
2. **The free-OpenAI LLM pool** in the backend (`free_llm_router.py`) — the
   24/7 autonomous agents now use real OpenAI models (GPT-4o / o4-mini) for
   free via GitHub Models.

One-command setup:

```bash
./scripts/setup_agents.sh          # installs CLIs, verifies pool, prints plugin commands
./scripts/setup_agents.sh --check  # verify only
```

---

## 1. Cross-provider plugins (Claude Code)

Anthropic's plugin system lets other vendors' CLIs run *inside* Claude Code.
You keep Claude Code as the orchestrator and delegate review/coding to other
models. Enable each from the Claude Code terminal:

| Plugin | Marketplace | Install | What it adds |
|--------|-------------|---------|--------------|
| **OpenAI Codex** | `/plugin marketplace add openai/codex-plugin-cc` | `/plugin install codex@openai-codex` | `/codex:review`, `/codex:adversarial-review`, `/codex:rescue`, task delegation |
| **Google Gemini** | `/plugin marketplace add sakibsadmanshajib/gemini-plugin-cc` | `/plugin install gemini@gemini-cc` | `/gemini:review`, visual/screenshot analysis |
| **xAI Grok** | `/plugin marketplace add zachdunn/grok-plugin-claude-code` | `/plugin install grok@grok-cc` | `/grok:review`, fast delegated coding |

After install, verify Codex and toggle the optional **review gate** (blocks
Claude from finalizing changes until Codex reviews them):

```
/codex:setup --enable-review-gate
```

Authenticate the CLIs once:

```bash
codex login     # ChatGPT account (free tier works) or OPENAI_API_KEY
gemini          # Google sign-in
grok            # xAI API key
```

> **Why these aren't auto-enabled:** enabling a third-party plugin executes its
> code at Claude Code startup, so Anthropic requires it to go through the
> trusted `/plugin` UI — it deliberately can't be written into `settings.json`
> by an agent. The marketplace add + install above is the supported path.

Also worth enabling from the official marketplace (`/plugin marketplace add
anthropics/claude-plugins-official`):

- **Superpowers** — multi-step agent workflows
- **Frontend Design** — UI/design agent (useful for the dashboard)
- **Context7** — up-to-date library docs injected into context

---

## 2. Free OpenAI models in the backend pool

`backend/app/tasks/free_llm_router.py` now includes **GitHub Models**, which
serves OpenAI's GPT-4o family through an OpenAI-compatible endpoint at no cost:

| Provider name | Model | Use |
|---------------|-------|-----|
| `github_gpt4o_mini` | `openai/gpt-4o-mini` | fast tasks, cheap reasoning |
| `github_gpt4o` | `openai/gpt-4o` | high-quality analysis |
| `github_o4_mini` | `openai/o4-mini` | code + step-by-step reasoning |

**Activation is automatic.** The router resolves the credential in this order:

```
GITHUB_MODELS_TOKEN  →  GITHUB_TOKEN  →  GH_TOKEN
```

In GitHub Actions the built-in `GITHUB_TOKEN` is always present, so every
autonomous-agent workflow gets free OpenAI models with zero extra config. For
local/Render use, set `GITHUB_MODELS_TOKEN` to any GitHub PAT with the `models`
scope.

These models are wired into `call_routed()`'s preferences:

- `task_type="code"`  → `o4-mini` and `gpt-4o` lead
- `task_type="analysis"` → `gpt-4o` mixed with Llama-70B providers
- `task_type="fast"` → `gpt-4o-mini` alongside Cerebras/Groq

The full free pool is now **13 providers** (Gemini, Groq, DeepSeek, SambaNova,
Cerebras, Together, Hyperbolic, NVIDIA NIM, OpenRouter, Gemini-Thinking, plus
the three GitHub Models OpenAI entries).

---

## 3. Parallel development — git worktree model

Never let two agents write to the same branch at once. Use worktrees so each
agent has an isolated checkout of the **same** repo (no re-cloning):

```bash
# one worktree per agent, each on its own branch
git worktree add ../qe-codex   -b agent/codex-task
git worktree add ../qe-gemini  -b agent/gemini-task
git worktree add ../qe-frontend -b agent/frontend-task

# point a different IDE / CLI at each directory
#   VS Code        → ../qe-frontend
#   Claude Code    → main repo (orchestrator)
#   Codex CLI      → ../qe-codex
#   Gemini CLI     → ../qe-gemini

# when an agent's branch is ready, open a PR and remove the worktree
git worktree remove ../qe-codex
```

Recommended topology:

```
Claude Code (orchestrator, this repo)
  ├── Agent tool → spawns sub-agents in temporary worktrees (isolation: "worktree")
  ├── /codex:review  → OpenAI second opinion
  ├── /gemini:review → Google visual + review
  └── /grok:review   → xAI fast review
```

You do **not** need to switch to VS Code. Claude Code's `Agent` tool already
spawns isolated worktree agents; the cross-provider plugins add other models'
reviews on top. Use VS Code only if you want a human-driven window alongside.
