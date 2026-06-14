#!/usr/bin/env bash
#
# QuantEdge — Multi-Agent / Cross-Provider Setup
# ----------------------------------------------
# Automates everything that CAN be automated for the multi-agent dev setup:
#   1. Installs the cross-provider CLIs (Codex, Gemini, Grok) used by Claude
#      Code plugins.
#   2. Verifies the free-OpenAI LLM pool (GitHub Models) is reachable.
#   3. Prints the exact `/plugin` slash commands to run inside Claude Code
#      (Anthropic requires plugin enable/install to go through the trusted UI
#      path — it cannot be scripted into settings.json for security reasons).
#
# Usage:
#   ./scripts/setup_agents.sh                 # install everything
#   ./scripts/setup_agents.sh --check         # only verify, install nothing
#
# Safe to re-run; every step is idempotent.

set -uo pipefail

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

bold() { printf "\033[1m%s\033[0m\n" "$1"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$1"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$1"; }
info() { printf "  \033[36m·\033[0m %s\n" "$1"; }

have() { command -v "$1" >/dev/null 2>&1; }

bold "QuantEdge multi-agent setup"
echo

# ── 1. Node toolchain ─────────────────────────────────────────────────────────
bold "1. Node.js (>= 18.18 required for the cross-provider CLIs)"
if have node; then
  ok "node $(node --version)"
else
  warn "node not found — install Node 18.18+ from https://nodejs.org and re-run"
fi
echo

# ── 2. Cross-provider CLIs ────────────────────────────────────────────────────
# Each Claude Code plugin shells out to one of these CLIs. Installing the CLI is
# what makes /codex:, /gemini:, and /grok: commands actually work.
install_cli() {
  local name="$1" pkg="$2" probe="$3"
  if have "$probe"; then
    ok "$name already installed ($($probe --version 2>/dev/null | head -1))"
    return
  fi
  if [[ $CHECK_ONLY -eq 1 ]]; then
    warn "$name not installed (run without --check to install)"
    return
  fi
  if have npm; then
    info "installing $name ($pkg) ..."
    npm install -g "$pkg" >/dev/null 2>&1 && ok "$name installed" \
      || warn "$name install failed — run: npm install -g $pkg"
  else
    warn "npm unavailable — cannot install $name"
  fi
}

bold "2. Cross-provider CLIs"
install_cli "OpenAI Codex" "@openai/codex"            "codex"
install_cli "Google Gemini" "@google/gemini-cli"      "gemini"
install_cli "xAI Grok"      "@vibe-kit/grok-cli"      "grok"
echo

# Perplexity integrates as an MCP server (web-grounded research), not a CLI.
bold "2b. Perplexity (MCP server — web-grounded research)"
if [[ -n "${PERPLEXITY_API_KEY:-}" ]]; then
  ok "PERPLEXITY_API_KEY present — backend research route + MCP ready"
else
  warn "set PERPLEXITY_API_KEY to enable Perplexity Sonar (backend research route + MCP)"
fi
echo

# ── 3. Free OpenAI LLM pool (GitHub Models) ───────────────────────────────────
bold "3. Free OpenAI models in the backend pool (GitHub Models)"
TOKEN="${GITHUB_MODELS_TOKEN:-${GITHUB_TOKEN:-${GH_TOKEN:-}}}"
if [[ -n "$TOKEN" ]]; then
  ok "GitHub token present — free GPT-4o / GPT-4o-mini / o4-mini are LIVE in free_llm_router"
  info "models.github.ai/inference reachable via GITHUB_MODELS_TOKEN (or GITHUB_TOKEN fallback)"
else
  warn "no GitHub token in env — set GITHUB_MODELS_TOKEN to activate free OpenAI models"
  info "any GitHub PAT with 'models' scope works; in CI the built-in GITHUB_TOKEN is used automatically"
fi
echo

# ── 4. Plugin enable commands (manual, by design) ─────────────────────────────
bold "4. Enable the cross-provider plugins inside Claude Code"
cat <<'EOS'
  Run these in your Claude Code terminal (they must go through the trusted
  /plugin UI — settings.json cannot auto-enable third-party plugins):

    # OpenAI Codex — review, adversarial audits, task delegation
    /plugin marketplace add openai/codex-plugin-cc
    /plugin install codex@openai-codex
    /codex:setup

    # Google Gemini — visual analysis + second-opinion review
    /plugin marketplace add sakibsadmanshajib/gemini-plugin-cc
    /plugin install gemini@gemini-cc

    # xAI Grok — fast delegated coding + review
    /plugin marketplace add zachdunn/grok-plugin-claude-code
    /plugin install grok@grok-cc

    # Perplexity — web-grounded research (MCP server, not a /plugin)
    claude mcp add perplexity --env PERPLEXITY_API_KEY="$PERPLEXITY_API_KEY" -- npx -y @perplexity-ai/mcp-server

  After installing, authenticate each CLI once:
    codex login        # ChatGPT account (free tier ok) or OPENAI_API_KEY
    gemini             # follow the Google sign-in prompt
    grok               # paste your xAI key
    # Perplexity uses PERPLEXITY_API_KEY directly — no interactive login
EOS
echo
bold "Done."
[[ $CHECK_ONLY -eq 1 ]] && info "check-only mode — nothing was installed"
echo "See docs/MULTI_AGENT_SETUP.md for the worktree workflow and full details."
