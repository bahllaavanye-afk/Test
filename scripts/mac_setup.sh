#!/usr/bin/env bash
#
# QuantEdge — one-command macOS bootstrap for the cross-provider agent stack.
#
# Run this ONCE in your Mac Terminal:
#     bash scripts/mac_setup.sh
#
# It does everything automatable:
#   • checks/installs Node 18.18+ (via Homebrew)
#   • installs the Codex, Gemini, and Grok CLIs
#   • installs the Claude Code CLI
#   • launches `codex login` (the ONE step only you can finish — it opens your
#     browser to sign in with your ChatGPT account; free tier works)
#
# After it finishes, run `claude` in your repo and enable the plugins with the
# three /plugin commands it prints at the end.

set -uo pipefail

bold() { printf "\033[1m%s\033[0m\n" "$1"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$1"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$1"; }
info() { printf "  \033[36m·\033[0m %s\n" "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

bold "QuantEdge — macOS agent bootstrap"
echo

# ── 1. Node 18.18+ ────────────────────────────────────────────────────────────
bold "1. Node.js"
need_node=1
if have node; then
  major=$(node -p "process.versions.node.split('.')[0]" 2>/dev/null || echo 0)
  minor=$(node -p "process.versions.node.split('.')[1]" 2>/dev/null || echo 0)
  if [ "$major" -gt 18 ] || { [ "$major" -eq 18 ] && [ "$minor" -ge 18 ]; }; then
    ok "node $(node --version)"; need_node=0
  else
    warn "node $(node --version) is too old (need >= 18.18)"
  fi
fi
if [ "$need_node" -eq 1 ]; then
  if have brew; then
    info "installing Node via Homebrew ..."
    brew install node && ok "Node installed"
  else
    warn "Homebrew not found. Install it from https://brew.sh then re-run, or"
    warn "download Node 18.18+ from https://nodejs.org"
    exit 1
  fi
fi
echo

# ── 2. CLIs ───────────────────────────────────────────────────────────────────
bold "2. Cross-provider + Claude Code CLIs"
npm_install() {
  local label="$1" pkg="$2" probe="$3"
  if have "$probe"; then ok "$label already installed"; return; fi
  info "installing $label ..."
  if npm install -g "$pkg" >/dev/null 2>&1; then
    ok "$label installed"
  else
    warn "global install failed (permissions?). Retrying with a user prefix ..."
    npm install -g "$pkg" --prefix "$HOME/.npm-global" >/dev/null 2>&1 \
      && { ok "$label installed to ~/.npm-global"; export PATH="$HOME/.npm-global/bin:$PATH"; } \
      || warn "could not install $label — run manually: npm install -g $pkg"
  fi
}
npm_install "Claude Code" "@anthropic-ai/claude-code" "claude"
npm_install "OpenAI Codex" "@openai/codex"            "codex"
npm_install "Google Gemini" "@google/gemini-cli"      "gemini"
npm_install "xAI Grok"      "@vibe-kit/grok-cli"      "grok"

# Persist ~/.npm-global on PATH for future shells (zsh is the Mac default).
if [[ ":$PATH:" == *":$HOME/.npm-global/bin:"* ]] && ! grep -q '.npm-global/bin' "$HOME/.zshrc" 2>/dev/null; then
  echo 'export PATH=~/.npm-global/bin:$PATH' >> "$HOME/.zshrc"
  info "added ~/.npm-global/bin to PATH in ~/.zshrc (open a new tab to pick it up)"
fi
echo

# ── 3. Codex login (the one manual moment) ────────────────────────────────────
bold "3. Codex login"
if codex login status >/dev/null 2>&1; then
  ok "Codex already logged in"
else
  info "Launching 'codex login' — your browser will open. Sign in with your"
  info "ChatGPT account (the free tier works). Come back here when done."
  echo
  codex login || warn "login did not complete — you can re-run: codex login"
  echo
  if codex login status >/dev/null 2>&1; then ok "Codex logged in"; else warn "still not logged in — re-run: codex login"; fi
fi
echo

# ── 4. Next steps ─────────────────────────────────────────────────────────────
bold "4. Enable the plugins (inside Claude Code)"
cat <<'EOS'
  Start Claude Code in your repo, then paste these into its message box:

    cd ~/path/to/Test        # your cloned repo
    claude                   # launches Claude Code

  Then, in Claude Code:
    /plugin marketplace add openai/codex-plugin-cc
    /plugin install codex@openai-codex
    /codex:setup             # green = fully working

  Optional — Gemini & Grok:
    /plugin marketplace add sakibsadmanshajib/gemini-plugin-cc
    /plugin install gemini@gemini-cc
    /plugin marketplace add zachdunn/grok-plugin-claude-code
    /plugin install grok@grok-cc
    (then once each:  gemini   # Google sign-in
                      grok     # paste xAI key)
EOS
echo
bold "Bootstrap complete."
