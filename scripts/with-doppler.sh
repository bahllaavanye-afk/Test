#!/usr/bin/env bash
# Run any command with secrets injected from Doppler (the single source of truth).
#
#   ./scripts/with-doppler.sh ./scripts/launch.sh paper
#   ./scripts/with-doppler.sh python scripts/slack_message_monitor.py
#
# Requires the Doppler CLI (https://docs.doppler.com/docs/install-cli) and either a prior
# `doppler setup` in this repo, or DOPPLER_TOKEN / DOPPLER_PROJECT+DOPPLER_CONFIG in the env.
# See SECRETS.md for the one-time setup.
set -euo pipefail

if ! command -v doppler >/dev/null 2>&1; then
  echo "error: doppler CLI not found. Install it: curl -Ls https://cli.doppler.com/install.sh | sh" >&2
  echo "       then see SECRETS.md for setup." >&2
  exit 127
fi

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

exec doppler run -- "$@"
