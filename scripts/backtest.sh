#!/bin/bash
# Usage: ./scripts/backtest.sh <strategy> <symbol> <interval> <start> <end>
# Example: ./scripts/backtest.sh momentum SPY 1d 2021-01-01 2024-01-01
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m backend.app.backtest.cli "$@"
