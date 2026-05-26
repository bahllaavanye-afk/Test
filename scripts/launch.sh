#!/bin/bash
# QuantEdge Master Launcher
# Usage: ./scripts/launch.sh [dev|paper|live|backtest|train|compare]
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

case "$1" in
  dev)
    echo "Starting QuantEdge in DEV mode..."
    cd "$ROOT_DIR"
    docker-compose up --build
    ;;
  paper)
    echo "Starting QuantEdge in PAPER TRADING mode..."
    cd "$ROOT_DIR"
    TRADING_MODE=paper docker-compose up
    ;;
  live)
    echo "WARNING: LIVE TRADING MODE"
    echo "Type CONFIRM to proceed:"
    read -r confirm
    if [ "$confirm" = "CONFIRM" ]; then
      cd "$ROOT_DIR"
      TRADING_MODE=live docker-compose up
    else
      echo "Aborted."
      exit 1
    fi
    ;;
  backtest)
    shift
    "$SCRIPT_DIR/backtest.sh" "$@"
    ;;
  train)
    shift
    "$SCRIPT_DIR/train.sh" "$@"
    ;;
  compare)
    shift
    "$SCRIPT_DIR/compare.sh" "$@"
    ;;
  *)
    echo "Usage: launch.sh [dev|paper|live|backtest|train|compare]"
    exit 1
    ;;
esac
