#!/bin/bash
# Usage: ./scripts/train.sh --config experiments/configs/lstm_btc_1h.yaml
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python experiments/run_experiment.py "$@"
