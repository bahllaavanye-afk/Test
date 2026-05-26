#!/bin/bash
# Usage: ./scripts/compare.sh <strategy> <symbol>
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m backend.app.comparison.cli "$@"
