#!/usr/bin/env bash
# Usage: ./scripts/monitor.sh [--interval 300] [--once]
# Runs QA monitoring loop. --once for single run (useful for CI).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_ROOT/backend"
INTERVAL=300
ONCE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) INTERVAL="$2"; shift 2 ;;
    --once) ONCE=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "QuantEdge QA Monitor starting (interval: ${INTERVAL}s)"

run_cycle() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "$(date '+%Y-%m-%d %H:%M:%S UTC') — QA Cycle"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  cd "$BACKEND_DIR"

  # Run tests
  echo "Running test suite..."
  if python -m pytest tests/ -q --tb=short --no-header --timeout=60 2>&1; then
    echo "All tests passed"
    TEST_STATUS="passed"
    FAILED_TESTS=0
  else
    echo "Test failures detected"
    TEST_STATUS="failed"
    # Count failures from pytest output
    FAILED_TESTS=$(python -m pytest tests/ -q --tb=no --no-header 2>&1 | grep -oP '^\d+ failed' | grep -oP '\d+' || echo "0")
  fi

  # Check imports
  echo ""
  echo "Checking module imports..."
  FAILED_IMPORTS=0
  for module in app.main app.config app.risk.manager app.strategies app.ml.features.engineer; do
    if python -c "import $module" 2>/dev/null; then
      echo "  OK: $module"
    else
      echo "  FAIL: $module — IMPORT ERROR"
      FAILED_IMPORTS=$((FAILED_IMPORTS + 1))
    fi
  done

  # Security scan (simple grep-based)
  echo ""
  echo "Security scan..."
  ISSUES=0
  if grep -rn "get_event_loop()" "$BACKEND_DIR/app/" --include="*.py" -q 2>/dev/null; then
    echo "  WARN: deprecated asyncio.get_event_loop() found"
    ISSUES=$((ISSUES + 1))
  fi
  if grep -rn "datetime.utcnow()" "$BACKEND_DIR/app/" --include="*.py" -q 2>/dev/null; then
    echo "  WARN: deprecated datetime.utcnow() found"
    ISSUES=$((ISSUES + 1))
  fi
  if grep -rn 'execute(f"' "$BACKEND_DIR/app/" --include="*.py" -q 2>/dev/null || \
     grep -rn "execute(f'" "$BACKEND_DIR/app/" --include="*.py" -q 2>/dev/null; then
    echo "  WARN: f-string in SQL execute() — potential injection risk"
    ISSUES=$((ISSUES + 1))
  fi
  if [ "$ISSUES" -eq 0 ]; then
    echo "  OK: no known security patterns found"
  fi

  # Determine overall status
  STATUS="healthy"
  if [ "$FAILED_IMPORTS" -gt 2 ] || [ "${FAILED_TESTS:-0}" -gt 10 ]; then
    STATUS="critical"
  elif [ "$FAILED_IMPORTS" -gt 0 ] || [ "${FAILED_TESTS:-0}" -gt 0 ] || [ "$ISSUES" -gt 3 ]; then
    STATUS="degraded"
  fi

  echo ""
  echo "Overall status: $STATUS"

  # Write health JSON
  cat > "$PROJECT_ROOT/qa_health_report.json" <<JSON
{
  "status": "$STATUS",
  "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "test_status": "$TEST_STATUS",
  "failed_tests": ${FAILED_TESTS:-0},
  "failed_imports": $FAILED_IMPORTS,
  "security_issues": $ISSUES
}
JSON
  echo "Health report written to $PROJECT_ROOT/qa_health_report.json"
}

if $ONCE; then
  run_cycle
  exit 0
fi

while true; do
  run_cycle
  echo ""
  echo "Next check in ${INTERVAL}s. Press Ctrl+C to stop."
  sleep "$INTERVAL"
done
