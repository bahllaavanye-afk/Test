#!/bin/bash
# QuantEdge deployment pre-flight check
# Run before deploying to Render/Vercel to catch missing config early.
# Usage: ./scripts/validate_deploy.sh [--strict]
set -e

STRICT="${1:-}"
PASS=0
FAIL=0
WARN=0

green() { printf "\033[32m✓\033[0m  %s\n" "$1"; ((PASS++)); }
red()   { printf "\033[31m✗\033[0m  %s\n" "$1"; ((FAIL++)); }
yellow(){ printf "\033[33m⚠\033[0m  %s\n" "$1"; ((WARN++)); }
section(){ echo ""; printf "\033[1m%s\033[0m\n" "── $1"; }

# ── Required environment variables ────────────────────────────────────────────
section "Required env vars"

for var in SECRET_KEY DATABASE_URL ALPACA_API_KEY ALPACA_SECRET_KEY; do
  if [[ -n "${!var}" ]]; then
    green "$var is set"
  else
    red "$var is NOT SET"
  fi
done

# Redis is optional (degrades gracefully)
if [[ -n "$REDIS_URL" ]]; then
  green "REDIS_URL is set"
else
  yellow "REDIS_URL not set — caching disabled (ok for dev)"
fi

# ── Secret quality checks ─────────────────────────────────────────────────────
section "Secret quality"

if [[ -n "$SECRET_KEY" ]]; then
  LEN=${#SECRET_KEY}
  if [[ $LEN -ge 32 ]]; then
    green "SECRET_KEY length OK ($LEN chars)"
  else
    red "SECRET_KEY too short ($LEN chars — need ≥ 32)"
  fi

  if [[ "$SECRET_KEY" == "dev-only"* ]]; then
    red "SECRET_KEY looks like a dev placeholder — rotate before deploy"
  fi
fi

# ── Database URL format ────────────────────────────────────────────────────────
section "Database URL"

if [[ -n "$DATABASE_URL" ]]; then
  if [[ "$DATABASE_URL" == postgresql+asyncpg://* ]] || [[ "$DATABASE_URL" == postgresql://* ]]; then
    green "DATABASE_URL uses PostgreSQL (production driver)"
  elif [[ "$DATABASE_URL" == sqlite* ]]; then
    yellow "DATABASE_URL is SQLite — OK for dev, use Supabase in production"
  else
    yellow "DATABASE_URL format unrecognised: ${DATABASE_URL:0:30}..."
  fi
fi

# ── .env not committed ────────────────────────────────────────────────────────
section ".env file safety"

if [[ -f ".env" ]]; then
  if git ls-files --error-unmatch .env &>/dev/null 2>&1; then
    red ".env is tracked by git — remove with: git rm --cached .env"
  else
    green ".env exists but is NOT tracked by git"
  fi
else
  green "No .env file (using runtime environment — correct for production)"
fi

if grep -q ".env" .gitignore 2>/dev/null; then
  green ".env is in .gitignore"
else
  red ".env is missing from .gitignore"
fi

# ── Alembic migrations ────────────────────────────────────────────────────────
section "Alembic migrations"

MIGRATION_COUNT=$(ls backend/alembic/versions/*.py 2>/dev/null | wc -l | tr -d ' ')
if [[ "$MIGRATION_COUNT" -gt 0 ]]; then
  green "$MIGRATION_COUNT migration files found"
else
  red "No migration files found in backend/alembic/versions/"
fi

# ── Frontend build ────────────────────────────────────────────────────────────
section "Frontend"

if command -v node &>/dev/null; then
  NODE_VER=$(node --version)
  green "Node.js $NODE_VER"

  if [[ -f "frontend/package.json" ]]; then
    if [[ -n "$VITE_API_URL" ]]; then
      green "VITE_API_URL is set: $VITE_API_URL"
    else
      yellow "VITE_API_URL not set — frontend will use relative /api paths"
    fi
    if [[ -n "$VITE_WS_URL" ]]; then
      green "VITE_WS_URL is set"
    else
      yellow "VITE_WS_URL not set — WebSocket will use relative path"
    fi
  fi
else
  yellow "Node.js not found — skipping frontend checks"
fi

# ── render.yaml ───────────────────────────────────────────────────────────────
section "Deployment config"

if [[ -f "backend/render.yaml" ]]; then
  green "render.yaml found"
else
  red "backend/render.yaml not found"
fi

if [[ -f "frontend/vercel.json" ]]; then
  green "vercel.json found"
else
  red "frontend/vercel.json not found"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
printf "Results: \033[32m%d passed\033[0m, \033[33m%d warnings\033[0m, \033[31m%d failed\033[0m\n" $PASS $WARN $FAIL
echo ""

if [[ $FAIL -gt 0 ]]; then
  echo "Fix the errors above before deploying."
  if [[ "$STRICT" == "--strict" ]]; then
    exit 1
  fi
elif [[ $WARN -gt 0 ]]; then
  echo "Warnings above are non-blocking. Review before production deploy."
else
  echo "All checks passed. Ready to deploy."
fi
