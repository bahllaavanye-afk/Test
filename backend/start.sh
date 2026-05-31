#!/usr/bin/env bash
# QuantEdge startup — Render free tier
# Does NOT use set -e so a failed migration doesn't kill the server.

set -o pipefail

echo "=== QuantEdge startup @ $(date -u '+%Y-%m-%d %H:%M UTC') ==="
echo "DATABASE_URL prefix : ${DATABASE_URL:0:60}..."
echo "PORT                : ${PORT:-8000}"
echo "TRADING_MODE        : ${TRADING_MODE:-paper}"

# ── Detect Supabase IPv6 direct-connection issue ─────────────────────────────
# Supabase direct URLs (db.PROJECT.supabase.co:5432) resolve to IPv6.
# Render does not support outbound IPv6.  Use the Session Pooler URL instead.
if echo "${DATABASE_URL}" | grep -qE 'db\.[a-z0-9]+\.supabase\.co'; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  ⚠  SUPABASE IPv6 WARNING — MIGRATIONS SKIPPED                  ║"
  echo "║                                                                  ║"
  echo "║  Your DATABASE_URL is the Supabase DIRECT connection.           ║"
  echo "║  It resolves to IPv6 which Render cannot reach.                 ║"
  echo "║                                                                  ║"
  echo "║  FIX (takes 2 minutes):                                          ║"
  echo "║  1. Supabase Dashboard → Settings → Database → Connection pooling║"
  echo "║  2. Copy the Transaction mode URL (port 6543)                   ║"
  echo "║     postgres://postgres.PROJECT:PASS@aws-0-REGION.pooler.       ║"
  echo "║     supabase.com:6543/postgres                                  ║"
  echo "║  3. Render Dashboard → quantedge-api → Environment              ║"
  echo "║  4. Set DATABASE_URL = <pooler URL above>                       ║"
  echo "╚══════════════════════════════════════════════════════════════════╝"
  echo ""
  echo "Starting server anyway (DB operations will fail until URL is fixed)"
  exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
fi

# ── Ensure ALEMBIC_DATABASE_URL uses psycopg2 sync driver ────────────────────
# asyncpg (async) is used by the app; psycopg2 (sync) is needed for alembic.
if [ -z "${ALEMBIC_DATABASE_URL}" ]; then
  ALEMBIC_DATABASE_URL="${DATABASE_URL}"
fi
# Convert async URL formats to psycopg2
ALEMBIC_DATABASE_URL="${ALEMBIC_DATABASE_URL/+asyncpg/+psycopg2}"
ALEMBIC_DATABASE_URL="${ALEMBIC_DATABASE_URL/postgresql+psycopg2:\/\//postgresql:\/\/}"
# postgres:// → postgresql://
ALEMBIC_DATABASE_URL="${ALEMBIC_DATABASE_URL/postgres:\/\//postgresql:\/\/}"
export ALEMBIC_DATABASE_URL
echo "ALEMBIC_DATABASE_URL prefix : ${ALEMBIC_DATABASE_URL:0:60}..."

# ── Run Alembic migrations ────────────────────────────────────────────────────
echo "Running database migrations..."
MIGRATION_OK=0
MAX_TRIES=5
for i in $(seq 1 $MAX_TRIES); do
  if python -m alembic upgrade head 2>&1; then
    MIGRATION_OK=1
    echo "✓ Migrations complete"
    break
  fi
  EXIT_CODE=$?
  if [ "$i" -eq "$MAX_TRIES" ]; then
    echo "✗ Migrations failed after $MAX_TRIES attempts (exit $EXIT_CODE) — starting server anyway"
  else
    WAIT=$((i * 5))
    echo "Migration attempt $i/$MAX_TRIES failed — retrying in ${WAIT}s..."
    sleep $WAIT
  fi
done

# ── Start API server ──────────────────────────────────────────────────────────
echo "Starting uvicorn on port ${PORT:-8000}..."
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers 1 \
  --timeout-keep-alive 30
