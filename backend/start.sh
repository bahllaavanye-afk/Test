#!/usr/bin/env bash
# QuantEdge startup — Render free tier
# Does NOT use set -e so a failed migration doesn't kill the server.

echo "=== QuantEdge startup @ $(date -u '+%Y-%m-%d %H:%M UTC') ==="
echo "DATABASE_URL prefix : ${DATABASE_URL:0:50}..."
echo "PORT                : ${PORT:-8000}"
echo "TRADING_MODE        : ${TRADING_MODE:-paper}"

# ── Detect Supabase IPv6 direct-connection issue ─────────────────────────────
# Supabase direct URLs (db.PROJECT.supabase.co:5432) resolve to IPv6.
# Render does not support outbound IPv6. Use the Session Pooler URL instead.
if echo "${DATABASE_URL}" | grep -qE 'db\.[a-z]+\.supabase\.co'; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  SUPABASE IPv6 WARNING                                           ║"
  echo "║                                                                  ║"
  echo "║  Your DATABASE_URL points to the Supabase DIRECT connection.     ║"
  echo "║  This resolves to IPv6 which Render cannot reach.                ║"
  echo "║                                                                  ║"
  echo "║  FIX (takes 2 minutes):                                          ║"
  echo "║  1. Supabase Dashboard → Settings → Database → Connection pooling║"
  echo "║  2. Copy the 'Session mode' URL  (port 5432 via pooler)          ║"
  echo "║     OR  'Transaction mode' URL   (port 6543)                     ║"
  echo "║  3. Render Dashboard → quantedge-api → Environment               ║"
  echo "║  4. Set DATABASE_URL = <pooler URL>                              ║"
  echo "║                                                                  ║"
  echo "║  Pooler URL format:                                              ║"
  echo "║  postgres://postgres.PROJECT:PASS@aws-0-REGION.pooler.supabase.com:6543/postgres║"
  echo "╚══════════════════════════════════════════════════════════════════╝"
  echo ""
  echo "Skipping migrations (IPv6 unreachable). Starting server anyway..."
  exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
fi

# ── Ensure Alembic uses the correct (sync) database URL ──────────────────────
# ALEMBIC_DATABASE_URL takes priority; env.py falls back to DATABASE_URL and
# converts async prefixes to psycopg2 automatically.
export ALEMBIC_DATABASE_URL="${ALEMBIC_DATABASE_URL:-${DATABASE_URL}}"

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
