#!/usr/bin/env bash
# Render startup script: run Alembic migrations then start the API server.
set -e

echo "=== QuantEdge startup ==="
echo "DATABASE_URL prefix: ${DATABASE_URL:0:30}..."

python -m alembic upgrade head

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
