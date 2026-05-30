#!/usr/bin/env bash
set -e

echo "=== QuantEdge startup ==="
echo "DATABASE_URL prefix: ${DATABASE_URL:0:30}..."

# Run Alembic migrations with retry (DB may not be ready immediately)
MAX_TRIES=5
for i in $(seq 1 $MAX_TRIES); do
    if python -m alembic upgrade head; then
        break
    fi
    if [ "$i" -eq "$MAX_TRIES" ]; then
        echo "ERROR: Alembic migrations failed after $MAX_TRIES attempts, starting anyway..."
    else
        echo "Migration attempt $i failed, retrying in $((i * 3))s..."
        sleep $((i * 3))
    fi
done

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
