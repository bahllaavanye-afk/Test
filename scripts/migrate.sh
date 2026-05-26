#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../backend"
alembic upgrade head
