"""Unit-test fixtures — does NOT load the FastAPI app (no DB/network required)."""

import sys
from pathlib import Path

# Resolve the backend root directory (two levels up from this file)
BACKEND_ROOT: Path = Path(__file__).parents[2].resolve()

# Ensure the backend root is on the import path for test modules
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

__all__ = ["BACKEND_ROOT"]