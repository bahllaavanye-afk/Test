"""Unit-test fixtures — does NOT load the FastAPI app (no DB/network required)."""
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
