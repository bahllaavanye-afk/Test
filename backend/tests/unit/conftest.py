"""Unit-test fixtures — does NOT load the FastAPI app (no DB/network required)."""
import sys
from pathlib import Path

# Constants
BACKEND_ROOT_LEVEL = 2
SYS_PATH_INSERT_INDEX = 0

BACKEND_ROOT = Path(__file__).parents[BACKEND_ROOT_LEVEL]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(SYS_PATH_INSERT_INDEX, str(BACKEND_ROOT))