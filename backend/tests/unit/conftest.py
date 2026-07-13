"""Unit-test fixtures — does NOT load the FastAPI app (no DB/network required)."""
import sys
from pathlib import Path

# Constants
PARENT_LEVEL = 2
SYS_PATH_INSERT_INDEX = 0

BACKEND_ROOT = Path(__file__).parents[PARENT_LEVEL]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(SYS_PATH_INSERT_INDEX, str(BACKEND_ROOT))