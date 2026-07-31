"""Unit-test fixtures — does NOT load the FastAPI app (no DB/network required)."""
import sys
from pathlib import Path

# Constants
BACKEND_ROOT_PARENT_LEVEL = 2

def _validate_backend_root(root: Path) -> None:
    """Validate that the provided backend root path exists and is a directory.

    Args:
        root: Path object pointing to the backend root.

    Raises:
        ValueError: If ``root`` does not exist or is not a directory.
    """
    if not isinstance(root, Path):
        raise ValueError(f"Backend root must be a pathlib.Path instance, got {type(root).__name__}")
    if not root.exists():
        raise ValueError(f"Backend root path does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Backend root path is not a directory: {root}")

def _validate_sys_path(path_list) -> None:
    """Validate that ``sys.path`` is a mutable sequence of strings.

    Args:
        path_list: The object to validate (expected to be ``sys.path``).

    Raises:
        ValueError: If ``path_list`` is not a list-like container of strings.
    """
    if not hasattr(path_list, "__iter__"):
        raise ValueError(f"sys.path must be an iterable, got {type(path_list).__name__}")
    for idx, entry in enumerate(path_list):
        if not isinstance(entry, str):
            raise ValueError(f"sys.path entry at index {idx} is not a string: {type(entry).__name__}")

BACKEND_ROOT = Path(__file__).parents[BACKEND_ROOT_PARENT_LEVEL]
_validate_backend_root(BACKEND_ROOT)

_validate_sys_path(sys.path)
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))