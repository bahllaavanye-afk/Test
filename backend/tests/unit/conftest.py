"""Unit-test fixtures — does NOT load the FastAPI app (no DB/network required)."""
import sys
from pathlib import Path
import pytest
import tempfile

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

BACKEND_ROOT = Path(__file__).parents[2]
_validate_backend_root(BACKEND_ROOT)

_validate_sys_path(sys.path)
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# -------------------------------------------------------------------------
# New unit test cases for edge conditions
# -------------------------------------------------------------------------

def test_validate_backend_root_non_path():
    """Ensure a non‑Path argument raises ValueError."""
    with pytest.raises(ValueError, match="Backend root must be a pathlib.Path"):
        _validate_backend_root("not-a-path")

def test_validate_backend_root_nonexistent_path():
    """Ensure a Path that does not exist raises ValueError."""
    non_existent = Path(tempfile.gettempdir()) / "definitely_not_a_dir_12345"
    # Guarantee it does not exist
    if non_existent.exists():
        non_existent.unlink()
    with pytest.raises(ValueError, match="Backend root path does not exist"):
        _validate_backend_root(non_existent)

def test_validate_backend_root_file_instead_of_dir():
    """Ensure a Path that points to a file (not a directory) raises ValueError."""
    with tempfile.NamedTemporaryFile() as tmp_file:
        file_path = Path(tmp_file.name)
        with pytest.raises(ValueError, match="Backend root path is not a directory"):
            _validate_backend_root(file_path)

def test_validate_sys_path_non_iterable():
    """Ensure passing a non‑iterable to _validate_sys_path raises ValueError."""
    with pytest.raises(ValueError, match="sys.path must be an iterable"):
        _validate_sys_path(12345)

def test_validate_sys_path_contains_non_string():
    """Ensure sys.path entries that are not strings raise ValueError."""
    bad_path = ["valid/string", 42, "another/valid"]
    with pytest.raises(ValueError, match="sys.path entry at index 1 is not a string"):
        _validate_sys_path(bad_path)