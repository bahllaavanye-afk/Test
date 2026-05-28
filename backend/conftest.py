"""Root conftest — ensures backend/ is on sys.path so `import app.*` works."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Override the integration-test conftest path so unit tests don't trigger app import
collect_ignore_glob = []
