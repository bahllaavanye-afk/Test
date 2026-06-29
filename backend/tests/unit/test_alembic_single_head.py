"""Alembic migration graph must have exactly one head.

A migration once pointed at a non-existent ``down_revision`` ("h3c4d5e6f7a8"),
which split the graph into two heads and broke ``alembic upgrade head``. This
guards against that returning: every ``down_revision`` must resolve to a real
revision, and there must be a single head.
"""
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

_VERSIONS = Path(__file__).resolve().parents[2] / "alembic" / "versions"


def _graph() -> Tuple[Dict[str, str], Dict[str, Optional[str]]]:
    """Collect revision identifiers and their down revisions.

    Returns:
        A tuple containing:
        - revs: mapping from revision id to filename.
        - downs: mapping from revision id to its down_revision (or None).
    """
    revs: Dict[str, str] = {}
    downs: Dict[str, Optional[str]] = {}
    for f in _VERSIONS.glob("*.py"):
        text = f.read_text()
        # Tolerate optional type annotations, e.g. ``revision: str = '...'``.
        r = re.search(r"^revision\s*(?::[^=]+)?=\s*['\"]([^'\"]+)['\"]", text, re.M)
        d = re.search(r"^down_revision\s*(?::[^=]+)?=\s*['\"]([^'\"]+)['\"]", text, re.M)
        if r:
            rev = r.group(1)
            revs[rev] = f.name
            downs[rev] = d.group(1) if d else None
    return revs, downs


def test_every_down_revision_exists():
    revs, downs = _graph()
    dangling = {rev: dr for rev, dr in downs.items() if dr and dr not in revs}
    assert not dangling, f"Migrations point at non‑existent parents: {dangling}"


def test_exactly_one_head():
    revs, downs = _graph()
    referenced = {dr for dr in downs.values() if dr}
    heads = [rev for rev in revs if rev not in referenced]
    assert len(heads) == 1, f"Expected exactly one alembic head, found: {heads}"


def test_no_duplicate_revisions():
    """Ensure each revision identifier appears only once across migration files."""
    rev_counts: Dict[str, int] = {}
    for f in _VERSIONS.glob("*.py"):
        text = f.read_text()
        m = re.search(r"^revision\s*(?::[^=]+)?=\s*['\"]([^'\"]+)['\"]", text, re.M)
        if m:
            rev = m.group(1)
            rev_counts[rev] = rev_counts.get(rev, 0) + 1
    duplicates = [rev for rev, cnt in rev_counts.items() if cnt > 1]
    assert not duplicates, f"Duplicate revision IDs found: {duplicates}"