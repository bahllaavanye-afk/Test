"""Alembic migration graph must have exactly one head.

A migration once pointed at a non-existent ``down_revision`` ("h3c4d5e6f7a8"), which
split the graph into two heads and broke ``alembic upgrade head``. This guards against
that returning: every ``down_revision`` must resolve to a real revision, and there must
be a single head.
"""
import re
from pathlib import Path

_VERSIONS = Path(__file__).resolve().parents[2] / "alembic" / "versions"


def _graph():
    revs, downs = {}, {}
    for f in _VERSIONS.glob("*.py"):
        text = f.read_text()
        # Tolerate optional type annotations, e.g. `revision: str = '...'`.
        r = re.search(r"^revision\s*(?::[^=]+)?=\s*['\"]([^'\"]+)['\"]", text, re.M)
        d = re.search(r"^down_revision\s*(?::[^=]+)?=\s*['\"]([^'\"]+)['\"]", text, re.M)
        if r:
            revs[r.group(1)] = f.name
            downs[r.group(1)] = d.group(1) if d else None
    return revs, downs


def test_every_down_revision_exists():
    revs, downs = _graph()
    dangling = {rev: dr for rev, dr in downs.items() if dr and dr not in revs}
    assert not dangling, f"migrations point at non-existent parents: {dangling}"


def test_exactly_one_head():
    revs, downs = _graph()
    referenced = {dr for dr in downs.values() if dr}
    heads = [rev for rev in revs if rev not in referenced]
    assert len(heads) == 1, f"expected one alembic head, found {heads}"
