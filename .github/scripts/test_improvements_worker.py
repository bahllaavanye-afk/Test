"""Improvements Worker guards: parsing, blocked detection, stable titles."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from improvements_worker import is_blocked, issue_title, parse_open_items

_MD = """# QuantEdge — Improvements & Task Tracker

- [ ] **[P1] Do a real thing** — implement the widget
  with a second line of detail.
- [x] **[P0] Already done** — shipped.
- [ ] **[P2] Blocked thing** — needs a data feed nobody wired.

Some prose in between.

- [ ] `#squad-qa` — test failures backlog.
"""


def test_parse_extracts_only_unchecked_and_folds_lines():
    items = parse_open_items(_MD)
    assert len(items) == 3
    assert "with a second line of detail." in items[0]
    assert all("Already done" not in i for i in items)


def test_blocked_detection():
    assert is_blocked("**[P2] X** — blocked on OA_SESSION_COOKIE")
    assert is_blocked("needs a data feed nobody wired")
    assert is_blocked("waiting on POLYMARKET_PRIVATE_KEY wiring")
    assert not is_blocked("**[P1] Do a real thing** — implement the widget")


def test_issue_title_is_stable_and_bounded():
    t1 = issue_title("**[P1] Do a real thing** — implement the widget")
    t2 = issue_title("**[P1] Do a real thing** — implement the widget")
    assert t1 == t2 == "improvement: [P1] Do a real thing"
    long = issue_title("no bold heading here " + "x" * 300)
    assert len(long) <= 120 and long.startswith("improvement: ")
