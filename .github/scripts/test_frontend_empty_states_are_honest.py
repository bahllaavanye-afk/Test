"""An empty panel must not blame the operator for a platform-level cause.

Audited 2026-08-06. The problem was not silence — most pages had an empty state
— it was MISDIRECTION. Two of them told the operator to do something that
cannot help:

    Experiments → "Click \"Train Model\" above to queue your first training run"
    Comparison  → "No comparison runs yet — run a strategy comparison in Backtest Lab"

The backend runs on an ephemeral SQLite fallback, so rows are wiped on every
redeploy; and ML training needs PyTorch, deliberately excluded from the
deployment image. Following either instruction produces the same empty table.

A panel that attributes a platform cause to user inaction is the UI version of
the defect this codebase keeps finding: output that does not depend on what is
actually happening.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAGES = REPO / "frontend" / "src" / "pages"
COMPONENT = REPO / "frontend" / "src" / "components" / "ui" / "EmptyState.tsx"


def _page_sources() -> list[Path]:
    return sorted(PAGES.glob("*.tsx"))


def test_the_scan_actually_finds_pages():
    """Guard on the guard. A broken glob would make every check below vacuous —
    it would find no offending page and pass having verified nothing."""
    pages = _page_sources()
    assert len(pages) >= 10, (
        f"only {len(pages)} page(s) found under {PAGES}; the scan is supposed to "
        f"cover the whole pages directory. An empty scan makes this file useless.")


def test_the_component_exists_and_covers_the_known_causes():
    assert COMPONENT.is_file(), "EmptyState.tsx is gone; the honest copy lived there"
    src = COMPONENT.read_text()
    for reason in ("ephemeral-db", "no-rows-yet", "subsystem-unreachable", "ml-runtime"):
        assert f"'{reason}'" in src, f"EmptyState no longer defines the {reason!r} reason"


def test_no_page_tells_the_operator_to_retry_something_that_cannot_persist():
    """The two exact regressions, plus the shape of them.

    Both removed messages instructed an action whose result is wiped by the
    ephemeral DB. Re-adding either is the regression this pins."""
    banned = [
        'Click "Train Model" above to queue your first training run',
        "run a strategy comparison in Backtest Lab",
    ]
    offenders = []
    for page in _page_sources():
        text = page.read_text()
        for phrase in banned:
            if phrase in text:
                offenders.append(f"{page.name}: {phrase!r}")
    assert not offenders, (
        "empty-state copy tells the operator to take an action that cannot "
        f"currently succeed: {offenders}. Name the platform cause instead — see "
        f"EmptyState's reasons.")


def test_the_two_audited_pages_use_the_component():
    """Named because these are the two that were actively misleading. A page
    that regresses to hand-rolled copy would slip past the phrase check above
    by rewording it."""
    for name, reason in (("Experiments.tsx", "ml-runtime"),
                         ("Comparison.tsx", "ephemeral-db")):
        src = (PAGES / name).read_text()
        assert "EmptyState" in src, f"{name} no longer uses EmptyState"
        assert reason in src, f"{name} no longer states the {reason!r} cause"


def test_every_reason_passed_by_a_page_is_one_the_component_defines():
    """A typo'd reason would render `undefined` copy — an empty panel again,
    which is precisely what this work removed."""
    defined = set(re.findall(r"'([a-z-]+)':\s*\{", COMPONENT.read_text()))
    assert defined, "could not parse any reasons out of EmptyState"
    used = set()
    for page in _page_sources():
        used |= set(re.findall(r'reason="([a-z-]+)"', page.read_text()))
    unknown = used - defined
    assert not unknown, f"pages pass reason(s) EmptyState does not define: {unknown}"
