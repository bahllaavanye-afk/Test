"""Broker selection must prefer the one that can run without a human.

QuantEdge runs on GitHub Actions overnight. Zerodha Kite — the broker asked for
by name — issues its `request_token` only through a browser redirect a human
completes, and the `access_token` expires daily around 06:00 IST. A Kite-backed
desk places orders only on days somebody logged in by hand, which is the exact
"green-looking absence" this codebase has spent the week paying down.

So selection is by SUITABILITY, not by first-found: if both Kite and AngelOne
were configured, the one that survives the night has to win.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import india_broker as B  # noqa: E402

ALL_VARS = [v for s in B.BROKERS for v in s.env_vars]


@pytest.fixture
def clean(monkeypatch):
    for v in ALL_VARS:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def _configure(mp, spec):
    for v in spec.env_vars:
        mp.setenv(v, "x")


def test_no_credentials_means_no_broker(clean):
    assert B.configured_broker() is None


def test_it_reports_what_is_missing(clean):
    assert set(B.missing_for("angelone")) == set(
        next(s for s in B.BROKERS if s.name == "angelone").env_vars)


def test_partial_credentials_do_not_count(clean):
    clean.setenv("ANGELONE_API_KEY", "x")
    assert B.configured_broker() is None, (
        "a half-configured broker was selected; the first order would fail auth"
    )


@pytest.mark.parametrize("name", [s.name for s in B.BROKERS])
def test_each_broker_is_selectable_on_its_own(clean, name):
    spec = next(s for s in B.BROKERS if s.name == name)
    _configure(clean, spec)
    got = B.configured_broker()
    assert got is not None and got.name == name


def test_an_unattended_broker_beats_a_daily_login_one(clean):
    """THE ordering test. Both configured — the overnight-capable one wins."""
    _configure(clean, next(s for s in B.BROKERS if s.name == "zerodha"))
    _configure(clean, next(s for s in B.BROKERS if s.name == "angelone"))
    got = B.configured_broker()
    assert got.name == "angelone", (
        f"selected {got.name}, which needs a human every morning, over an "
        "equally-configured broker that does not. Selection is by suitability, "
        "not by list position or first-found."
    )


def test_the_unattended_brokers_come_first_in_the_table():
    """Ordering is the mechanism — a reshuffle silently breaks selection."""
    flags = [s.unattended for s in B.BROKERS]
    assert flags == sorted(flags, reverse=True), (
        f"BROKERS is no longer ordered unattended-first: {flags}"
    )


def test_zerodha_is_recorded_as_needing_a_daily_login():
    z = next(s for s in B.BROKERS if s.name == "zerodha")
    assert z.unattended is False, (
        "Kite Connect is marked unattended. Its access_token expires daily and "
        "the request_token comes from a browser redirect — marking it "
        "automatable would put the fleet back on a human's morning routine."
    )


def test_the_status_line_warns_when_a_manual_broker_is_selected(clean):
    _configure(clean, next(s for s in B.BROKERS if s.name == "zerodha"))
    line = B.status_line()
    assert "zerodha" in line and "DAILY manual login" in line


def test_the_status_line_names_the_cheapest_path_when_unconfigured(clean):
    line = B.status_line()
    assert "NO BROKER CONFIGURED" in line and "angelone" in line
    assert "research-only" in line


def test_nothing_here_places_an_order():
    """Live order routing to a real Indian account is not an AFK change."""
    src = (Path(__file__).resolve().parent / "india_broker.py").read_text()
    for bad in ("urlopen", "requests.post", "http.client", "place_order"):
        assert bad not in src, (
            f"india_broker.py now contains {bad!r} — it is a capability/spec "
            "module. Routing real orders to a real Indian brokerage account is "
            "a deliberate, supervised change."
        )
