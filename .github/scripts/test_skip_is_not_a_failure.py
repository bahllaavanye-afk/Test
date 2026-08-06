"""A deliberate skip was being logged as a broker failure.

Measured live 2026-08-06 13:35 — the first desk run after the account recovered
and the market opened, i.e. the first time this path had run with real orders
in ~9 hours:

    · META sell 0.56 would be a fractional SHORT (held 0.0) — Alpaca rejects
      those, and flooring gives 0. Skipping instead of failing at the broker.
    ✗ order placement returned no ID

    · TSLA sell 0.62 would be a fractional SHORT (held 0.0) — ... Skipping ...
    ✗ order placement returned no ID

The skip line explains a correct decision. The ✗ on the next line contradicts
it. `_place_order` returned None both when it CHOSE not to place and when a
placement genuinely failed, and the caller could not tell the two apart.

Same defect class as the order-origin audit and the brain-health canary: the
message did not depend on what actually happened.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("SECRET_KEY", "x" * 64)
os.environ.setdefault("DATABASE_URL", "")
import desk_order_placer as dop  # noqa: E402


def _coro(value):
    async def _c():
        return value
    return _c()

SCRIPTS = Path(__file__).resolve().parent


def test_the_sentinel_is_distinguishable_from_a_failure():
    assert dop._was_skipped(dop.SKIPPED_BY_DESIGN) is True
    assert dop._was_skipped(None) is False                    # genuine failure
    assert dop._was_skipped({"id": "abc"}) is False           # real order
    assert dop._was_skipped({}) is False


def test_a_fractional_short_is_a_skip_not_a_failure(monkeypatch, capsys):
    """The exact META/TSLA case: sell 0.56 of something held 0.0."""
    monkeypatch.setattr(dop, "_cached_position_map", lambda: _coro({}))
    monkeypatch.setattr(dop, "_is_shortable", lambda sym: _coro(True))
    qty = asyncio.run(dop._equity_short_safe_qty("META", "sell", 0.56, False))
    assert qty is None
    out = capsys.readouterr().out
    assert "fractional SHORT" in out and "Skipping" in out


def test_an_unshortable_asset_is_a_skip_not_a_failure(monkeypatch, capsys):
    monkeypatch.setattr(dop, "_cached_position_map", lambda: _coro({}))
    monkeypatch.setattr(dop, "_is_shortable", lambda sym: _coro(False))
    qty = asyncio.run(dop._equity_short_safe_qty("EIDO", "sell", 44.0, False))
    assert qty is None
    assert "not shortable" in capsys.readouterr().out


def test_ensure_filled_passes_a_skip_straight_through():
    """Nothing was placed, so there is no order to cancel-replace."""
    out = asyncio.run(dop._ensure_filled(dop.SKIPPED_BY_DESIGN, "META", "sell", 330.0))
    assert dop._was_skipped(out)


def test_a_real_failure_still_reports_no_ID():
    """The guard must not swallow the case it was written for."""
    src = (SCRIPTS / "desk_order_placer.py").read_text()
    assert 'print(f"    ✗ order placement returned no ID"' in src, (
        "the genuine-failure branch is gone — a placement that actually fails "
        "would now be silent, which is worse than the bug being fixed")


def test_the_caller_checks_for_a_skip_before_calling_it_a_failure():
    src = (SCRIPTS / "desk_order_placer.py").read_text()
    tail = src[src.index("submitted ("):][:4000]
    assert tail.index("_was_skipped(order)") < tail.index("returned no ID"), (
        "the skip check must precede the failure message, or deliberate skips "
        "keep printing ✗")


def test_every_deliberate_skip_inside_place_order_returns_the_sentinel():
    """Scans the function rather than naming the three known sites, so a fourth
    skip added later cannot quietly go back to reporting a failure."""
    import ast
    src = (SCRIPTS / "desk_order_placer.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "_place_order")
    # Every `return None` in the body of the try block is suspect: the function
    # returns either an order dict, the sentinel, or None-on-exception.
    bare_none = [n for n in ast.walk(fn)
                 if isinstance(n, ast.Return)
                 and isinstance(n.value, ast.Constant) and n.value.value is None]
    assert len(bare_none) <= 1, (
        f"{len(bare_none)} bare `return None` paths in _place_order; a deliberate "
        f"skip must return SKIPPED_BY_DESIGN so the caller does not log it as a "
        f"broker failure. Only the exception handler may return None.")
