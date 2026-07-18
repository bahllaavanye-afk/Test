"""Polymarket real-data feed guards: pure conversion + fail-soft fetching."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
sys.path.insert(0, str(Path(__file__).parent))

import polymarket_data as pm


def _points(n=48, start=0.40, step=0.005):
    return [{"t": 1_784_000_000 + i * 3600, "p": start + i * step} for i in range(n)]


def test_history_to_bars_scales_probability_to_percent():
    df = pm.history_to_bars(_points())
    assert df is not None and len(df) == 48
    assert df["close"].iloc[0] == pytest.approx(40.0)      # 0.40 → 40.0
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
    assert (df["high"] >= df["low"]).all()
    assert df.index.tz is not None                          # tz-aware timestamps


def test_history_to_bars_rejects_short_or_malformed():
    assert pm.history_to_bars(_points(10)) is None          # <30 points
    assert pm.history_to_bars([{"x": 1}] * 40) is None      # wrong keys


def test_fetch_top_markets_parses_and_fails_soft(monkeypatch):
    def fake_get(url, timeout=20):
        assert "gamma-api" in url
        return [{"question": "Will X happen?", "volume24hr": 123.0,
                 "clobTokenIds": json.dumps(["tok1", "tok2"])},
                {"question": "broken", "clobTokenIds": "not-json"}]
    monkeypatch.setattr(pm, "_get", fake_get)
    out = pm.fetch_top_markets(2)
    assert len(out) == 1 and out[0]["token_id"] == "tok1"   # broken row skipped

    monkeypatch.setattr(pm, "_get",
                        lambda url, timeout=20: (_ for _ in ()).throw(OSError("down")))
    assert pm.fetch_top_markets(2) == []                    # fail-soft, never raise


def test_desk_feed_builds_pm_symbols(monkeypatch):
    monkeypatch.setattr(pm, "fetch_top_markets",
                        lambda limit=6: [{"question": "Will rates fall in 2026?",
                                          "token_id": "tokA", "volume24h": 9.9}])
    monkeypatch.setattr(pm, "fetch_price_bars",
                        lambda tid: pm.history_to_bars(_points()))
    feed = pm.desk_feed(1)
    assert list(feed) == ["PM:Will rates fall in 2026?"]
    assert len(feed["PM:Will rates fall in 2026?"]) == 48
