"""Desk posts → shared brain (the missing CONSUME side).

Desks post funnel/orders/P&L to Discord every run, but nothing fed those
outcomes back into the company brain, so employees discussed status theater
instead of what the desks actually did. company_brain.fetch_desk_knowledge()
reads the durable Discord history and feeds it into the brain; get_company_context()
then surfaces it into every employee prompt. These tests mock the Discord read
and the LLM; no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import notify  # noqa: E402
import company_brain  # noqa: E402
import llm_common  # noqa: E402


# ── fetch_desk_knowledge: reads newest substantive post per channel ──────────

def test_fetch_desk_knowledge_one_post_per_channel(monkeypatch):
    def fake_read(channel, limit=8):
        return {
            "pnl-daily": [{"content": "funnel: 49 → 5 placed, +$12.30"}],
            "desk-fx-rates": [{"content": "FX — 2 orders EUR_USD BUY@1.08"}],
        }.get(str(channel).lstrip("#"), [])
    monkeypatch.setattr(notify, "read_channel_recent", fake_read)

    items = company_brain.fetch_desk_knowledge()
    chans = {i["channel"] for i in items}
    assert "pnl-daily" in chans and "desk-fx-rates" in chans
    assert all(i["source"] == "desk" for i in items)
    # one representative post per channel
    assert len([i for i in items if i["channel"] == "pnl-daily"]) == 1


def test_fetch_desk_knowledge_skips_trivial_posts(monkeypatch):
    monkeypatch.setattr(notify, "read_channel_recent",
                        lambda ch, limit=8: [{"content": "ok"}, {"content": "real desk summary here"}]
                        if str(ch).lstrip("#") == "pnl-daily" else [])
    items = company_brain.fetch_desk_knowledge()
    pnl = [i for i in items if i["channel"] == "pnl-daily"]
    assert pnl and pnl[0]["text"] == "real desk summary here"  # trivial "ok" skipped


def test_fetch_desk_knowledge_fails_soft(monkeypatch):
    def boom(ch, limit=8):
        raise RuntimeError("api down")
    monkeypatch.setattr(notify, "read_channel_recent", boom)
    assert company_brain.fetch_desk_knowledge() == []


# ── synthesize_insights: desk items reach the CIO synthesis prompt ───────────

def test_synthesize_includes_desk_results(monkeypatch):
    captured = {}
    def fake_llm(prompt, **kw):
        captured["prompt"] = prompt
        return '[{"insight": "FX desk carrying EUR longs", "category": "strategy", "priority": 2}]'
    monkeypatch.setattr(company_brain, "llm", fake_llm)

    out = company_brain.synthesize_insights(
        slack_msgs=[], github_items=[], code_findings=[],
        desk_items=[{"source": "desk", "channel": "pnl-daily", "text": "funnel 49→5, +$12"}],
    )
    assert "DESK RESULTS" in captured["prompt"]
    assert "funnel 49→5" in captured["prompt"]
    assert out["insights"][0]["category"] == "strategy"


def test_synthesize_empty_when_all_sources_empty(monkeypatch):
    # desk_items defaulting to None must not crash and must short-circuit.
    assert company_brain.synthesize_insights([], [], []) == {}


# ── the loop closes: desk_outcomes surface in the injected company context ───

def test_desk_outcomes_surface_in_company_context(monkeypatch, tmp_path):
    brain_file = tmp_path / "company_brain.json"
    monkeypatch.setattr(llm_common, "_BRAIN_FILE", brain_file)
    # bust the module cache so our fresh file is read
    monkeypatch.setattr(llm_common, "_brain_cache", {}, raising=False)
    monkeypatch.setattr(llm_common, "_brain_cache_ts", 0.0, raising=False)

    llm_common.memory_write("desk_outcomes", {
        "channel": "pnl-daily", "summary": "crypto desk placed 3 orders, +$8.40",
        "source": "desk_run_summary",
    })
    ctx = llm_common.get_company_context()
    assert "Desk results:" in ctx
    assert "crypto desk placed 3 orders" in ctx
