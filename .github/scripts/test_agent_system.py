"""
Integration tests for the multi-agent LLM system.

Covers the components added for: task→model routing, Claude-for-seniors gating,
the Render secret sync, and the 3-hourly member standup. All tests are pure
(no network, no Slack, no LLM calls) so CI catches regressions fast.

Run: cd .github/scripts && python -m pytest test_agent_system.py -q
"""
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))


# ── model_router ──────────────────────────────────────────────────────────────

def test_task_tiers_map_to_known_provider_chains():
    import model_router as m
    for task, tier in m.TASK_TIERS.items():
        if tier == "deterministic":
            continue
        assert tier in m.TIER_PROVIDERS, f"{task} → unknown tier {tier}"


def test_deterministic_tasks_request_no_llm():
    import model_router as m
    plan = m.route("heartbeat", "cto")
    assert plan["deterministic"] is True
    assert plan["providers"] == []
    assert plan["claude_eligible"] is False


def test_free_first_for_every_tier():
    import model_router as m
    for task in ("standup", "alpha_mining", "risk_signoff"):
        plan = m.route(task, "cro")
        assert plan["providers"], f"{task} has no free provider chain"
        # First provider is always a free one, never claude.
        assert plan["providers"][0] != "claude"


def test_claude_gated_to_senior_high_stakes(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ALLOW_CLAUDE_SENIOR", "true")
    monkeypatch.delenv("CLAUDE_DAILY_BUDGET", raising=False)
    import model_router as m
    importlib.reload(m)
    # senior + high-stakes → eligible
    assert m.route("risk_signoff", "cro")["claude_eligible"] is True
    assert m.route("architecture", "vp_eng")["claude_eligible"] is True
    # non-senior + high-stakes → NOT eligible
    assert m.route("risk_signoff", "data_engineer_2")["claude_eligible"] is False
    # senior + low-tier task → NOT eligible (don't waste Claude on chatter)
    assert m.route("standup", "cro")["claude_eligible"] is False


def test_claude_disabled_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import model_router as m
    importlib.reload(m)
    assert m.route("risk_signoff", "cro")["claude_eligible"] is False


def test_claude_flag_off_blocks_escalation(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ALLOW_CLAUDE_SENIOR", "false")
    import model_router as m
    importlib.reload(m)
    assert m.route("risk_signoff", "cro")["claude_eligible"] is False


def test_claude_budget_exhaustion_blocks(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ALLOW_CLAUDE_SENIOR", "true")  # opt-in, so budget is the gate under test
    monkeypatch.setenv("CLAUDE_DAILY_BUDGET", "0")
    import model_router as m
    importlib.reload(m)
    assert m.claude_budget_remaining() == 0
    assert m.route("risk_signoff", "cro")["claude_eligible"] is False


def test_claude_fail_closed_by_default(monkeypatch):
    # Key present but ALLOW_CLAUDE_SENIOR unset → must NOT escalate (fail-closed).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("ALLOW_CLAUDE_SENIOR", raising=False)
    monkeypatch.delenv("CLAUDE_DAILY_BUDGET", raising=False)
    import model_router as m
    importlib.reload(m)
    assert m.route("risk_signoff", "cro")["claude_eligible"] is False


# ── render_sync_secrets ─────────────────────────────────────────────────────

def test_numbered_key_maps_to_canonical(monkeypatch):
    for v in ("GROQ_API_KEY", "GROQ_API_KEY_1", "GEMINI_API_KEY", "GEMINI_API_KEY_1"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("GROQ_API_KEY_1", "groq-numbered")
    import render_sync_secrets as r
    importlib.reload(r)
    vals = r._resolve_values()
    # numbered variant mirrored verbatim
    assert vals.get("GROQ_API_KEY_1") == "groq-numbered"
    # canonical name populated from the numbered variant (gateway reads this)
    assert vals.get("GROQ_API_KEY") == "groq-numbered"


def test_empty_secrets_are_skipped(monkeypatch):
    for v in list(os.environ):
        if v.endswith("_API_KEY") or "_API_KEY_" in v:
            monkeypatch.delenv(v, raising=False)
    import render_sync_secrets as r
    importlib.reload(r)
    assert r._resolve_values() == {}


def test_canonical_not_overwritten_when_present(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "canonical")
    monkeypatch.setenv("GEMINI_API_KEY_1", "numbered")
    import render_sync_secrets as r
    importlib.reload(r)
    vals = r._resolve_values()
    assert vals["GEMINI_API_KEY"] == "canonical"   # bare value wins
    assert vals["GEMINI_API_KEY_1"] == "numbered"  # variant still mirrored


# ── member_standup ───────────────────────────────────────────────────────────

def test_standup_channel_teams_reference_real_personas():
    import member_standup as s
    from slack_agent_team import _EMPLOYEE_PERSONAS
    for channel, (lead, _name, members) in s.CHANNEL_TEAMS.items():
        for role in [lead, *members]:
            assert role in _EMPLOYEE_PERSONAS or role == "frontend", \
                f"#{channel}: role '{role}' has no persona"


def test_standup_skips_without_llm_key(monkeypatch):
    for v in s_keys():
        monkeypatch.delenv(v, raising=False)
    import member_standup as s
    importlib.reload(s)
    out = s.run_member_standup()
    assert out.get("skipped") == "no_llm_key"
    assert out["member_replies"] == 0


def s_keys():
    import member_standup as s
    return list(s._LLM_KEY_VARS)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
