"""Guard: any workflow that can post to Discord must pass the BOT token too.

Why this exists — the bug it would have caught:
notify.py routes each message to its real Discord channel using
DISCORD_BOT_TOKEN, and only falls back to the single default webhook (which
dumps into #general with a `[#channel]` text prefix) when the bot token is
absent. 25 of 28 Discord-posting workflows passed only DISCORD_WEBHOOK_URL in
their env, so every one of their messages fell back to #general and the real
channels stayed empty. Nothing tested the workflow env, so it went unnoticed.

This test fails if a workflow provides `secrets.DISCORD_WEBHOOK_URL` to a step
without also providing `secrets.DISCORD_BOT_TOKEN`. Pure text scan — no yaml
dependency, so it runs anywhere pytest does.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"


def _workflow_files() -> list[Path]:
    return sorted(p for p in _WORKFLOWS.glob("*.yml"))


def test_workflows_dir_exists():
    assert _WORKFLOWS.is_dir(), f"missing {_WORKFLOWS}"
    assert _workflow_files(), "no workflow files found"


@pytest.mark.parametrize("wf", _workflow_files(), ids=lambda p: p.name)
def test_webhook_workflows_also_pass_bot_token(wf: Path):
    """If a workflow wires the Discord webhook, it must wire the bot token too,
    otherwise notify.py can't route and everything lands in #general."""
    text = wf.read_text()
    uses_webhook = "secrets.DISCORD_WEBHOOK_URL" in text
    if not uses_webhook:
        pytest.skip("workflow does not post to Discord")
    assert "secrets.DISCORD_BOT_TOKEN" in text, (
        f"{wf.name} passes DISCORD_WEBHOOK_URL but not DISCORD_BOT_TOKEN — its "
        f"messages will fall back to the default webhook (#general) instead of "
        f"routing to the real channel. Add "
        f"`DISCORD_BOT_TOKEN: ${{{{ secrets.DISCORD_BOT_TOKEN }}}}` to the step env."
    )
