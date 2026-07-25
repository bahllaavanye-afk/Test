"""Tests for app.notifications.discord — the backend's only notification path.

Added when Slack was removed (2026-07-25). The old Slack client had NO tests at
all, which is how two real defects survived in it:
  * the Discord failover flattened the rich coloured attachment into plain text,
    so once Slack died every alert rendered as `**title**\\n• k: v`;
  * `_enabled` was frozen at construction, so configuring credentials after
    import could never turn notifications on.
Both are covered below, alongside the routing and typed-helper behaviour.
"""
from __future__ import annotations

import pytest

from app.notifications.discord import (
    CHANNEL_MAP,
    COLORS,
    DEFAULT_COLOR,
    MAX_EMBED_FIELDS,
    DiscordClient,
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-bot-token")
    monkeypatch.setenv("DISCORD_GUILD_ID", "999")
    c = DiscordClient()
    c._bot_token = "test-bot-token"
    c._guild_id = "999"
    c._default_webhook = ""
    return c


# ── embed construction ───────────────────────────────────────────────────────

def test_embed_is_structured_not_flattened_text():
    """The whole point of the rewrite: Discord gets a real embed."""
    embed = DiscordClient._build_embed(
        "order_filled", "✅ Filled: AAPL", "some text", {"Symbol": "AAPL", "Qty": "1.0"}
    )
    assert embed["title"] == "✅ Filled: AAPL"
    assert embed["description"] == "some text"
    assert embed["color"] == COLORS["order_filled"]
    assert [f["name"] for f in embed["fields"]] == ["Symbol", "Qty"]
    assert [f["value"] for f in embed["fields"]] == ["AAPL", "1.0"]


def test_embed_colour_differs_per_event_type():
    filled = DiscordClient._build_embed("order_filled", "t", None, None)["color"]
    risk = DiscordClient._build_embed("risk_event", "t", None, None)["color"]
    unknown = DiscordClient._build_embed("not_a_real_event", "t", None, None)["color"]
    assert filled != risk
    assert unknown == DEFAULT_COLOR


def test_embed_respects_discord_field_cap():
    fields = {f"k{i}": i for i in range(MAX_EMBED_FIELDS + 10)}
    embed = DiscordClient._build_embed("system", "t", None, fields)
    assert len(embed["fields"]) == MAX_EMBED_FIELDS


def test_embed_omits_description_when_no_text():
    assert "description" not in DiscordClient._build_embed("system", "t", None, None)


def test_embed_truncates_overlong_values():
    embed = DiscordClient._build_embed("system", "T" * 500, "D" * 9000, {"k": "v" * 4000})
    assert len(embed["title"]) <= 256
    assert len(embed["description"]) <= 4096
    assert len(embed["fields"][0]["value"]) <= 1024


# ── enablement ───────────────────────────────────────────────────────────────

def test_enabled_is_dynamic_not_frozen_at_construction(monkeypatch):
    """Regression: the old client cached this, so late config never took effect."""
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    c = DiscordClient()
    c._bot_token = ""
    c._default_webhook = ""
    assert c._enabled is False
    c._default_webhook = "https://discord.com/api/webhooks/x/y"
    assert c._enabled is True


@pytest.mark.asyncio
async def test_send_is_a_noop_when_nothing_configured(monkeypatch):
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    c = DiscordClient()
    c._bot_token = ""
    c._default_webhook = ""
    assert await c.send("alerts", "risk_event", "title") is False


# ── routing ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_logical_stream_maps_to_its_own_channel(client, monkeypatch):
    seen: dict = {}

    async def fake_channel_id(channel):
        seen["channel"] = channel
        return "chan-1"

    async def fake_post_bot(cid, embed):
        seen["cid"], seen["embed"] = cid, embed
        return True

    monkeypatch.setattr(client, "_channel_id", fake_channel_id)
    monkeypatch.setattr(client, "_post_bot", fake_post_bot)

    assert await client.send("alerts", "risk_event", "⚠️ Risk") is True
    # "alerts" must be routed to the real risk channel, not a generic dump
    assert seen["channel"] == CHANNEL_MAP["alerts"] == "risk-alerts"
    assert seen["cid"] == "chan-1"


@pytest.mark.asyncio
async def test_unknown_channel_passes_through_verbatim(client, monkeypatch):
    seen = {}

    async def fake_channel_id(channel):
        seen["channel"] = channel
        return None

    async def fake_webhook(channel, embed):
        return True

    monkeypatch.setattr(client, "_channel_id", fake_channel_id)
    monkeypatch.setattr(client, "_post_webhook", fake_webhook)
    await client.send("desk-crypto", "system", "hi")
    assert seen["channel"] == "desk-crypto"


@pytest.mark.asyncio
async def test_falls_back_to_webhook_when_bot_post_fails(client, monkeypatch):
    calls = []

    async def fake_channel_id(channel):
        return "chan-1"

    async def failing_bot(cid, embed):
        calls.append("bot")
        return False

    async def ok_webhook(channel, embed):
        calls.append("webhook")
        return True

    monkeypatch.setattr(client, "_channel_id", fake_channel_id)
    monkeypatch.setattr(client, "_post_bot", failing_bot)
    monkeypatch.setattr(client, "_post_webhook", ok_webhook)

    assert await client.send("system", "system", "t") is True
    assert calls == ["bot", "webhook"]


def test_per_channel_webhook_beats_the_catch_all(client, monkeypatch):
    client._default_webhook = "https://discord.com/api/webhooks/default"
    monkeypatch.setenv("DISCORD_WEBHOOK_URL_RISK_ALERTS", "https://discord.com/api/webhooks/risk")
    url, is_default = client._webhook_for("risk-alerts")
    assert url.endswith("/risk")
    assert is_default is False


def test_catch_all_webhook_is_flagged_so_channel_can_be_labelled(client):
    client._default_webhook = "https://discord.com/api/webhooks/default"
    url, is_default = client._webhook_for("some-channel")
    assert url.endswith("/default")
    assert is_default is True


# ── typed helpers ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_typed_helpers_route_and_label_correctly(client, monkeypatch):
    captured: list[tuple] = []

    async def fake_send(channel, event_type, title, fields=None, text=None):
        captured.append((channel, event_type, title, fields, text))
        return True

    monkeypatch.setattr(client, "send", fake_send)

    await client.notify_order_filled("AAPL", "buy", 2.0, 190.5, slippage_bps=1.25, algo="twap")
    await client.notify_signal("momentum", "SPY", "sell", 0.83, target_price=500.0)
    await client.notify_risk_event("drawdown", "too deep", value=0.12)
    await client.notify_circuit_breaker("daily_loss", 0.05, 0.02)
    await client.notify_daily_summary(123.45, 7, 0.571, best_strategy="momentum")
    await client.notify_system("all good", level="warning")

    channels = [c[0] for c in captured]
    assert channels == ["orders", "signals", "alerts", "alerts", "system", "system"]

    # order fill carries the numbers a human needs
    fill_fields = captured[0][3]
    assert fill_fields["Symbol"] == "AAPL"
    assert fill_fields["Side"] == "BUY"
    assert fill_fields["Slippage"] == "1.25 bps"
    assert fill_fields["Algo"] == "twap"

    # circuit breaker must be unmistakable and explain the halt
    assert "CIRCUIT BREAKER" in captured[3][2]
    assert "halted" in captured[3][4].lower()

    # warning level picks the warning glyph
    assert captured[5][2].startswith("⚠️")


@pytest.mark.asyncio
async def test_optional_fields_are_omitted_when_absent(client, monkeypatch):
    captured = []

    async def fake_send(channel, event_type, title, fields=None, text=None):
        captured.append(fields)
        return True

    monkeypatch.setattr(client, "send", fake_send)
    await client.notify_order_filled("AAPL", "buy", 1.0, 100.0)
    assert "Slippage" not in captured[0]
    assert "Algo" not in captured[0]

    await client.notify_experiment_done("exp", None, None, None)
    assert list(captured[1]) == ["Name"]
