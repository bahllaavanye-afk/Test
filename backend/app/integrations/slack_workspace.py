"""Full Slack workspace integration for the engineering team.

Posts standups, alpha reviews, daily P&L, deploys, incidents, and CI failures
to dedicated channels. Each channel has its own webhook URL so cost is zero
on Slack's free tier (10k messages, 90‑day retention).

Channels (create these in your Slack workspace):
  #engineering-standup    — daily standups (one post per squad)
  #alpha-research         — new strategy proposals, paper reviews
  #pnl-daily              — EOD P&L attribution
  #risk-alerts            — VaR breaches, circuit breaker fires
  #incidents              — P0/P1 incidents + postmortems
  #deploys                — deploy notifications
  #ci-failures            — CI / test failures
  #ml-experiments         — training run completions, model leaderboard

Config: set the channel‑specific webhook URLs as env vars
  SLACK_WEBHOOK_STANDUP, SLACK_WEBHOOK_ALPHA, SLACK_WEBHOOK_PNL,
  SLACK_WEBHOOK_RISK, SLACK_WEBHOOK_INCIDENTS, SLACK_WEBHOOK_DEPLOYS,
  SLACK_WEBHOOK_CI, SLACK_WEBHOOK_ML
Or use the legacy SLACK_WEBHOOK_DEFAULT fallback.

Get webhook URLs at: https://api.slack.com/apps → Create New App → From Scratch
→ Incoming Webhooks → Add per‑channel webhooks → copy each URL.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Tuple, List

import httpx
from pydantic import BaseModel, Field, HttpUrl, validator

from app.utils.logging import logger


class SlackChannel:
    STANDUP = "standup"
    ALPHA = "alpha"
    PNL = "pnl"
    RISK = "risk"
    INCIDENTS = "incidents"
    DEPLOYS = "deploys"
    CI = "ci"
    ML = "ml"


_ENV_BY_CHANNEL = {
    SlackChannel.STANDUP: "SLACK_WEBHOOK_STANDUP",
    SlackChannel.ALPHA: "SLACK_WEBHOOK_ALPHA",
    SlackChannel.PNL: "SLACK_WEBHOOK_PNL",
    SlackChannel.RISK: "SLACK_WEBHOOK_RISK",
    SlackChannel.INCIDENTS: "SLACK_WEBHOOK_INCIDENTS",
    SlackChannel.DEPLOYS: "SLACK_WEBHOOK_DEPLOYS",
    SlackChannel.CI: "SLACK_WEBHOOK_CI",
    SlackChannel.ML: "SLACK_WEBHOOK_ML",
}


def _resolve_webhook(channel: str) -> str | None:
    """Pick the most specific webhook env var, fall back to the default."""
    env_name = _ENV_BY_CHANNEL.get(channel)
    if env_name:
        url = os.getenv(env_name, "").strip()
        if url:
            return url
    return os.getenv("SLACK_WEBHOOK_DEFAULT", "").strip() or None


async def post(
    channel: str,
    text: str,
    *,
    blocks: List[dict] | None = None,
    color: str | None = None,
) -> bool:
    """Post a message to the given channel. Returns ``True`` on success."""
    webhook = _resolve_webhook(channel)
    if not webhook:
        logger.debug("slack: no webhook for channel", channel=channel)
        return False

    payload: dict[str, Any] = {"text": text}
    if blocks:
        payload["blocks"] = blocks
    if color:
        # Slack attachment color shows on the left side
        payload["attachments"] = [{"color": color, "text": text}]

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(webhook, json=payload)
            r.raise_for_status()
            return True
    except Exception as e:
        logger.warning("slack post failed", channel=channel, error=str(e))
        return False


# ── Pydantic schemas for validation ──────────────────────────────────────


class StandupPayload(BaseModel):
    """Payload for a daily standup post."""

    squad: str = Field(..., description="Team or squad identifier", example="Alpha")
    shipped: List[str] = Field(
        default_factory=list,
        description="List of items shipped since last standup",
        example=["feature‑x", "bug‑123"],
    )
    planned: List[str] = Field(
        default_factory=list,
        description="List of items planned for the next iteration",
        example=["feature‑y", "research‑z"],
    )
    blockers: List[str] = Field(
        default_factory=list,
        description="List of current blockers; empty if none",
        example=["awaiting data from DB"],
    )

    @validator("squad")
    def squad_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("squad must be a non‑empty string")
        return v


class AlphaReviewPayload(BaseModel):
    """Payload for an alpha strategy review."""

    strategy: str = Field(..., description="Strategy name", example="MeanReversionV2")
    sharpe: float = Field(..., description="Sharpe ratio (annualized)", example=1.23, ge=0)
    maxdd: float = Field(
        ...,
        description="Maximum drawdown as a proportion (0‑1)",
        example=0.15,
        ge=0,
        le=1,
    )
    decision: Literal["promoted", "iterate", "reject"] = Field(
        ...,
        description="Decision after review",
        example="promoted",
    )


class EODPNLPayload(BaseModel):
    """Payload for end‑of‑day P&L summary."""

    date: str = Field(
        ...,
        description="Date of the P&L report in ISO format (YYYY‑MM‑DD)",
        example="2024-07-21",
    )
    total_pnl: float = Field(..., description="Total P&L for the day", example=12345.67)
    top: List[Tuple[str, float]] = Field(
        ...,
        description="Top performers as (symbol, pnl) tuples",
        example=[("AAPL", 5000.0), ("MSFT", 3000.0)],
    )
    bottom: List[Tuple[str, float]] = Field(
        ...,
        description="Bottom performers as (symbol, pnl) tuples",
        example=[("TSLA", -2000.0), ("AMZN", -1500.0)],
    )

    @validator("date")
    def validate_date(cls, v: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError("date must be in YYYY-MM-DD format")
        return v


class RiskAlertPayload(BaseModel):
    """Payload for a risk alert."""

    severity: Literal["P0", "P1", "P2"] = Field(..., description="Severity level", example="P0")
    message: str = Field(..., description="Human‑readable alert message", example="VaR breach")
    metric: Optional[str] = Field(None, description="Metric name if applicable", example="VaR")
    value: Optional[float] = Field(
        None,
        description="Metric value associated with the alert",
        example=0.025,
    )

    @validator("message")
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be empty")
        return v


class DeployPayload(BaseModel):
    """Payload for a deployment notification."""

    service: str = Field(..., description="Service name", example="order‑router")
    version: str = Field(..., description="Version identifier", example="v2.3.1")
    status: Literal["success", "failure"] = Field(..., description="Deployment status", example="success")
    url: Optional[HttpUrl] = Field(None, description="Optional URL to view deployment details", example="https://ci.example.com/deploy/123")

    @validator("service", "version")
    def non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field cannot be empty")
        return v


class CIFailurePayload(BaseModel):
    """Payload for a CI failure notification."""

    branch: str = Field(..., description="Git branch name", example="main")
    run_url: HttpUrl = Field(..., description="URL to the CI run logs", example="https://ci.example.com/run/456")
    failing_step: str = Field(..., description="Name of the failing step", example="tests")


class IncidentPayload(BaseModel):
    """Payload for an incident report."""

    severity: Literal["P0", "P1", "P2"] = Field(..., description="Incident severity", example="P1")
    component: str = Field(..., description="Component or service impacted", example="pricing‑engine")
    description: str = Field(..., description="Detailed description of the incident", example="Timeouts in market data feed")
    oncall: str = Field(..., description="On‑call engineer identifier", example="jdoe")


class MLRunCompletePayload(BaseModel):
    """Payload for a completed ML training run."""

    model: str = Field(..., description="Model name", example="XGBoostRegressor")
    symbol: str = Field(..., description="Trading symbol the model was trained on", example="EURUSD")
    val_sharpe: float = Field(..., description="Validation Sharpe ratio", example=1.45, ge=0)
    run_id: str = Field(..., description="Unique identifier of the training run", example="run-20240721-001")


# ── High‑level helpers used across the codebase ───────────────────────────


async def post_standup(squad: str, shipped: List[str], planned: List[str], blockers: List[str]) -> bool:
    payload = StandupPayload(squad=squad, shipped=shipped, planned=planned, blockers=blockers)
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"🌅 {payload.squad} standup — {datetime.now(timezone.utc):%Y-%m-%d}"},
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Shipped*\n" + ("\n".join(f"• {x}" for x in payload.shipped) or "_nothing yet_"),
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Planned*\n" + ("\n".join(f"• {x}" for x in payload.planned) or "_to be set_"),
                },
            ],
        },
    ]
    if payload.blockers:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*🚧 Blockers*\n" + "\n".join(f"• {x}" for x in payload.blockers)},
            }
        )
    return await post(SlackChannel.STANDUP, f"{payload.squad} standup", blocks=blocks)


async def post_alpha_review(strategy: str, sharpe: float, maxdd: float, decision: str) -> bool:
    payload = AlphaReviewPayload(strategy=strategy, sharpe=sharpe, maxdd=maxdd, decision=decision)
    color = "good" if payload.decision == "promoted" else "warning" if payload.decision == "iterate" else "danger"
    text = f"📈 *{payload.strategy}* — Sharpe {payload.sharpe:.2f}, MaxDD {payload.maxdd:.1%}, decision: {payload.decision}"
    return await post(SlackChannel.ALPHA, text, color=color)


async def post_eod_pnl(date: str, total_pnl: float, top: List[Tuple[str, float]], bottom: List[Tuple[str, float]]) -> bool:
    payload = EODPNLPayload(date=date, total_pnl=total_pnl, top=top, bottom=bottom)
    sign = "+" if payload.total_pnl >= 0 else ""
    top_str = "\n".join(f"  🟢 {s}: {sign}${p:,.0f}" for s, p in payload.top[:5])
    bot_str = "\n".join(f"  🔴 {s}: ${p:,.0f}" for s, p in payload.bottom[:5])
    text = (
        f"💰 *EOD P&L {payload.date}* — Total: {sign}${payload.total_pnl:,.0f}\n\n"
        f"*Top 5*\n{top_str}\n\n*Bottom 5*\n{bot_str}"
    )
    color = "good" if payload.total_pnl >= 0 else "danger"
    return await post(SlackChannel.PNL, text, color=color)


async def post_risk_alert(severity: str, message: str, metric: str | None = None, value: float | None = None) -> bool:
    payload = RiskAlertPayload(severity=severity, message=message, metric=metric, value=value)
    color = {"P0": "danger", "P1": "warning", "P2": "good"}.get(payload.severity, "warning")
    detail = f" ({payload.metric}={payload.value:.4f})" if payload.metric and payload.value is not None else ""
    return await post(SlackChannel.RISK, f"⚠️ *[{payload.severity}] Risk*: {payload.message}{detail}", color=color)


async def post_deploy(service: str, version: str, status: str, url: str | None = None) -> bool:
    payload = DeployPayload(service=service, version=version, status=status, url=url)
    color = "good" if payload.status == "success" else "danger"
    link = f"\n<{payload.url}|View deploy>" if payload.url else ""
    return await post(SlackChannel.DEPLOYS, f"🚀 *{payload.service}* deploy `{payload.version}` → {payload.status}{link}", color=color)


async def post_ci_failure(branch: str, run_url: str, failing_step: str) -> bool:
    payload = CIFailurePayload(branch=branch, run_url=run_url, failing_step=failing_step)
    text = f"❌ CI failed on `{payload.branch}` — step *{payload.failing_step}*\n<{payload.run_url}|View logs>"
    return await post(SlackChannel.CI, text, color="danger")


async def post_incident(severity: str, component: str, description: str, oncall: str) -> bool:
    payload = IncidentPayload(severity=severity, component=component, description=description, oncall=oncall)
    color = {"P0": "danger", "P1": "warning"}.get(payload.severity, "good")
    text = f"🚨 *[{payload.severity}] Incident* — {payload.component}\n{payload.description}\n_On-call: {payload.oncall}_"
    return await post(SlackChannel.INCIDENTS, text, color=color)


async def post_ml_run_complete(model: str, symbol: str, val_sharpe: float, run_id: str) -> bool:
    payload = MLRunCompletePayload(model=model, symbol=symbol, val_sharpe=val_sharpe, run_id=run_id)
    color = "good" if payload.val_sharpe > 1.0 else "warning"
    text = (
        f"🧠 ML training complete — *{payload.model}* on *{payload.symbol}* — "
        f"val Sharpe {payload.val_sharpe:.2f} (run `{payload.run_id}`)"
    )
    return await post(SlackChannel.ML, text, color=color)