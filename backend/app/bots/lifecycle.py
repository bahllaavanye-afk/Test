"""Autonomous bot lifecycle — the 'human using Options Alpha' loop, no human.

Deterministic policy (no LLM in the trading path), run periodically by the
scheduler:

  • DISABLE  enabled bots with enough closed-trade evidence that they're losing
             (paper capital rotates away from them, like archiving in Options Alpha)
  • ENABLE   disabled bots whose record has turned positive (second chances are
             cheap on paper; a re-enabled bot re-earns its slot)
  • CREATE   template bots that aren't instantiated yet, bounded per run, so the
             fleet grows itself toward full template coverage

`decide_bot_actions` is a pure function over per-bot stats so the policy is
unit-testable without a DB. `run_bot_lifecycle` is the thin I/O wrapper the
scheduler calls.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.utils.logging import logger

# Evidence thresholds — conservative so bots aren't churned on noise.
MIN_TRADES_TO_JUDGE = 8       # need this many closed trades before disabling
DISABLE_WIN_RATE = 0.45       # losing money AND winning < 45% → disable
MIN_TRADES_TO_PROMOTE = 5     # positive record this size → re-enable
MAX_CREATES_PER_RUN = 4       # bounded fleet growth per cycle (11 OA clones → full rollout in ~3 cycles)


@dataclass
class BotStats:
    bot_id: str
    name: str
    is_enabled: bool
    is_archived: bool
    trades: int          # closed trades in the lookback window
    total_pnl: float
    win_rate: float | None


def decide_bot_actions(stats: list[BotStats], uninstantiated_templates: list[str]) -> dict:
    """Pure policy: given per-bot records, decide enable/disable/create actions."""
    disable: list[BotStats] = []
    enable: list[BotStats] = []

    for s in stats:
        if s.is_archived:
            continue
        if s.is_enabled:
            if (
                s.trades >= MIN_TRADES_TO_JUDGE
                and s.total_pnl < 0
                and (s.win_rate or 0) < DISABLE_WIN_RATE
            ):
                disable.append(s)
        else:
            if s.trades >= MIN_TRADES_TO_PROMOTE and s.total_pnl > 0:
                enable.append(s)

    return {
        "disable": disable,
        "enable": enable,
        "create": uninstantiated_templates[:MAX_CREATES_PER_RUN],
    }


async def _bot_stats(db, lookback_days: int = 30) -> list[BotStats]:
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import case, func, select

    from app.models.bot import Bot
    from app.models.trade import Trade

    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    bots = (await db.execute(select(Bot))).scalars().all()

    rows = (await db.execute(
        select(
            Trade.strategy_name,
            func.count(Trade.id),
            func.coalesce(func.sum(Trade.realized_pnl), 0.0),
            func.sum(case((Trade.realized_pnl > 0, 1), else_=0)),
        )
        .where(Trade.closed_at >= since, Trade.strategy_name.isnot(None))
        .group_by(Trade.strategy_name)
    )).all()
    perf = {r[0]: (int(r[1]), float(r[2]), int(r[3] or 0)) for r in rows}

    out: list[BotStats] = []
    for b in bots:
        n, pnl, wins = perf.get(b.name, (0, 0.0, 0))
        out.append(BotStats(
            bot_id=b.id, name=b.name,
            is_enabled=bool(b.is_enabled),
            is_archived=bool(getattr(b, "is_archived", False)),
            trades=n, total_pnl=pnl,
            win_rate=(wins / n) if n else None,
        ))
    return out


async def run_bot_lifecycle(db_session_factory=None) -> dict:
    """Apply the lifecycle policy to the real fleet. Returns an action summary."""
    from sqlalchemy import select

    from app.bots.templates import BOT_TEMPLATES
    from app.models.bot import Bot

    if db_session_factory is None:
        from app.database import AsyncSessionLocal as db_session_factory  # type: ignore

    summary = {"disabled": [], "enabled": [], "created": []}
    try:
        async with db_session_factory() as db:
            stats = await _bot_stats(db)
            existing_names = {s.name for s in stats}
            # BOT_TEMPLATES: {template_id: template dict}
            missing = {tid: t for tid, t in BOT_TEMPLATES.items()
                       if t.get("name") and t["name"] not in existing_names}
            actions = decide_bot_actions(stats, list(missing.keys()))

            for s in actions["disable"] + actions["enable"]:
                bot = (await db.execute(select(Bot).where(Bot.id == s.bot_id))).scalar_one_or_none()
                if bot is None:
                    continue
                if s in actions["disable"]:
                    bot.is_enabled = False
                    summary["disabled"].append(s.name)
                else:
                    bot.is_enabled = True
                    summary["enabled"].append(s.name)

            # Fleet growth: instantiate missing templates under the fleet's owner
            # (same construction as app/bots/seed.py — user + account of an existing bot).
            if actions["create"]:
                import uuid as _uuid

                anchor = (await db.execute(
                    select(Bot.user_id, Bot.account_id).limit(1)
                )).first()
                if anchor:
                    for tid in actions["create"]:
                        t = missing[tid]
                        try:
                            db.add(Bot(
                                id=str(_uuid.uuid4()),
                                user_id=anchor.user_id,
                                account_id=anchor.account_id,
                                name=t["name"],
                                description=t.get("description", ""),
                                symbol=t["symbol"],
                                market_type=t.get("market_type", "equity"),
                                trigger=t["trigger"],
                                conditions=t.get("conditions", []),
                                condition_logic=t.get("condition_logic", "ALL"),
                                action=t["action"],
                                exit_rules=t.get("exit_rules", []),
                                is_enabled=True,
                                template_id=tid,
                            ))
                            summary["created"].append(t["name"])
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Bot lifecycle: create failed", template=tid, error=str(exc))

            if summary["disabled"] or summary["enabled"] or summary["created"]:
                await db.commit()

        if any(summary.values()):
            logger.info("Bot lifecycle actions", **{k: v for k, v in summary.items() if v})
            try:
                from app.notifications.slack import slack
                lines = []
                if summary["disabled"]:
                    lines.append("⛔ Disabled (losing record): " + ", ".join(summary["disabled"]))
                if summary["enabled"]:
                    lines.append("✅ Re-enabled (record turned positive): " + ", ".join(summary["enabled"]))
                if summary["created"]:
                    lines.append("🆕 Created from templates: " + ", ".join(summary["created"]))
                await slack.send(
                    channel="orders", event_type="info",
                    title="🤖 Bot lifecycle manager", text="\n".join(lines),
                )
            except Exception:  # notification is best-effort
                pass
    except Exception as exc:  # noqa: BLE001
        logger.error("Bot lifecycle failed", error=str(exc))
    return summary
