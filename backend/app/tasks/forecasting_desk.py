"""
Forecasting desk — projects weekly / monthly / yearly profit from the realized
PnL of every desk, and reports a summary to leadership every 6 hours.

It reads closed Trade records (the same source the analytics performance endpoint
uses), builds a daily-PnL series overall and per strategy/desk, and runs the
statistical projection in analytics/forecasting.py. The 6-hourly report goes to
the leadership-summary Slack channel and is broadcast on the agent bus so other
desks and the leadership dashboard can consume it. Real data only: with fewer
than 5 trading days of history it reports "insufficient data" rather than
projecting from noise.
"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.utils.logging import logger

STATE_PATH = Path(__file__).parents[3] / "experiments" / "results" / "forecast.json"
REPORT_INTERVAL_SECONDS = 6 * 3600  # every 6 hours


class ForecastingDesk:
    def __init__(self, lookback_days: int = 90, interval_seconds: int = REPORT_INTERVAL_SECONDS):
        self.lookback_days = lookback_days
        self.interval_seconds = interval_seconds
        self._running = False
        self.last_forecast: dict | None = None

    async def _daily_series(self, db_session_factory) -> tuple[list[float], dict[str, list[float]]]:
        """Return (overall daily PnL series, per-desk daily PnL series) from Trades."""
        from sqlalchemy import func, select

        from app.models.trade import Trade

        cutoff = datetime.now(UTC) - timedelta(days=self.lookback_days)
        async with db_session_factory() as db:
            # Overall: daily total PnL.
            day = func.date(Trade.closed_at)
            overall_rows = (await db.execute(
                select(day.label("d"), func.sum(Trade.realized_pnl).label("pnl"))
                .where(Trade.closed_at >= cutoff, Trade.realized_pnl.isnot(None))
                .group_by(day).order_by(day)
            )).all()
            overall = [float(r.pnl) for r in overall_rows]

            # Per-desk: daily total PnL per strategy_name.
            desk_rows = (await db.execute(
                select(Trade.strategy_name, day.label("d"),
                       func.sum(Trade.realized_pnl).label("pnl"))
                .where(Trade.closed_at >= cutoff, Trade.realized_pnl.isnot(None))
                .group_by(Trade.strategy_name, day).order_by(day)
            )).all()
        by_desk: dict[str, list[float]] = {}
        for name, _d, pnl in desk_rows:
            by_desk.setdefault(name or "unattributed", []).append(float(pnl))
        return overall, by_desk

    async def compute(self, db_session_factory=None) -> dict:
        from app.analytics.forecasting import build_forecast
        if db_session_factory is None:
            from app.database import AsyncSessionLocal as db_session_factory
        try:
            overall, by_desk = await self._daily_series(db_session_factory)
        except Exception as e:  # noqa: BLE001
            logger.warning("forecasting_desk: query failed", error=str(e))
            return {"sufficient_data": False, "error": str(e)}
        forecast = build_forecast(overall, by_desk=by_desk)
        forecast["computed_at"] = datetime.now(UTC).isoformat()
        forecast["lookback_days"] = self.lookback_days
        self.last_forecast = forecast
        return forecast

    def _format_report(self, f: dict) -> str:
        if not f.get("sufficient_data"):
            return ("*Profit Forecast* — insufficient history "
                    f"({f.get('stats', {}).get('n_days', 0)} trading days). "
                    "Need ≥5 days of closed trades to project.")
        s = f["stats"]
        p = f["projections"]
        return (
            "*Profit Forecast (leadership)* :crystal_ball:\n"
            f"• History: {s['n_days']}d | realized PnL: {s['total_pnl']:.0f} | "
            f"Sharpe(ann): {s['sharpe_annual']:.2f} | positive days: {s['positive_day_rate']*100:.0f}%\n"
            f"• Weekly:  {p['weekly']['expected_pnl']:.0f} "
            f"(range {p['weekly']['low_pnl']:.0f}…{p['weekly']['high_pnl']:.0f})\n"
            f"• Monthly: {p['monthly']['expected_pnl']:.0f} "
            f"(range {p['monthly']['low_pnl']:.0f}…{p['monthly']['high_pnl']:.0f})\n"
            f"• Yearly:  {p['yearly']['expected_pnl']:.0f} "
            f"(range {p['yearly']['low_pnl']:.0f}…{p['yearly']['high_pnl']:.0f})\n"
            f"_{f['method']}_"
        )

    async def run_and_report(self, db_session_factory=None) -> dict:
        forecast = await self.compute(db_session_factory)
        report_text = self._format_report(forecast)

        # Persist for the dashboard / API.
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            STATE_PATH.write_text(json.dumps(forecast, indent=2, default=str))
        except Exception as e:  # noqa: BLE001
            logger.debug("forecasting_desk: state write failed", error=str(e))

        # Report to leadership via Slack + agent bus.
        try:
            from app.notifications.slack import slack
            await slack.send("leadership-summary", "system", "Profit Forecast", text=report_text)
        except Exception as e:  # noqa: BLE001
            logger.debug("forecasting_desk: slack report failed", error=str(e))
        try:
            from app.tasks.agent_bus import get_bus
            await get_bus().broadcast_signal(
                {"type": "profit_forecast", "forecast": forecast},
                from_agent="forecasting_desk",
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("forecasting_desk: broadcast failed", error=str(e))

        logger.info("forecasting_desk: report sent",
                    sufficient=forecast.get("sufficient_data"))
        return forecast

    async def run(self) -> None:
        self._running = True
        logger.info("ForecastingDesk started", interval_hours=self.interval_seconds / 3600)
        while self._running:
            try:
                await self.run_and_report()
            except asyncio.CancelledError:
                logger.info("ForecastingDesk cancelled — shutting down")
                break
            except Exception as e:  # noqa: BLE001
                logger.error("ForecastingDesk cycle crashed", error=str(e))
            if self._running:
                await asyncio.sleep(self.interval_seconds)

    def stop(self) -> None:
        self._running = False


_desk: ForecastingDesk | None = None


def get_forecasting_desk() -> ForecastingDesk:
    global _desk
    if _desk is None:
        _desk = ForecastingDesk()
    return _desk
