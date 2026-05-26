"""Analyze realized vs expected slippage per execution algorithm."""
from __future__ import annotations
import sys
import asyncio
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))


async def analyze_slippage(days: int = 30) -> None:
    from sqlalchemy import select, func
    from app.database import AsyncSessionLocal
    from app.models.slippage import SlippageRecord
    from datetime import datetime, timezone, timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(
                SlippageRecord.execution_algo,
                func.avg(SlippageRecord.slippage_bps).label("avg_bps"),
                func.min(SlippageRecord.slippage_bps).label("min_bps"),
                func.max(SlippageRecord.slippage_bps).label("max_bps"),
                func.count(SlippageRecord.id).label("count"),
            )
            .where(SlippageRecord.created_at >= cutoff)
            .group_by(SlippageRecord.execution_algo)
        )
        rows = result.all()

    print(f"\nSlippage Analysis (last {days} days):")
    print(f"{'Algorithm':<20} {'Avg bps':>10} {'Min bps':>10} {'Max bps':>10} {'Orders':>8}")
    print("-" * 65)
    for r in sorted(rows, key=lambda x: x.avg_bps or 0):
        print(f"{r.execution_algo or 'unknown':<20} {r.avg_bps or 0:>10.2f} {r.min_bps or 0:>10.2f} {r.max_bps or 0:>10.2f} {r.count:>8}")

    if rows:
        market_bps = next((r.avg_bps for r in rows if r.execution_algo == "market"), None)
        if market_bps:
            print("\nSavings vs market orders:")
            for r in rows:
                if r.execution_algo != "market" and r.avg_bps is not None:
                    saving = market_bps - r.avg_bps
                    print(f"  {r.execution_algo}: {saving:+.2f} bps vs market")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    asyncio.run(analyze_slippage(args.days))
