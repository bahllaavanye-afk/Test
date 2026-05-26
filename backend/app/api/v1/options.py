"""Options flow, wheel strategy, and macro calendar endpoints."""
from fastapi import APIRouter, Depends, Query
from app.api.deps import get_current_user
from app.models.user import User
from app.options.flow import scanner
from app.options.wheel import find_wheel_opportunities
from app.options.macro_calendar import get_upcoming_events, get_next_fomc

router = APIRouter(prefix="/options", tags=["options"])


@router.get("/flow")
async def get_options_flow(
    unusual_only: bool = Query(False),
    current_user: User = Depends(get_current_user),
):
    flows = await scanner.scan()
    if unusual_only:
        flows = [f for f in flows if f.is_unusual]
    return [f.to_dict() for f in flows[:50]]


@router.get("/put-call-ratio")
async def get_put_call_ratio(current_user: User = Depends(get_current_user)):
    await scanner.scan()
    return scanner.put_call_ratio()


@router.get("/wheel")
async def get_wheel_opportunities(
    tickers: str = Query(None, description="Comma-separated tickers, e.g. AAPL,MSFT"),
    current_user: User = Depends(get_current_user),
):
    ticker_list = tickers.split(",") if tickers else None
    return [s.to_dict() for s in find_wheel_opportunities(ticker_list)]


@router.get("/macro-calendar")
async def get_macro_calendar(
    days_ahead: int = Query(90, le=365),
    current_user: User = Depends(get_current_user),
):
    return get_upcoming_events(days_ahead)


@router.get("/next-fomc")
async def next_fomc():
    """Public endpoint — no auth required for next FOMC date."""
    return get_next_fomc()
