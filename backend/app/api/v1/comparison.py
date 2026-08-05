"""Manual vs ML strategy comparison endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.comparison import ComparisonResult as ComparisonModel
from app.models.user import User
from app.comparison.benchmarks import get_benchmark_stats
from pydantic import BaseModel, ConfigDict

# Constants
PREFIX = "/comparison"
TAG = "comparison"
ENDPOINT_BENCHMARKS = "/benchmarks"
ENDPOINT_RESULTS = "/results"
ENDPOINT_ROOT = "/"
MIN_MANUAL_SHARPE = 1e-9
IMPROVEMENT_PRECISION = 4
DEFAULT_LIMIT = 20

router = APIRouter(prefix=PREFIX, tags=[TAG])


class ComparisonOut(BaseModel):
    id: str
    strategy_name: str
    symbol: str
    manual_sharpe: float | None
    ml_sharpe: float | None
    is_significant: bool | None
    winner: str | None
    spy_sharpe: float | None
    ml_improvement_pct: float | None = None

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_model(cls, m) -> "ComparisonOut":
        improvement = None
        if m.manual_sharpe is not None and m.ml_sharpe is not None:
            base = float(m.manual_sharpe) or MIN_MANUAL_SHARPE
            improvement = (float(m.ml_sharpe) - float(m.manual_sharpe)) / abs(base)
        return cls(
            id=m.id,
            strategy_name=m.strategy_name,
            symbol=m.symbol,
            manual_sharpe=float(m.manual_sharpe) if m.manual_sharpe else None,
            ml_sharpe=float(m.ml_sharpe) if m.ml_sharpe else None,
            is_significant=m.is_significant,
            winner=m.winner,
            spy_sharpe=float(m.spy_sharpe) if m.spy_sharpe else None,
            ml_improvement_pct=round(improvement, IMPROVEMENT_PRECISION) if improvement else None,
        )


@router.get(ENDPOINT_BENCHMARKS)
async def get_benchmarks():
    return get_benchmark_stats()


@router.get(ENDPOINT_RESULTS, response_model=list[ComparisonOut])
@router.get(ENDPOINT_ROOT, response_model=list[ComparisonOut])
async def list_comparisons(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(ComparisonModel).order_by(ComparisonModel.created_at.desc()).limit(DEFAULT_LIMIT)
    )
    rows = result.scalars().all()
    return [ComparisonOut.from_model(r) for r in rows]


# --------------------------- Unit Tests ---------------------------------
import pytest

# Helper mock model mimicking the ORM entity
class _MockComparisonModel:
    def __init__(
        self,
        id: str,
        strategy_name: str,
        symbol: str,
        manual_sharpe: float | None = None,
        ml_sharpe: float | None = None,
        is_significant: bool | None = None,
        winner: str | None = None,
        spy_sharpe: float | None = None,
    ):
        self.id = id
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.manual_sharpe = manual_sharpe
        self.ml_sharpe = ml_sharpe
        self.is_significant = is_significant
        self.winner = winner
        self.spy_sharpe = spy_sharpe


@pytest.mark.parametrize(
    "manual, ml, expected_pct",
    [
        (0.0, 0.2, round((0.2 - 0.0) / MIN_MANUAL_SHARPE, IMPROVEMENT_PRECISION)),
        (None, None, None),
        (0.5, 0.75, round((0.75 - 0.5) / 0.5, IMPROVEMENT_PRECISION)),
    ],
)
def test_comparison_out_improvement_calculation(manual, ml, expected_pct):
    """Verify ml_improvement_pct calculation, especially when manual_sharpe is zero or None."""
    model = _MockComparisonModel(
        id="test",
        strategy_name="test_strat",
        symbol="TEST",
        manual_sharpe=manual,
        ml_sharpe=ml,
        is_significant=True,
        winner="ml",
        spy_sharpe=0.3,
    )
    out = ComparisonOut.from_model(model)
    assert out.ml_improvement_pct == expected_pct
    # Ensure manual_sharpe and ml_sharpe are correctly typed or None
    if manual is None:
        assert out.manual_sharpe is None
    else:
        assert out.manual_sharpe == float(manual)
    if ml is None:
        assert out.ml_sharpe is None
    else:
        assert out.ml_sharpe == float(ml)


@pytest.mark.asyncio
async def test_list_comparisons_respects_default_limit(monkeypatch):
    """Ensure list_comparisons returns at most DEFAULT_LIMIT items even if DB yields more."""
    # Create more mock rows than DEFAULT_LIMIT
    mock_rows = [
        _MockComparisonModel(
            id=f"id_{i}",
            strategy_name="strat",
            symbol="SYM",
            manual_sharpe=0.1 + i * 0.01,
            ml_sharpe=0.2 + i * 0.01,
            is_significant=False,
            winner="ml",
            spy_sharpe=0.3,
        )
        for i in range(DEFAULT_LIMIT + 5)
    ]

    class _MockResult:
        def scalars(self):
            class _Scalars:
                def all(self):
                    return mock_rows
            return _Scalars()

    class _MockAsyncSession:
        async def execute(self, *_args, **_kwargs):
            return _MockResult()

    async def _mock_get_current_user():
        return User(id=1, username="test")  # type: ignore

    # Patch dependencies
    monkeypatch.setattr("app.api.deps.get_current_user", lambda: _mock_get_current_user())
    monkeypatch.setattr("app.database.get_db", lambda: _MockAsyncSession())

    # Call the endpoint directly
    result = await list_comparisons(db=_MockAsyncSession(), current_user=await _mock_get_current_user())
    assert isinstance(result, list)
    assert len(result) == DEFAULT_LIMIT
    # Verify ordering by checking the first element corresponds to the last inserted (highest created_at)
    # Since we mock without timestamps, we just ensure the list contains the expected IDs.
    expected_ids = [row.id for row in mock_rows[:DEFAULT_LIMIT]]
    actual_ids = [item.id for item in result]
    assert actual_ids == expected_ids