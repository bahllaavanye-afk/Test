"""
Regression tests for the 6 bugs found in the 2026-06-14 parallel principal-engineer
review. Each test fails against the pre-fix code and passes after the fix, so the
suite catches these classes of bug going forward.
"""
from __future__ import annotations

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Bug 1: kelly.size_from_kelly divided by price with no zero guard
#        → ZeroDivisionError on halted / stale quotes (price == 0).
# ─────────────────────────────────────────────────────────────────────────────
class TestKellyPriceGuard:
    def test_zero_price_returns_zero_not_crash(self):
        from app.risk.kelly import size_from_kelly
        # Must not raise ZeroDivisionError; sizing is impossible without a price.
        assert size_from_kelly(100_000, 0.6, 0.05, 0.03, price=0.0) == 0

    def test_negative_price_returns_zero(self):
        from app.risk.kelly import size_from_kelly
        assert size_from_kelly(100_000, 0.6, 0.05, 0.03, price=-1.0) == 0

    def test_valid_price_still_sizes(self):
        from app.risk.kelly import size_from_kelly
        shares = size_from_kelly(100_000, 0.6, 0.05, 0.03, price=100.0)
        assert shares >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Bug 2: kelly_fraction used exact float `avg_loss == 0`; a near-zero avg_loss
#        (e.g. 1e-300) slipped through and produced inf/NaN.
# ─────────────────────────────────────────────────────────────────────────────
class TestKellyNearZeroLoss:
    def test_tiny_avg_loss_returns_finite(self):
        from app.risk.kelly import kelly_fraction
        f = kelly_fraction(win_rate=0.6, avg_win=0.05, avg_loss=1e-300)
        assert f == 0.0  # guarded as effectively zero loss
        assert np.isfinite(f)

    def test_exact_zero_loss_returns_zero(self):
        from app.risk.kelly import kelly_fraction
        assert kelly_fraction(0.6, 0.05, 0.0) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Bug 3: EnsembleModel.forward() mutated self.weights["_gnn"] every call,
#        so the GNN weight accumulated and skewed normalization over time.
# ─────────────────────────────────────────────────────────────────────────────
class _FakeGNN:
    def predict(self, returns_df, node_features):
        return np.array([0.7])


class TestEnsembleWeightsImmutableAcrossCalls:
    def test_forward_does_not_mutate_self_weights(self):
        from app.ml.models.ensemble_model import EnsembleModel
        ens = EnsembleModel(weights={}, gnn_weight=0.5)
        ens.register_gnn(_FakeGNN())
        weights_before = dict(ens.weights)

        x = {"gnn": (None, None)}
        ens.forward(x)
        ens.forward(x)
        ens.forward(x)

        # self.weights must be unchanged — the GNN weight is applied per-call only.
        assert ens.weights == weights_before
        assert "_gnn" not in ens.weights

    def test_empty_predictions_fallback_is_half(self):
        from app.ml.models.ensemble_model import EnsembleModel
        ens = EnsembleModel(weights={})
        out = ens.forward({})  # no models, no gnn
        assert float(np.asarray(out).ravel()[0]) == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Bug 4: LimitFirstExecution re-submitted the FULL quantity as a market order
#        after a poll-timeout cancel, double-executing any partially-filled qty.
# ─────────────────────────────────────────────────────────────────────────────
class _PartialFillBroker:
    """Limit order partially fills 40 of 100, never completes → forces fallback."""
    def __init__(self):
        self.market_orders: list = []

    async def get_quote(self, symbol):
        from types import SimpleNamespace
        return SimpleNamespace(bid=99.0, ask=100.0)

    async def place_order(self, req):
        from types import SimpleNamespace
        if req.order_type == "market":
            self.market_orders.append(req)
            return SimpleNamespace(status="filled", broker_order_id="mkt",
                                   filled_qty=float(req.quantity), avg_fill_price=100.0)
        # limit order — accepted but only partially fills
        return SimpleNamespace(status="new", broker_order_id="lim",
                               filled_qty=40.0, avg_fill_price=100.0)

    async def get_order(self, order_id):
        return {"status": "partially_filled", "filled_qty": 40.0}

    async def cancel_order(self, order_id):
        return {"status": "cancelled"}


class TestLimitFirstNoDoubleExecution:
    @pytest.mark.asyncio
    async def test_only_remainder_market_ordered(self):
        from app.brokers.base import OrderRequest
        from app.execution.limit_first import LimitFirstExecution

        broker = _PartialFillBroker()
        algo = LimitFirstExecution(broker, fallback_seconds=1)
        req = OrderRequest(symbol="AAPL", side="buy", quantity=100.0,
                           order_type="limit", risk_bucket="directional")
        await algo.execute(req)

        # Exactly one market order, for the 60 unfilled shares — not the full 100.
        assert len(broker.market_orders) == 1
        assert float(broker.market_orders[0].quantity) == 60.0


# ─────────────────────────────────────────────────────────────────────────────
# Bug 5/6: transformer registry imports (name-mismatch warnings on every boot).
# ─────────────────────────────────────────────────────────────────────────────
class TestModelRegistryImports:
    def test_transformer_aliases_import_cleanly(self):
        from app.ml.models import TransformerPredictor, iTransformerPredictor
        assert TransformerPredictor is not None
        assert iTransformerPredictor is not None
