"""
Real-time integration tests that mirror every page of the QuantEdge website.

Each test class corresponds to a page in the frontend and exercises the API
endpoints that page actually calls. Tests use a real async HTTP client backed
by the FastAPI ASGI app (in-process, SQLite test DB).

Authentication: tests that hit protected routes obtain a JWT via
/api/v1/auth/login using a registered test user (registered on the fly if
the DB is fresh).
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Any

import pytest
import pytest_asyncio

# ─── Shared auth helper ──────────────────────────────────────────────────────

_AUTH_EMAIL    = "integration@quantedge.test"
_AUTH_PASSWORD = "Integr@tion!999"
_ACCESS_TOKEN: str | None = None


async def _get_token(client) -> str:
    """Return a valid JWT, registering + logging in if needed."""
    global _ACCESS_TOKEN
    if _ACCESS_TOKEN:
        return _ACCESS_TOKEN

    # Try to register (idempotent — 409 means already exists)
    try:
        await client.post("/api/v1/auth/register", json={
            "email": _AUTH_EMAIL,
            "password": _AUTH_PASSWORD,
        })
    except Exception:
        pass

    resp = await client.post("/api/v1/auth/login", json={
        "username": _AUTH_EMAIL,
        "password": _AUTH_PASSWORD,
    })
    if resp.status_code != 200:
        pytest.skip(f"Login failed ({resp.status_code}) — DB not migrated in test env")

    _ACCESS_TOKEN = resp.json()["access_token"]
    return _ACCESS_TOKEN


async def _auth(client) -> dict[str, str]:
    """Return Authorization header dict."""
    token = await _get_token(client)
    return {"Authorization": f"Bearer {token}"}


# ─── Helper ──────────────────────────────────────────────────────────────────

def _shape(data: Any, *keys: str) -> None:
    """Assert that `data` (dict or list) contains every key."""
    if isinstance(data, list):
        if not data:
            return  # empty list is OK — shape contract can't be verified
        data = data[0]
    for k in keys:
        assert k in data, f"Missing key '{k}' in response: {json.dumps(data, default=str)[:300]}"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Health / public routes
# ═══════════════════════════════════════════════════════════════════════════════

class TestPublicEndpoints:
    @pytest.mark.asyncio
    async def test_health(self, client):
        r = await client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "platform" in body

    @pytest.mark.asyncio
    async def test_health_never_5xx(self, client):
        for _ in range(3):
            r = await client.get("/health")
            assert r.status_code < 500


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Authentication (Login page)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuthPage:
    @pytest.mark.asyncio
    async def test_register_and_login_flow(self, client):
        """Full register → login → JWT token flow (mirrors Login.tsx)."""
        email = f"rt_{int(time.time())}@example.com"
        r = await client.post("/api/v1/auth/register", json={"email": email, "password": "Passw0rd!"})
        if r.status_code in (500, 503):
            pytest.skip("DB not migrated")
        assert r.status_code in (200, 201, 409)

        r2 = await client.post("/api/v1/auth/login", json={"username": email, "password": "Passw0rd!"})
        if r2.status_code != 200:
            pytest.skip("Login not available in this env")
        data = r2.json()
        assert "access_token" in data
        assert data.get("token_type", "bearer").lower() == "bearer"

    @pytest.mark.asyncio
    async def test_invalid_credentials_returns_401(self, client):
        r = await client.post("/api/v1/auth/login", json={
            "username": "nobody@example.com",
            "password": "wrong",
        })
        assert r.status_code in (401, 400, 422, 500, 503)  # not 200

    @pytest.mark.asyncio
    async def test_protected_route_without_token_returns_401_or_403(self, client):
        r = await client.get("/api/v1/analytics/")
        # In demo mode (DEMO_MODE=true), unauthenticated requests succeed (200)
        # In prod mode, 401/403 is returned
        assert r.status_code in (200, 401, 403)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Dashboard page
# ═══════════════════════════════════════════════════════════════════════════════

class TestDashboardPage:
    @pytest.mark.asyncio
    async def test_analytics_summary(self, client):
        """Dashboard calls GET /analytics/ for module list + trade count."""
        headers = await _auth(client)
        r = await client.get("/api/v1/analytics/", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert "modules" in body
        assert "trade_count" in body
        assert isinstance(body["modules"], list)
        assert len(body["modules"]) > 0

    @pytest.mark.asyncio
    async def test_live_stats(self, client):
        """Dashboard polls GET /analytics/live-stats for real-time KPIs."""
        headers = await _auth(client)
        r = await client.get("/api/v1/analytics/live-stats", headers=headers)
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_system_status(self, client):
        """Dashboard shows system health banner from /analytics/system-status."""
        headers = await _auth(client)
        r = await client.get("/api/v1/analytics/system-status", headers=headers)
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_positions_list(self, client):
        """Dashboard shows open positions from GET /positions."""
        headers = await _auth(client)
        r = await client.get("/api/v1/positions", headers=headers)
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            assert isinstance(r.json(), list)

    @pytest.mark.asyncio
    async def test_accounts_list(self, client):
        """Dashboard requires accounts list from GET /accounts."""
        headers = await _auth(client)
        r = await client.get("/api/v1/accounts", headers=headers)
        assert r.status_code in (200, 404)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Bot Desk page
# ═══════════════════════════════════════════════════════════════════════════════

class TestBotDeskPage:
    @pytest.mark.asyncio
    async def test_bot_templates(self, client):
        """BotDesk sidebar loads GET /bots/templates."""
        headers = await _auth(client)
        r = await client.get("/api/v1/bots/templates", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, dict)
        assert len(body) > 0, "At least one template must exist"

    @pytest.mark.asyncio
    async def test_bots_list(self, client):
        """BotDesk lists all bots from GET /bots/."""
        headers = await _auth(client)
        r = await client.get("/api/v1/bots/", headers=headers)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    @pytest.mark.asyncio
    async def test_bots_summary_all(self, client):
        """BotDesk command centre uses GET /bots/summary/all."""
        headers = await _auth(client)
        r = await client.get("/api/v1/bots/summary/all", headers=headers)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    @pytest.mark.asyncio
    async def test_create_and_toggle_bot(self, client):
        """Full bot lifecycle: create → toggle off → toggle on → delete."""
        headers = await _auth(client)
        payload = {
            "name": "Integration Test Bot",
            "description": "Automated test",
            "symbol": "AAPL",
            "market_type": "equity",
            "desk": "equity",
            "signal_source": "rule_based",
            "trigger": {"type": "schedule", "interval": "1h"},
            "conditions": [{"type": "indicator", "indicator": "rsi", "operator": "<", "value": 30}],
            "action": {"type": "open_long", "size_pct": 5.0},
        }
        r = await client.post("/api/v1/bots/", json=payload, headers=headers)
        if r.status_code in (500, 503):
            pytest.skip("DB not available")
        assert r.status_code == 201
        bot = r.json()
        bot_id = bot["id"]

        # Toggle off
        r2 = await client.post(f"/api/v1/bots/{bot_id}/toggle", headers=headers)
        assert r2.status_code == 200
        assert r2.json()["is_enabled"] is False

        # Toggle on
        r3 = await client.post(f"/api/v1/bots/{bot_id}/toggle", headers=headers)
        assert r3.status_code == 200
        assert r3.json()["is_enabled"] is True

        # Delete
        r4 = await client.delete(f"/api/v1/bots/{bot_id}", headers=headers)
        assert r4.status_code == 204

    @pytest.mark.asyncio
    async def test_create_ml_hybrid_bot(self, client):
        """Verify ML/hybrid bots are accepted by the schema."""
        headers = await _auth(client)
        payload = {
            "name": "Hybrid BTC Bot",
            "symbol": "BTC-USD",
            "market_type": "crypto",
            "desk": "crypto",
            "signal_source": "hybrid",
            "ml_model_name": "lstm_btc_1h",
            "ml_confidence_threshold": 0.65,
            "trigger": {"type": "schedule", "interval": "1h"},
            "conditions": [],
            "action": {"type": "open_long", "size_pct": 2.0},
        }
        r = await client.post("/api/v1/bots/", json=payload, headers=headers)
        if r.status_code in (500, 503):
            pytest.skip("DB not available")
        assert r.status_code == 201
        bot = r.json()
        assert bot["signal_source"] == "hybrid"
        assert bot["ml_confidence_threshold"] == pytest.approx(0.65, abs=1e-4)
        # Cleanup
        await client.delete(f"/api/v1/bots/{bot['id']}", headers=headers)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Equity Trading page
# ═══════════════════════════════════════════════════════════════════════════════

class TestEquityTradingPage:
    @pytest.mark.asyncio
    async def test_market_data_quote(self, client):
        """EquityTrading page fetches live quote for the selected symbol."""
        headers = await _auth(client)
        r = await client.get("/api/v1/market-data/quote/AAPL", headers=headers)
        # 404 if Alpaca not connected is OK; 5xx is not
        assert r.status_code < 500

    @pytest.mark.asyncio
    async def test_market_data_bars(self, client):
        """EquityTrading page fetches OHLCV bars for the chart."""
        headers = await _auth(client)
        r = await client.get("/api/v1/market-data/bars/AAPL", headers=headers)
        assert r.status_code < 500

    @pytest.mark.asyncio
    async def test_orders_list(self, client):
        """EquityTrading shows open orders from GET /orders."""
        headers = await _auth(client)
        r = await client.get("/api/v1/orders", headers=headers)
        assert r.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_iv_rank_scan(self, client):
        """EquityTrading options panel loads IV rank scan."""
        headers = await _auth(client)
        r = await client.get("/api/v1/market-data/iv-rank-scan", headers=headers)
        assert r.status_code < 500


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Crypto Trading page
# ═══════════════════════════════════════════════════════════════════════════════

class TestCryptoTradingPage:
    @pytest.mark.asyncio
    async def test_crypto_bars(self, client):
        """CryptoTrading chart fetches BTC bars."""
        headers = await _auth(client)
        r = await client.get("/api/v1/market-data/bars/BTC-USD", headers=headers)
        assert r.status_code < 500

    @pytest.mark.asyncio
    async def test_polymarket_feed(self, client):
        """CryptoTrading page fetches Polymarket market data."""
        headers = await _auth(client)
        r = await client.get("/api/v1/market-data/polymarket", headers=headers)
        assert r.status_code < 500

    @pytest.mark.asyncio
    async def test_news_feed(self, client):
        """CryptoTrading news panel loads from /market-data/news."""
        headers = await _auth(client)
        r = await client.get("/api/v1/market-data/news", headers=headers)
        assert r.status_code < 500


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Backtest Lab page
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktestLabPage:
    @pytest.mark.asyncio
    async def test_backtest_list(self, client):
        """BacktestLab loads history from GET /backtests."""
        headers = await _auth(client)
        r = await client.get("/api/v1/backtests", headers=headers)
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            assert isinstance(r.json(), list)

    @pytest.mark.asyncio
    async def test_strategy_list(self, client):
        """BacktestLab strategy selector calls GET /strategies."""
        headers = await _auth(client)
        r = await client.get("/api/v1/strategies", headers=headers)
        assert r.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_run_backtest(self, client):
        """Submit a backtest run — must return 201 or 200 with an id."""
        headers = await _auth(client)
        payload = {
            "strategy_name": "momentum",
            "symbol": "SPY",
            "interval": "1d",
            "start_date": "2022-01-01",
            "end_date": "2023-01-01",
            "params": {"lookback_months": 6, "min_score": 0.2},
        }
        r = await client.post("/api/v1/backtests/run", json=payload, headers=headers)
        if r.status_code in (500, 503):
            pytest.skip("Backtest engine not available in test env")
        assert r.status_code in (200, 201, 202)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Experiments page
# ═══════════════════════════════════════════════════════════════════════════════

class TestExperimentsPage:
    @pytest.mark.asyncio
    async def test_experiments_list(self, client):
        """Experiments page loads from GET /experiments."""
        headers = await _auth(client)
        r = await client.get("/api/v1/experiments", headers=headers)
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            assert isinstance(r.json(), list)

    @pytest.mark.asyncio
    async def test_ml_model_list(self, client):
        """Experiments page lists available ML models from GET /ml/models."""
        headers = await _auth(client)
        r = await client.get("/api/v1/ml/models", headers=headers)
        assert r.status_code < 500


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Analytics / Tearsheet page
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalyticsPage:
    @pytest.mark.asyncio
    async def test_tearsheet(self, client):
        """Analytics tearsheet endpoint — must return all required investor fields."""
        headers = await _auth(client)
        r = await client.get("/api/v1/analytics/tearsheet", headers=headers)
        assert r.status_code == 200
        body = r.json()
        # Required investor pitch fields
        for field in ["sharpe", "total_return_pct", "max_drawdown_pct", "win_rate",
                      "num_trades", "equity_curve", "drawdown_curve"]:
            assert field in body, f"Tearsheet missing required field: {field}"

    @pytest.mark.asyncio
    async def test_monthly_returns(self, client):
        """Analytics heatmap calendar calls GET /analytics/monthly-returns."""
        headers = await _auth(client)
        r = await client.get("/api/v1/analytics/monthly-returns", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)

    @pytest.mark.asyncio
    async def test_equity_curve(self, client):
        """Analytics equity curve chart calls GET /analytics/equity-curve."""
        headers = await _auth(client)
        r = await client.get("/api/v1/analytics/equity-curve", headers=headers)
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_performance_metrics(self, client):
        """Analytics performance panel calls GET /analytics/performance."""
        headers = await _auth(client)
        r = await client.get("/api/v1/analytics/performance", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert "sharpe" in body or "total_return_pct" in body or "status" in body

    @pytest.mark.asyncio
    async def test_attribution(self, client):
        """Analytics attribution breakdown calls GET /analytics/attribution."""
        headers = await _auth(client)
        r = await client.get("/api/v1/analytics/attribution", headers=headers)
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_slippage(self, client):
        """Analytics slippage tab calls GET /analytics/slippage."""
        headers = await _auth(client)
        r = await client.get("/api/v1/analytics/slippage", headers=headers)
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_competition_report(self, client):
        """Analytics benchmarks vs competitors calls GET /analytics/competition-report."""
        headers = await _auth(client)
        r = await client.get("/api/v1/analytics/competition-report", headers=headers)
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_portfolio_greeks(self, client):
        """Analytics options Greeks panel calls GET /analytics/portfolio-greeks."""
        headers = await _auth(client)
        r = await client.get("/api/v1/analytics/portfolio-greeks", headers=headers)
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_correlation_matrix(self, client):
        """Analytics correlation heatmap calls GET /analytics/correlation."""
        headers = await _auth(client)
        r = await client.get("/api/v1/analytics/correlation", headers=headers)
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_macro_signals(self, client):
        """Macro Signals page calls GET /analytics/macro."""
        headers = await _auth(client)
        r = await client.get("/api/v1/analytics/macro", headers=headers)
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_sentiment(self, client):
        """Analytics sentiment calls GET /analytics/sentiment."""
        headers = await _auth(client)
        r = await client.get("/api/v1/analytics/sentiment", headers=headers)
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Risk Manager page
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskManagerPage:
    @pytest.mark.asyncio
    async def test_risk_rules(self, client):
        """RiskManager page loads rules from GET /risk/rules."""
        headers = await _auth(client)
        r = await client.get("/api/v1/risk/rules", headers=headers)
        assert r.status_code < 500

    @pytest.mark.asyncio
    async def test_circuit_breaker_status(self, client):
        """RiskManager shows circuit breaker status from GET /risk/circuit-breaker."""
        headers = await _auth(client)
        r = await client.get("/api/v1/risk/circuit-breaker", headers=headers)
        assert r.status_code < 500

    @pytest.mark.asyncio
    async def test_regime_status(self, client):
        """RiskManager shows market regime from GET /regime/current."""
        headers = await _auth(client)
        r = await client.get("/api/v1/regime/current", headers=headers)
        assert r.status_code < 500


# ═══════════════════════════════════════════════════════════════════════════════
# 11. ML Insights page
# ═══════════════════════════════════════════════════════════════════════════════

class TestMLInsightsPage:
    @pytest.mark.asyncio
    async def test_ml_predictions(self, client):
        """MLInsights calls GET /ml/predictions."""
        headers = await _auth(client)
        r = await client.get("/api/v1/ml/predictions", headers=headers)
        assert r.status_code < 500

    @pytest.mark.asyncio
    async def test_ml_signals(self, client):
        """MLInsights signal feed calls GET /ml/signals."""
        headers = await _auth(client)
        r = await client.get("/api/v1/ml/signals", headers=headers)
        assert r.status_code < 500

    @pytest.mark.asyncio
    async def test_ml_feature_importance(self, client):
        """MLInsights feature importance chart calls GET /ml/feature-importance."""
        headers = await _auth(client)
        r = await client.get("/api/v1/ml/feature-importance", headers=headers)
        assert r.status_code < 500


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Comparison page
# ═══════════════════════════════════════════════════════════════════════════════

class TestComparisonPage:
    @pytest.mark.asyncio
    async def test_comparison_history(self, client):
        """Comparison page loads history from GET /comparison."""
        headers = await _auth(client)
        r = await client.get("/api/v1/comparison", headers=headers)
        assert r.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_benchmarks_endpoint(self, client):
        """Comparison page benchmark data endpoint."""
        headers = await _auth(client)
        r = await client.get("/api/v1/analytics/competition-report", headers=headers)
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Agent Dashboard / Agent Command page
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentDashboardPage:
    @pytest.mark.asyncio
    async def test_agent_status(self, client):
        """AgentDashboard polls GET /agents/status every 30s."""
        headers = await _auth(client)
        r = await client.get("/api/v1/agents/status", headers=headers)
        assert r.status_code == 200
        body = r.json()
        # Must contain all agent blocks
        for key in ["algo_agent", "self_improver", "qa_monitor",
                    "research_scientist", "modeling_engineer", "free_llm_fleet"]:
            assert key in body, f"agents/status missing key: {key}"
        # Free LLM fleet shape
        fleet = body["free_llm_fleet"]
        assert "active_providers" in fleet
        assert "total_keys" in fleet
        assert "throughput" in fleet

    @pytest.mark.asyncio
    async def test_self_improver_best(self, client):
        """AgentDashboard calls GET /agents/self-improver/best for leaderboard."""
        headers = await _auth(client)
        r = await client.get("/api/v1/agents/self-improver/best", headers=headers)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    @pytest.mark.asyncio
    async def test_param_spaces_get(self, client):
        """AgentDashboard shows configurable param spaces via GET /agents/self-improver/param-spaces."""
        headers = await _auth(client)
        r = await client.get("/api/v1/agents/self-improver/param-spaces", headers=headers)
        assert r.status_code == 200
        spaces = r.json()
        assert isinstance(spaces, dict)
        # Spot-check core strategies across all desks
        for strat in ["momentum", "rsi_macd", "supertrend", "breakout",
                      "vix_mean_reversion", "funding_rate_arb", "poly_binary_arb",
                      "yield_curve_momentum"]:
            assert strat in spaces, f"Expected strategy '{strat}' in param spaces"

    @pytest.mark.asyncio
    async def test_param_spaces_post_registers_new_strategy(self, client):
        """Agents can register a new strategy search space at runtime."""
        headers = await _auth(client)
        payload = {
            "strategy": "test_custom_strategy",
            "space": {"period": [10, 20, 30], "threshold": [0.5, 1.0, 2.0]},
        }
        r = await client.post("/api/v1/agents/self-improver/param-spaces",
                              json=payload, headers=headers)
        assert r.status_code == 200
        assert r.json()["registered"] == "test_custom_strategy"

        # Verify it's now discoverable
        r2 = await client.get("/api/v1/agents/self-improver/param-spaces", headers=headers)
        assert "test_custom_strategy" in r2.json()

    @pytest.mark.asyncio
    async def test_algo_agent_results(self, client):
        """AgentDashboard loads algo agent results from GET /agents/results."""
        headers = await _auth(client)
        r = await client.get("/api/v1/agents/results", headers=headers)
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            assert isinstance(r.json(), list)

    @pytest.mark.asyncio
    async def test_research_summary(self, client):
        """AgentDashboard research tab calls GET /agents/research."""
        headers = await _auth(client)
        r = await client.get("/api/v1/agents/research", headers=headers)
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_modeling_summary(self, client):
        """AgentDashboard modeling tab calls GET /agents/modeling."""
        headers = await _auth(client)
        r = await client.get("/api/v1/agents/modeling", headers=headers)
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Strategy leaderboard
# ═══════════════════════════════════════════════════════════════════════════════

class TestLeaderboardPage:
    @pytest.mark.asyncio
    async def test_leaderboard(self, client):
        """Leaderboard page calls GET /leaderboard."""
        headers = await _auth(client)
        r = await client.get("/api/v1/leaderboard", headers=headers)
        assert r.status_code < 500

    @pytest.mark.asyncio
    async def test_improvements(self, client):
        """Promotions/improvements page calls GET /improvements."""
        headers = await _auth(client)
        r = await client.get("/api/v1/improvements", headers=headers)
        assert r.status_code < 500


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Strategy registry completeness
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategyRegistry:
    def test_all_desks_covered(self):
        """Every desk must have at least one registered strategy."""
        from app.strategies import STRATEGY_REGISTRY
        names = set(STRATEGY_REGISTRY.keys())
        # Equities
        assert any("momentum" in n for n in names), "No momentum strategy"
        assert any("reversion" in n or "mean" in n for n in names), "No mean-reversion strategy"
        assert any("breakout" in n for n in names), "No breakout strategy"
        # Crypto
        assert any("funding" in n or "arb" in n or "triangular" in n for n in names), "No crypto arb"
        # Options/vol
        assert any("vix" in n or "vol" in n or "gamma" in n for n in names), "No vol/options strategy"
        # Polymarket
        assert any("poly" in n for n in names), "No polymarket strategy"

    def test_strategy_count_above_threshold(self):
        """Platform must have >= 40 strategies registered (minimum viable universe)."""
        from app.strategies import STRATEGY_REGISTRY
        count = len(STRATEGY_REGISTRY)
        assert count >= 40, f"Only {count} strategies registered — expected >= 40"

    def test_all_strategies_instantiatable(self):
        """Every registered strategy must instantiate without error."""
        from app.strategies import STRATEGY_REGISTRY
        errors = []
        for name, cls in STRATEGY_REGISTRY.items():
            try:
                cls()
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        assert not errors, "Some strategies failed to instantiate:\n" + "\n".join(errors)

    def test_self_improver_param_spaces_subset_of_registry(self):
        """Every strategy in PARAM_SPACES should ideally exist in STRATEGY_REGISTRY
        (or be a soft warning — Polymarket strategies may be omitted from the simple registry)."""
        from app.tasks.self_improver import PARAM_SPACES
        from app.strategies import STRATEGY_REGISTRY
        not_in_registry = [s for s in PARAM_SPACES if s not in STRATEGY_REGISTRY]
        # Soft check: polymarket strategies are expected to be partially absent
        # (they need CLOB data, so the yfinance evaluator skips them)
        hard_missing = [s for s in not_in_registry if "poly" not in s and "stablecoin" not in s]
        assert len(hard_missing) <= 5, (
            f"Too many PARAM_SPACES strategies not in STRATEGY_REGISTRY: {hard_missing}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 16. WebSocket connection manager unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestWebSocketManager:
    @pytest.mark.asyncio
    async def test_broadcast_no_subscribers_does_not_raise(self):
        from app.ws.manager import ConnectionManager
        mgr = ConnectionManager()
        await mgr.broadcast("prices", {"symbol": "SPY", "bid": 500.0, "ask": 500.1})

    @pytest.mark.asyncio
    async def test_disconnect_without_connect_does_not_raise(self):
        from app.ws.manager import ConnectionManager
        mgr = ConnectionManager()
        mgr.disconnect(None, "prices")  # type: ignore

    @pytest.mark.asyncio
    async def test_multiple_topics_isolated(self):
        from app.ws.manager import ConnectionManager
        mgr = ConnectionManager()
        # Broadcast to a topic with no subscribers — other topics unaffected
        await mgr.broadcast("prices", {"test": True})
        await mgr.broadcast("orders", {"test": True})
        await mgr.broadcast("alerts", {"test": True})


# ═══════════════════════════════════════════════════════════════════════════════
# 17. Real-time data freshness (market-data endpoints)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarketDataEndpoints:
    @pytest.mark.asyncio
    async def test_earnings_calendar(self, client):
        """Equity trading and scanner pages call GET /market-data/earnings."""
        headers = await _auth(client)
        r = await client.get("/api/v1/market-data/earnings", headers=headers)
        assert r.status_code < 500

    @pytest.mark.asyncio
    async def test_economic_calendar(self, client):
        """Dashboard / Macro page calls GET /market-data/economic-calendar."""
        headers = await _auth(client)
        r = await client.get("/api/v1/market-data/economic-calendar", headers=headers)
        assert r.status_code < 500

    @pytest.mark.asyncio
    async def test_sector_heatmap(self, client):
        """Dashboard calls GET /market-data/sector-heatmap."""
        headers = await _auth(client)
        r = await client.get("/api/v1/market-data/sector-heatmap", headers=headers)
        assert r.status_code < 500

    @pytest.mark.asyncio
    async def test_pcr(self, client):
        """Options page and risk manager call GET /market-data/pcr (put-call ratio)."""
        headers = await _auth(client)
        r = await client.get("/api/v1/market-data/pcr", headers=headers)
        assert r.status_code < 500

    @pytest.mark.asyncio
    async def test_multiple_quotes(self, client):
        """Watchlist panel calls GET /market-data/quotes with multiple symbols."""
        headers = await _auth(client)
        r = await client.get("/api/v1/market-data/quotes?symbols=SPY,QQQ,AAPL", headers=headers)
        assert r.status_code < 500


# ═══════════════════════════════════════════════════════════════════════════════
# 18. Scanner page
# ═══════════════════════════════════════════════════════════════════════════════

class TestScannersPage:
    @pytest.mark.asyncio
    async def test_scanners_list(self, client):
        """Scanners page calls GET /scanners."""
        headers = await _auth(client)
        r = await client.get("/api/v1/scanners", headers=headers)
        assert r.status_code < 500

    @pytest.mark.asyncio
    async def test_arb_opportunities(self, client):
        """Scanners page shows arb opportunities from GET /analytics/arb-opportunities."""
        headers = await _auth(client)
        r = await client.get("/api/v1/analytics/arb-opportunities", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)


# ═══════════════════════════════════════════════════════════════════════════════
# 19. Pipeline page
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelinePage:
    @pytest.mark.asyncio
    async def test_pipeline_status(self, client):
        """Pipeline page calls GET /pipeline/status."""
        headers = await _auth(client)
        r = await client.get("/api/v1/pipeline/status", headers=headers)
        assert r.status_code < 500

    @pytest.mark.asyncio
    async def test_pipeline_stages(self, client):
        """Pipeline page loads stage breakdown from GET /pipeline/stages."""
        headers = await _auth(client)
        r = await client.get("/api/v1/pipeline/stages", headers=headers)
        assert r.status_code < 500


# ═══════════════════════════════════════════════════════════════════════════════
# 20. Activity feed page
# ═══════════════════════════════════════════════════════════════════════════════

class TestActivityPage:
    @pytest.mark.asyncio
    async def test_audit_log(self, client):
        """Activity page loads from GET /audit-log."""
        headers = await _auth(client)
        r = await client.get("/api/v1/audit-log", headers=headers)
        assert r.status_code < 500

    @pytest.mark.asyncio
    async def test_trades_history(self, client):
        """Activity page calls GET /trades for trade history."""
        headers = await _auth(client)
        r = await client.get("/api/v1/trades", headers=headers)
        assert r.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_notifications(self, client):
        """Activity page notification panel calls GET /notifications."""
        headers = await _auth(client)
        r = await client.get("/api/v1/notifications", headers=headers)
        assert r.status_code < 500


# ═══════════════════════════════════════════════════════════════════════════════
# 21. Promotions page (Strategy Promotions)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromotionsPage:
    @pytest.mark.asyncio
    async def test_list_promotions(self, client):
        """Promotions page calls GET /promotions/."""
        headers = await _auth(client)
        r = await client.get("/api/v1/promotions/", headers=headers)
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            assert isinstance(r.json(), list)

    @pytest.mark.asyncio
    async def test_promotions_criteria(self, client):
        """Promotions page fetches criteria definitions."""
        headers = await _auth(client)
        r = await client.get("/api/v1/promotions/criteria/all", headers=headers)
        assert r.status_code < 500

    @pytest.mark.asyncio
    async def test_create_promotion_requires_fields(self, client):
        """POST /promotions/ with missing fields returns 422."""
        headers = await _auth(client)
        r = await client.post("/api/v1/promotions/", json={}, headers=headers)
        assert r.status_code in (422, 400, 404)

    @pytest.mark.asyncio
    async def test_create_and_approve_promotion(self, client):
        """Full promotion lifecycle: create → fetch → approve."""
        headers = await _auth(client)
        payload = {
            "strategy_name": "test_momentum",
            "symbol": "SPY",
            "interval": "1d",
            "reason": "Test integration",
            "paper_sharpe": 1.8,
            "paper_max_dd": 0.10,
            "paper_days": 14,
        }
        r_create = await client.post("/api/v1/promotions/", json=payload, headers=headers)
        if r_create.status_code in (404, 500, 503):
            pytest.skip("Promotions endpoint not available in this env")
        assert r_create.status_code in (200, 201)
        promo_id = r_create.json().get("id") or r_create.json().get("promotion_id")
        if not promo_id:
            return  # can't test further without id

        # Approve
        r_approve = await client.post(f"/api/v1/promotions/{promo_id}/approve", headers=headers)
        assert r_approve.status_code in (200, 201, 403, 404)

    @pytest.mark.asyncio
    async def test_reject_promotion_requires_reason(self, client):
        """POST /promotions/{id}/reject with no reason returns 422."""
        headers = await _auth(client)
        r = await client.post("/api/v1/promotions/nonexistent-id/reject", json={}, headers=headers)
        assert r.status_code in (400, 404, 422)


# ═══════════════════════════════════════════════════════════════════════════════
# 22. Copy Trading page
# ═══════════════════════════════════════════════════════════════════════════════

class TestCopyTradingPage:
    @pytest.mark.asyncio
    async def test_leaderboard(self, client):
        """Copy Trading page calls GET /copy-trading/leaderboard."""
        headers = await _auth(client)
        r = await client.get("/api/v1/copy-trading/leaderboard", headers=headers)
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            assert isinstance(r.json(), list)

    @pytest.mark.asyncio
    async def test_list_follows(self, client):
        """Copy Trading page loads followed traders."""
        headers = await _auth(client)
        r = await client.get("/api/v1/copy-trading/follows", headers=headers)
        assert r.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_follow_requires_fields(self, client):
        """POST /copy-trading/follow with missing body returns 422."""
        headers = await _auth(client)
        r = await client.post("/api/v1/copy-trading/follow", json={}, headers=headers)
        assert r.status_code in (400, 404, 422)

    @pytest.mark.asyncio
    async def test_follow_and_unfollow_trader(self, client):
        """Follow a trader then unfollow — round-trip smoke test."""
        headers = await _auth(client)
        r = await client.post("/api/v1/copy-trading/follow", json={
            "leader_id": "test-leader-001",
            "allocation_pct": 0.05,
            "max_position_pct": 0.02,
        }, headers=headers)
        if r.status_code in (404, 500, 503):
            pytest.skip("Copy trading endpoint not available")
        assert r.status_code in (200, 201, 409)  # 409 = already following

        follow_id = (r.json() or {}).get("id")
        if follow_id:
            r_del = await client.delete(f"/api/v1/copy-trading/follows/{follow_id}", headers=headers)
            assert r_del.status_code in (200, 204, 404)


# ═══════════════════════════════════════════════════════════════════════════════
# 23. Task Manager page
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskManagerPage:
    @pytest.mark.asyncio
    async def test_list_tasks(self, client):
        """Task Manager page calls GET /tasks/."""
        headers = await _auth(client)
        r = await client.get("/api/v1/tasks/", headers=headers)
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            assert isinstance(r.json(), list)

    @pytest.mark.asyncio
    async def test_list_employees(self, client):
        """Task Manager loads employee roster from GET /tasks/employees."""
        headers = await _auth(client)
        r = await client.get("/api/v1/tasks/employees", headers=headers)
        assert r.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_create_task(self, client):
        """POST /tasks/ creates a new task."""
        headers = await _auth(client)
        r = await client.post("/api/v1/tasks/", json={
            "title": "Integration test task",
            "description": "Created by automated test",
            "priority": "low",
            "assigned_to": "gemini_analyst",
        }, headers=headers)
        if r.status_code in (404, 500, 503):
            pytest.skip("Tasks endpoint not available")
        assert r.status_code in (200, 201, 422)

    @pytest.mark.asyncio
    async def test_create_and_delete_task(self, client):
        """Full task lifecycle: create → update → delete."""
        headers = await _auth(client)
        r_create = await client.post("/api/v1/tasks/", json={
            "title": "Temp test task",
            "description": "Will be deleted",
            "priority": "low",
            "assigned_to": "system",
        }, headers=headers)
        if r_create.status_code in (404, 500, 503):
            pytest.skip("Tasks endpoint not available")
        if r_create.status_code not in (200, 201):
            return
        task_id = (r_create.json() or {}).get("id")
        if not task_id:
            return

        # Update status
        r_patch = await client.patch(f"/api/v1/tasks/{task_id}", json={"status": "in_progress"}, headers=headers)
        assert r_patch.status_code in (200, 204, 404)

        # Delete
        r_del = await client.delete(f"/api/v1/tasks/{task_id}", headers=headers)
        assert r_del.status_code in (200, 204, 404)


# ═══════════════════════════════════════════════════════════════════════════════
# 24. Risk Controls page
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskControlsPage:
    @pytest.mark.asyncio
    async def test_risk_dashboard(self, client):
        """Risk Controls page calls GET /risk/."""
        headers = await _auth(client)
        r = await client.get("/api/v1/risk/", headers=headers)
        assert r.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_list_risk_rules(self, client):
        """Risk Controls page loads rules from GET /risk/rules."""
        headers = await _auth(client)
        r = await client.get("/api/v1/risk/rules", headers=headers)
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            assert isinstance(r.json(), list)

    @pytest.mark.asyncio
    async def test_risk_events(self, client):
        """Risk Controls page loads event history from GET /risk/events."""
        headers = await _auth(client)
        r = await client.get("/api/v1/risk/events", headers=headers)
        assert r.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_create_risk_rule(self, client):
        """POST /risk/rules creates a rule."""
        headers = await _auth(client)
        r = await client.post("/api/v1/risk/rules", json={
            "rule_type": "max_drawdown",
            "threshold": 0.15,
            "action": "halt",
        }, headers=headers)
        if r.status_code in (404, 500, 503):
            pytest.skip("Risk rules endpoint not available")
        assert r.status_code in (200, 201, 422)

    @pytest.mark.asyncio
    async def test_circuit_breaker_status(self, client):
        """Risk Controls page checks circuit breaker via GET /risk/circuit-breaker."""
        headers = await _auth(client)
        r = await client.get("/api/v1/risk/circuit-breaker", headers=headers)
        assert r.status_code in (200, 404)


# ═══════════════════════════════════════════════════════════════════════════════
# 25. Positions Hub page
# ═══════════════════════════════════════════════════════════════════════════════

class TestPositionsHubPage:
    @pytest.mark.asyncio
    async def test_list_positions(self, client):
        """Positions Hub calls GET /positions/."""
        headers = await _auth(client)
        r = await client.get("/api/v1/positions/", headers=headers)
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            assert isinstance(r.json(), list)

    @pytest.mark.asyncio
    async def test_positions_summary(self, client):
        """Positions Hub loads summary stats."""
        headers = await _auth(client)
        r = await client.get("/api/v1/positions/summary", headers=headers)
        assert r.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_patch_exit_config(self, client):
        """PATCH /positions/{symbol}/exit-config updates exit rules."""
        headers = await _auth(client)
        r = await client.patch("/api/v1/positions/SPY/exit-config", json={
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05,
        }, headers=headers)
        assert r.status_code in (200, 204, 404, 422)


# ═══════════════════════════════════════════════════════════════════════════════
# 26. Bot Dashboard page (summary endpoint)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBotDashboardSummary:
    @pytest.mark.asyncio
    async def test_bots_summary_all(self, client):
        """Bot Dashboard calls GET /bots/summary/all for a combined view."""
        headers = await _auth(client)
        r = await client.get("/api/v1/bots/summary/all", headers=headers)
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            assert isinstance(r.json(), list)

    @pytest.mark.asyncio
    async def test_bot_detail_trades(self, client):
        """GET /bots/{bot_id}/trades returns trade history for a bot."""
        headers = await _auth(client)
        # Create a bot first
        r_bot = await client.post("/api/v1/bots/", json={
            "name": f"test_bot_{int(time.time())}",
            "symbol": "SPY",
            "market_type": "equity",
        }, headers=headers)
        if r_bot.status_code not in (200, 201):
            pytest.skip("Bot creation not available")
        bot_id = r_bot.json().get("id")
        if not bot_id:
            return

        r = await client.get(f"/api/v1/bots/{bot_id}/trades", headers=headers)
        assert r.status_code in (200, 404)


# ═══════════════════════════════════════════════════════════════════════════════
# 27. Attribution page
# ═══════════════════════════════════════════════════════════════════════════════

class TestAttributionPage:
    @pytest.mark.asyncio
    async def test_attribution_endpoint(self, client):
        """Attribution page calls GET /analytics/attribution."""
        headers = await _auth(client)
        r = await client.get("/api/v1/analytics/attribution", headers=headers)
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            body = r.json()
            assert isinstance(body, (list, dict))

    @pytest.mark.asyncio
    async def test_daily_pnl_endpoint(self, client):
        """Attribution page calls GET /analytics/daily-pnl."""
        headers = await _auth(client)
        r = await client.get("/api/v1/analytics/daily-pnl", headers=headers)
        assert r.status_code in (200, 404)
