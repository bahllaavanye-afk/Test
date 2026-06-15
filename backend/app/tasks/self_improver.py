"""
Self-improvement autoloop. Runs forever, looking for ways to improve the platform:
  1. Take the top-3 strategies from AlgoAgent leaderboard
  2. Sweep their parameters (Optuna-style) — run 5 random configs each
  3. If a config beats the current best Sharpe by > 10%, promote it
  4. Log everything to experiments/results/self_improver.json
  5. Sleep, then repeat
"""
from __future__ import annotations

import asyncio
import json
import random
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.utils.logging import logger

RESULTS_FILE = Path(__file__).parents[3] / "experiments" / "results" / "self_improver.json"
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Parameter search spaces per strategy — covers all major strategies across all desks.
# Add a new entry here to make any strategy auto-tunable by the LLM-guided sweep.
PARAM_SPACES: dict[str, dict[str, list]] = {
    # ── Equities — directional ──────────────────────────────────────────────
    "momentum": {
        "lookback_months": [3, 6, 9, 12],
        "min_score": [0.1, 0.2, 0.3, 0.5],
    },
    "cross_sectional_momentum": {
        "formation_months": [3, 6, 9, 12],
        "holding_months": [1, 3, 6],
        "skip_months": [0, 1],
    },
    "mean_reversion": {
        "bb_period": [10, 20, 30],
        "bb_std": [1.5, 2.0, 2.5],
        "rsi_oversold": [20, 25, 30],
    },
    "rsi_macd": {
        "rsi_period": [9, 14, 21],
        "rsi_oversold": [25, 30, 35],
        "macd_fast": [8, 12, 16],
        "macd_slow": [21, 26, 30],
    },
    "breakout": {
        "high_period": [50, 100, 252],
        "volume_mult": [1.2, 1.5, 2.0],
    },
    "supertrend": {
        "atr_period": [10, 14, 20],
        "multiplier": [2.0, 3.0, 4.0],
    },
    "opening_range_breakout": {
        "range_minutes": [15, 30, 60],
        "volume_mult": [1.2, 1.5, 2.0],
        "stop_loss_pct": [0.5, 1.0, 2.0],
    },
    "vwap_reversion": {
        "std_bands": [1.0, 1.5, 2.0, 2.5],
        "exit_band": [0.1, 0.25, 0.5],
    },
    "pairs_trading": {
        "zscore_entry": [1.5, 2.0, 2.5],
        "zscore_exit": [0.0, 0.25, 0.5],
        "window": [20, 30, 60],
    },
    "pca_stat_arb": {
        "n_components": [3, 5, 10],
        "zscore_entry": [1.5, 2.0, 2.5],
        "lookback": [60, 120, 252],
    },
    "low_volatility": {
        "lookback_days": [21, 63, 126, 252],
        "rebalance_freq": [5, 10, 21],
    },
    "sector_rotation": {
        "momentum_window": [20, 60, 120],
        "rebalance_days": [21, 63],
    },
    "multi_factor_equity": {
        "momentum_weight": [0.2, 0.3, 0.4],
        "value_weight": [0.2, 0.3, 0.4],
        "quality_weight": [0.2, 0.3, 0.4],
    },
    # ── Equities — volatility / options desk ───────────────────────────────
    "vix_mean_reversion": {
        "vix_low": [12, 14, 16],
        "vix_high": [25, 30, 35],
        "hold_days": [3, 5, 10],
    },
    "vol_carry_short": {
        "vix_threshold": [18, 20, 25],
        "holding_days": [5, 10, 21],
    },
    "vol_term_structure": {
        "slope_threshold": [0.02, 0.05, 0.1],
        "term1_days": [30, 45],
        "term2_days": [60, 90],
    },
    "gamma_exposure": {
        "net_gamma_threshold": [0.0, 0.5e9, 1e9],
        "signal_window": [1, 3, 5],
    },
    "skew_arb": {
        "skew_z_threshold": [1.5, 2.0, 2.5],
        "lookback_days": [20, 30, 60],
    },
    "dispersion_trading": {
        "corr_threshold": [0.5, 0.6, 0.7],
        "vol_spread_pct": [0.05, 0.10, 0.15],
    },
    # ── Crypto desk ────────────────────────────────────────────────────────
    "funding_rate_arb": {
        "funding_threshold": [0.0001, 0.0003, 0.0005],
        "hold_hours": [8, 24, 72],
    },
    "crypto_basis_roll": {
        "basis_z_entry": [1.0, 1.5, 2.0],
        "roll_days_before": [1, 3, 7],
    },
    "triangular_arb": {
        "min_profit_bps": [5, 10, 20],
        "max_slippage_bps": [2, 5, 10],
    },
    "dex_cex_arb": {
        "min_spread_bps": [10, 20, 30],
        "gas_budget_usd": [5, 10, 20],
    },
    "stablecoin_depeg_arb": {
        "depeg_threshold_pct": [0.1, 0.2, 0.5],
        "max_hold_hours": [1, 4, 24],
    },
    "btc_eth_stat_arb": {
        "window": [20, 30, 60],
        "zscore_entry": [1.5, 2.0, 2.5],
        "zscore_exit": [0.25, 0.5, 0.75],
    },
    "liquidation_cascade_fade": {
        "liq_vol_mult": [2.0, 3.0, 5.0],
        "fade_window_bars": [3, 5, 10],
    },
    "on_chain_exchange_netflow": {
        "inflow_z_threshold": [1.5, 2.0, 2.5],
        "window_hours": [24, 48, 72],
    },
    # ── Fixed income / macro desk ──────────────────────────────────────────
    "yield_curve_momentum": {
        "curve_window": [20, 60, 120],
        "spread_threshold": [0.1, 0.25, 0.5],
    },
    "bond_equity_rotation": {
        "momentum_window": [20, 60, 120],
        "rebalance_days": [5, 10, 21],
    },
    "tlt_spy_rotation": {
        "momentum_window": [20, 60, 120],
        "volatility_window": [20, 60],
    },
    "duration_momentum": {
        "lookback_days": [20, 60, 120],
        "rebalance_freq": [5, 21],
    },
    # ── Polymarket desk ───────────────────────────────────────────────────
    "poly_binary_arb": {
        "max_spread_pct": [2, 3, 5],
        "min_liquidity": [100, 500, 1000],
    },
    "poly_calibration_arb": {
        "min_edge_pct": [2, 4, 6],
        "kelly_fraction": [0.1, 0.25, 0.5],
    },
    "poly_near_resolution": {
        "days_to_resolution": [1, 3, 7],
        "min_edge_pct": [1, 2, 5],
    },
    "poly_market_maker": {
        "spread_bps": [100, 200, 300],
        "max_position_pct": [5, 10, 20],
    },
}


class SelfImprover:
    def __init__(self, algo_agent=None, interval_seconds: int = 900):
        self.algo_agent = algo_agent
        self.interval_seconds = interval_seconds
        self._best_params: dict[str, dict] = {}    # strategy → best params dict
        self._best_sharpe: dict[str, float] = {}   # strategy → best Sharpe
        self._running = False
        self._iteration = 0

    def _sample_params(self, strategy: str) -> dict:
        """Random sample from PARAM_SPACES."""
        space = PARAM_SPACES.get(strategy, {})
        return {k: random.choice(v) for k, v in space.items()}

    async def _propose_params_llm(
        self, strategy: str, space: dict, tried: list[tuple[dict, float]]
    ) -> dict | None:
        """
        Ask the free-LLM fleet to propose the next promising config given the
        search space and results tried so far. Returns a valid params dict drawn
        strictly from `space`, or None if no LLM keys / invalid response.
        Degrades gracefully to random sampling when no free-LLM keys are set.
        """
        try:
            from app.tasks.free_llm_router import available_keys, call_routed
            if not available_keys():
                return None  # No free-LLM keys configured — caller falls back to random

            history = "\n".join(
                f"  params={json.dumps(p)} -> sharpe={s:.3f}" for p, s in tried[-8:]
            ) or "  (none yet)"
            prompt = (
                f"You are a quant hyperparameter optimizer for the '{strategy}' trading strategy.\n"
                f"Search space (choose ONE value per key from these lists):\n{json.dumps(space, indent=2)}\n"
                f"Results so far (maximize Sharpe):\n{history}\n"
                "Propose the next single config most likely to beat the best Sharpe. "
                "Respond with ONLY a JSON object mapping each key to one allowed value."
            )
            raw = await call_routed(
                [{"role": "user", "content": prompt}], task_type="fast", max_tokens=256
            )
            if not raw:
                return None
            # Extract the JSON object from the response defensively
            start, end = raw.find("{"), raw.rfind("}")
            if start < 0 or end <= start:
                return None
            proposed = json.loads(raw[start:end + 1])
            # Validate: every key present and value is within the allowed list
            cleaned: dict = {}
            for k, allowed in space.items():
                v = proposed.get(k)
                cleaned[k] = v if v in allowed else random.choice(allowed)
            return cleaned
        except Exception as e:
            logger.debug("LLM param proposal failed", strategy=strategy, error=str(e))
            return None

    async def _evaluate(self, strategy: str, symbol: str, params: dict) -> float:
        """Run a quick backtest with the given params. Returns Sharpe."""
        try:
            import pandas as pd
            import yfinance as yf

            from app.backtest.engine import run_backtest
            from app.strategies import STRATEGY_REGISTRY

            end = datetime.now(UTC)
            start = end - timedelta(days=730)
            loop = asyncio.get_running_loop()
            hist = await loop.run_in_executor(
                None,
                lambda: yf.download(symbol, start=str(start.date()), end=str(end.date()),
                                    interval="1d", auto_adjust=True, progress=False)
            )
            if hist is None or len(hist) < 60:
                return 0.0

            close = hist["Close"].squeeze() if hasattr(hist["Close"], "squeeze") else hist["Close"]

            cls = STRATEGY_REGISTRY.get(strategy)
            if not cls:
                return 0.0

            try:
                strat = cls(**params)
            except TypeError:
                strat = cls()  # ignore params if constructor doesn't accept them

            signals = strat.backtest_signals(hist)
            if signals is None or (hasattr(signals, "__len__") and len(signals) < 30):
                return 0.0

            sig_series = signals if hasattr(signals, "values") else pd.Series(signals, index=hist.index)
            metrics = run_backtest(sig_series, close)
            return float(metrics.sharpe)
        except Exception as e:
            logger.debug("Self-improver eval failed", strategy=strategy, error=str(e))
            return 0.0

    async def _improve_strategy(self, strategy: str, symbol: str) -> dict | None:
        """Sweep params for one strategy. Returns promoted result or None."""
        space = PARAM_SPACES.get(strategy)
        if not space:
            return None

        current_best = self._best_sharpe.get(f"{strategy}:{symbol}", 0.0)
        best_iter_sharpe = current_best
        best_iter_params = None

        # 5 configs per iteration. When free-LLM keys are present, the fleet
        # proposes guided configs informed by results so far; otherwise random.
        tried: list[tuple[dict, float]] = []
        for i in range(5):
            params = None
            if i >= 1:  # let the LLM learn from at least one prior result
                params = await self._propose_params_llm(strategy, space, tried)
            if params is None:
                params = self._sample_params(strategy)
            sharpe = await self._evaluate(strategy, symbol, params)
            tried.append((params, sharpe))
            if sharpe > best_iter_sharpe:
                best_iter_sharpe = sharpe
                best_iter_params = params

        # Promote if improvement > 10%
        if best_iter_params and best_iter_sharpe > current_best * 1.10 and best_iter_sharpe > 0.5:
            key = f"{strategy}:{symbol}"
            self._best_params[key] = best_iter_params
            self._best_sharpe[key] = best_iter_sharpe
            promotion = {
                "id": str(uuid.uuid4()),
                "strategy": strategy,
                "symbol": symbol,
                "params": best_iter_params,
                "new_sharpe": round(best_iter_sharpe, 4),
                "previous_sharpe": round(current_best, 4),
                "improvement_pct": round((best_iter_sharpe - current_best) / max(abs(current_best), 0.1), 4),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            self._persist(promotion)
            logger.info("Self-improver PROMOTED params", **promotion)
            return promotion
        return None

    def _persist(self, entry: dict) -> None:
        try:
            history = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else []
            history.append(entry)
            history = history[-300:]
            RESULTS_FILE.write_text(json.dumps(history, indent=2))
        except Exception as exc:
            logger.debug("self_improver persist failed", error=str(exc))

    def get_best_params(self, strategy: str, symbol: str) -> dict | None:
        return self._best_params.get(f"{strategy}:{symbol}")

    def get_all_best(self) -> list[dict]:
        """Return all promoted configs across every strategy/symbol pair, sorted by Sharpe."""
        return sorted(
            [
                {
                    "key": k,
                    "strategy": k.split(":")[0],
                    "symbol": k.split(":", 1)[1],
                    "best_sharpe": self._best_sharpe.get(k, 0.0),
                    "best_params": v,
                }
                for k, v in self._best_params.items()
            ],
            key=lambda x: x["best_sharpe"],
            reverse=True,
        )

    @staticmethod
    def get_param_spaces() -> dict:
        """Return the full parameter search space for all strategies (read-only view)."""
        return PARAM_SPACES

    @staticmethod
    def register_param_space(strategy: str, space: dict) -> None:
        """Add or replace a strategy's search space at runtime (used by agents / API)."""
        PARAM_SPACES[strategy] = space

    def get_history(self) -> list[dict]:
        if not RESULTS_FILE.exists():
            return []
        try:
            return json.loads(RESULTS_FILE.read_text())
        except Exception:
            return []

    async def run(self) -> None:
        self._running = True
        logger.info("SelfImprover started", interval=self.interval_seconds)

        # Cross-desk coverage: (strategy, symbol) pairs cycled each iteration.
        # Each pair is tried once per interval; pairs with no PARAM_SPACES entry are skipped.
        TARGETS = [
            # Equities — trend / momentum
            ("momentum",                "SPY"),
            ("momentum",                "QQQ"),
            ("cross_sectional_momentum", "SPY"),
            ("rsi_macd",               "AAPL"),
            ("rsi_macd",               "MSFT"),
            ("breakout",               "NVDA"),
            ("supertrend",             "SPY"),
            ("supertrend",             "QQQ"),
            ("opening_range_breakout", "SPY"),
            ("vwap_reversion",         "SPY"),
            # Equities — stat arb
            ("pairs_trading",          "SPY"),
            ("pca_stat_arb",           "SPY"),
            ("mean_reversion",         "AAPL"),
            # Equities — factor / low vol
            ("low_volatility",         "SPY"),
            ("sector_rotation",        "SPY"),
            ("multi_factor_equity",    "SPY"),
            # Volatility / options desk
            ("vix_mean_reversion",     "SPY"),
            ("vol_carry_short",        "SPY"),
            ("vol_term_structure",     "SPY"),
            ("skew_arb",               "SPY"),
            ("dispersion_trading",     "SPY"),
            # Crypto desk
            ("funding_rate_arb",       "BTC-USD"),
            ("btc_eth_stat_arb",       "BTC-USD"),
            ("liquidation_cascade_fade", "BTC-USD"),
            ("on_chain_exchange_netflow", "BTC-USD"),
            # Fixed income / macro
            ("yield_curve_momentum",   "TLT"),
            ("bond_equity_rotation",   "TLT"),
            ("tlt_spy_rotation",       "TLT"),
            ("duration_momentum",      "TLT"),
            # Polymarket desk (no yfinance data — evaluator will return 0.0; harmless)
            ("poly_binary_arb",        "POLYMARKET"),
            ("poly_calibration_arb",   "POLYMARKET"),
        ]

        while self._running:
            self._iteration += 1
            logger.info("SelfImprover iteration", n=self._iteration)
            for strategy, symbol in TARGETS:
                try:
                    await self._improve_strategy(strategy, symbol)
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    logger.warning("Self-improver target failed", strategy=strategy, symbol=symbol, error=str(e))
            await asyncio.sleep(self.interval_seconds)

    async def stop(self) -> None:
        self._running = False
