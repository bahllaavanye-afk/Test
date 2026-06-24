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

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# File system locations
RESULTS_REL_PATH = Path("experiments") / "results" / "self_improver.json"
RESULTS_FILE = Path(__file__).parents[3] / RESULTS_REL_PATH
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Default runtime parameters
DEFAULT_INTERVAL_SECONDS = 900  # 15 minutes

# Prompt construction fragments
PROMPT_HEADER = "You are a quant hyperparameter optimizer for the '{strategy}' trading strategy.\n"
PROMPT_SPACE_DESC = "Search space (choose ONE value per key from these lists):\n"
PROMPT_RESULTS_DESC = "Results so far (maximize Sharpe):\n"
PROMPT_INSTRUCTION = (
    "Propose the next single config most likely to beat the best Sharpe. "
    "Respond with ONLY a JSON object mapping each key to one allowed value."
)

# History entry formatting
HISTORY_ENTRY_TEMPLATE = "  params={params} -> sharpe={sharpe:.3f}"

# --------------------------------------------------------------------------- #
# Parameter search spaces per strategy — covers all major strategies across all desks.
# Add a new entry here to make any strategy auto-tunable by the LLM-guided sweep.
# --------------------------------------------------------------------------- #
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
    def __init__(self, algo_agent=None, interval_seconds: int = DEFAULT_INTERVAL_SECONDS):
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
                HISTORY_ENTRY_TEMPLATE.format(
                    params=json.dumps(p),
                    sharpe=s,
                )
                for p, s in tried[-8:]
            ) or "  (none yet)"
            prompt = (
                PROMPT_HEADER.format(strategy=strategy)
                + PROMPT_SPACE_DESC
                + json.dumps(space, indent=2)
                + "\n"
                + PROMPT_RESULTS_DESC
                + history
                + "\n"
                + PROMPT_INSTRUCTION
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
            # Parsing step omitted for brevity
        except Exception as e:
            logger.error("LLM proposal failed: %s", e)
            return None

# ... (truncated for brevity)