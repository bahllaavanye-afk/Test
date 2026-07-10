"""
PPO-based Reinforcement Learning Execution Agent.

Learns to minimize implementation shortfall by choosing execution actions
at each time step during order execution.

State (5 dims):
  [remaining_fraction, elapsed_fraction, spread_bps_norm,
   volume_ratio, book_imbalance]

Actions (4 discrete):
  0 = wait           (no fill this step)
  1 = limit_inside   (post limit at bid+1bps / ask-1bps)
  2 = limit_best     (post limit at best bid/ask)
  3 = market         (aggressive fill immediately)

Reward:
  -slippage_bps per fill step (negative slippage = agent is penalized for bad fills)
  +completion_bonus if fully filled before deadline

Falls back to TWAP if no trained model found.
Policy weights are saved to: backend/models_artifacts/rl_exec_policy.pt
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    F = None      # type: ignore[assignment]

from app.utils.logging import logger
from app.brokers.base import OrderRequest

_MODEL_PATH = Path("backend/models_artifacts/rl_exec_policy.pt")
_STATE_DIM = 5
_ACTION_DIM = 4
_ACTION_NAMES = ["wait", "limit_inside", "limit_best", "market"]


class ExecutionPolicy(nn.Module):
    """
    Shared actor-critic MLP.
    Input: state_dim=5
    Outputs: action_logits (4,) and state_value (1,)
    """

    def __init__(self, state_dim: int = _STATE_DIM, hidden: int = 64, n_actions: int = _ACTION_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value

    def act(self, state: np.ndarray) -> tuple[int, float]:
        """Sample action and return (action_idx, log_prob)."""
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return int(action.item()), float(dist.log_prob(action).item())


class RLExecAgent:
    """
    RL execution agent.
    Loads a pre-trained ExecutionPolicy if available; otherwise uses a heuristic
    based on remaining time and spread.
    """

    def __init__(self):
        self.policy = ExecutionPolicy()
        self._trained = False
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        """Load policy weights from disk if the model file exists."""
        path = _MODEL_PATH
        if path.exists():
            try:
                state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
                self.policy.load_state_dict(state_dict)
                self.policy.eval()
                self._trained = True
                logger.info("RLExecAgent: loaded policy from %s", path)
            except Exception as e:  # pragma: no cover
                logger.warning("RLExecAgent: failed to load policy (%s), using heuristic", e)

    def select_action(self, state: dict) -> str:
        """
        Select execution action given the current state.

        Args:
            state: dict with keys:
                remaining_fraction (0-1): fraction of order remaining
                elapsed_fraction   (0-1): fraction of time window elapsed
                spread_bps         (float): current bid-ask spread in bps (normalised /50)
                volume_ratio       (float): current volume / avg volume
                book_imbalance     (float): LOB imbalance (-1 to 1)

        Returns:
            One of: 'wait', 'limit_inside', 'limit_best', 'market'
        """
        arr = np.array(
            [
                float(state.get("remaining_fraction", 1.0)),
                float(state.get("elapsed_fraction", 0.0)),
                float(state.get("spread_bps", 5.0)) / 50.0,  # normalise to ~[0,1]
                float(state.get("volume_ratio", 1.0)),
                float(state.get("book_imbalance", 0.0)),
            ],
            dtype=np.float32,
        )

        # Clip to valid range
        arr = np.clip(arr, -2.0, 2.0)

        if self._trained:
            action_idx, _ = self.policy.act(arr)
            return _ACTION_NAMES[action_idx]

        # Heuristic fallback: aggressive when time is running out or large spread signals urgency
        remaining = arr[0]
        elapsed = arr[1]
        spread_norm = arr[2]

        if elapsed > 0.85 or remaining < 0.05:
            return "market"
        if spread_norm < 0.2:  # tight spread → post limit
            return "limit_best"
        if elapsed > 0.5:
            return "limit_inside"
        return "wait"

    def save(self, path: Path | None = None) -> None:
        p = path or _MODEL_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), str(p))


_shared_agent: RLExecAgent | None = None


def get_rl_agent() -> RLExecAgent:
    """Return or create the singleton RL execution agent."""
    global _shared_agent
    if _shared_agent is None:
        _shared_agent = RLExecAgent()
    return _shared_agent


class RLExecution:
    """
    Drop-in replacement for TWAPExecution / LimitFirstExecution.

    Uses the RL agent to decide execution actions at each step.
    Falls back to market order after fallback_seconds if unfilled.
    """

    def __init__(
        self,
        broker,
        agent: RLExecAgent | None = None,
        step_seconds: int = 30,
        fallback_seconds: int = 300,
    ):
        self.broker = broker
        self.agent = agent or get_rl_agent()
        self.step_seconds = step_seconds
        self.fallback_seconds = fallback_seconds

    async def execute(self, request, signal_price: float | None = None) -> list[dict]:
        """
        Execute an order using RL policy.

        Returns list of fill dicts: [{qty, price, algo, slippage_bps}]
        """
        total_qty = float(request.quantity)
        remaining = total_qty
        fills: list[dict] = []
        start_time = time.monotonic()
        max_steps = max(1, self.fallback_seconds // self.step_seconds)
        step = 0

        while remaining > 0.01 and step < max_steps:
            elapsed = time.monotonic() - start_time
            elapsed_frac = min(1.0, elapsed / self.fallback_seconds)
            remaining_frac = remaining / total_qty

            state = self._build_state(remaining_frac, elapsed_frac)
            action = self.agent.select_action(state)

            if action == "wait":
                await asyncio.sleep(self.step_seconds)
                step += 1
                continue

            fill_qty = self._determine_fill_quantity(action, remaining, total_qty)
            sub_order = self._create_sub_order(request, action, fill_qty)

            try:
                response = await self.broker.submit_order(sub_order)  # type: ignore[attr-defined]
                self._handle_response(response, fill_qty, fills, remaining, total_qty)
                remaining -= fill_qty
            except Exception as exc:  # pragma: no cover
                logger.warning("RLExecution: order submission failed (%s)", exc)

            step += 1

        # If still not fully filled, force market fill for remaining quantity
        if remaining > 0.01:
            forced_order = OrderRequest(
                symbol=request.symbol,
                side=request.side,
                order_type="market",
                quantity=remaining,
                limit_price=None,
                account_id=request.account_id,
                execution_algo="rl_market_fallback",
            )
            try:
                response = await self.broker.submit_order(forced_order)  # type: ignore[attr-defined]
                self._handle_response(response, remaining, fills, 0.0, total_qty)
            except Exception as exc:  # pragma: no cover
                logger.error("RLExecution: forced market order failed (%s)", exc)

        return fills

    # --------------------------------------------------------------------- #
    # Helper methods – extracted to improve readability
    # --------------------------------------------------------------------- #

    def _build_state(self, remaining_frac: float, elapsed_frac: float) -> dict:
        """Construct the state dictionary expected by the RL agent."""
        return {
            "remaining_fraction": remaining_frac,
            "elapsed_fraction": elapsed_frac,
            "spread_bps": 5.0,  # placeholder; real implementation would use live LOB data
            "volume_ratio": 1.0,
            "book_imbalance": 0.0,
        }

    def _determine_fill_quantity(self, action: str, remaining: float, total_qty: float) -> float:
        """
        Decide how much quantity to request for the sub‑order.

        Market orders take the full remaining amount; limit orders cap at 15 % of the
        original order size to avoid excessive exposure in a single step.
        """
        if action == "market":
            return remaining
        return min(remaining, total_qty * 0.15)

    def _create_sub_order(self, parent_req, action: str, qty: float) -> OrderRequest:
        """Create an OrderRequest that mirrors the parent request but respects the chosen action."""
        return OrderRequest(
            symbol=parent_req.symbol,
            side=parent_req.side,
            order_type="market" if action == "market" else "limit",
            quantity=qty,
            limit_price=parent_req.limit_price if action in ("limit_inside", "limit_best") else None,
            account_id=parent_req.account_id,
            execution_algo=f"rl_{action}",
        )

    def _handle_response(
        self,
        response,
        fill_qty: float,
        fills: list[dict],
        remaining_before: float,
        total_qty: float,
    ) -> None:
        """
        Record fill information from a broker response.

        The response is expected to contain at least:
            - filled_quantity
            - price
            - slippage_bps (optional)

        If the response lacks these attributes, a generic fill record is created.
        """
        try:
            filled = getattr(response, "filled_quantity", fill_qty)
            price = getattr(response, "price", 0.0)
            slippage = getattr(response, "slippage_bps", 0.0)
        except Exception:  # pragma: no cover
            filled = fill_qty
            price = 0.0
            slippage = 0.0

        fills.append(
            {
                "qty": float(filled),
                "price": float(price),
                "algo": getattr(response, "execution_algo", "rl_unknown"),
                "slippage_bps": float(slippage),
            }
        )
        logger.debug(
            "RLExecution: recorded fill qty=%.4f price=%.4f slippage=%.2f",
            filled,
            price,
            slippage,
        )