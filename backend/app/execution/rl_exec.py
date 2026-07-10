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
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from app.utils.logging import logger

_MODEL_PATH = Path("backend/models_artifacts/rl_exec_policy.pt")
_STATE_DIM = 5
_ACTION_DIM = 4
_ACTION_NAMES = ["wait", "limit_inside", "limit_best", "market"]


class ExecutionPolicy(nn.Module):
    """
    Shared actor‑critic multilayer perceptron.

    The network maps a five‑dimensional state vector to:
    * ``action_logits`` – unnormalised scores for each of the four discrete actions.
    * ``state_value`` – a scalar value estimate used by PPO during training.

    Parameters
    ----------
    state_dim: int, default ``_STATE_DIM``
        Dimensionality of the input state.
    hidden: int, default ``64``
        Number of hidden units in each hidden layer.
    n_actions: int, default ``_ACTION_DIM``
        Number of discrete actions the policy can emit.
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
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(batch, state_dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(logits, value)`` where ``logits`` has shape ``(batch, n_actions)`` and
            ``value`` has shape ``(batch, 1)``.
        """
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value

    def act(self, state: np.ndarray) -> tuple[int, float]:
        """
        Sample an action from the policy given a raw state vector.

        Parameters
        ----------
        state: np.ndarray
            One‑dimensional array of length ``_STATE_DIM`` containing the current
            market/execution state.

        Returns
        -------
        tuple[int, float]
            ``(action_index, log_probability)`` where ``action_index`` is an integer in
            ``[0, _ACTION_DIM)`` and ``log_probability`` is the natural logarithm of the
            probability of the sampled action.
        """
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return int(action.item()), float(dist.log_prob(action).item())


class RLExecAgent:
    """
    Reinforcement‑learning execution agent.

    The agent tries to load a pre‑trained :class:`ExecutionPolicy`.  If the model file
    is not present or cannot be loaded, a simple heuristic fallback is used.
    """

    def __init__(self) -> None:
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

    def select_action(self, state: Dict[str, Any]) -> str:
        """
        Choose an execution action based on the current market state.

        Parameters
        ----------
        state: dict
            Mapping with the following keys:
            * ``remaining_fraction`` (float, 0‑1) – fraction of the order still to fill.
            * ``elapsed_fraction``   (float, 0‑1) – fraction of the allotted time elapsed.
            * ``spread_bps``         (float) – current spread in basis points (normalised /50).
            * ``volume_ratio``       (float) – current volume divided by average volume.
            * ``book_imbalance``     (float, -1‑1) – limit‑order‑book imbalance.

        Returns
        -------
        str
            One of ``'wait'``, ``'limit_inside'``, ``'limit_best'`` or ``'market'``.
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

        # Clip to a reasonable range to avoid pathological inputs.
        arr = np.clip(arr, -2.0, 2.0)

        if self._trained:
            action_idx, _ = self.policy.act(arr)
            return _ACTION_NAMES[action_idx]

        # Heuristic fallback: become aggressive when time is short or spread is wide.
        remaining = arr[0]
        elapsed = arr[1]
        spread_norm = arr[2]

        if elapsed > 0.85 or remaining < 0.05:
            return "market"
        if spread_norm < 0.2:
            return "limit_best"
        if elapsed > 0.5:
            return "limit_inside"
        return "wait"

    def save(self, path: Optional[Path] = None) -> None:
        """
        Persist the current policy weights to disk.

        Parameters
        ----------
        path: pathlib.Path, optional
            Destination path.  If omitted, the default ``_MODEL_PATH`` is used.
        """
        p = path or _MODEL_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), str(p))


_shared_agent: RLExecAgent | None = None


def get_rl_agent() -> RLExecAgent:
    """Return the singleton RL execution agent, creating it on first use."""
    global _shared_agent
    if _shared_agent is None:
        _shared_agent = RLExecAgent()
    return _shared_agent


class RLExecution:
    """
    Execution algorithm that delegates order‑placement decisions to a reinforcement‑learning
    agent.

    The algorithm runs in discrete steps (``step_seconds``) until either the order is fully
    filled or ``fallback_seconds`` elapse, at which point any remaining quantity is submitted
    as a market order.
    """

    def __init__(
        self,
        broker: Any,
        agent: RLExecAgent | None = None,
        step_seconds: int = 30,
        fallback_seconds: int = 300,
    ) -> None:
        """
        Parameters
        ----------
        broker: Any
            Object responsible for communicating with the market/exchange.
        agent: RLExecAgent, optional
            Pre‑instantiated RL agent.  If omitted, the global singleton is used.
        step_seconds: int, default ``30``
            Length of each decision interval.
        fallback_seconds: int, default ``300``
            Maximum time allowed before forcing a market fill.
        """
        self.broker = broker
        self.agent = agent or get_rl_agent()
        self.step_seconds = step_seconds
        self.fallback_seconds = fallback_seconds

    async def execute(self, request: Any, signal_price: float | None = None) -> List[Dict[str, Any]]:
        """
        Run the RL‑driven execution loop for a single order request.

        Parameters
        ----------
        request: Any
            An order request object that provides at least ``symbol``, ``side``,
            ``quantity``, ``limit_price`` (optional), ``account_id`` and ``execution_algo``.
        signal_price: float, optional
            Reference price that could be used by more sophisticated strategies; currently
            unused.

        Returns
        -------
        list[dict]
            A list of fill dictionaries each containing ``qty``, ``price``, ``algo`` and
            ``slippage_bps``.
        """
        from app.brokers.base import OrderRequest  # Local import to avoid circular dependencies.

        total_qty = float(request.quantity)
        remaining = total_qty
        fills: List[Dict[str, Any]] = []
        start_time = time.monotonic()
        max_steps = max(1, self.fallback_seconds // self.step_seconds)
        step = 0

        while remaining > 0.01 and step < max_steps:
            elapsed = time.monotonic() - start_time
            elapsed_frac = min(1.0, elapsed / self.fallback_seconds)
            remaining_frac = remaining / total_qty

            state = {
                "remaining_fraction": remaining_frac,
                "elapsed_fraction": elapsed_frac,
                "spread_bps": 5.0,  # Placeholder; in production this would be live LOB data.
                "volume_ratio": 1.0,
                "book_imbalance": 0.0,
            }

            action = self.agent.select_action(state)

            if action == "wait":
                await asyncio.sleep(self.step_seconds)
                step += 1
                continue

            # Determine fill quantity based on the chosen action.
            fill_qty = remaining if action == "market" else min(remaining, total_qty * 0.15)

            sub = OrderRequest(
                symbol=request.symbol,
                side=request.side,
                order_type="market" if action == "market" else "limit",
                quantity=fill_qty,
                limit_price=request.limit_price if action in ("limit_inside", "limit_best") else None,
                account_id=request.account_id,
                execution_algo=f"rl_{action}",
            )

            try:
                # Submit the sub‑order to the broker and await a fill response.
                fill = await self.broker.submit_order(sub)  # Expected to be awaitable.
                fills.append(
                    {
                        "qty": fill.quantity,
                        "price": fill.price,
                        "algo": f"rl_{action}",
                        "slippage_bps": fill.slippage_bps if hasattr(fill, "slippage_bps") else None,
                    }
                )
                remaining -= float(fill.quantity)
            except Exception as exc:  # pragma: no cover
                logger.error("RLExecution: order submission failed (%s)", exc)
                # If the submission fails, break out to avoid an infinite loop.
                break

            # Small pause before the next decision step.
            await asyncio.sleep(self.step_seconds)
            step += 1

        # If any quantity remains after the loop, force a market fill.
        if remaining > 0.01:
            market_sub = OrderRequest(
                symbol=request.symbol,
                side=request.side,
                order_type="market",
                quantity=remaining,
                limit_price=None,
                account_id=request.account_id,
                execution_algo="rl_fallback_market",
            )
            try:
                fill = await self.broker.submit_order(market_sub)
                fills.append(
                    {
                        "qty": fill.quantity,
                        "price": fill.price,
                        "algo": "rl_fallback_market",
                        "slippage_bps": fill.slippage_bps if hasattr(fill, "slippage_bps") else None,
                    }
                )
            except Exception as exc:  # pragma: no cover
                logger.error("RLExecution: fallback market order failed (%s)", exc)

        return fills