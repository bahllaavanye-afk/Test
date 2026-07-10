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
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    F = None      # type: ignore[assignment]

from app.utils.logging import logger

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
            except (OSError, RuntimeError) as e:
                logger.warning(
                    "RLExecAgent: failed to load policy from %s (%s); using heuristic fallback",
                    path,
                    e,
                )
            except Exception as e:  # pragma: no cover
                logger.exception(
                    "RLExecAgent: unexpected error while loading policy from %s", path
                )
                # Keep _trained as False to ensure heuristic fallback

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
            try:
                action_idx, _ = self.policy.act(arr)
                return _ACTION_NAMES[action_idx]
            except Exception as e:
                logger.exception("RLExecAgent: error during policy inference (%s)", e)
                # Fallback to heuristic if inference fails
                self._trained = False

        # Heuristic fallback: aggressive when time is running out or large spread signals urgency
        remaining = arr[0]
        elapsed = arr[1]
        spread_norm = arr[2]

        if elapsed > 0.85 or remaining < 0.05:
            return "market"
        elif spread_norm < 0.2:  # tight spread → post limit
            return "limit_best"
        elif elapsed > 0.5:
            return "limit_inside"
        else:
            return "wait"

    def save(self, path: Path | None = None) -> None:
        """Persist the current policy weights."""
        p = path or _MODEL_PATH
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.policy.state_dict(), str(p))
            logger.info("RLExecAgent: policy saved to %s", p)
        except OSError as e:
            logger.error("RLExecAgent: failed to write policy to %s (%s)", p, e)
        except Exception as e:  # pragma: no cover
            logger.exception("RLExecAgent: unexpected error while saving policy to %s", p)


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
        from app.brokers.base import OrderRequest

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

            state = {
                "remaining_fraction": remaining_frac,
                "elapsed_fraction": elapsed_frac,
                "spread_bps": 5.0,  # default; would come from live LOB in production
                "volume_ratio": 1.0,
                "book_imbalance": 0.0,
            }

            action = self.agent.select_action(state)

            if action == "wait":
                await asyncio.sleep(self.step_seconds)
                step += 1
                continue

            # Build sub-order request
            fill_qty = remaining if action == "market" else min(remaining, total_qty * 0.15)
            sub = OrderRequest(
                symbol=request.symbol,
                side=request.side,
                order_type="market" if action == "market" else "limit",
                quantity=fill_qty,
                limit_price=request.limit_price
                if action in ("limit_inside", "limit_best")
                else None,
                account_id=request.account_id,
                execution_algo=f"rl_{action}",
            )

            try:
                # Assume broker has an async method `place_order` returning a dict with
                # keys: qty (filled quantity), price (average fill price), slippage_bps.
                fill_result = await self.broker.place_order(sub)
            except (ConnectionError, TimeoutError) as e:
                logger.error(
                    "RLExecution: network error while placing %s order for %s (step %d): %s",
                    action,
                    request.symbol,
                    step,
                    e,
                )
                # On network issues, retry after a short pause
                await asyncio.sleep(self.step_seconds)
                step += 1
                continue
            except Exception as e:
                logger.exception(
                    "RLExecution: unexpected error while placing %s order for %s (step %d)",
                    action,
                    request.symbol,
                    step,
                )
                # Abort execution on unrecoverable errors
                break

            # Process successful fill
            try:
                filled_qty = float(fill_result.get("qty", 0))
                fill_price = float(fill_result.get("price", 0))
                slippage_bps = float(fill_result.get("slippage_bps", 0))
                fills.append(
                    {
                        "qty": filled_qty,
                        "price": fill_price,
                        "algo": f"rl_{action}",
                        "slippage_bps": slippage_bps,
                    }
                )
                remaining -= filled_qty
            except (KeyError, TypeError, ValueError) as e:
                logger.error(
                    "RLExecution: malformed fill result for %s order at step %d (%s)",
                    action,
                    step,
                    e,
                )
                # Treat as no fill and continue
                await asyncio.sleep(self.step_seconds)
                step += 1
                continue

            step += 1

        # Final fallback: if still remaining, attempt a market order
        if remaining > 0.01:
            fallback_qty = remaining
            fallback_order = OrderRequest(
                symbol=request.symbol,
                side=request.side,
                order_type="market",
                quantity=fallback_qty,
                limit_price=None,
                account_id=request.account_id,
                execution_algo="rl_fallback_market",
            )
            try:
                fill_result = await self.broker.place_order(fallback_order)
                fills.append(
                    {
                        "qty": float(fill_result.get("qty", fallback_qty)),
                        "price": float(fill_result.get("price", 0)),
                        "algo": "rl_fallback_market",
                        "slippage_bps": float(fill_result.get("slippage_bps", 0)),
                    }
                )
            except Exception as e:
                logger.exception(
                    "RLExecution: fallback market order failed for %s (remaining %.4f)",
                    request.symbol,
                    remaining,
                )
        return fills