"""
RL-based dynamic position sizing agent.

This module provides a lightweight wrapper around a PPO policy network that maps a
portfolio state vector to a discrete Kelly multiplier. The network architecture
and the mapping from actions to multipliers are defined as constants. If a trained
model file is present, it is loaded; otherwise the sizer falls back to a neutral
1.0x multiplier.

State vector (dimension 7):
    - portfolio_heat: float
    - strategy_sharpe_30d: float
    - drawdown_pct: float
    - regime: int
    - volatility_ratio: float
    - win_streak: int
    - loss_streak: int

Action space (dimension 4):
    0 → 0.5× Kelly multiplier
    1 → 1.0× Kelly multiplier
    2 → 1.5× Kelly multiplier
    3 → 2.0× Kelly multiplier
"""

from pathlib import Path
from typing import Mapping, Tuple

import torch
import torch.nn as nn

STATE_DIM = 7
ACTION_DIM = 4
KELLY_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0]
MODEL_PATH = Path(__file__).parent.parent.parent / "models_artifacts" / "rl_position_sizer.pt"


class _PolicyNet(nn.Module):
    """Neural network representing the PPO policy and value heads.

    The network consists of a shared trunk followed by separate linear heads for
    policy logits and state‑value estimation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(64, ACTION_DIM)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(batch_size, STATE_DIM)``.

        Returns:
            A tuple ``(logits, value)`` where ``logits`` has shape
            ``(batch_size, ACTION_DIM)`` and ``value`` has shape ``(batch_size, 1)``.
        """
        h = self.shared(x)
        return self.policy_head(h), self.value_head(h)


class RLPositionSizer:
    """Wrapper that loads a trained policy network and provides a Kelly multiplier.

    The class lazily loads the model from ``MODEL_PATH`` on instantiation. If the
    model file is missing or loading fails, the sizer will always return a neutral
    multiplier of ``1.0``.
    """

    def __init__(self) -> None:
        self._net = _PolicyNet()
        self._loaded = False
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        """Load the trained policy network if the model file exists.

        The method sets ``self._loaded`` to ``True`` on successful load. Any
        exception during loading is silently ignored to keep the sizer operational
        with the fallback multiplier.
        """
        if MODEL_PATH.exists():
            try:
                state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
                self._net.load_state_dict(state["policy"])
                self._loaded = True
            except Exception:
                # Loading failed – keep ``_loaded`` as ``False`` to use fallback.
                pass

    def scale_factor(self, state: Mapping[str, float]) -> float:
        """Return the Kelly multiplier for the given portfolio state.

        The method converts the supplied ``state`` mapping to a tensor, runs the
        policy network, selects the action with the highest logit, and maps it to
        the corresponding multiplier.

        Args:
            state: Mapping containing the required keys:
                ``portfolio_heat``, ``strategy_sharpe_30d``, ``drawdown_pct``,
                ``regime``, ``volatility_ratio``, ``win_streak``, ``loss_streak``.
                Missing keys default to the values shown in the original code.

        Returns:
            A float multiplier in ``[0.5, 2.0]``. If the model is not loaded,
            ``1.0`` is returned.
        """
        if not self._loaded:
            return 1.0

        vec = torch.tensor(
            [
                float(state.get("portfolio_heat", 0.5)),
                float(state.get("strategy_sharpe_30d", 1.0)),
                float(state.get("drawdown_pct", 0.0)),
                float(state.get("regime", 1)),
                float(state.get("volatility_ratio", 1.0)),
                float(state.get("win_streak", 0)),
                float(state.get("loss_streak", 0)),
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

        self._net.eval()
        with torch.no_grad():
            logits, _ = self._net(vec)
            action = int(logits.argmax(dim=-1).item())

        return KELLY_MULTIPLIERS[action]