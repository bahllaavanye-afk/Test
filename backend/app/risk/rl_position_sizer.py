"""
PPO-based dynamic position sizing agent.
State dim=7: [portfolio_heat, strategy_sharpe_30d, drawdown_pct, regime, volatility_ratio, win_streak, loss_streak]
Action: 0=0.5x, 1=1.0x, 2=1.5x, 3=2.0x Kelly multiplier
Reward: excess return over fixed Kelly baseline, penalized by -0.1*drawdown
Falls back to 1.0x when no trained policy found.
"""
from pathlib import Path

import torch
import torch.nn as nn

STATE_DIM = 7
ACTION_DIM = 4
KELLY_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0]
MODEL_PATH = Path(__file__).parent.parent.parent / "models_artifacts" / "rl_position_sizer.pt"


class _PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(STATE_DIM, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.policy_head = nn.Linear(64, ACTION_DIM)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.policy_head(h), self.value_head(h)


class RLPositionSizer:
    def __init__(self):
        self._net = _PolicyNet()
        self._loaded = False
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if MODEL_PATH.exists():
            try:
                state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
                self._net.load_state_dict(state["policy"])
                self._loaded = True
            except Exception:
                pass

    def scale_factor(self, state: dict) -> float:
        """Returns Kelly multiplier (0.5-2.0) given current portfolio state."""
        if not self._loaded:
            return 1.0
        vec = torch.tensor([
            float(state.get("portfolio_heat", 0.5)),
            float(state.get("strategy_sharpe_30d", 1.0)),
            float(state.get("drawdown_pct", 0.0)),
            float(state.get("regime", 1)),
            float(state.get("volatility_ratio", 1.0)),
            float(state.get("win_streak", 0)),
            float(state.get("loss_streak", 0)),
        ], dtype=torch.float32).unsqueeze(0)
        self._net.eval()
        with torch.no_grad():
            logits, _ = self._net(vec)
            action = int(logits.argmax(dim=-1).item())
        return KELLY_MULTIPLIERS[action]
