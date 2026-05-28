"""
Training script for the A3C-LSTM RL trading agent.

Usage:
    python -m app.ml.training.train_rl  (uses a synthetic demo dataset)

Or import and call directly:
    from app.ml.training.train_rl import train_rl_agent
    await train_rl_agent(ohlcv_df, n_episodes=2000)
"""
import asyncio
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from app.ml.models.a3c_lstm import A3CLSTMAgent

logger = logging.getLogger(__name__)

_CHECKPOINT_DIR = Path(__file__).parents[3] / "checkpoints"

# Feature builder (mirrors rl_trader.py feature construction)
_SEQ_LEN = 30


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _build_features(df: pd.DataFrame) -> np.ndarray:
    """Return (T, n_features) feature matrix with no lookahead."""
    close = df["close"]
    volume = df["volume"]
    returns = close.pct_change().fillna(0.0)
    log_vol = np.log1p(volume).diff().fillna(0.0)
    rsi_norm = (_rsi(close).fillna(50.0) - 50.0) / 50.0
    return np.stack([returns.values, log_vol.values, rsi_norm.values], axis=1)


def _step_reward(df: pd.DataFrame, action: int, t: int) -> float:
    """
    Simple reward: profit/loss of the action taken at step t.
    Action 0=buy, 1=hold, 2=sell.
    Returns the next-bar return scaled by direction.
    """
    if t + 1 >= len(df):
        return 0.0
    next_ret = float(df["close"].iloc[t + 1] / df["close"].iloc[t] - 1.0)
    if action == 0:    # buy — reward is positive return
        return next_ret
    elif action == 2:  # sell — reward is negative return (profit from short)
        return -next_ret
    else:              # hold
        return 0.0


async def train_rl_agent(
    ohlcv_df: pd.DataFrame,
    n_episodes: int = 1000,
    gamma: float = 0.99,
    lr: float = 1e-4,
    grad_clip: float = 0.5,
    checkpoint_every: int = 100,
    n_features: int = 3,
    hidden_size: int = 128,
    model_path: str | None = None,
) -> A3CLSTMAgent:
    """
    Train an A3C-LSTM agent on OHLCV data.

    Single-threaded A3C (no multiprocessing): runs episodes sequentially.
    Each episode walks the full price history, collecting (s, a, r) tuples,
    then performs one gradient update per episode.

    Args:
        ohlcv_df:         DataFrame with columns [open, high, low, close, volume]
        n_episodes:       Number of training episodes
        gamma:            Discount factor
        lr:               Adam learning rate
        grad_clip:        Gradient clipping max norm
        checkpoint_every: Save checkpoint every N episodes
        n_features:       Feature dimension (must match model architecture)
        hidden_size:      LSTM hidden size
        model_path:       Where to save the final model; defaults to checkpoints dir

    Returns:
        Trained A3CLSTMAgent
    """
    features = _build_features(ohlcv_df)  # (T, n_features_raw)
    T = len(features)

    if T < _SEQ_LEN + 2:
        raise ValueError(f"DataFrame too short ({T} rows); need at least {_SEQ_LEN + 2}")

    # Pad or trim to expected n_features
    raw_dim = features.shape[1]
    if raw_dim < n_features:
        pad = np.zeros((T, n_features - raw_dim))
        features = np.hstack([features, pad])
    else:
        features = features[:, :n_features]

    agent = A3CLSTMAgent(n_features=n_features, hidden_size=hidden_size, n_actions=3)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    save_path = model_path or str(_CHECKPOINT_DIR / "a3c_lstm_latest.pt")

    total_rewards: list[float] = []

    for episode in range(1, n_episodes + 1):
        states: list[torch.Tensor] = []
        actions: list[int] = []
        rewards: list[float] = []

        agent.eval()
        with torch.no_grad():
            for t in range(_SEQ_LEN, T - 1):
                window = features[t - _SEQ_LEN : t]  # (seq_len, n_features)
                x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, n_feat)
                action = agent.select_action(x)
                reward = _step_reward(ohlcv_df, action, t)

                states.append(x.squeeze(0))  # (seq_len, n_features)
                actions.append(action)
                rewards.append(reward)

        if not states:
            continue

        # Stack trajectory
        states_tensor = torch.stack(states)           # (T', seq_len, n_features)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        dones = [False] * len(rewards)

        # Single gradient update
        agent.train()
        optimizer.zero_grad()
        loss_dict = agent.actor_critic_loss(
            states_tensor, actions_tensor, rewards, dones, gamma=gamma
        )
        loss_dict["loss"].backward()
        nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
        optimizer.step()

        ep_reward = float(sum(rewards))
        total_rewards.append(ep_reward)

        if episode % 10 == 0:
            avg = np.mean(total_rewards[-10:])
            logger.info(
                "Episode %d/%d  reward=%.4f  avg10=%.4f  loss=%.4f",
                episode,
                n_episodes,
                ep_reward,
                avg,
                loss_dict["loss"].item(),
            )

        # Save checkpoint
        if episode % checkpoint_every == 0:
            ckpt_path = save_path.replace(".pt", f"_ep{episode:04d}.pt")
            agent.save(
                ckpt_path,
                metadata={
                    "episode": episode,
                    "avg_reward": float(np.mean(total_rewards[-100:])),
                    "n_features": n_features,
                    "hidden_size": hidden_size,
                },
            )
            logger.info("Checkpoint saved → %s", ckpt_path)

    # Save final model as the "latest" checkpoint
    agent.save(
        save_path,
        metadata={
            "episode": n_episodes,
            "avg_reward": float(np.mean(total_rewards[-100:]) if total_rewards else 0.0),
            "n_features": n_features,
            "hidden_size": hidden_size,
        },
    )
    logger.info("Training complete. Final model saved → %s", save_path)
    return agent


# ------------------------------------------------------------------
# CLI entry point — runs a quick smoke-test with synthetic data
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(42)
    price = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, 300))
    demo_df = pd.DataFrame(
        {
            "open": price * 0.999,
            "high": price * 1.005,
            "low": price * 0.995,
            "close": price,
            "volume": rng.integers(100_000, 500_000, 300).astype(float),
        }
    )

    trained = asyncio.run(train_rl_agent(demo_df, n_episodes=50, checkpoint_every=25))
    print(f"Trained agent: {trained}")
