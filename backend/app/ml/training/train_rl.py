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
# Transaction cost applied per trade (as a fraction of price)
_TRANSACTION_COST = 0.001
# Maximum holding period before forced exit (in bars)
_MAX_HOLDING_PERIOD = 5
# RSI thresholds for entry confirmation (normalized)
_RSI_BUY_THRESHOLD = -0.3
_RSI_SELL_THRESHOLD = 0.3


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


def _step_reward(
    df: pd.DataFrame,
    action: int,
    t: int,
    position: int,
    entry_idx: int | None,
    rsi_norm: pd.Series,
    median_volume: float,
) -> tuple[float, int, int | None]:
    """
    Enhanced reward logic with entry confirmation, transaction cost,
    and forced exit after a maximum holding period.

    Returns a tuple of (reward, updated_position, updated_entry_idx).
    """
    # Default reward (no position change)
    reward = 0.0

    # Helper to compute profit for closing a position
    def _close_profit(pos: int, entry: int, cur: int) -> float:
        ret = float(df["close"].iloc[cur] / df["close"].iloc[entry] - 1.0)
        return ret if pos == 1 else -ret

    # Entry logic with confirmation filters
    if position == 0:
        if action == 0:  # attempt long
            if (
                rsi_norm.iloc[t] < _RSI_BUY_THRESHOLD
                and df["volume"].iloc[t] > median_volume
            ):
                # Open long position, incur transaction cost
                position = 1
                entry_idx = t
                reward -= _TRANSACTION_COST
            else:
                # Invalid entry, treat as hold with small penalty
                reward -= 0.0005
        elif action == 2:  # attempt short
            if (
                rsi_norm.iloc[t] > _RSI_SELL_THRESHOLD
                and df["volume"].iloc[t] > median_volume
            ):
                position = -1
                entry_idx = t
                reward -= _TRANSACTION_COST
            else:
                reward -= 0.0005
        # action == 1 (hold) yields zero reward
        return reward, position, entry_idx

    # Position is open – evaluate exit conditions
    hold_len = t - entry_idx if entry_idx is not None else 0
    close_signal = False

    # Forced exit after max holding period
    if hold_len >= _MAX_HOLDING_PERIOD:
        close_signal = True
    else:
        # RSI reversal as a soft exit cue
        if position == 1 and rsi_norm.iloc[t] > 0.0:
            close_signal = True
        if position == -1 and rsi_norm.iloc[t] < 0.0:
            close_signal = True

    # Opposite action also triggers exit
    if (position == 1 and action == 2) or (position == -1 and action == 0):
        close_signal = True

    if close_signal:
        profit = _close_profit(position, entry_idx, t)
        reward += profit - _TRANSACTION_COST  # apply cost on exit as well
        position = 0
        entry_idx = None
    else:
        # Holding without exit yields no immediate reward
        reward = 0.0

    return reward, position, entry_idx


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

    # Pre‑compute auxiliary series used for confirmation filters
    rsi_norm_series = (_rsi(ohlcv_df["close"]).fillna(50.0) - 50.0) / 50.0
    median_vol = float(ohlcv_df["volume"].median())

    agent = A3CLSTMAgent(n_features=n_features, hidden_size=hidden_size, n_actions=3)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    save_path = model_path or str(_CHECKPOINT_DIR / "a3c_lstm_latest.pt")

    total_rewards: list[float] = []

    for episode in range(1, n_episodes + 1):
        states: list[torch.Tensor] = []
        actions: list[int] = []
        rewards: list[float] = []

        # Position tracking for the episode
        position = 0
        entry_idx: int | None = None

        agent.eval()
        with torch.no_grad():
            for t in range(_SEQ_LEN, T - 1):
                window = features[t - _SEQ_LEN : t]  # (seq_len, n_features)
                x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, n_feat)
                action = agent.select_action(x)

                reward, position, entry_idx = _step_reward(
                    ohlcv_df,
                    action,
                    t,
                    position,
                    entry_idx,
                    rsi_norm_series,
                    median_vol,
                )

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