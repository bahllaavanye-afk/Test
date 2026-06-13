"""
Train RL position sizer and dynamic exit agents.
Uses synthetic GBM price paths (no broker connection required).
Saves best policy to models_artifacts/rl_*.pt
"""
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "experiments" / "results"
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "models_artifacts"

def generate_gbm_episode(n_steps=252, mu=0.0008, sigma=0.018, start_price=100.0):
    """Generate synthetic daily price path via Geometric Brownian Motion."""
    returns = np.random.normal(mu, sigma, n_steps)
    prices = start_price * np.cumprod(1 + returns)
    return prices, returns

def train_position_sizer(n_episodes=500, lr=3e-4):
    """PPO training for position sizer. Returns final policy state_dict."""
    from app.risk.rl_position_sizer import KELLY_MULTIPLIERS, _PolicyNet
    net = _PolicyNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    episode_rewards = []
    best_reward = -float("inf")
    best_state = None

    for ep in range(n_episodes):
        prices, returns = generate_gbm_episode()
        states, actions, rewards, log_probs, values = [], [], [], [], []
        portfolio_value = 1.0
        peak_value = 1.0
        win_streak = loss_streak = 0

        for i in range(len(returns) - 1):
            drawdown = (peak_value - portfolio_value) / peak_value
            state_vec = torch.tensor([
                min(portfolio_value / 2.0, 1.0),  # portfolio_heat proxy
                1.0,  # strategy_sharpe placeholder
                drawdown,
                1.0,  # regime: bull
                abs(returns[i]) / 0.018,  # volatility ratio
                float(min(win_streak, 10)) / 10,
                float(min(loss_streak, 10)) / 10,
            ], dtype=torch.float32).unsqueeze(0)

            logits, value = net(state_vec)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            multiplier = KELLY_MULTIPLIERS[action.item()]
            kelly_base = 0.25  # fixed Kelly fraction
            position_size = kelly_base * multiplier
            step_return = returns[i + 1] * position_size
            portfolio_value *= (1 + step_return)
            peak_value = max(peak_value, portfolio_value)

            # Reward: step return minus drawdown penalty
            reward = step_return - 0.1 * drawdown
            if step_return > 0:
                win_streak += 1; loss_streak = 0
            else:
                loss_streak += 1; win_streak = 0

            states.append(state_vec)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(dist.log_prob(action))
            values.append(value)

        # PPO update (simplified single epoch)
        if len(rewards) < 2:
            continue
        returns_t = torch.zeros(len(rewards))
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + 0.99 * G
            returns_t[t] = G
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        old_log_probs = torch.stack(log_probs).detach()
        vals = torch.stack(values).squeeze().detach()
        advantages = returns_t - vals

        total_loss = torch.tensor(0.0)
        for _ in range(4):  # PPO epochs
            new_logits, new_vals = net(torch.cat(states))
            new_dist = torch.distributions.Categorical(logits=new_logits)
            new_log_probs = new_dist.log_prob(torch.stack(actions).squeeze())
            ratio = (new_log_probs - old_log_probs.squeeze()).exp()
            clip_ratio = torch.clamp(ratio, 0.8, 1.2)
            policy_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
            value_loss = (new_vals.squeeze() - returns_t).pow(2).mean()
            entropy = new_dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            total_loss = loss

        ep_reward = sum(rewards)
        episode_rewards.append(ep_reward)
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

        if (ep + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"Episode {ep+1}/{n_episodes} | avg_reward={avg:.4f} | best={best_reward:.4f}")

    return best_state, episode_rewards

def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Training RL Position Sizer...")
    best_state, rewards = train_position_sizer(n_episodes=500)
    if best_state:
        torch.save({"policy": best_state}, ARTIFACTS_DIR / "rl_position_sizer.pt")
        print(f"Saved policy. Best episode reward: {max(rewards):.4f}")
    results = {
        "trained_at": datetime.now(UTC).isoformat(),
        "n_episodes": 500,
        "best_reward": float(max(rewards)),
        "final_avg_reward": float(np.mean(rewards[-50:])),
        "episode_rewards": [float(r) for r in rewards[-100:]],  # last 100 for chart
    }
    (RESULTS_DIR / "rl_training.json").write_text(json.dumps(results, indent=2))
    print("Results saved to experiments/results/rl_training.json")

if __name__ == "__main__":
    main()
