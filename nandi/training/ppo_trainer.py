"""
PPO Trainer — trains Nandi agents on historical forex data.

Supports both single-pair and sequential multi-pair training.
Features SAC-style automatic entropy tuning instead of fixed entropy floor.
"""

import os
import time
import logging

import numpy as np
import torch

from nandi.config import PPO_CONFIG, TRAINING_CONFIG, MODEL_DIR
from nandi.models.agent import NandiAgent
from nandi.environment.single_pair_env import MultiEpisodeEnv

logger = logging.getLogger(__name__)

# Device selection
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class RolloutBuffer:
    """Stores experience for PPO updates."""

    def __init__(self):
        self.market_states = []
        self.position_infos = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, market_state, position_info, action, log_prob, reward, value, done):
        self.market_states.append(market_state)
        self.position_infos.append(position_info)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, last_value, gamma=0.99, lam=0.95):
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)

        last_gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_done = 0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = (
                self.rewards[t]
                + gamma * next_value * (1 - next_done)
                - self.values[t]
            )
            last_gae = delta + gamma * lam * (1 - next_done) * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + np.array(self.values, dtype=np.float32)

    def get_batches(self, batch_size):
        n = len(self.rewards)
        indices = np.random.permutation(n)

        ms = np.array(self.market_states, dtype=np.float32)
        pi = np.array(self.position_infos, dtype=np.float32)
        act = np.array(self.actions, dtype=np.float32).reshape(-1, 1)
        lp = np.array(self.log_probs, dtype=np.float32)
        adv = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        ret = self.returns

        for start in range(0, n, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            yield ms[idx], pi[idx], act[idx], lp[idx], adv[idx], ret[idx]

    def clear(self):
        self.__init__()


class NandiTrainer:
    """PPO training loop for Nandi agent with automatic entropy tuning."""

    def __init__(self, agent, train_env, eval_env=None,
                 ppo_config=None, training_config=None):
        self.agent = agent
        self.train_env = train_env
        self.eval_env = eval_env
        self.ppo = ppo_config or PPO_CONFIG
        self.config = training_config or TRAINING_CONFIG
        self.device = DEVICE

        self.agent.to(self.device)
        logger.info(f"Using device: {self.device}")

        self.optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=self.ppo["learning_rate"],
        )

        total_timesteps = self.config["total_timesteps"]
        rollout_length = self.ppo["rollout_length"]
        n_updates = total_timesteps // rollout_length
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=n_updates,
        )

        # SAC-style automatic entropy tuning
        self.target_entropy = -1.0  # target entropy for 1D continuous action
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.total_steps = 0
        self.best_eval_return = -np.inf
        self.evals_without_improvement = 0
        self.early_stop_patience = self.config.get("early_stop_patience", 10)
        self.training_stats = []

    def train(self):
        total_timesteps = self.config["total_timesteps"]
        rollout_length = self.ppo["rollout_length"]

        logger.info(f"Training for {total_timesteps:,} timesteps...")
        start_time = time.time()
        state = self.train_env.reset()

        while self.total_steps < total_timesteps:
            buffer = RolloutBuffer()
            episode_rewards = []
            ep_reward = 0

            for _ in range(rollout_length):
                market_state, position_info = state
                action, log_prob, value, unc = self.agent.get_action(
                    market_state, position_info
                )

                next_state, reward, done, info = self.train_env.step(action)
                buffer.add(
                    market_state, position_info,
                    action, log_prob, reward, value, float(done)
                )

                ep_reward += reward
                self.total_steps += 1

                if done:
                    episode_rewards.append(ep_reward)
                    ep_reward = 0
                    state = self.train_env.reset()
                else:
                    state = next_state

            ms, pi = state
            _, _, last_value, _ = self.agent.get_action(ms, pi)
            buffer.compute_returns(
                last_value, gamma=self.ppo["gamma"], lam=self.ppo["lambda_gae"],
            )

            stats = self._ppo_update(buffer)
            self.scheduler.step()

            if episode_rewards:
                stats["mean_episode_reward"] = np.mean(episode_rewards)

            elapsed = time.time() - start_time
            fps = self.total_steps / elapsed

            if self.total_steps % 5000 < rollout_length:
                lr = self.optimizer.param_groups[0]["lr"]
                alpha = self.log_alpha.exp().item()
                logger.info(
                    f"Step {self.total_steps:>7,}/{total_timesteps:,} | "
                    f"FPS: {fps:.0f} | LR: {lr:.1e} | alpha: {alpha:.3f} | "
                    f"P-loss: {stats['policy_loss']:.4f} | "
                    f"V-loss: {stats['value_loss']:.4f} | "
                    f"Entropy: {stats['entropy']:.4f}"
                )

            if (self.eval_env is not None
                    and self.total_steps % self.config["eval_interval"] < rollout_length):
                eval_stats = self.evaluate()
                self.training_stats.append({
                    "step": self.total_steps, **stats, **eval_stats,
                })

                if eval_stats["eval_return"] > self.best_eval_return:
                    self.best_eval_return = eval_stats["eval_return"]
                    self.evals_without_improvement = 0
                    self.agent.save_agent()
                    logger.info(f"New best eval return: {eval_stats['eval_return']:.2f}%")
                else:
                    self.evals_without_improvement += 1
                    if self.evals_without_improvement >= self.early_stop_patience:
                        logger.info(f"Early stopping after {self.early_stop_patience} evals")
                        break

            if self.total_steps % self.config["save_interval"] < rollout_length:
                self.agent.save_agent(
                    os.path.join(MODEL_DIR, f"nandi_checkpoint_{self.total_steps}")
                )

        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time / 60:.1f} minutes")
        self.agent.save_agent()
        return self.training_stats

    def _ppo_update(self, buffer):
        self.agent.train()
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(self.ppo["n_epochs"]):
            for batch in buffer.get_batches(self.ppo["batch_size"]):
                ms, pi, actions, old_log_probs, advantages, returns = batch

                ms = torch.tensor(ms, dtype=torch.float32, device=self.device)
                pi = torch.tensor(pi, dtype=torch.float32, device=self.device)
                actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
                old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
                advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
                returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

                log_probs, values, entropy = self.agent.evaluate_actions(ms, pi, actions)

                ratio = torch.exp(log_probs - old_log_probs)
                clipped = torch.clamp(
                    ratio,
                    1 - self.ppo["clip_ratio"],
                    1 + self.ppo["clip_ratio"],
                )
                policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
                value_loss = (values - returns).pow(2).mean()
                entropy_loss = -entropy.mean()

                # Auto entropy tuning: learn entropy coefficient via dual gradient
                alpha = self.log_alpha.exp().detach()
                alpha_loss = -(self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                loss = (
                    policy_loss
                    + self.ppo["value_coef"] * value_loss
                    + alpha * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.ppo["max_grad_norm"],
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                n_updates += 1

        buffer.clear()
        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
            "alpha": self.log_alpha.exp().item(),
        }

    def evaluate(self, n_episodes=None):
        self.agent.eval()
        n_episodes = n_episodes or self.config["n_eval_episodes"]
        returns = []
        drawdowns = []
        trade_counts = []
        win_rates = []

        for _ in range(n_episodes):
            state = self.eval_env.reset()
            done = False
            while not done:
                ms, pi = state
                action, _, _, unc = self.agent.get_action(ms, pi, deterministic=True)
                if unc > 0.7:
                    action *= 0.3
                state, _, done, info = self.eval_env.step(action)

            returns.append(info["return_pct"])
            drawdowns.append(info["drawdown"])
            trade_counts.append(info["total_trades"])
            win_rates.append(info["win_rate"])

        mean_return = np.mean(returns)
        mean_dd = np.mean(drawdowns)

        logger.info(
            f"EVAL | Return: {mean_return:+.2f}% | MaxDD: {mean_dd:.2%} | "
            f"Trades: {np.mean(trade_counts):.0f} | WR: {np.mean(win_rates):.1%}"
        )

        return {
            "eval_return": mean_return,
            "eval_drawdown": mean_dd,
            "eval_trades": np.mean(trade_counts),
            "eval_win_rate": np.mean(win_rates),
        }
