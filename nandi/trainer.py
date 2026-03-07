"""
Nandi PPO Trainer — Trains the RL agent on historical forex data.
"""

import os
import time
import logging

import numpy as np
import tensorflow as tf

from nandi.config import PPO_CONFIG, TRAINING_CONFIG, MODEL_DIR
from nandi.model import NandiAgent
from nandi.environment import MultiEpisodeEnv

logger = logging.getLogger(__name__)


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
        """Compute GAE advantages and returns."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

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
            returns[t] = advantages[t] + self.values[t]

        self.advantages = advantages
        self.returns = returns

    def get_batches(self, batch_size):
        """Yield random mini-batches."""
        n = len(self.rewards)
        indices = np.random.permutation(n)

        ms = np.array(self.market_states, dtype=np.float32)
        pi = np.array(self.position_infos, dtype=np.float32)
        act = np.array(self.actions, dtype=np.float32).reshape(-1, 1)
        lp = np.array(self.log_probs, dtype=np.float32)
        adv = self.advantages
        ret = self.returns

        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for start in range(0, n, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            yield (
                ms[idx], pi[idx], act[idx],
                lp[idx], adv[idx], ret[idx],
            )

    def clear(self):
        self.__init__()


class NandiTrainer:
    """PPO training loop for Nandi agent."""

    def __init__(self, agent, train_env, eval_env=None,
                 ppo_config=None, training_config=None):
        self.agent = agent
        self.train_env = train_env
        self.eval_env = eval_env
        self.ppo = ppo_config or PPO_CONFIG
        self.config = training_config or TRAINING_CONFIG

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.ppo["learning_rate"],
            clipnorm=self.ppo["max_grad_norm"],
        )

        self.total_steps = 0
        self.best_eval_return = -np.inf
        self.training_stats = []

    def train(self):
        """Main training loop."""
        total_timesteps = self.config["total_timesteps"]
        rollout_length = self.ppo["rollout_length"]

        logger.info(f"Training Nandi for {total_timesteps:,} timesteps...")
        logger.info(f"Rollout length: {rollout_length} | Batch size: {self.ppo['batch_size']}")

        start_time = time.time()
        state = self.train_env.reset()

        while self.total_steps < total_timesteps:
            # Collect rollout
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

            # Compute returns with last value
            ms, pi = state
            _, _, last_value, _ = self.agent.get_action(ms, pi)
            buffer.compute_returns(
                last_value,
                gamma=self.ppo["gamma"],
                lam=self.ppo["lambda_gae"],
            )

            # PPO update
            stats = self._ppo_update(buffer)

            # Logging
            if episode_rewards:
                mean_reward = np.mean(episode_rewards)
                stats["mean_episode_reward"] = mean_reward

            elapsed = time.time() - start_time
            fps = self.total_steps / elapsed

            if self.total_steps % 5000 < rollout_length:
                logger.info(
                    f"Step {self.total_steps:>7,}/{total_timesteps:,} | "
                    f"FPS: {fps:.0f} | "
                    f"Policy loss: {stats['policy_loss']:.4f} | "
                    f"Value loss: {stats['value_loss']:.4f} | "
                    f"Entropy: {stats['entropy']:.4f} | "
                    f"Ep reward: {stats.get('mean_episode_reward', 0):.4f}"
                )

            # Evaluate
            if (self.eval_env is not None
                    and self.total_steps % self.config["eval_interval"] < rollout_length):
                eval_stats = self.evaluate()
                self.training_stats.append({
                    "step": self.total_steps,
                    **stats,
                    **eval_stats,
                })

                if eval_stats["eval_return"] > self.best_eval_return:
                    self.best_eval_return = eval_stats["eval_return"]
                    self.agent.save_agent()
                    logger.info(f"New best eval return: {eval_stats['eval_return']:.2f}% — saved!")

            # Periodic save
            if self.total_steps % self.config["save_interval"] < rollout_length:
                self.agent.save_agent(
                    os.path.join(MODEL_DIR, f"nandi_checkpoint_{self.total_steps}")
                )

        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time / 60:.1f} minutes")
        self.agent.save_agent()

        return self.training_stats

    def _ppo_update(self, buffer):
        """Run PPO epochs on collected rollout."""
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(self.ppo["n_epochs"]):
            for batch in buffer.get_batches(self.ppo["batch_size"]):
                ms, pi, actions, old_log_probs, advantages, returns = batch

                ms = tf.constant(ms)
                pi = tf.constant(pi)
                actions = tf.constant(actions)
                old_log_probs = tf.constant(old_log_probs)
                advantages = tf.constant(advantages)
                returns = tf.constant(returns)

                with tf.GradientTape() as tape:
                    log_probs, values, entropy = self.agent.evaluate_actions(
                        ms, pi, actions
                    )

                    # PPO clipped objective
                    ratio = tf.exp(log_probs - old_log_probs)
                    clipped = tf.clip_by_value(
                        ratio,
                        1 - self.ppo["clip_ratio"],
                        1 + self.ppo["clip_ratio"],
                    )
                    policy_loss = -tf.reduce_mean(
                        tf.minimum(ratio * advantages, clipped * advantages)
                    )

                    # Value loss
                    value_loss = tf.reduce_mean(
                        tf.square(values - returns)
                    )

                    # Entropy bonus
                    entropy_loss = -tf.reduce_mean(entropy)

                    # Total loss
                    loss = (
                        policy_loss
                        + self.ppo["value_coef"] * value_loss
                        + self.ppo["entropy_coef"] * entropy_loss
                    )

                grads = tape.gradient(loss, self.agent.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(grads, self.agent.trainable_variables)
                )

                total_policy_loss += float(policy_loss)
                total_value_loss += float(value_loss)
                total_entropy += float(-entropy_loss)
                n_updates += 1

        buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
        }

    def evaluate(self, n_episodes=None):
        """Run evaluation episodes on eval environment."""
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
                action, _, _, unc = self.agent.get_action(
                    ms, pi, deterministic=True
                )
                # Gate by uncertainty: reduce position if uncertain
                if unc > 0.7:
                    action *= 0.3

                state, _, done, info = self.eval_env.step(action)

            returns.append(info["return_pct"])
            drawdowns.append(info["drawdown"])
            trade_counts.append(info["total_trades"])
            win_rates.append(info["win_rate"])

        mean_return = np.mean(returns)
        mean_dd = np.mean(drawdowns)
        mean_trades = np.mean(trade_counts)
        mean_wr = np.mean(win_rates)

        logger.info(
            f"EVAL | Return: {mean_return:+.2f}% | "
            f"MaxDD: {mean_dd:.2%} | "
            f"Trades: {mean_trades:.0f} | "
            f"WinRate: {mean_wr:.1%}"
        )

        return {
            "eval_return": mean_return,
            "eval_drawdown": mean_dd,
            "eval_trades": mean_trades,
            "eval_win_rate": mean_wr,
        }
