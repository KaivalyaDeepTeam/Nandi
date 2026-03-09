"""Soft Actor-Critic (SAC) trainer for Nandi agent.

SAC is better suited for continuous action spaces than PPO:
- Off-policy (more sample efficient)
- Automatic entropy tuning
- Twin critics for stability
"""

import os
import time
import copy
import logging
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from nandi.config import PPO_CONFIG, TRAINING_CONFIG, MODEL_DIR
from nandi.models.agent import NandiAgent
from nandi.models.critic import TwinCritic

logger = logging.getLogger(__name__)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, market_state, position_info, action, reward,
            next_market_state, next_position_info, done):
        self.buffer.append((
            market_state, position_info, action, reward,
            next_market_state, next_position_info, done,
        ))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        ms = np.array([b[0] for b in batch], dtype=np.float32)
        pi = np.array([b[1] for b in batch], dtype=np.float32)
        actions = np.array([b[2] for b in batch], dtype=np.float32).reshape(-1, 1)
        rewards = np.array([b[3] for b in batch], dtype=np.float32)
        next_ms = np.array([b[4] for b in batch], dtype=np.float32)
        next_pi = np.array([b[5] for b in batch], dtype=np.float32)
        dones = np.array([b[6] for b in batch], dtype=np.float32)

        return ms, pi, actions, rewards, next_ms, next_pi, dones

    def __len__(self):
        return len(self.buffer)


class SACTrainer:
    """Soft Actor-Critic training loop."""

    def __init__(self, agent, train_env, eval_env=None,
                 config=None, training_config=None,
                 replay_size=100000, batch_size=256, tau=0.005,
                 gamma=0.99, lr=3e-4, warmup_steps=1000):
        self.agent = agent
        self.train_env = train_env
        self.eval_env = eval_env
        self.config = training_config or TRAINING_CONFIG
        self.device = DEVICE

        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.warmup_steps = warmup_steps

        self.agent.to(self.device)

        # Get encoder output dims by running a forward pass
        d = agent.encoder.output_dense.out_features  # 128
        pos_dim = 32  # from position_embed

        # Twin critics
        self.critic = TwinCritic(state_dim=d, pos_dim=pos_dim).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.agent.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Auto entropy tuning
        self.target_entropy = -1.0
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.replay_buffer = ReplayBuffer(replay_size)
        self.total_steps = 0
        self.best_eval_return = -np.inf
        self.training_stats = []

    def _get_encoding(self, market_state, position_info):
        """Get encoder output and position embedding."""
        projected = self.agent.feature_proj(market_state)
        encoded = self.agent.encoder(projected)
        pos_emb = self.agent.position_embed(position_info)
        return encoded, pos_emb

    def _soft_update(self):
        """Polyak averaging for target critic."""
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def _update(self):
        """Single SAC update step."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        ms, pi, actions, rewards, next_ms, next_pi, dones = \
            self.replay_buffer.sample(self.batch_size)

        ms_t = torch.tensor(ms, dtype=torch.float32, device=self.device)
        pi_t = torch.tensor(pi, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rew_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_ms_t = torch.tensor(next_ms, dtype=torch.float32, device=self.device)
        next_pi_t = torch.tensor(next_pi, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        alpha = self.log_alpha.exp().detach()

        # --- Critic update ---
        with torch.no_grad():
            next_pre_tanh, next_std, _ = self.agent(next_ms_t, next_pi_t)
            next_noise = torch.randn_like(next_pre_tanh) * next_std
            next_pre_tanh_action = next_pre_tanh + next_noise
            next_action = torch.tanh(next_pre_tanh_action)

            next_log_prob = self.agent._tanh_log_prob(
                next_pre_tanh_action, next_action, next_pre_tanh, next_std
            )

            next_enc, next_pos = self._get_encoding(next_ms_t, next_pi_t)
            q1_next, q2_next = self.target_critic(next_enc, next_pos, next_action)
            q_next = torch.min(q1_next, q2_next).squeeze(-1)

            target_q = rew_t + (1 - done_t) * self.gamma * (q_next - alpha * next_log_prob)

        enc, pos = self._get_encoding(ms_t, pi_t)
        q1, q2 = self.critic(enc.detach(), pos.detach(), act_t)
        critic_loss = F.mse_loss(q1.squeeze(-1), target_q) + \
                     F.mse_loss(q2.squeeze(-1), target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        pre_tanh_mean, std, _ = self.agent(ms_t, pi_t)
        noise = torch.randn_like(pre_tanh_mean) * std
        pre_tanh_action = pre_tanh_mean + noise
        action_new = torch.tanh(pre_tanh_action)

        log_prob = self.agent._tanh_log_prob(
            pre_tanh_action, action_new, pre_tanh_mean, std
        )

        enc_new, pos_new = self._get_encoding(ms_t, pi_t)
        q1_new = self.critic.q1_forward(enc_new, pos_new, action_new)

        actor_loss = (alpha * log_prob - q1_new.squeeze(-1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.actor_optimizer.step()

        # --- Alpha update ---
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Target update ---
        self._soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": alpha.item(),
            "entropy": -log_prob.mean().item(),
        }

    def train(self):
        """Main SAC training loop."""
        total_timesteps = self.config["total_timesteps"]
        logger.info(f"SAC training for {total_timesteps:,} timesteps...")
        start_time = time.time()

        state = self.train_env.reset()
        episode_reward = 0
        episode_rewards = []

        while self.total_steps < total_timesteps:
            ms, pi = state

            if self.total_steps < self.warmup_steps:
                action = np.random.uniform(-1, 1)
            else:
                action, _, _, _ = self.agent.get_action(ms, pi, deterministic=False)

            next_state, reward, done, info = self.train_env.step(action)
            next_ms, next_pi = next_state

            self.replay_buffer.add(ms, pi, action, reward, next_ms, next_pi, float(done))

            episode_reward += reward
            self.total_steps += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                state = self.train_env.reset()
            else:
                state = next_state

            # Update after warmup
            if self.total_steps >= self.warmup_steps:
                stats = self._update()

                if self.total_steps % 5000 == 0 and stats:
                    logger.info(
                        f"Step {self.total_steps:>7,}/{total_timesteps:,} | "
                        f"C-loss: {stats.get('critic_loss', 0):.4f} | "
                        f"A-loss: {stats.get('actor_loss', 0):.4f} | "
                        f"alpha: {stats.get('alpha', 0):.3f}"
                    )

            # Evaluate
            if (self.eval_env is not None and
                self.total_steps % self.config["eval_interval"] == 0):
                eval_stats = self._evaluate()
                if eval_stats["eval_return"] > self.best_eval_return:
                    self.best_eval_return = eval_stats["eval_return"]
                    self.agent.save_agent()
                    logger.info(f"New best: {eval_stats['eval_return']:.2f}%")

        total_time = time.time() - start_time
        logger.info(f"SAC training complete in {total_time / 60:.1f} minutes")
        self.agent.save_agent()
        return self.training_stats

    def _evaluate(self, n_episodes=None):
        """Evaluate current policy."""
        self.agent.eval()
        n_episodes = n_episodes or self.config.get("n_eval_episodes", 10)
        returns = []

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

        mean_return = float(np.mean(returns))
        logger.info(f"SAC EVAL | Return: {mean_return:+.2f}%")
        return {"eval_return": mean_return}
