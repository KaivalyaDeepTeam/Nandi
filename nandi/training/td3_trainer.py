"""Twin Delayed DDPG (TD3) trainer for Nandi agent.

TD3 features: twin critics, delayed policy updates, target policy smoothing.
"""

import os
import time
import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F

from nandi.config import TRAINING_CONFIG, MODEL_DIR
from nandi.models.critic import TwinCritic
from nandi.training.sac_trainer import ReplayBuffer, DEVICE

logger = logging.getLogger(__name__)


class TD3Trainer:
    """Twin Delayed DDPG training loop."""

    def __init__(self, agent, train_env, eval_env=None,
                 config=None, replay_size=100000, batch_size=256,
                 tau=0.005, gamma=0.99, lr=3e-4,
                 policy_delay=2, noise_clip=0.5, policy_noise=0.2,
                 warmup_steps=1000):
        self.agent = agent
        self.train_env = train_env
        self.eval_env = eval_env
        self.config = config or TRAINING_CONFIG
        self.device = DEVICE

        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.policy_delay = policy_delay
        self.noise_clip = noise_clip
        self.policy_noise = policy_noise
        self.warmup_steps = warmup_steps

        self.agent.to(self.device)

        d = agent.encoder.output_dense.out_features
        pos_dim = 32

        self.critic = TwinCritic(state_dim=d, pos_dim=pos_dim).to(self.device)
        self.target_agent = copy.deepcopy(agent).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.agent.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(replay_size)
        self.total_steps = 0
        self.update_count = 0
        self.best_eval_return = -np.inf

    def _get_encoding(self, model, market_state, position_info):
        projected = model.feature_proj(market_state)
        encoded = model.encoder(projected)
        pos_emb = model.position_embed(position_info)
        return encoded, pos_emb

    def _soft_update(self, source, target):
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def _update(self):
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

        # Target policy smoothing
        with torch.no_grad():
            next_pre_tanh, _, _ = self.target_agent(next_ms_t, next_pi_t)
            noise = (torch.randn_like(next_pre_tanh) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = torch.tanh(next_pre_tanh + noise)

            next_enc, next_pos = self._get_encoding(self.target_agent, next_ms_t, next_pi_t)
            q1_next, q2_next = self.target_critic(next_enc, next_pos, next_action)
            q_next = torch.min(q1_next, q2_next).squeeze(-1)
            target_q = rew_t + (1 - done_t) * self.gamma * q_next

        # Critic update
        enc, pos = self._get_encoding(self.agent, ms_t, pi_t)
        q1, q2 = self.critic(enc.detach(), pos.detach(), act_t)
        critic_loss = F.mse_loss(q1.squeeze(-1), target_q) + \
                     F.mse_loss(q2.squeeze(-1), target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.update_count += 1
        stats = {"critic_loss": critic_loss.item()}

        # Delayed policy update
        if self.update_count % self.policy_delay == 0:
            pre_tanh, _, _ = self.agent(ms_t, pi_t)
            action_new = torch.tanh(pre_tanh)
            enc_new, pos_new = self._get_encoding(self.agent, ms_t, pi_t)
            q1_new = self.critic.q1_forward(enc_new, pos_new, action_new)
            actor_loss = -q1_new.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            self.actor_optimizer.step()

            self._soft_update(self.critic, self.target_critic)
            self._soft_update(self.agent, self.target_agent)
            stats["actor_loss"] = actor_loss.item()

        return stats

    def train(self):
        total_timesteps = self.config["total_timesteps"]
        logger.info(f"TD3 training for {total_timesteps:,} timesteps...")
        start_time = time.time()
        state = self.train_env.reset()
        episode_reward = 0

        while self.total_steps < total_timesteps:
            ms, pi = state
            if self.total_steps < self.warmup_steps:
                action = np.random.uniform(-1, 1)
            else:
                action, _, _, _ = self.agent.get_action(ms, pi, deterministic=False)
                action += np.random.normal(0, 0.1)
                action = float(np.clip(action, -1, 1))

            next_state, reward, done, info = self.train_env.step(action)
            next_ms, next_pi = next_state
            self.replay_buffer.add(ms, pi, action, reward, next_ms, next_pi, float(done))
            episode_reward += reward
            self.total_steps += 1

            if done:
                episode_reward = 0
                state = self.train_env.reset()
            else:
                state = next_state

            if self.total_steps >= self.warmup_steps:
                stats = self._update()
                if self.total_steps % 5000 == 0 and stats:
                    logger.info(f"TD3 Step {self.total_steps:>7,} | C-loss: {stats.get('critic_loss', 0):.4f}")

            if (self.eval_env and self.total_steps % self.config["eval_interval"] == 0):
                eval_stats = self._evaluate()
                if eval_stats["eval_return"] > self.best_eval_return:
                    self.best_eval_return = eval_stats["eval_return"]
                    self.agent.save_agent()

        logger.info(f"TD3 complete in {(time.time() - start_time) / 60:.1f} min")
        self.agent.save_agent()

    def _evaluate(self, n_episodes=10):
        self.agent.eval()
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
        mean_ret = float(np.mean(returns))
        logger.info(f"TD3 EVAL | Return: {mean_ret:+.2f}%")
        return {"eval_return": mean_ret}
