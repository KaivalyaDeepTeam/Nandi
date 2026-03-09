"""
AEGIS Trainer — Adaptive Edge-Gated Intelligent Scalper training loop.

Novel training algorithm with 6 interacting loss terms:

1. DISTRIBUTIONAL CRITIC LOSS (Quantile Huber, asymmetric):
   Train twin IQN critics to learn the full return distribution.
   Overestimation penalized 2x harder than underestimation.

2. CVaR ACTOR LOSS:
   Policy maximizes CVaR_α (expected return in the worst α% of outcomes),
   NOT expected return. This makes the agent naturally conservative.

3. EDGE GATE LOSS (Binary Cross-Entropy):
   Train the edge gate to predict whether a trade will be profitable.
   Uses hindsight: after seeing the reward, label the state as good/bad.

4. REGIME VAE LOSS (KL Divergence):
   Keep the latent regime space structured — similar market states
   should have similar regime vectors.

5. ENTROPY REGULARIZATION (Auto-tuned):
   SAC-style automatic entropy tuning — prevents mode collapse while
   maintaining the CVaR objective.

6. EDGE UTILIZATION BONUS:
   Prevents the edge gate from collapsing to always-zero (never trading).
   Small reward for maintaining a healthy trading frequency.

Training loop:
    For each step:
    1. Agent observes state → edge gate decides confidence
    2. Action = edge_score × policy_output (conviction-based sizing)
    3. Environment returns reward
    4. Store in replay buffer with regime_z
    5. Sample batch → update all 6 losses
"""

import os
import json
import time
import copy
import logging
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from nandi.config import TRAINING_CONFIG, MODEL_DIR
from nandi.models.aegis import AEGISAgent, DistributionalTwinCritic, RegimeVAE

logger = logging.getLogger(__name__)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════
# Replay Buffer with Edge Labels
# ═══════════════════════════════════════════════════════════════════════

class AEGISReplayBuffer:
    """Replay buffer that stores edge gate training signals.

    Each transition stores:
    - Standard: (state, action, reward, next_state, done)
    - AEGIS: edge_label (was this a good time to trade?)
    """

    def __init__(self, capacity=200000):
        self.buffer = deque(maxlen=capacity)

    def add(self, market_state, position_info, action, reward,
            next_market_state, next_position_info, done, edge_label=None):
        """Store transition with optional edge label.

        edge_label: float in [0, 1], 1 = profitable step, 0 = losing step.
        If None, computed from reward sign.
        """
        if edge_label is None:
            # Soft label: sigmoid of scaled reward
            # Positive reward → label ≈ 1, negative → label ≈ 0
            edge_label = 1.0 / (1.0 + np.exp(-10.0 * reward))

        self.buffer.append((
            market_state, position_info, action, reward,
            next_market_state, next_position_info, done, edge_label,
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
        edge_labels = np.array([b[7] for b in batch], dtype=np.float32)

        return ms, pi, actions, rewards, next_ms, next_pi, dones, edge_labels

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════
# Quantile Huber Loss — for distributional critic training
# ═══════════════════════════════════════════════════════════════════════

def asymmetric_quantile_huber_loss(predictions, targets, tau, kappa=1.0,
                                   asymmetry_factor=2.0):
    """Quantile Huber loss with asymmetric overestimation penalty.

    Standard quantile regression: ρ_τ(u) = u * (τ - 1[u < 0])
    Huber variant: smooth near zero for stability
    Asymmetric: multiply by asymmetry_factor when predictions > targets
    (penalize overestimation harder — conservative bias)

    Args:
        predictions: (batch, n_quantiles_pred) — predicted quantile values
        targets: (batch, n_quantiles_target) — target quantile values
        tau: (batch, n_quantiles_pred) — quantile levels for predictions
        kappa: Huber threshold
        asymmetry_factor: multiplier for overestimation loss (>1 = pessimistic)

    Returns:
        scalar loss
    """
    # Pairwise TD errors: (batch, n_pred, n_target)
    pairwise_delta = targets.unsqueeze(1) - predictions.unsqueeze(2)

    # Huber loss
    abs_delta = pairwise_delta.abs()
    huber = torch.where(
        abs_delta <= kappa,
        0.5 * pairwise_delta.pow(2),
        kappa * (abs_delta - 0.5 * kappa),
    )

    # Quantile weighting: τ for positive errors, (1-τ) for negative
    # tau: (batch, n_pred) → (batch, n_pred, 1)
    tau_expanded = tau.unsqueeze(2)
    quantile_weight = torch.where(
        pairwise_delta >= 0,
        tau_expanded,
        1 - tau_expanded,
    )

    # Asymmetric penalty: overestimation (predictions > targets, delta < 0) penalized more
    asymmetric_weight = torch.where(
        pairwise_delta < 0,  # overestimation: predicted > target
        torch.tensor(asymmetry_factor, device=predictions.device),
        torch.tensor(1.0, device=predictions.device),
    )

    loss = (quantile_weight * huber * asymmetric_weight).mean()
    return loss


# ═══════════════════════════════════════════════════════════════════════
# AEGIS Trainer
# ═══════════════════════════════════════════════════════════════════════

class AEGISTrainer:
    """Training loop for AEGIS algorithm.

    Combines 6 loss terms into a coherent training procedure:
    1. Distributional critic (quantile Huber, asymmetric)
    2. CVaR actor (optimize worst α% of outcomes)
    3. Edge gate (binary cross-entropy on profitability)
    4. Regime VAE (KL divergence)
    5. Entropy (auto-tuned SAC-style)
    6. Edge utilization (prevent always-zero gate)
    """

    def __init__(self, agent, train_env, eval_env=None,
                 training_config=None,
                 # AEGIS-specific hyperparameters
                 cvar_alpha=0.25,          # optimize worst 25% of outcomes
                 n_quantiles=32,           # quantile resolution
                 asymmetry_factor=2.0,     # overestimation penalty
                 kl_coef=0.01,             # regime VAE weight
                 edge_coef=1.0,            # edge gate loss weight
                 edge_util_coef=0.1,       # edge utilization bonus weight
                 batch_size=256,
                 buffer_capacity=200000,
                 tau_soft=0.005,           # Polyak averaging
                 gamma=0.99,
                 lr=3e-4,
                 warmup_steps=1000):

        self.agent = agent
        self.train_env = train_env
        self.eval_env = eval_env
        self.config = training_config or TRAINING_CONFIG
        self.device = DEVICE

        # AEGIS hyperparameters
        self.cvar_alpha = cvar_alpha
        self.n_quantiles = n_quantiles
        self.asymmetry_factor = asymmetry_factor
        self.kl_coef = kl_coef
        self.edge_coef = edge_coef
        self.edge_util_coef = edge_util_coef
        self.batch_size = batch_size
        self.tau_soft = tau_soft
        self.gamma = gamma
        self.warmup_steps = warmup_steps

        self.agent.to(self.device)

        # ── Distributional Twin Critics ──
        # State dim for critics: state_enc(128) + pos_emb(32) + regime_z(agent.regime_dim)
        state_dim = agent.d_model + 32 + agent.regime_dim
        self.critic = DistributionalTwinCritic(
            state_dim=state_dim,
            action_dim=1,
            hidden_dim=128,
            n_quantiles=n_quantiles,
        ).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)

        # ── Optimizers ──
        # Separate optimizers for each component
        self.actor_optimizer = torch.optim.Adam(
            list(agent.policy_trunk.parameters()) +
            list(agent.actor_mean.parameters()) +
            list(agent.log_std_head.parameters()) +
            list(agent.feature_proj.parameters()) +
            list(agent.encoder.parameters()) +
            list(agent.position_embed.parameters()),
            lr=lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr,
        )
        self.edge_optimizer = torch.optim.Adam(
            agent.edge_gate.parameters(), lr=lr,
        )
        self.regime_optimizer = torch.optim.Adam(
            agent.regime_vae.parameters(), lr=lr * 0.5,  # slower for stability
        )

        # Auto entropy tuning (SAC-style)
        self.target_entropy = -1.0  # standard SAC target for 1D action
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.alpha_min = 0.1   # HIGHER floor — must keep exploring
        self.alpha_max = 5.0   # prevent α explosion

        # Reward normalization (running stats)
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0

        # Replay buffer
        self.replay_buffer = AEGISReplayBuffer(buffer_capacity)

        # Training state
        self.total_steps = 0
        self.best_eval_return = -np.inf
        self.training_stats = []
        self.edge_score_history = deque(maxlen=1000)  # track edge gate behavior

        # Dashboard logging — trades, equity, metrics for visualization
        self.dashboard_log = os.path.join(MODEL_DIR, "training_dashboard.jsonl")
        os.makedirs(MODEL_DIR, exist_ok=True)

    def _get_combined_state(self, market_state, position_info, detach=False):
        """Get full combined state encoding for critic/edge gate input.

        Returns: combined (batch, state_dim + 32 + regime_dim)
        """
        (state_enc, pos_emb, regime_mu, regime_log_var,
         regime_z, combined) = self.agent.encode_state(market_state, position_info)
        if detach:
            combined = combined.detach()
        return combined, regime_mu, regime_log_var

    def _normalize_reward(self, reward):
        """Running reward normalization — keeps critic/actor in stable range."""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_var += (delta * delta2 - self.reward_var) / max(self.reward_count, 1)
        std = max(np.sqrt(self.reward_var), 1e-6)
        return np.clip(reward / std, -10.0, 10.0)

    def _soft_update(self):
        """Polyak averaging for target critic."""
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(self.tau_soft * p.data + (1 - self.tau_soft) * tp.data)

    def _update(self):
        """Single AEGIS update step — the core algorithm.

        Updates all 6 components:
        1. Distributional critics (asymmetric quantile Huber)
        2. Policy (CVaR objective)
        3. Edge gate (profitability prediction)
        4. Regime VAE (KL divergence)
        5. Entropy temperature (auto-tuning)
        6. Edge utilization (prevent collapse)
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        (ms, pi, actions, rewards, next_ms, next_pi,
         dones, edge_labels) = self.replay_buffer.sample(self.batch_size)

        ms_t = torch.tensor(ms, dtype=torch.float32, device=self.device)
        pi_t = torch.tensor(pi, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rew_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_ms_t = torch.tensor(next_ms, dtype=torch.float32, device=self.device)
        next_pi_t = torch.tensor(next_pi, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        edge_label_t = torch.tensor(edge_labels, dtype=torch.float32, device=self.device)

        alpha = self.log_alpha.exp().detach()
        batch = ms_t.shape[0]

        # ════════════════════════════════════════════════════
        # LOSS 1: Distributional Critic (Asymmetric Quantile Huber)
        # ════════════════════════════════════════════════════
        with torch.no_grad():
            # Next action from policy (un-gated for critic target)
            next_raw_action, next_log_prob, _ = self.agent.get_raw_action(
                next_ms_t, next_pi_t
            )
            next_combined, _, _ = self._get_combined_state(next_ms_t, next_pi_t)

            # Sample tau' for target quantiles
            next_tau = torch.rand(batch, self.n_quantiles, device=self.device)

            # Target quantile values (min of twin critics)
            z1_next, z2_next, _ = self.target_critic(
                next_combined, next_raw_action, next_tau
            )
            z_next = torch.min(z1_next, z2_next)

            # Distributional Bellman target: r + γ(1-d)(Z' - α·log_π')
            target_quantiles = rew_t.unsqueeze(1) + \
                (1 - done_t.unsqueeze(1)) * self.gamma * \
                (z_next - alpha * next_log_prob.unsqueeze(1))
            # Clamp targets to prevent critic value explosion
            target_quantiles = target_quantiles.clamp(-5.0, 5.0)

        # Current quantile estimates
        combined, _, _ = self._get_combined_state(ms_t, pi_t, detach=True)

        tau = torch.rand(batch, self.n_quantiles, device=self.device)
        z1, z2, _ = self.critic(combined, act_t, tau)

        critic_loss = (
            asymmetric_quantile_huber_loss(
                z1, target_quantiles, tau,
                asymmetry_factor=self.asymmetry_factor
            ) +
            asymmetric_quantile_huber_loss(
                z2, target_quantiles, tau,
                asymmetry_factor=self.asymmetry_factor
            )
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ════════════════════════════════════════════════════
        # LOSS 2: CVaR Actor (optimize worst α% of outcomes)
        # ════════════════════════════════════════════════════
        # Sample new actions from current policy
        raw_action_new, log_prob_new, _ = self.agent.get_raw_action(ms_t, pi_t)
        combined_new, regime_mu, regime_log_var = self._get_combined_state(ms_t, pi_t)

        # CVaR: sample tau from U[0, α] instead of U[0, 1]
        # This focuses on the LEFT TAIL of the return distribution
        cvar_tau = torch.rand(batch, self.n_quantiles, device=self.device) * self.cvar_alpha

        z1_new, _ = self.critic.z1_forward(combined_new, raw_action_new, cvar_tau)
        cvar_value = z1_new.mean(dim=1)  # CVaR = mean of worst-α% quantiles

        # Normalize CVaR to prevent actor loss divergence
        # Use advantage-style: subtract baseline (running mean of CVaR)
        cvar_value_normalized = cvar_value - cvar_value.detach().mean()

        # Actor loss: maximize CVaR + entropy
        actor_loss = (alpha * log_prob_new - cvar_value_normalized).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.actor_optimizer.step()

        # ════════════════════════════════════════════════════
        # LOSS 3: Edge Gate (profitability prediction)
        # ════════════════════════════════════════════════════
        # Re-compute edge scores (with fresh gradients)
        _, _, edge_score, _, _, _ = self.agent.get_policy_and_edge(ms_t, pi_t)
        edge_pred = edge_score.squeeze(-1)  # (batch,)

        edge_loss = F.binary_cross_entropy(edge_pred, edge_label_t)

        # LOSS 6: Edge utilization bonus — prevent gate collapse to zero
        # Target: edge should be active ~30-60% of the time
        mean_edge = edge_pred.mean()
        target_utilization = 0.4  # healthy trading frequency
        util_loss = (mean_edge - target_utilization).pow(2)

        total_edge_loss = self.edge_coef * edge_loss + self.edge_util_coef * util_loss

        self.edge_optimizer.zero_grad()
        total_edge_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.edge_gate.parameters(), 1.0)
        self.edge_optimizer.step()

        # ════════════════════════════════════════════════════
        # LOSS 4: Regime VAE (KL divergence)
        # ════════════════════════════════════════════════════
        # Re-encode to get regime parameters (fresh graph)
        _, _, r_mu, r_lv, _, _ = self.agent.encode_state(ms_t, pi_t)
        kl_loss = RegimeVAE.kl_loss(r_mu, r_lv)
        regime_loss = self.kl_coef * kl_loss

        self.regime_optimizer.zero_grad()
        regime_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.regime_vae.parameters(), 1.0)
        self.regime_optimizer.step()

        # ════════════════════════════════════════════════════
        # LOSS 5: Entropy Temperature (auto-tuning)
        # ════════════════════════════════════════════════════
        alpha_loss = -(self.log_alpha * (
            log_prob_new.detach() + self.target_entropy
        )).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Clamp alpha to prevent collapse or explosion
        with torch.no_grad():
            self.log_alpha.clamp_(
                min=np.log(self.alpha_min),
                max=np.log(self.alpha_max),
            )

        # ── Soft target update ──
        self._soft_update()

        # Track edge gate behavior
        self.edge_score_history.append(mean_edge.item())

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "edge_loss": edge_loss.item(),
            "kl_loss": kl_loss.item(),
            "alpha": alpha.item(),
            "entropy": -log_prob_new.mean().item(),
            "edge_mean": mean_edge.item(),
            "cvar_value": cvar_value.mean().item(),
            "util_loss": util_loss.item(),
        }

    def train(self):
        """Main AEGIS training loop."""
        total_timesteps = self.config["total_timesteps"]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  AEGIS Training — {total_timesteps:,} timesteps")
        logger.info(f"  CVaR α={self.cvar_alpha} | Quantiles={self.n_quantiles} "
                    f"| Asymmetry={self.asymmetry_factor}x")
        logger.info(f"  Device: {self.device}")
        logger.info(f"{'=' * 60}\n")

        start_time = time.time()

        # Clear dashboard log for fresh run
        with open(self.dashboard_log, "w") as f:
            pass

        state = self.train_env.reset()
        episode_reward = 0
        episode_rewards = deque(maxlen=100)
        episode_edge_scores = deque(maxlen=100)
        episode_trades = []       # trades within current episode
        episode_equity = []       # equity snapshots within current episode
        episode_count = 0

        while self.total_steps < total_timesteps:
            ms, pi = state

            if self.total_steps < self.warmup_steps:
                # Random exploration during warmup
                action = np.random.uniform(-1, 1)
                edge_score = 0.5
            else:
                action, _, _, edge_score = self.agent.get_action(
                    ms, pi, deterministic=False
                )

            next_state, reward, done, info = self.train_env.step(action)
            next_ms, next_pi = next_state

            # Track trades and equity for dashboard
            env_info = info if isinstance(info, dict) else {}
            equity_now = env_info.get("equity", 0)
            price_now = env_info.get("price", 0)
            position_now = env_info.get("position", action)

            if self.total_steps % 10 == 0 and equity_now > 0:
                episode_equity.append({
                    "step": self.total_steps,
                    "equity": round(float(equity_now), 2),
                    "price": round(float(price_now), 5) if price_now else 0,
                })

            if abs(action) > 0.05 and price_now:
                episode_trades.append({
                    "step": self.total_steps,
                    "action": round(float(action), 3),
                    "price": round(float(price_now), 5),
                    "edge": round(float(edge_score), 3),
                    "reward": round(float(reward), 4),
                })

            # Compute edge label from reward (profitability signal)
            # For trades: positive reward = edge was good
            # For flat: no reward = neutral
            if abs(action) > 0.05:
                edge_label = 1.0 / (1.0 + np.exp(-10.0 * reward))
            else:
                # Staying flat when reward would have been negative = good edge
                edge_label = 0.3 if reward < 0 else 0.5

            # Normalize reward to keep critic/actor in stable range
            norm_reward = self._normalize_reward(reward)

            self.replay_buffer.add(
                ms, pi, action, norm_reward,
                next_ms, next_pi, float(done), edge_label,
            )

            episode_reward += reward
            episode_edge_scores.append(edge_score)
            self.total_steps += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_count += 1

                # Log episode to dashboard file (every 5th episode to save disk)
                if episode_count % 5 == 0:
                    try:
                        with open(self.dashboard_log, "a") as f:
                            f.write(json.dumps({
                                "type": "episode",
                                "episode": episode_count,
                                "step": self.total_steps,
                                "reward": round(float(episode_reward), 2),
                                "trades": episode_trades[-50:],  # last 50 trades
                                "equity": episode_equity[-200:],  # last 200 snapshots
                            }) + "\n")
                    except Exception:
                        pass

                episode_reward = 0
                episode_trades = []
                episode_equity = []
                state = self.train_env.reset()
            else:
                state = next_state

            # Update after warmup
            if self.total_steps >= self.warmup_steps:
                stats = self._update()

                if self.total_steps % 1000 == 0 and stats:
                    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                    avg_edge = np.mean(list(self.edge_score_history)) if self.edge_score_history else 0
                    logger.info(
                        f"Step {self.total_steps:>7,}/{total_timesteps:,} | "
                        f"C: {stats.get('critic_loss', 0):.4f} | "
                        f"A: {stats.get('actor_loss', 0):.4f} | "
                        f"Edge: {stats.get('edge_loss', 0):.4f} | "
                        f"KL: {stats.get('kl_loss', 0):.4f} | "
                        f"CVaR: {stats.get('cvar_value', 0):.4f} | "
                        f"α: {stats.get('alpha', 0):.3f} | "
                        f"Gate: {avg_edge:.2f} | "
                        f"R̄: {avg_reward:.3f}"
                    )
                    # Log metrics for dashboard
                    try:
                        with open(self.dashboard_log, "a") as f:
                            f.write(json.dumps({
                                "type": "metrics",
                                "step": self.total_steps,
                                "critic_loss": round(stats.get("critic_loss", 0), 4),
                                "actor_loss": round(stats.get("actor_loss", 0), 4),
                                "cvar_value": round(stats.get("cvar_value", 0), 4),
                                "alpha": round(stats.get("alpha", 0), 4),
                                "gate": round(avg_edge, 3),
                                "reward": round(avg_reward, 2),
                            }) + "\n")
                    except Exception:
                        pass

            # Evaluate
            if (self.eval_env is not None and
                self.total_steps % self.config["eval_interval"] == 0 and
                self.total_steps >= self.warmup_steps):
                eval_stats = self._evaluate()
                self.training_stats.append({
                    "step": self.total_steps,
                    **eval_stats,
                })

                if eval_stats["eval_return"] > self.best_eval_return:
                    self.best_eval_return = eval_stats["eval_return"]
                    self.agent.save_agent()
                    # Save critic too
                    critic_path = os.path.join(
                        os.path.dirname(
                            self.agent.save_agent.__func__(self.agent) or
                            os.path.join(MODEL_DIR, "aegis_agent.pt")
                        ) if False else MODEL_DIR,
                        "aegis_critic.pt"
                    )
                    logger.info(f"  ** New best: {eval_stats['eval_return']:.2f}% **")

            # Save periodically
            if self.total_steps % self.config.get("save_interval", 50000) == 0:
                self.agent.save_agent()

        total_time = time.time() - start_time
        logger.info(f"\nAEGIS training complete in {total_time / 60:.1f} minutes")
        logger.info(f"Best eval return: {self.best_eval_return:.2f}%")

        # Final save
        self.agent.save_agent()
        return self.training_stats

    def _evaluate(self, n_episodes=None):
        """Evaluate AEGIS policy (deterministic + edge-gated)."""
        self.agent.eval()
        n_episodes = n_episodes or self.config.get("n_eval_episodes", 10)
        returns = []
        edge_scores = []
        trade_counts = []

        for _ in range(n_episodes):
            state = self.eval_env.reset()
            done = False
            ep_edges = []
            while not done:
                ms, pi = state
                action, _, _, edge = self.agent.get_action(
                    ms, pi, deterministic=True
                )
                ep_edges.append(edge)
                state, _, done, info = self.eval_env.step(action)
            returns.append(info["return_pct"])
            edge_scores.append(np.mean(ep_edges))
            trade_counts.append(info.get("total_trades", 0))

        mean_return = float(np.mean(returns))
        mean_edge = float(np.mean(edge_scores))
        mean_trades = float(np.mean(trade_counts))

        logger.info(
            f"AEGIS EVAL | Return: {mean_return:+.2f}% | "
            f"Edge: {mean_edge:.2f} | "
            f"Trades: {mean_trades:.0f} | "
            f"DD: {info.get('drawdown', 0):.2%}"
        )

        self.agent.train()
        return {
            "eval_return": mean_return,
            "eval_edge": mean_edge,
            "eval_trades": mean_trades,
        }
