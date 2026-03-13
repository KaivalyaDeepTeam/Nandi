"""
Phase 2: Rainbow-IQN-DQN Trainer.

Combines: Double DQN + Dueling + IQN + Prioritized Replay + N-step + NoisyNet.

Key design: Optimize EXPECTED RETURN (not CVaR). Risk is enforced via
environment constraints (DD limits, session loss cooldown, position caps).
Agent learns to maximize profit while respecting hard risk limits.

Multi-pair training: cycles through all pairs uniformly. Each batch samples
from a shared replay buffer containing all pairs. Pair embedding differentiates
pair-specific patterns.
"""

import os
import copy
import json
import time
import logging
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from nandi.config import DQN_CONFIG, TRAINING_CONFIG, MODEL_DIR
from nandi.training.replay_buffer import PrioritizedReplayBuffer, NStepBuffer

logger = logging.getLogger(__name__)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DQNTrainer:
    """Rainbow-IQN-DQN trainer for discrete-action forex trading.

    Features:
    - Double DQN: online network selects action, target evaluates
    - Dueling architecture: value + advantage streams
    - IQN: implicit quantile network for distributional RL
    - PER: prioritized experience replay
    - N-step: multi-step bootstrapped returns
    - NoisyNet: learned exploration (no ε-greedy)
    """

    def __init__(self, agent, train_envs, eval_envs=None, dqn_config=None,
                 training_config=None, device=None, freeze_encoder=False):
        """
        Args:
            agent: NandiDQNAgent instance (pre-trained via HOA or fresh)
            train_envs: list of MultiEpisodeDiscreteEnv (one per pair)
            eval_envs: list of MultiEpisodeDiscreteEnv for evaluation
            dqn_config: override DQN_CONFIG
            training_config: override TRAINING_CONFIG
            device: torch.device
            freeze_encoder: if True, freeze encoder+feature_proj during RL
                (preserves HOA feature representations)
        """
        self.cfg = dqn_config or DQN_CONFIG
        self.train_cfg = training_config or TRAINING_CONFIG
        self.device = device or get_device()

        self.agent = agent.to(self.device)
        self.target_agent = copy.deepcopy(agent).to(self.device)
        self.target_agent.eval()

        self.train_envs = train_envs
        self.eval_envs = eval_envs or []

        # Freeze encoder if requested (preserve HOA features)
        if freeze_encoder:
            frozen_count = 0
            for name, param in self.agent.named_parameters():
                if name.startswith(("encoder.", "feature_proj.")):
                    param.requires_grad = False
                    frozen_count += 1
            logger.info(f"Froze {frozen_count} encoder/feature_proj parameters")

        # Optimizer (only trainable params)
        trainable_params = [p for p in self.agent.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            trainable_params, lr=self.cfg["lr"],
        )

        # Replay buffer (shared across all pairs)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.cfg["buffer_capacity"],
            alpha=self.cfg["per_alpha"],
            beta_start=self.cfg["per_beta_start"],
            beta_end=self.cfg["per_beta_end"],
            beta_steps=self.cfg["total_steps"],
        )

        # N-step buffers (one per pair to avoid cross-episode contamination)
        self.n_step_buffers = {
            env.pair_idx: NStepBuffer(
                n_step=self.cfg["n_step"],
                gamma=self.cfg["gamma"],
            )
            for env in train_envs
        }

        # Training state
        self.total_steps = 0
        self.best_eval_return = -np.inf
        self.training_stats = []

        # Dashboard
        self.dashboard_log = os.path.join(MODEL_DIR, "dqn_training.jsonl")
        os.makedirs(MODEL_DIR, exist_ok=True)

    def _hard_target_update(self):
        """Copy online network weights to target network."""
        self.target_agent.load_state_dict(self.agent.state_dict())

    def _update(self):
        """Single DQN update step.

        Returns:
            stats: dict with loss and metrics
        """
        if len(self.replay_buffer) < self.cfg["batch_size"]:
            return {}

        batch, indices, is_weights = self.replay_buffer.sample(
            self.cfg["batch_size"],
        )

        ms_t = torch.tensor(batch["market_state"], device=self.device)
        pi_t = torch.tensor(batch["position_info"], device=self.device)
        act_t = torch.tensor(batch["actions"], device=self.device).long()
        rew_t = torch.tensor(batch["rewards"], device=self.device)
        next_ms_t = torch.tensor(batch["next_market_state"], device=self.device)
        next_pi_t = torch.tensor(batch["next_position_info"], device=self.device)
        done_t = torch.tensor(batch["dones"], device=self.device)
        pair_ids = torch.tensor(batch["pair_ids"], device=self.device).long()
        is_w = torch.tensor(is_weights, device=self.device)

        if batch["action_masks"] is not None:
            mask_t = torch.tensor(batch["action_masks"], device=self.device).bool()
        else:
            mask_t = None

        B = ms_t.shape[0]
        n_tau = self.cfg["n_tau"]
        gamma = self.cfg["gamma"] ** self.cfg["n_step"]  # n-step discount

        # ── Target first (all no_grad, safe to reset noise) ──
        with torch.no_grad():
            # Double DQN: online selects best action
            self.agent.reset_noise()
            next_q = self.agent.get_q_values(
                next_ms_t, next_pi_t, pair_ids,
                action_mask=mask_t,
            )
            next_actions = next_q.argmax(dim=-1)  # (B,)

            # Target evaluates that action's quantiles
            tau_target = torch.rand(B, n_tau, device=self.device)
            target_quantiles, _ = self.target_agent(
                next_ms_t, next_pi_t, pair_ids, tau=tau_target,
            )
            na_expanded = next_actions.unsqueeze(1).unsqueeze(2).expand(-1, n_tau, -1)
            target_q = target_quantiles.gather(2, na_expanded).squeeze(2)

            # Bellman target: R + γ^n * Q_target(s', a*)
            target = rew_t.unsqueeze(1) + (1 - done_t.unsqueeze(1)) * gamma * target_q
            target = target.clamp(-5.0, 5.0)

        # ── Online network: Q-quantiles for taken actions (single grad pass) ──
        # Reset noise ONCE, then no more in-place modifications until backward
        self.agent.reset_noise()
        tau = torch.rand(B, n_tau, device=self.device)
        q_quantiles, _ = self.agent(ms_t, pi_t, pair_ids, tau=tau)
        act_expanded = act_t.unsqueeze(1).unsqueeze(2).expand(-1, n_tau, -1)
        q_taken = q_quantiles.gather(2, act_expanded).squeeze(2)  # (B, n_tau)

        # ── Quantile Huber loss with IS weighting (single pass) ──
        kappa = self.cfg["kappa"]
        # Pairwise TD errors: (B, n_pred, n_target)
        delta = target.unsqueeze(1) - q_taken.unsqueeze(2)
        abs_delta = delta.abs()
        huber = torch.where(
            abs_delta <= kappa,
            0.5 * delta.pow(2),
            kappa * (abs_delta - 0.5 * kappa),
        )
        tau_exp = tau.unsqueeze(2)
        qw = torch.where(delta >= 0, tau_exp, 1 - tau_exp)
        per_sample_loss = (qw * huber).mean(dim=2).sum(dim=1)  # (B,)

        # IS-weighted loss
        weighted_loss = (is_w * per_sample_loss).mean()

        # TD errors for PER (detached)
        td_errors = abs_delta.mean(dim=(1, 2)).detach().cpu().numpy()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 10.0)
        self.optimizer.step()

        # Update PER priorities
        self.replay_buffer.update_priorities(indices, td_errors)

        # Hard target update
        if self.total_steps % self.cfg["target_update_freq"] == 0:
            self._hard_target_update()

        return {
            "loss": weighted_loss.item(),
            "td_error": float(np.mean(td_errors)),
            "q_mean": float(q_taken.detach().mean().item()),
        }

    def train(self):
        """Main DQN training loop.

        Cycles through environments, collects experience, trains.

        Returns:
            training_stats: list of eval dicts
        """
        total_steps = self.cfg["total_steps"]

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Phase 2: Rainbow-IQN-DQN Training")
        logger.info(f"  Steps: {total_steps:,} | Pairs: {len(self.train_envs)}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"{'=' * 60}\n")

        start_time = time.time()

        # Clear dashboard
        with open(self.dashboard_log, "w") as f:
            pass

        # Initialize environments
        env_states = []
        for env in self.train_envs:
            state = env.reset()
            env_states.append(state)

        episode_rewards = deque(maxlen=100)
        episode_trades = deque(maxlen=100)
        current_rewards = [0.0] * len(self.train_envs)
        env_idx = 0  # round-robin through envs
        action_counts = np.zeros(4, dtype=np.int64)

        eval_interval = self.train_cfg.get("eval_interval", 25_000)
        save_interval = self.train_cfg.get("save_interval", 50_000)

        while self.total_steps < total_steps:
            env = self.train_envs[env_idx]
            ms, pi = env_states[env_idx]
            pair_idx = env.pair_idx

            # Get action (NoisyNet + small epsilon-greedy for diversity)
            mask = env.get_action_mask()
            valid_actions = np.where(mask)[0]
            if self.total_steps < self.cfg["warmup_steps"]:
                # Random action during warmup
                action = int(np.random.choice(valid_actions))
                q_values = np.zeros(4)
            else:
                self.agent.train()
                self.agent.reset_noise()
                action, q_values = self.agent.get_action(
                    ms, pi, pair_idx, action_mask=mask,
                )
                # Small epsilon-greedy on top of NoisyNet (decays over training)
                progress = min(1.0, self.total_steps / self.cfg["total_steps"])
                eps = max(0.02, 0.1 * (1.0 - progress))  # 10% → 2%
                if np.random.random() < eps:
                    action = int(np.random.choice(valid_actions))

            action_counts[action] += 1

            # Step environment
            next_state, reward, done, info = env.step(action)
            next_ms, next_pi = next_state

            # N-step buffer
            nstep_buf = self.n_step_buffers[pair_idx]
            transitions = nstep_buf.add(
                ms, pi, action, reward,
                next_ms, next_pi, float(done),
                pair_idx=pair_idx,
                action_mask=mask,
            )

            # Add n-step transitions to PER buffer
            for t in transitions:
                self.replay_buffer.add(*t)

            current_rewards[env_idx] += reward
            self.total_steps += 1

            if done:
                episode_rewards.append(current_rewards[env_idx])
                episode_trades.append(info.get("total_trades", 0))
                current_rewards[env_idx] = 0.0
                # Reset n-step buffer for this pair
                self.n_step_buffers[pair_idx].reset()
                env_states[env_idx] = env.reset()
            else:
                env_states[env_idx] = next_state

            # Round-robin to next environment
            env_idx = (env_idx + 1) % len(self.train_envs)

            # Update after warmup
            if self.total_steps >= self.cfg["warmup_steps"]:
                stats = self._update()

                if self.total_steps % 1000 == 0 and stats:
                    avg_r = np.mean(episode_rewards) if episode_rewards else 0
                    avg_trades = np.mean(episode_trades) if episode_trades else 0
                    total_actions = max(1, action_counts.sum())
                    act_dist = action_counts / total_actions * 100

                    logger.info(
                        f"Step {self.total_steps:>7,}/{total_steps:,} | "
                        f"Loss: {stats.get('loss', 0):.4f} | "
                        f"TD: {stats.get('td_error', 0):.4f} | "
                        f"Q̄: {stats.get('q_mean', 0):.3f} | "
                        f"R̄: {avg_r:.3f} | "
                        f"Trades: {avg_trades:.0f} | "
                        f"H/L/S/C: {act_dist[0]:.0f}/{act_dist[1]:.0f}/"
                        f"{act_dist[2]:.0f}/{act_dist[3]:.0f}%"
                    )

                    try:
                        with open(self.dashboard_log, "a") as f:
                            f.write(json.dumps({
                                "step": self.total_steps,
                                "loss": round(stats.get("loss", 0), 4),
                                "td_error": round(stats.get("td_error", 0), 4),
                                "q_mean": round(stats.get("q_mean", 0), 4),
                                "reward": round(avg_r, 3),
                                "trades": round(avg_trades, 1),
                                "action_dist": act_dist.tolist(),
                            }) + "\n")
                    except Exception:
                        pass

            # Evaluate
            if (self.eval_envs and
                    self.total_steps % eval_interval == 0 and
                    self.total_steps >= self.cfg["warmup_steps"]):
                eval_stats = self._evaluate()
                self.training_stats.append({
                    "step": self.total_steps, **eval_stats,
                })
                if eval_stats["eval_return"] > self.best_eval_return:
                    self.best_eval_return = eval_stats["eval_return"]
                    self.agent.save_agent()
                    logger.info(
                        f"  ** New best: {eval_stats['eval_return']:.2f}% | "
                        f"Trades: {eval_stats['eval_trades']:.0f} **"
                    )

            # Save periodically
            if self.total_steps % save_interval == 0:
                self.agent.save_agent()

        elapsed = time.time() - start_time
        logger.info(f"\nDQN training complete in {elapsed / 60:.1f} minutes")
        logger.info(f"Best eval return: {self.best_eval_return:.2f}%")

        self.agent.save_agent()
        return self.training_stats

    def _evaluate(self, n_episodes=None):
        """Evaluate DQN agent across all eval envs.

        Uses greedy (argmax) action selection. Q-value spreads are small
        (~0.04), so softmax with any reasonable temperature degrades to
        near-random. Argmax correctly reflects what the agent has learned.
        """
        self.agent.eval()
        n_episodes = n_episodes or self.train_cfg.get("n_eval_episodes", 10)

        returns = []
        trade_counts = []
        win_rates = []
        action_dists = np.zeros(4)
        max_dd = 0.0
        q_spreads = []

        for eval_env in self.eval_envs:
            for _ in range(max(1, n_episodes // len(self.eval_envs))):
                state = eval_env.reset()
                done = False
                ep_actions = []

                while not done:
                    ms, pi = state
                    mask = eval_env.get_action_mask()
                    _, q_values = self.agent.get_action(
                        ms, pi, eval_env.pair_idx,
                        action_mask=mask, deterministic=True,
                    )
                    # Greedy action selection (argmax)
                    q_masked = q_values.copy()
                    q_masked[~mask] = -1e8
                    action = int(np.argmax(q_masked))

                    # Track Q-value spread for diagnostics
                    valid_q = q_values[mask]
                    if len(valid_q) > 1:
                        q_spreads.append(float(valid_q.max() - valid_q.min()))

                    ep_actions.append(action)
                    state, _, done, info = eval_env.step(action)

                returns.append(info.get("return_pct", 0))
                trade_counts.append(info.get("total_trades", 0))
                win_rates.append(info.get("win_rate", 0))
                max_dd = max(max_dd, info.get("drawdown", 0))

                for a in ep_actions:
                    action_dists[a] += 1

        total_a = max(1, action_dists.sum())
        mean_return = float(np.mean(returns)) if returns else 0
        mean_trades = float(np.mean(trade_counts)) if trade_counts else 0
        mean_wr = float(np.mean(win_rates)) if win_rates else 0
        mean_q_spread = float(np.mean(q_spreads)) if q_spreads else 0

        logger.info(
            f"DQN EVAL | Return: {mean_return:+.2f}% | "
            f"Trades: {mean_trades:.0f} | "
            f"WR: {mean_wr:.1%} | "
            f"MaxDD: {max_dd:.2%} | "
            f"Q-spread: {mean_q_spread:.4f} | "
            f"H/L/S/C: {action_dists[0]/total_a*100:.0f}/"
            f"{action_dists[1]/total_a*100:.0f}/"
            f"{action_dists[2]/total_a*100:.0f}/"
            f"{action_dists[3]/total_a*100:.0f}%"
        )

        self.agent.train()
        return {
            "eval_return": mean_return,
            "eval_trades": mean_trades,
            "eval_win_rate": mean_wr,
            "eval_max_dd": max_dd,
            "q_spread": mean_q_spread,
        }
