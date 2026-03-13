"""
Phase 2 (PPO): Proximal Policy Optimization for discrete-action trading.

Key advantage over DQN: PPO outputs action PROBABILITIES via policy gradient.
- No Q-value dominance problem (DQN's argmax always picks HOLD)
- Entropy bonus prevents collapse to single action
- On-policy: learns from its own distribution (no off-policy bias)
- GAE-lambda for low-variance advantage estimation

Multi-pair training: collects rollouts from all pairs, computes GAE per-pair
to avoid cross-episode contamination, then shuffles for minibatch updates.
"""

import os
import time
import logging

import numpy as np
import torch

from nandi.config import PPO_CONFIG, TRAINING_CONFIG, MODEL_DIR

logger = logging.getLogger(__name__)


class RolloutBuffer:
    """Stores on-policy transitions for PPO updates.

    Collects (state, action, reward, value, log_prob, done) tuples,
    then computes GAE-lambda advantages and discounted returns.
    """

    def __init__(self):
        self.market_states = []
        self.position_infos = []
        self.pair_ids = []
        self.actions = []
        self.action_masks = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self._advantages = None
        self._returns = None

    def add(self, ms, pi, pair_id, action, mask, log_prob, reward, value, done):
        self.market_states.append(ms)
        self.position_infos.append(pi)
        self.pair_ids.append(pair_id)
        self.actions.append(action)
        self.action_masks.append(mask)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value, gamma, lam):
        """GAE-Lambda advantage estimation.

        Args:
            last_value: V(s_T) bootstrap value for last state
            gamma: discount factor
            lam: GAE lambda parameter
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            next_non_terminal = 1.0 - float(self.dones[t])
            delta = (self.rewards[t] + gamma * next_value * next_non_terminal
                     - self.values[t])
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[t] = gae

        self._advantages = advantages
        self._returns = advantages + np.array(self.values, dtype=np.float32)

    def get_batches(self, batch_size):
        """Yield shuffled mini-batches as numpy arrays."""
        n = len(self.rewards)
        indices = np.random.permutation(n)

        ms = np.array(self.market_states)
        pi = np.array(self.position_infos)
        pair_ids = np.array(self.pair_ids, dtype=np.int64)
        actions = np.array(self.actions, dtype=np.int64)
        masks = np.array(self.action_masks)
        old_log_probs = np.array(self.log_probs, dtype=np.float32)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            yield {
                "market_states": ms[idx],
                "position_infos": pi[idx],
                "pair_ids": pair_ids[idx],
                "actions": actions[idx],
                "action_masks": masks[idx],
                "old_log_probs": old_log_probs[idx],
                "advantages": self._advantages[idx],
                "returns": self._returns[idx],
            }

    def __len__(self):
        return len(self.rewards)


class PPOTrainer:
    """PPO trainer for discrete-action forex trading.

    Collects rollouts per-env (correct GAE), shuffles across envs
    for minibatch updates, with entropy bonus to prevent HOLD collapse.
    """

    def __init__(self, agent, train_envs, eval_envs=None, ppo_config=None,
                 training_config=None, device=None, freeze_encoder=False):
        self.cfg = ppo_config or PPO_CONFIG
        self.train_cfg = training_config or TRAINING_CONFIG
        self.device = device or torch.device("cpu")

        self.agent = agent.to(self.device)
        self.train_envs = train_envs
        self.eval_envs = eval_envs or []

        # PPO hyperparameters
        self.gamma = self.cfg["gamma"]
        self.lam = self.cfg["lambda_gae"]
        self.clip_ratio = self.cfg["clip_ratio"]
        self.entropy_coef = self.cfg["entropy_coef"]
        self.value_coef = self.cfg["value_coef"]
        self.max_grad_norm = self.cfg["max_grad_norm"]
        self.n_epochs = self.cfg["n_epochs"]
        self.batch_size = self.cfg["batch_size"]
        self.rollout_length = self.cfg["rollout_length"]
        self.min_entropy = self.cfg.get("min_entropy", 0.3)

        self.total_steps = self.train_cfg.get("total_timesteps", 500_000)
        self.eval_interval = self.train_cfg.get("eval_interval", 25_000)
        self.save_interval = self.train_cfg.get("save_interval", 50_000)
        self.n_eval_episodes = self.train_cfg.get("n_eval_episodes", 5)

        # Differential LR: encoder at 10x lower LR to preserve HOA features
        if freeze_encoder:
            encoder_params = []
            head_params = []
            for name, param in self.agent.named_parameters():
                if name.startswith(("encoder.", "feature_proj.")):
                    encoder_params.append(param)
                else:
                    head_params.append(param)
            encoder_lr = self.cfg["learning_rate"] * 0.1
            self.optimizer = torch.optim.Adam([
                {"params": encoder_params, "lr": encoder_lr},
                {"params": head_params, "lr": self.cfg["learning_rate"]},
            ])
            logger.info(
                f"Differential LR: {len(encoder_params)} encoder params "
                f"at {encoder_lr:.1e}, {len(head_params)} head params "
                f"at {self.cfg['learning_rate']:.1e}"
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.agent.parameters(), lr=self.cfg["learning_rate"],
            )

        # LR scheduler
        n_updates = max(1, self.total_steps // self.rollout_length)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_updates,
        )

        # Per-env state tracking
        self.env_states = [None] * len(train_envs)
        self.env_dones = [True] * len(train_envs)

    def collect_rollout(self):
        """Collect rollouts from all envs, compute GAE per-env, merge.

        Per-env collection ensures correct GAE (sequential transitions).
        Merging shuffles across envs for diverse minibatches.
        """
        self.agent.eval()

        n_envs = len(self.train_envs)
        steps_per_env = max(1, self.rollout_length // n_envs)
        env_buffers = []

        # Action distribution tracking
        action_counts = np.zeros(4, dtype=np.int64)
        episode_rewards = []
        current_ep_reward = [0.0] * n_envs

        for env_idx, env in enumerate(self.train_envs):
            # Reset if needed
            if self.env_dones[env_idx] or self.env_states[env_idx] is None:
                state = env.reset()
                self.env_states[env_idx] = state
                self.env_dones[env_idx] = False
                current_ep_reward[env_idx] = 0.0

            buf = RolloutBuffer()

            for _ in range(steps_per_env):
                ms, pi = self.env_states[env_idx]
                mask = env.get_action_mask()
                pair_id = env.pair_idx

                action, log_prob, value, probs = self.agent.get_action(
                    ms, pi, pair_id, action_mask=mask, deterministic=False,
                )

                next_state, reward, done, info = env.step(action)

                buf.add(
                    ms=ms.copy(), pi=pi.copy(), pair_id=pair_id,
                    action=action, mask=mask.copy(), log_prob=log_prob,
                    reward=reward, value=value, done=done,
                )
                action_counts[action] += 1
                current_ep_reward[env_idx] += reward

                if done:
                    episode_rewards.append(current_ep_reward[env_idx])
                    state = env.reset()
                    self.env_states[env_idx] = state
                    current_ep_reward[env_idx] = 0.0
                else:
                    self.env_states[env_idx] = next_state

            # Bootstrap value for GAE
            last_ms, last_pi = self.env_states[env_idx]
            _, _, last_value, _ = self.agent.get_action(
                last_ms, last_pi, env.pair_idx,
                action_mask=env.get_action_mask(),
                deterministic=False,
            )
            buf.compute_returns_and_advantages(last_value, self.gamma, self.lam)
            env_buffers.append(buf)

        # Merge all env buffers
        merged = RolloutBuffer()
        for buf in env_buffers:
            merged.market_states.extend(buf.market_states)
            merged.position_infos.extend(buf.position_infos)
            merged.pair_ids.extend(buf.pair_ids)
            merged.actions.extend(buf.actions)
            merged.action_masks.extend(buf.action_masks)
            merged.log_probs.extend(buf.log_probs)
            merged.rewards.extend(buf.rewards)
            merged.values.extend(buf.values)
            merged.dones.extend(buf.dones)

        merged._advantages = np.concatenate(
            [b._advantages for b in env_buffers]
        )
        merged._returns = np.concatenate(
            [b._returns for b in env_buffers]
        )

        # Compute rollout stats
        total = action_counts.sum()
        stats = {
            "action_dist": action_counts / max(1, total),
            "mean_reward": np.mean(
                episode_rewards) if episode_rewards else 0.0,
            "n_episodes": len(episode_rewards),
        }

        return merged, stats

    def ppo_update(self, buffer, progress=0.0):
        """Run PPO clipped objective update.

        Args:
            buffer: RolloutBuffer with collected transitions
            progress: float in [0, 1], training progress for entropy scheduling

        Returns dict with policy_loss, value_loss, entropy, clip_fraction.
        """
        self.agent.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch in buffer.get_batches(self.batch_size):
                ms = torch.tensor(
                    batch["market_states"],
                    dtype=torch.float32, device=self.device,
                )
                pi = torch.tensor(
                    batch["position_infos"],
                    dtype=torch.float32, device=self.device,
                )
                pair_ids = torch.tensor(
                    batch["pair_ids"],
                    dtype=torch.long, device=self.device,
                )
                actions = torch.tensor(
                    batch["actions"],
                    dtype=torch.long, device=self.device,
                )
                masks = torch.tensor(
                    batch["action_masks"],
                    dtype=torch.bool, device=self.device,
                )
                old_log_probs = torch.tensor(
                    batch["old_log_probs"],
                    dtype=torch.float32, device=self.device,
                )
                advantages = torch.tensor(
                    batch["advantages"],
                    dtype=torch.float32, device=self.device,
                )
                returns = torch.tensor(
                    batch["returns"],
                    dtype=torch.float32, device=self.device,
                )

                # Normalize advantages
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # Forward pass
                new_log_probs, values, entropy = self.agent.evaluate_actions(
                    ms, pi, pair_ids, actions, action_masks=masks,
                )

                # Policy loss (clipped PPO)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio,
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clip fraction (diagnostic)
                clip_frac = (
                    (ratio - 1.0).abs() > self.clip_ratio
                ).float().mean().item()

                # Value loss
                value_loss = 0.5 * (values - returns).pow(2).mean()

                # Entropy: scheduled + adaptive targeting
                # Schedule: start 4x base (explore), decay to 1x (selective)
                entropy_mean = entropy.mean()
                ent_val = entropy_mean.item()
                schedule_mult = 1.0 + 3.0 * (1.0 - progress)  # 4x→1x
                ent_coef = self.entropy_coef * schedule_mult
                max_entropy = self.min_entropy * 1.5  # target: [0.3, 0.45]
                if ent_val < self.min_entropy:
                    # Too low: boost to prevent collapse
                    ratio = self.min_entropy / max(ent_val, 0.05)
                    ent_coef *= min(ratio ** 2, 5.0)
                elif ent_val > max_entropy:
                    # Too high: penalize to prevent random trading
                    ent_coef *= -1.0

                loss = (policy_loss
                        + self.value_coef * value_loss
                        - ent_coef * entropy_mean)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_mean.item()
                total_clip_frac += clip_frac
                n_updates += 1

        self.scheduler.step()

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
            "clip_frac": total_clip_frac / max(1, n_updates),
        }

    def evaluate(self):
        """Evaluate agent with stochastic policy on eval envs.

        Uses stochastic sampling (not argmax) — PPO's learned distribution
        IS the policy. Argmax would collapse to HOLD.

        Runs n_eval_episodes per env with random starts to reduce variance.
        """
        self.agent.eval()
        results = []

        for env in self.eval_envs:
            pair_name = (env.env.pair_name
                         if hasattr(env, "env") else "unknown")

            ep_returns = []
            ep_rewards = []
            ep_trades = []
            ep_wins = []
            ep_hold_pcts = []

            for _ep in range(self.n_eval_episodes):
                state = env.reset()
                done = False
                ep_reward = 0.0
                trades = 0
                wins = 0
                action_counts = np.zeros(4, dtype=np.int64)

                while not done:
                    ms, pi = state
                    mask = env.get_action_mask()
                    action, _, _, probs = self.agent.get_action(
                        ms, pi, env.pair_idx,
                        action_mask=mask, deterministic=False,
                    )
                    state, reward, done, info = env.step(action)
                    ep_reward += reward
                    action_counts[action] += 1
                    if info.get("trade_closed", False):
                        trades += 1
                        if info.get("raw_pnl", 0.0) > 0:
                            wins += 1

                total_actions = action_counts.sum()
                ret_pct = info.get("return_pct", 0.0)
                ep_returns.append(ret_pct / 100.0)
                ep_rewards.append(ep_reward)
                ep_trades.append(trades)
                ep_wins.append(wins)
                ep_hold_pcts.append(
                    100.0 * action_counts[0] / max(1, total_actions)
                )

            total_trades = sum(ep_trades)
            total_wins = sum(ep_wins)
            results.append({
                "pair": pair_name,
                "return": np.mean(ep_returns),
                "return_std": np.std(ep_returns),
                "reward": np.mean(ep_rewards),
                "trades": np.mean(ep_trades),
                "win_rate": total_wins / max(1, total_trades),
                "hold_pct": np.mean(ep_hold_pcts),
                "n_episodes": self.n_eval_episodes,
            })

        if not results:
            return {
                "eval_return": 0.0, "eval_trades": 0,
                "eval_hold_pct": 100.0,
            }

        avg_return = np.mean([r["return"] for r in results])
        avg_trades = np.mean([r["trades"] for r in results])
        avg_hold = np.mean([r["hold_pct"] for r in results])
        avg_wr = np.mean([r["win_rate"] for r in results])

        for r in results:
            logger.info(
                f"  {r['pair']:>8s}: ret={r['return']:+.2%} "
                f"(±{r['return_std']:.2%}, {r['n_episodes']}ep) "
                f"trades={r['trades']:.0f} WR={r['win_rate']:.1%} "
                f"hold={r['hold_pct']:.0f}%"
            )

        return {
            "eval_return": avg_return,
            "eval_trades": avg_trades,
            "eval_hold_pct": avg_hold,
            "eval_win_rate": avg_wr,
            "details": results,
        }

    def train(self):
        """Main PPO training loop.

        Returns:
            dict with best_eval_return and training stats
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Phase 2: PPO Training (Discrete Actions)")
        logger.info(f"{'=' * 60}")
        logger.info(f"  γ={self.gamma} λ={self.lam} clip={self.clip_ratio}")
        logger.info(f"  entropy_coef={self.entropy_coef} "
                     f"min_entropy={self.min_entropy}")
        logger.info(f"  rollout={self.rollout_length} "
                     f"epochs={self.n_epochs} batch={self.batch_size}")
        logger.info(f"  total_steps={self.total_steps:,}")
        logger.info(f"{'=' * 60}\n")

        self.agent.to(self.device)
        start_time = time.time()

        total_steps = 0
        best_eval_return = -float("inf")
        n_iterations = max(1, self.total_steps // self.rollout_length)

        for iteration in range(n_iterations):
            iter_start = time.time()

            # Collect rollout
            buffer, rollout_stats = self.collect_rollout()
            total_steps += len(buffer)

            # PPO update
            progress = total_steps / self.total_steps
            update_stats = self.ppo_update(buffer, progress=progress)

            # Log every 5 iterations
            if (iteration + 1) % 5 == 0 or iteration == 0:
                ad = rollout_stats["action_dist"]
                elapsed = time.time() - start_time
                fps = total_steps / max(1, elapsed)
                logger.info(
                    f"[{total_steps:>7,}/{self.total_steps:,}] "
                    f"π={update_stats['policy_loss']:.4f} "
                    f"V={update_stats['value_loss']:.4f} "
                    f"H={update_stats['entropy']:.3f} "
                    f"clip={update_stats['clip_frac']:.2f} | "
                    f"HOLD={ad[0]:.0%} L={ad[1]:.0%} S={ad[2]:.0%} "
                    f"C={ad[3]:.0%} | "
                    f"R̄={rollout_stats['mean_reward']:.1f} "
                    f"fps={fps:.0f}"
                )

            # Evaluate
            if (total_steps % self.eval_interval < self.rollout_length
                    or iteration == n_iterations - 1):
                logger.info(f"\n--- Eval at step {total_steps:,} ---")
                eval_stats = self.evaluate()
                logger.info(
                    f"Eval: ret={eval_stats['eval_return']:+.2%} "
                    f"trades={eval_stats['eval_trades']:.0f} "
                    f"WR={eval_stats.get('eval_win_rate', 0):.1%} "
                    f"hold={eval_stats['eval_hold_pct']:.0f}%\n"
                )

                if eval_stats["eval_return"] > best_eval_return:
                    best_eval_return = eval_stats["eval_return"]
                    self.agent.save_agent(
                        os.path.join(MODEL_DIR, "ppo_agent_best.pt")
                    )
                    logger.info(
                        f"  *** New best: {best_eval_return:+.2%} ***"
                    )

            # Save checkpoint
            if total_steps % self.save_interval < self.rollout_length:
                self.agent.save_agent(
                    os.path.join(MODEL_DIR, f"ppo_agent_{total_steps}.pt")
                )

        elapsed = time.time() - start_time
        logger.info(f"\nPPO Training complete in {elapsed / 60:.1f} min")
        logger.info(f"Best eval return: {best_eval_return:+.2%}")

        return {
            "best_eval_return": best_eval_return,
            "total_steps": total_steps,
            "elapsed_min": elapsed / 60,
        }
