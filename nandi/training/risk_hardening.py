"""
Phase 3: Risk Hardening — adversarial training for robustness.

Continues DQN training at low LR with adversarial perturbations:
- 70% normal episodes, 30% adversarial
- Flash crashes (0.2% of bars, 2% moves)
- Spread widening (5% of bars, 2-5x spread)
- Slippage spikes
- Feature noise

Agent learns to HOLD/CLOSE during adverse conditions.
"""

import logging
import copy

import numpy as np

from nandi.config import DQN_CONFIG
from nandi.training.dqn_trainer import DQNTrainer

logger = logging.getLogger(__name__)


class AdversarialDiscreteEnv:
    """Wraps a MultiEpisodeDiscreteEnv with adversarial perturbations.

    Perturbation types:
    1. Flash crash: sudden large price move (simulated via reward shock)
    2. Spread widening: increased transaction costs via reward penalty
    3. Feature noise: Gaussian noise on market state features
    """

    def __init__(self, base_env,
                 flash_crash_prob=0.002,
                 flash_crash_magnitude=0.02,
                 spread_widen_prob=0.05,
                 spread_widen_range=(2.0, 5.0),
                 feature_noise_std=0.05):
        self.env = base_env
        self.flash_crash_prob = flash_crash_prob
        self.flash_crash_magnitude = flash_crash_magnitude
        self.spread_widen_prob = spread_widen_prob
        self.spread_widen_range = spread_widen_range
        self.feature_noise_std = feature_noise_std

    def reset(self):
        state = self.env.reset()
        return self._perturb_state(state)

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        # Flash crash: sudden adverse reward
        if np.random.random() < self.flash_crash_prob:
            crash_penalty = -self.flash_crash_magnitude * np.random.uniform(0.5, 1.5)
            reward += crash_penalty
            reward = float(np.clip(reward, -1.0, 1.0))
            info["adversarial"] = "flash_crash"

        # Spread widening: extra cost penalty proportional to action
        if np.random.random() < self.spread_widen_prob:
            widen = np.random.uniform(*self.spread_widen_range)
            if action in (1, 2, 3):  # LONG, SHORT, CLOSE — involve a trade
                spread_penalty = -0.001 * widen
                reward += spread_penalty
                reward = float(np.clip(reward, -1.0, 1.0))
            info["adversarial"] = info.get("adversarial", "") + "spread_widen"

        state = self._perturb_state(state)
        return state, reward, done, info

    def _perturb_state(self, state):
        market_state, position_info = state
        noise = np.random.normal(0, self.feature_noise_std, market_state.shape)
        noisy_state = (market_state + noise).astype(np.float32)
        return noisy_state, position_info

    def get_action_mask(self):
        return self.env.get_action_mask()

    @property
    def market_state_shape(self):
        return self.env.market_state_shape

    @property
    def position_info_dim(self):
        return self.env.position_info_dim

    @property
    def pair_idx(self):
        return self.env.pair_idx


class RiskHardeningTrainer:
    """Phase 3: Continue DQN training with adversarial perturbations.

    Mixes normal and adversarial environments (70/30 split).
    Uses lower learning rate for fine-tuning without catastrophic forgetting.
    """

    def __init__(self, agent, train_envs, eval_envs=None, dqn_config=None,
                 training_config=None, device=None):
        """
        Args:
            agent: NandiDQNAgent (pre-trained from Phase 2)
            train_envs: list of MultiEpisodeDiscreteEnv
            eval_envs: list for evaluation
            dqn_config: override DQN_CONFIG
            training_config: override TRAINING_CONFIG
            device: torch.device
        """
        self.cfg = dqn_config or DQN_CONFIG
        self.device = device

        # Create mixed environments: 70% normal, 30% adversarial
        mixed_envs = []
        for env in train_envs:
            mixed_envs.append(env)  # normal

        # Add adversarial versions of some envs
        n_adversarial = max(1, int(len(train_envs) * self.cfg.get(
            "adversarial_ratio", 0.3) / (1.0 - self.cfg.get("adversarial_ratio", 0.3))
        ))
        for i in range(n_adversarial):
            base = train_envs[i % len(train_envs)]
            adv = AdversarialDiscreteEnv(base)
            mixed_envs.append(adv)

        logger.info(
            f"Risk hardening: {len(train_envs)} normal + "
            f"{n_adversarial} adversarial = {len(mixed_envs)} total envs"
        )

        # Override config for hardening phase
        hardening_config = dict(self.cfg)
        hardening_config["lr"] = self.cfg.get("hardening_lr", 3e-5)
        hardening_config["total_steps"] = self.cfg.get("hardening_steps", 50_000)
        hardening_config["warmup_steps"] = 0  # no warmup needed — buffer from Phase 2

        # Create DQN trainer with mixed envs
        self.trainer = DQNTrainer(
            agent=agent,
            train_envs=mixed_envs,
            eval_envs=eval_envs,
            dqn_config=hardening_config,
            training_config=training_config,
            device=device,
        )

    def train(self):
        """Run risk hardening phase.

        Returns:
            training_stats from the DQN trainer
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Phase 3: Risk Hardening")
        logger.info(f"  LR: {self.cfg.get('hardening_lr', 3e-5)} | "
                     f"Steps: {self.cfg.get('hardening_steps', 50_000):,}")
        logger.info(f"{'=' * 60}\n")

        return self.trainer.train()
