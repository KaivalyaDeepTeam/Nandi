"""Adversarial training wrapper for robustness testing."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class AdversarialEnvironment:
    """Wraps a trading environment to inject adversarial perturbations.

    Tests agent robustness by:
    1. Randomly injecting adverse price moves
    2. Adding noise to features
    3. Randomly increasing transaction costs
    """

    def __init__(self, base_env, perturbation_prob=0.05,
                 feature_noise_std=0.03, cost_multiplier_range=(0.5, 3.0)):
        self.env = base_env
        self.perturbation_prob = perturbation_prob
        self.feature_noise_std = feature_noise_std
        self.cost_multiplier_range = cost_multiplier_range

        self._original_cost_bps = None

    def reset(self, start_idx=None):
        """Reset with randomized environment parameters."""
        state = self.env.reset(start_idx)

        # Randomize transaction cost
        cost_mult = np.random.uniform(*self.cost_multiplier_range)
        from nandi.config import TRANSACTION_COST_BPS
        self.env.env.transaction_cost_bps = TRANSACTION_COST_BPS * cost_mult if hasattr(self.env, 'env') else TRANSACTION_COST_BPS * cost_mult

        return self._perturb_state(state)

    def step(self, action):
        """Step with adversarial perturbations."""
        state, reward, done, info = self.env.step(action)

        # Randomly inject adverse reward perturbation
        if np.random.random() < self.perturbation_prob:
            adversarial_penalty = -abs(action) * 0.02  # penalty proportional to position
            reward += adversarial_penalty
            reward = float(np.clip(reward, -1.0, 1.0))

        state = self._perturb_state(state)
        return state, reward, done, info

    def _perturb_state(self, state):
        """Add Gaussian noise to market state features."""
        if self.feature_noise_std <= 0:
            return state

        market_state, position_info = state
        noise = np.random.normal(0, self.feature_noise_std, market_state.shape)
        noisy_state = (market_state + noise).astype(np.float32)
        return noisy_state, position_info

    @property
    def market_state_shape(self):
        return self.env.market_state_shape

    @property
    def position_info_dim(self):
        return self.env.position_info_dim
