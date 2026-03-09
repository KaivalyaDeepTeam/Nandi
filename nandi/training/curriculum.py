"""Curriculum learning for progressive training difficulty."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class CurriculumScheduler:
    """Progressively increases training difficulty.

    Stages:
    1. Low volatility periods with reduced leverage (easy)
    2. Medium volatility with moderate leverage
    3. High volatility with full leverage (hard)
    4. Crisis periods with reduced leverage (adversarial)
    """

    STAGES = [
        {"name": "easy", "vol_percentile_max": 40, "leverage_scale": 0.5,
         "steps": 100_000, "description": "Low-vol periods, reduced leverage"},
        {"name": "medium", "vol_percentile_max": 70, "leverage_scale": 0.8,
         "steps": 150_000, "description": "Medium-vol, moderate leverage"},
        {"name": "hard", "vol_percentile_max": 100, "leverage_scale": 1.0,
         "steps": 150_000, "description": "All periods, full leverage"},
        {"name": "adversarial", "vol_percentile_max": 100, "leverage_scale": 0.8,
         "steps": 100_000, "description": "All periods with noise injection"},
    ]

    def __init__(self):
        self.current_stage_idx = 0
        self.steps_in_stage = 0

    @property
    def current_stage(self):
        return self.STAGES[min(self.current_stage_idx, len(self.STAGES) - 1)]

    @property
    def stage_name(self):
        return self.current_stage["name"]

    def step(self):
        """Advance one timestep. Returns True if stage changed."""
        self.steps_in_stage += 1
        if self.steps_in_stage >= self.current_stage["steps"]:
            if self.current_stage_idx < len(self.STAGES) - 1:
                self.current_stage_idx += 1
                self.steps_in_stage = 0
                logger.info(f"Curriculum advanced to stage: {self.stage_name} "
                          f"({self.current_stage['description']})")
                return True
        return False

    def filter_data_by_volatility(self, features, vol_column_idx=0):
        """Filter training data to match current curriculum stage.

        Args:
            features: (n_samples, n_features) array
            vol_column_idx: index of the volatility feature column

        Returns:
            boolean mask for valid training indices
        """
        max_pct = self.current_stage["vol_percentile_max"]
        if max_pct >= 100:
            return np.ones(len(features), dtype=bool)

        vol = np.abs(features[:, vol_column_idx])
        threshold = np.percentile(vol[~np.isnan(vol)], max_pct)
        return vol <= threshold

    def get_leverage_scale(self):
        """Get leverage multiplier for current stage."""
        return self.current_stage["leverage_scale"]

    def should_inject_noise(self):
        """Whether to add feature noise (adversarial stage)."""
        return self.current_stage["name"] == "adversarial"

    def get_noise_std(self):
        """Noise standard deviation for adversarial training."""
        if self.should_inject_noise():
            return 0.05
        return 0.0

    def reset(self):
        """Reset to beginning."""
        self.current_stage_idx = 0
        self.steps_in_stage = 0
