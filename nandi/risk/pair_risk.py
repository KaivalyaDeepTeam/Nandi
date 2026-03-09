"""
Per-pair position and drawdown limits.
"""

import logging
import numpy as np

from nandi.config import RISK_LIMITS

logger = logging.getLogger(__name__)


class PairRiskManager:
    """Per-pair risk limits — mirrors the single-pair env hard limits for live use."""

    def __init__(self, pair_name, config=None):
        self.pair_name = pair_name
        self.config = config or RISK_LIMITS
        self.peak_equity = None
        self.daily_start_equity = None

    def reset(self, initial_equity):
        self.peak_equity = initial_equity
        self.daily_start_equity = initial_equity

    def new_day(self, current_equity):
        self.daily_start_equity = current_equity

    def check_position(self, proposed_position, current_equity):
        """Adjust a single pair's position based on risk limits.

        Returns:
            adjusted_position: float
            risk_info: dict
        """
        if self.peak_equity is None:
            self.peak_equity = current_equity
        self.peak_equity = max(self.peak_equity, current_equity)

        dd = (self.peak_equity - current_equity) / self.peak_equity
        daily_loss = 0.0
        if self.daily_start_equity:
            daily_loss = (self.daily_start_equity - current_equity) / self.daily_start_equity

        info = {
            "pair": self.pair_name,
            "drawdown": dd,
            "daily_loss": daily_loss,
            "action_taken": None,
        }

        position = proposed_position

        # Force flat at max drawdown
        if dd > self.config["max_drawdown"]:
            position = 0.0
            info["action_taken"] = "force_flat_max_dd"
            logger.warning(f"[{self.pair_name}] Force flat: DD {dd:.2%}")

        # Force flat on daily loss
        elif daily_loss > self.config["max_daily_loss"]:
            position = 0.0
            info["action_taken"] = "force_flat_daily_loss"

        # Scale down at threshold
        elif dd > self.config["scale_down_threshold"]:
            position *= 0.5
            info["action_taken"] = "scale_down"

        return float(np.clip(position, -1.0, 1.0)), info
