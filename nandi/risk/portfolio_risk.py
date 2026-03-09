"""
Portfolio-level risk management — hard limits + regime scaling.
"""

import logging
import numpy as np

from nandi.config import PORTFOLIO_RISK

logger = logging.getLogger(__name__)


class PortfolioRiskManager:
    """Enforces portfolio-level risk limits across all pairs."""

    def __init__(self, config=None):
        self.config = config or PORTFOLIO_RISK
        self.peak_equity = None
        self.daily_start_equity = None
        self.is_halted = False

    def reset(self, initial_equity):
        self.peak_equity = initial_equity
        self.daily_start_equity = initial_equity
        self.is_halted = False

    def new_day(self, current_equity):
        """Call at start of each trading day."""
        self.daily_start_equity = current_equity

    def check_and_adjust(self, positions, current_equity):
        """Check portfolio-level constraints and adjust positions.

        Args:
            positions: {pair: float} current/proposed positions.
            current_equity: current portfolio equity.

        Returns:
            adjusted_positions: {pair: float} positions after risk adjustments.
            risk_status: dict with flags and metrics.
        """
        if self.peak_equity is None:
            self.peak_equity = current_equity
        self.peak_equity = max(self.peak_equity, current_equity)

        portfolio_dd = (self.peak_equity - current_equity) / self.peak_equity
        daily_loss = 0.0
        if self.daily_start_equity:
            daily_loss = (self.daily_start_equity - current_equity) / self.daily_start_equity

        status = {
            "portfolio_dd": portfolio_dd,
            "daily_loss": daily_loss,
            "is_halted": False,
            "scale_factor": 1.0,
            "reason": None,
        }

        adjusted = dict(positions)

        # Circuit breaker: max portfolio drawdown
        if portfolio_dd > self.config["max_portfolio_dd"]:
            adjusted = {p: 0.0 for p in positions}
            status["is_halted"] = True
            status["reason"] = f"Portfolio DD {portfolio_dd:.2%} > {self.config['max_portfolio_dd']:.0%}"
            self.is_halted = True
            logger.warning(f"CIRCUIT BREAKER: {status['reason']}")
            return adjusted, status

        # Daily loss limit
        if daily_loss > self.config["max_daily_loss"]:
            adjusted = {p: 0.0 for p in positions}
            status["is_halted"] = True
            status["reason"] = f"Daily loss {daily_loss:.2%} > {self.config['max_daily_loss']:.0%}"
            logger.warning(f"DAILY LIMIT: {status['reason']}")
            return adjusted, status

        # Scale down at DD threshold
        if portfolio_dd > self.config["scale_down_dd"]:
            scale = self.config["scale_down_factor"]
            adjusted = {p: v * scale for p, v in adjusted.items()}
            status["scale_factor"] = scale
            logger.info(f"Scale down: DD {portfolio_dd:.2%}, positions *= {scale}")

        # Enforce per-pair limits
        max_single = self.config["max_single_pair"]
        adjusted = {
            p: float(np.clip(v, -max_single, max_single))
            for p, v in adjusted.items()
        }

        # Enforce total exposure limit
        total_exposure = sum(abs(v) for v in adjusted.values())
        if total_exposure > self.config["max_total_exposure"]:
            ratio = self.config["max_total_exposure"] / total_exposure
            adjusted = {p: v * ratio for p, v in adjusted.items()}

        return adjusted, status
