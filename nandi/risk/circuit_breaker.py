"""
Portfolio-level circuit breaker — emergency stop on extreme drawdown.
"""

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Emergency stop when portfolio drawdown exceeds hard limits.

    Once triggered, trading is halted for a cooldown period.
    """

    def __init__(self, max_dd=0.12, cooldown_hours=24):
        self.max_dd = max_dd
        self.cooldown_hours = cooldown_hours
        self.triggered = False
        self.trigger_time = None
        self.trigger_dd = None

    def check(self, portfolio_dd):
        """Check if circuit breaker should trigger.

        Args:
            portfolio_dd: current portfolio drawdown (0 to 1).

        Returns:
            bool: True if trading should be halted.
        """
        # Check cooldown
        if self.triggered and self.trigger_time:
            elapsed = datetime.now() - self.trigger_time
            if elapsed < timedelta(hours=self.cooldown_hours):
                return True
            else:
                logger.info(f"Circuit breaker cooldown expired after {elapsed}")
                self.triggered = False

        # Check DD threshold
        if portfolio_dd > self.max_dd:
            self.triggered = True
            self.trigger_time = datetime.now()
            self.trigger_dd = portfolio_dd
            logger.warning(
                f"CIRCUIT BREAKER TRIGGERED: Portfolio DD {portfolio_dd:.2%} > {self.max_dd:.0%}"
            )
            return True

        return False

    def reset(self):
        self.triggered = False
        self.trigger_time = None
        self.trigger_dd = None

    @property
    def status(self):
        if not self.triggered:
            return "OK"
        remaining = None
        if self.trigger_time:
            elapsed = datetime.now() - self.trigger_time
            remaining = timedelta(hours=self.cooldown_hours) - elapsed
        return f"HALTED (DD={self.trigger_dd:.2%}, remaining={remaining})"
