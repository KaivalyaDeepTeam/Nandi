"""Portfolio Value at Risk (VaR) and Conditional VaR (CVaR) computation."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class PortfolioVaR:
    """Computes historical VaR and CVaR for portfolio risk assessment."""

    def __init__(self, confidence=0.95, window=60):
        self.confidence = confidence
        self.window = window
        self.returns_history = []

    def update(self, portfolio_return):
        """Add latest portfolio return to history."""
        self.returns_history.append(portfolio_return)
        if len(self.returns_history) > self.window * 5:
            self.returns_history = self.returns_history[-self.window * 5:]

    def compute_var(self):
        """Compute historical Value at Risk."""
        if len(self.returns_history) < 20:
            return 0.0
        returns = np.array(self.returns_history[-self.window:])
        return float(-np.percentile(returns, (1 - self.confidence) * 100))

    def compute_cvar(self):
        """Compute Conditional VaR (Expected Shortfall)."""
        var = self.compute_var()
        if var == 0:
            return 0.0
        returns = np.array(self.returns_history[-self.window:])
        tail = returns[returns <= -var]
        return float(-np.mean(tail)) if len(tail) > 0 else var

    def get_position_scalar(self, target_var=0.02):
        """Compute position scalar to target a specific daily VaR.

        If current VaR exceeds target, returns a scalar < 1.0 to reduce positions.
        """
        current_var = self.compute_var()
        if current_var <= 0 or current_var <= target_var:
            return 1.0
        return target_var / current_var

    def get_risk_report(self):
        """Generate a risk report."""
        if len(self.returns_history) < 20:
            return {"status": "insufficient_data", "n_observations": len(self.returns_history)}

        returns = np.array(self.returns_history)
        return {
            "var_95": self.compute_var(),
            "cvar_95": self.compute_cvar(),
            "mean_return": float(np.mean(returns)),
            "vol_annual": float(np.std(returns) * np.sqrt(252)),
            "worst_day": float(np.min(returns)),
            "best_day": float(np.max(returns)),
            "n_observations": len(returns),
            "skewness": float(self._skewness(returns)),
            "kurtosis": float(self._kurtosis(returns)),
        }

    @staticmethod
    def _skewness(x):
        n = len(x)
        if n < 3:
            return 0.0
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        if std < 1e-10:
            return 0.0
        return float(n / ((n-1) * (n-2)) * np.sum(((x - mean) / std) ** 3))

    @staticmethod
    def _kurtosis(x):
        n = len(x)
        if n < 4:
            return 0.0
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((x - mean) / std) ** 4) - 3.0)

    def reset(self):
        self.returns_history = []
