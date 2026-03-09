"""ML-based alpha signal combination with regime-adaptive weighting."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class AlphaCombiner:
    """Combines alpha signals using rolling-Sharpe weighted averaging."""

    def __init__(self, alpha_names):
        self.alpha_names = alpha_names
        self.weights = {name: 1.0 / len(alpha_names) for name in alpha_names}
        self.performance_history = {name: [] for name in alpha_names}

    def record_performance(self, alpha_name, return_value):
        """Record a return for an alpha source."""
        if alpha_name in self.performance_history:
            self.performance_history[alpha_name].append(return_value)
            # Keep last 252 days
            if len(self.performance_history[alpha_name]) > 252:
                self.performance_history[alpha_name] = \
                    self.performance_history[alpha_name][-252:]

    def update_weights(self):
        """Update weights based on rolling Sharpe ratio of each alpha."""
        sharpes = {}
        for name in self.alpha_names:
            history = self.performance_history[name]
            if len(history) < 20:
                sharpes[name] = 0.0
                continue
            recent = np.array(history[-60:])
            std = np.std(recent)
            sharpes[name] = float(np.mean(recent) / (std + 1e-10) * np.sqrt(252)) if std > 1e-10 else 0.0

        # Softmax-like weighting (only positive Sharpes)
        clipped = {k: max(0, v) for k, v in sharpes.items()}
        total = sum(clipped.values())
        if total > 0:
            self.weights = {k: v / total for k, v in clipped.items()}

        logger.info(f"Alpha weights updated: {self.weights}")

    def combine(self, signals_by_alpha):
        """Combine signals from multiple alphas into weighted positions.

        Args:
            signals_by_alpha: {alpha_name: [AlphaSignal, ...]}

        Returns:
            {pair: combined_position}
        """
        combined = {}
        for alpha_name, signals in signals_by_alpha.items():
            w = self.weights.get(alpha_name, 0.0)
            for sig in signals:
                if sig.pair not in combined:
                    combined[sig.pair] = 0.0
                combined[sig.pair] += sig.weighted_signal * w
        return combined


class RegimeAdaptiveCombiner(AlphaCombiner):
    """Extends AlphaCombiner with regime-specific weight profiles."""

    REGIME_WEIGHTS = {
        "trending":  {"rl": 0.4, "momentum": 0.4, "mean_reversion": 0.1, "stat_arb": 0.1},
        "ranging":   {"rl": 0.3, "momentum": 0.1, "mean_reversion": 0.4, "stat_arb": 0.2},
        "volatile":  {"rl": 0.2, "momentum": 0.2, "mean_reversion": 0.1, "stat_arb": 0.5},
        "choppy":    {"rl": 0.5, "momentum": 0.0, "mean_reversion": 0.0, "stat_arb": 0.0},
    }

    def combine_with_regime(self, signals_by_alpha, regime="trending"):
        """Combine signals using regime-specific weights."""
        regime_weights = self.REGIME_WEIGHTS.get(regime, self.weights)

        combined = {}
        for alpha_name, signals in signals_by_alpha.items():
            w = regime_weights.get(alpha_name, 0.0)
            for sig in signals:
                if sig.pair not in combined:
                    combined[sig.pair] = 0.0
                combined[sig.pair] += sig.weighted_signal * w
        return combined
