"""Monte Carlo permutation testing for strategy significance."""

import logging
import numpy as np
from nandi.utils.metrics import sharpe_ratio

logger = logging.getLogger(__name__)


class MonteCarloValidator:
    """Tests if strategy performance is statistically significant via return shuffling."""

    def __init__(self, n_permutations=1000):
        self.n_permutations = n_permutations

    def test_significance(self, strategy_returns, benchmark_returns=None):
        """Test if strategy Sharpe is significantly better than random.

        Args:
            strategy_returns: array of daily strategy returns
            benchmark_returns: optional benchmark (default: shuffled strategy returns)

        Returns:
            dict with real_sharpe, p_value, null_mean, null_std, is_significant
        """
        real_sharpe = sharpe_ratio(strategy_returns)
        null_sharpes = []

        for _ in range(self.n_permutations):
            shuffled = np.random.permutation(strategy_returns)
            null_sharpes.append(sharpe_ratio(shuffled))

        null_sharpes = np.array(null_sharpes)
        p_value = float(np.mean(null_sharpes >= real_sharpe))

        result = {
            "real_sharpe": real_sharpe,
            "p_value": p_value,
            "null_mean": float(np.mean(null_sharpes)),
            "null_std": float(np.std(null_sharpes)),
            "is_significant": p_value < 0.05,
            "significance_level": "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else "ns",
        }

        logger.info(f"Monte Carlo: Sharpe={real_sharpe:.3f}, p={p_value:.4f} "
                    f"({'SIGNIFICANT' if result['is_significant'] else 'NOT significant'})")
        return result
