"""
HMM Regime Detector — 4-state Gaussian Hidden Markov Model.

Phase 2 implementation: detects ranging, trending, volatile, choppy regimes.
Requires: pip install hmmlearn
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Regime labels sorted by volatility (post-fit)
REGIME_LABELS = ["ranging", "trending", "volatile", "choppy"]

# Regime scaling factors for position sizing
REGIME_SCALES = {
    "trending": 1.0,
    "ranging": 0.8,
    "volatile": 0.5,
    "choppy": 0.0,  # no trading
}


class HMMRegimeDetector:
    """4-state Gaussian HMM for market regime detection."""

    def __init__(self, n_states=4):
        self.n_states = n_states
        self.model = None
        self.state_labels = None
        self.fitted = False

    def fit(self, features):
        """Fit HMM on historical features.

        Args:
            features: (N, 5) array of [vol_5d, vol_20d, adx, hurst, entropy].
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning("hmmlearn not installed. Regime detection disabled.")
            return

        features = np.asarray(features, dtype=np.float64)
        features = features[~np.any(np.isnan(features), axis=1)]

        if len(features) < 100:
            logger.warning("Not enough data for HMM fitting")
            return

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        self.model.fit(features)

        # Sort states by mean volatility for consistent labeling
        mean_vols = self.model.means_[:, 0]  # vol_5d column
        sorted_indices = np.argsort(mean_vols)

        self.state_labels = {}
        for i, idx in enumerate(sorted_indices):
            self.state_labels[idx] = REGIME_LABELS[min(i, len(REGIME_LABELS) - 1)]

        self.fitted = True
        logger.info(f"HMM fitted with {self.n_states} states on {len(features)} samples")

    def predict(self, features):
        """Predict current regime.

        Args:
            features: (N, 5) or (5,) array.

        Returns:
            regime label string.
        """
        if not self.fitted or self.model is None:
            return "trending"  # default

        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        state = self.model.predict(features)[-1]
        return self.state_labels.get(state, "trending")

    def get_regime_scale(self, regime_label):
        """Get position sizing scale for a regime."""
        return REGIME_SCALES.get(regime_label, 1.0)

    def predict_scale(self, features):
        """Predict regime and return scaling factor directly."""
        label = self.predict(features)
        return self.get_regime_scale(label), label
