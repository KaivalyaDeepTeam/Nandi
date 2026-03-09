"""Out-of-Distribution detection using Mahalanobis distance."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class OODDetector:
    """Detects when live data is out-of-distribution relative to training data."""

    def __init__(self):
        self.train_mean = None
        self.train_cov_inv = None
        self.n_features = 0
        self.threshold = None
        self.is_fitted = False

    def fit(self, train_features):
        """Fit on training data to learn the in-distribution manifold.

        Args:
            train_features: (n_samples, n_features) array of training features
        """
        self.train_mean = np.mean(train_features, axis=0)
        cov = np.cov(train_features.T)
        # Regularize to prevent singular matrix
        cov += np.eye(cov.shape[0]) * 1e-6
        self.train_cov_inv = np.linalg.inv(cov)
        self.n_features = train_features.shape[1]

        # Compute threshold from training data (99th percentile)
        train_scores = np.array([self.score(x) for x in train_features])
        self.threshold = float(np.percentile(train_scores, 99))
        self.is_fitted = True

        logger.info(f"OOD detector fitted on {len(train_features)} samples, "
                    f"{self.n_features} features, threshold={self.threshold:.2f}")

    def score(self, feature_vector):
        """Compute Mahalanobis distance score for a single observation."""
        if not self.is_fitted:
            return 0.0
        diff = feature_vector - self.train_mean
        return float(diff @ self.train_cov_inv @ diff)

    def is_ood(self, feature_vector, threshold=None):
        """Check if a feature vector is out-of-distribution."""
        if not self.is_fitted:
            return False
        t = threshold or self.threshold
        return self.score(feature_vector) > t

    def batch_score(self, features):
        """Score multiple observations at once."""
        if not self.is_fitted:
            return np.zeros(len(features))
        diff = features - self.train_mean
        return np.array([float(d @ self.train_cov_inv @ d) for d in diff])

    def get_ood_fraction(self, features, threshold=None):
        """Get fraction of observations that are OOD."""
        scores = self.batch_score(features)
        t = threshold or self.threshold
        return float(np.mean(scores > t))
