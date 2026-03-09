"""
Portfolio Optimizer — Signal combination + Kelly sizing + constraint enforcement.

Phase 1: Equal-weight per-pair RL agents with exposure caps.
Phase 2 adds: inverse-variance weighting, Half-Kelly sizing, correlation adjustment.
"""

import logging
import numpy as np

from nandi.config import PORTFOLIO_RISK

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Combines per-pair signals into portfolio positions with constraints."""

    def __init__(self, pairs, config=None):
        self.pairs = pairs
        self.config = config or PORTFOLIO_RISK
        # Phase 2: per-alpha rolling stats for inverse-variance weighting
        self.alpha_stats = {}

    def optimize(self, raw_signals, regime_scales=None):
        """Convert raw per-pair signals to constrained portfolio positions.

        Args:
            raw_signals: {pair: float} raw signal in [-1, 1] from RL agents.
            regime_scales: {pair: float} optional regime scaling (0.0-1.0).

        Returns:
            {pair: float} final positions after constraints.
        """
        positions = dict(raw_signals)

        # Apply regime scaling if provided
        if regime_scales:
            for pair in positions:
                if pair in regime_scales:
                    positions[pair] *= regime_scales.get(pair, 1.0)

        # Enforce constraints
        positions = self._enforce_constraints(positions)

        return positions

    def optimize_with_kelly(self, raw_signals, pair_stats, regime_scales=None):
        """Phase 2: Kelly-sized positions.

        Args:
            raw_signals: {pair: float} raw signal direction.
            pair_stats: {pair: {"win_rate": float, "avg_win": float, "avg_loss": float}}.
            regime_scales: {pair: float}.
        """
        positions = {}

        for pair, signal in raw_signals.items():
            stats = pair_stats.get(pair, {})
            wr = stats.get("win_rate", 0.5)
            avg_win = stats.get("avg_win", 1.0)
            avg_loss = stats.get("avg_loss", 1.0)

            # Half-Kelly sizing
            if avg_win > 0:
                kelly = max(0, (wr * avg_win - (1 - wr) * avg_loss) / avg_win) / 2
            else:
                kelly = 0.0

            kelly = min(kelly, 0.5)  # cap at 50%
            positions[pair] = signal * kelly

        if regime_scales:
            for pair in positions:
                if pair in regime_scales:
                    positions[pair] *= regime_scales.get(pair, 1.0)

        return self._enforce_constraints(positions)

    def _enforce_constraints(self, positions):
        """Apply portfolio-level and per-pair constraints."""
        max_single = self.config["max_single_pair"]
        max_total = self.config["max_total_exposure"]

        # Clip per-pair
        clipped = {
            p: float(np.clip(v, -max_single, max_single))
            for p, v in positions.items()
        }

        # Clip total exposure
        total = sum(abs(v) for v in clipped.values())
        if total > max_total:
            scale = max_total / total
            clipped = {p: v * scale for p, v in clipped.items()}

        return clipped
