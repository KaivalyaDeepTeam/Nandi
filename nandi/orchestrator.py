"""
Trading Orchestrator — Central pipeline connecting all Nandi subsystems.

Wires together: data → features → regime → alphas → combine → optimize → risk.
"""

import logging
import numpy as np
import pandas as pd

from nandi.config import PAIRS, PORTFOLIO_RISK
from nandi.alpha.rl_alpha import RLAlpha
from nandi.alpha.momentum_alpha import MomentumAlpha
from nandi.alpha.mean_reversion_alpha import MeanReversionAlpha
from nandi.alpha.stat_arb_alpha import StatArbAlpha
from nandi.portfolio.optimizer import PortfolioOptimizer
from nandi.portfolio.correlation import CorrelationTracker
from nandi.risk.portfolio_risk import PortfolioRiskManager
from nandi.risk.circuit_breaker import CircuitBreaker
from nandi.regime.hmm_detector import REGIME_SCALES

logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """Central pipeline connecting all trading subsystems."""

    def __init__(self, pairs, agents=None, regime_detector=None):
        self.pairs = pairs
        self.agents = agents or {}
        self.regime_detector = regime_detector

        # Alpha sources
        self.alphas = []
        if agents:
            self.alphas.append(RLAlpha(pairs, agents=agents))
        self.alphas.append(MomentumAlpha(pairs))
        self.alphas.append(MeanReversionAlpha(pairs))
        self.alphas.append(StatArbAlpha(pairs))

        # Alpha weights (default equal, can be updated)
        self.alpha_weights = {
            "rl": 0.4,
            "momentum": 0.2,
            "mean_reversion": 0.2,
            "stat_arb": 0.2,
        }

        # Portfolio components
        self.optimizer = PortfolioOptimizer(pairs)
        self.correlation_tracker = CorrelationTracker(pairs)
        self.risk_manager = PortfolioRiskManager()
        self.circuit_breaker = CircuitBreaker()

        # State tracking
        self.current_positions = {pair: 0.0 for pair in pairs}
        self.current_equity = 10000.0
        self.pair_stats = {}  # for Kelly sizing

    def generate_signals(self, features_by_pair, spread_zscores=None,
                         equity=None, positions=None):
        """Full signal generation pipeline.

        Args:
            features_by_pair: {pair: (market_state, position_info)} or {pair: feature_dict}
            spread_zscores: DataFrame of spread z-scores for stat arb
            equity: current portfolio equity
            positions: current positions dict

        Returns:
            dict of {pair: target_position}
        """
        if equity is not None:
            self.current_equity = equity
        if positions is not None:
            self.current_positions = positions

        # Check circuit breaker
        if self.circuit_breaker.triggered:
            # Re-check to respect cooldown expiry
            if self.circuit_breaker.check(0.0):
                logger.warning("Circuit breaker active — all positions zeroed")
                return {pair: 0.0 for pair in self.pairs}

        # Detect regime if detector available
        regime_scales = None
        if self.regime_detector is not None:
            regime_scales = {}
            for pair in self.pairs:
                feat = features_by_pair.get(pair)
                if feat is not None:
                    regime = self.regime_detector.predict(feat)
                    regime_scales[pair] = REGIME_SCALES.get(regime, 0.5)

        # Generate signals from all alphas
        combined_signals = {pair: 0.0 for pair in self.pairs}

        for alpha in self.alphas:
            try:
                kwargs = {}
                if hasattr(alpha, 'name') and alpha.name == 'stat_arb' and spread_zscores is not None:
                    kwargs['spread_zscores'] = spread_zscores

                signals = alpha.generate(features_by_pair, **kwargs)
                weight = self.alpha_weights.get(
                    alpha.name if hasattr(alpha, 'name') else 'unknown', 0.0
                )
                for sig in signals:
                    combined_signals[sig.pair] = (
                        combined_signals.get(sig.pair, 0.0)
                        + sig.weighted_signal * weight
                    )
            except Exception as e:
                logger.warning(f"Alpha {alpha} failed: {e}")

        # Clip combined signals to [-1, 1]
        for pair in combined_signals:
            combined_signals[pair] = float(np.clip(combined_signals[pair], -1.0, 1.0))

        # Portfolio optimization with Kelly sizing if stats available
        if self.pair_stats:
            target_positions = self.optimizer.optimize_with_kelly(
                combined_signals, self.pair_stats, regime_scales
            )
        else:
            target_positions = self.optimizer.optimize(
                combined_signals, regime_scales
            )

        # Correlation adjustment
        target_positions = self.correlation_tracker.adjust_for_correlation(
            target_positions
        )

        # Portfolio risk check
        target_positions, risk_info = self.risk_manager.check_and_adjust(
            target_positions, self.current_equity
        )

        if risk_info.get("is_halted"):
            self.circuit_breaker.check(risk_info.get("portfolio_dd", 0))

        return target_positions

    def update_pair_stats(self, pair, win_rate, avg_win, avg_loss):
        """Update rolling performance stats for Kelly sizing."""
        self.pair_stats[pair] = {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        }

    def update_correlations(self, pair_returns):
        """Update correlation tracker with latest returns."""
        self.correlation_tracker.update(pair_returns)

    def check_circuit_breaker(self, portfolio_dd):
        """Check and update circuit breaker status."""
        return self.circuit_breaker.check(portfolio_dd)
