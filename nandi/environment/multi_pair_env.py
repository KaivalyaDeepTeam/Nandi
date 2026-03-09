"""
Multi-Pair Portfolio Environment — wraps N single-pair environments.

Each pair gets its own RL agent (Phase 1: independent agents).
The multi-pair env coordinates portfolio-level constraints.
"""

import numpy as np
import logging

from nandi.config import LOOKBACK_WINDOW, INITIAL_BALANCE, PORTFOLIO_RISK
from nandi.environment.single_pair_env import ForexTradingEnv
from nandi.environment.market_sim import MarketSimulator

logger = logging.getLogger(__name__)


class MultiPairEnv:
    """Coordinates multiple single-pair environments for portfolio backtesting.

    Phase 1: Each pair runs independently with portfolio-level exposure caps.
    """

    def __init__(self, pair_data_dict, lookback=LOOKBACK_WINDOW,
                 initial_balance=INITIAL_BALANCE):
        """
        Args:
            pair_data_dict: {pair_name: {"train_features": ..., "train_prices": ...}} or
                           {pair_name: {"features": ..., "prices": ...}}
            lookback: lookback window for each env.
            initial_balance: total portfolio starting balance.
        """
        self.pairs = list(pair_data_dict.keys())
        self.n_pairs = len(self.pairs)
        self.initial_balance = initial_balance
        self.per_pair_balance = initial_balance  # each pair trades as if it has full balance

        self.envs = {}
        for pair, data in pair_data_dict.items():
            features = data.get("features", data.get("train_features"))
            prices = data.get("prices", data.get("train_prices"))
            market_sim = MarketSimulator(pair_name=pair)
            self.envs[pair] = ForexTradingEnv(
                features=features, prices=prices,
                lookback=lookback, initial_balance=self.per_pair_balance,
                pair_name=pair, market_sim=market_sim,
            )

        self.positions = {pair: 0.0 for pair in self.pairs}
        self.portfolio_equity = initial_balance
        self.peak_portfolio_equity = initial_balance

    def reset(self):
        """Reset all environments."""
        states = {}
        for pair in self.pairs:
            states[pair] = self.envs[pair].reset()
        self.positions = {pair: 0.0 for pair in self.pairs}
        self.portfolio_equity = self.initial_balance
        self.peak_portfolio_equity = self.initial_balance
        return states

    def step(self, actions):
        """Step all environments with portfolio-level constraints.

        Args:
            actions: {pair_name: float} target positions.

        Returns:
            states, rewards, dones, infos (all dicts keyed by pair)
        """
        # Enforce portfolio exposure limit
        actions = self._enforce_exposure_limits(actions)

        states = {}
        rewards = {}
        dones = {}
        infos = {}
        all_done = True

        for pair in self.pairs:
            action = actions.get(pair, 0.0)
            state, reward, done, info = self.envs[pair].step(action)
            states[pair] = state
            rewards[pair] = reward
            dones[pair] = done
            infos[pair] = info
            self.positions[pair] = info["position"]
            if not done:
                all_done = False

        # Update portfolio equity
        self.portfolio_equity = sum(
            self.envs[p].equity for p in self.pairs
        ) / self.n_pairs * (self.initial_balance / self.per_pair_balance)

        self.peak_portfolio_equity = max(self.peak_portfolio_equity, self.portfolio_equity)

        # Check portfolio-level circuit breaker
        portfolio_dd = (self.peak_portfolio_equity - self.portfolio_equity) / self.peak_portfolio_equity
        if portfolio_dd > PORTFOLIO_RISK["max_portfolio_dd"]:
            logger.warning(f"Portfolio DD {portfolio_dd:.2%} exceeds limit — forcing all flat")
            for pair in self.pairs:
                self.envs[pair].position = 0.0
                dones[pair] = True

        return states, rewards, dones, infos

    def _enforce_exposure_limits(self, actions):
        """Scale down positions if total exposure exceeds limit."""
        max_exposure = PORTFOLIO_RISK["max_total_exposure"]
        max_single = PORTFOLIO_RISK["max_single_pair"]

        # Clip individual pairs
        clipped = {}
        for pair, action in actions.items():
            clipped[pair] = float(np.clip(action, -max_single, max_single))

        # Check total exposure
        total_exposure = sum(abs(a) for a in clipped.values())
        if total_exposure > max_exposure:
            scale = max_exposure / total_exposure
            clipped = {p: a * scale for p, a in clipped.items()}

        return clipped

    def get_portfolio_info(self):
        """Portfolio-level statistics."""
        pair_equities = {p: self.envs[p].equity for p in self.pairs}
        pair_returns = {
            p: (self.envs[p].equity / self.per_pair_balance - 1) * 100
            for p in self.pairs
        }
        portfolio_dd = (
            (self.peak_portfolio_equity - self.portfolio_equity)
            / self.peak_portfolio_equity
        )
        total_exposure = sum(abs(self.positions[p]) for p in self.pairs)

        return {
            "portfolio_equity": self.portfolio_equity,
            "portfolio_dd": portfolio_dd,
            "total_exposure": total_exposure,
            "pair_equities": pair_equities,
            "pair_returns": pair_returns,
            "positions": dict(self.positions),
        }
