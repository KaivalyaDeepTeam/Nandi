"""Multi-algorithm ensemble — combines PPO, SAC, and TD3 agents."""

import logging
import numpy as np
import torch

from nandi.models.agent import NandiAgent

logger = logging.getLogger(__name__)


class EnsembleAgent:
    """Combines multiple trained agents via weighted averaging.

    Uncertainty is measured as disagreement (std) between agents.
    """

    def __init__(self, agents, weights=None):
        self.agents = agents
        self.weights = weights or [1.0 / len(agents)] * len(agents)

    def get_action(self, market_state, position_info, deterministic=True):
        """Get ensemble action — weighted average of all agents."""
        actions = []
        values = []

        for agent in self.agents:
            action, _, value, _ = agent.get_action(
                market_state, position_info, deterministic=deterministic
            )
            actions.append(action)
            values.append(value)

        # Weighted average action
        action = sum(w * a for w, a in zip(self.weights, actions))
        action = float(np.clip(action, -1.0, 1.0))

        # Uncertainty = disagreement between agents
        uncertainty = float(np.std(actions))

        # Average value
        value = sum(w * v for w, v in zip(self.weights, values))

        return action, 0.0, value, uncertainty

    def update_weights(self, agent_sharpes):
        """Update weights based on recent Sharpe ratios."""
        sharpes = np.array(agent_sharpes)
        sharpes = np.clip(sharpes, 0, None)  # only positive weights
        total = sharpes.sum()
        if total > 0:
            self.weights = (sharpes / total).tolist()
        logger.info(f"Ensemble weights updated: {self.weights}")

    def eval(self):
        for agent in self.agents:
            agent.eval()

    def train(self):
        for agent in self.agents:
            agent.train()

    def save_agent(self, path=None):
        for i, agent in enumerate(self.agents):
            agent_path = f"{path}_agent{i}" if path else None
            agent.save_agent(agent_path)
