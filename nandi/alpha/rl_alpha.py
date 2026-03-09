"""
RL Alpha — wraps NandiAgent as an alpha signal source.

Phase 2: integrates trained RL agents into the alpha framework.
"""

from typing import List

import numpy as np

from nandi.alpha.base import BaseAlpha, AlphaSignal


class RLAlpha(BaseAlpha):
    """Wraps per-pair NandiAgent instances as alpha sources."""

    def __init__(self, pairs, agents=None):
        """
        Args:
            pairs: list of pair names.
            agents: {pair: NandiAgent} dict of loaded agents. Set later if needed.
        """
        super().__init__(name="rl", pairs=pairs)
        self.agents = agents or {}

    def set_agent(self, pair, agent):
        self.agents[pair] = agent

    def generate(self, features, position_infos=None, **kwargs) -> List[AlphaSignal]:
        """Generate RL signals for all pairs.

        Args:
            features: {pair: (market_state, position_info)} or {pair: market_state_array}.
            position_infos: {pair: position_info_array} if not bundled in features.
        """
        signals = []

        for pair in self.pairs:
            if pair not in self.agents:
                continue

            agent = self.agents[pair]
            feat = features.get(pair)
            if feat is None:
                continue

            if isinstance(feat, tuple):
                market_state, pos_info = feat
            else:
                market_state = feat
                pos_info = position_infos.get(pair, np.zeros(4, dtype=np.float32)) if position_infos else np.zeros(4, dtype=np.float32)

            action, _, _, uncertainty = agent.get_action(
                market_state, pos_info, deterministic=True
            )

            # Confidence is inverse of uncertainty
            confidence = max(0.0, min(1.0, 1.0 - uncertainty))

            signals.append(AlphaSignal(
                pair=pair,
                direction=float(np.clip(action, -1, 1)),
                confidence=confidence,
                alpha_name=self.name,
                metadata={"uncertainty": float(uncertainty)},
            ))

        return signals
