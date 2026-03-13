"""
NandiPPOAgent — Discrete-Action Actor-Critic with shared MSFAN encoder.

Key advantage over DQN: outputs action PROBABILITIES, not Q-values.
PPO can learn π(LONG|s)=0.05 globally but π(LONG|s)=0.8 on specific bars,
without Q-value dominance forcing argmax = HOLD everywhere.

Architecture (shared encoder with DQN):
    market_state → feature_proj → MSFAN → (B, 128)
    position_info → position_embed → (B, 32)
    pair_id → pair_embed → (B, 16)
    concat(176) → trunk(176→256→128)
    ├── actor_head(128→64→4) → Categorical logits
    └── critic_head(128→64→1) → V(s) scalar

Action space: {HOLD=0, LONG=1, SHORT=2, CLOSE=3} with masking.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from nandi.config import ENCODER_CONFIG, DQN_CONFIG, MODEL_DIR, PAIRS
from nandi.models.msfan import MultiScaleEncoder


class NandiPPOAgent(nn.Module):
    """Discrete-action PPO agent with shared MSFAN encoder.

    Same encoder pipeline as NandiDQNAgent, but replaces IQN dueling
    streams with simple actor (logits) and critic (value) heads.
    """

    ACTIONS = {"HOLD": 0, "LONG": 1, "SHORT": 2, "CLOSE": 3}
    N_ACTIONS = 4

    def __init__(self, n_features, encoder_config=None, dqn_config=None,
                 encoder_type="msfan"):
        super().__init__()
        enc_cfg = encoder_config or ENCODER_CONFIG
        cfg = dqn_config or DQN_CONFIG
        d = enc_cfg["d_model"]  # 128

        self.n_features = n_features
        self.d_model = d
        self.n_actions = cfg["n_actions"]  # 4
        self.trunk_out = cfg["trunk_out"]  # 128

        # ── Shared encoder (identical to DQN) ──
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, d), nn.GELU(), nn.LayerNorm(d),
        )
        self.encoder = MultiScaleEncoder(enc_cfg)

        self.position_embed = nn.Sequential(
            nn.Linear(cfg["position_dim"], 32), nn.GELU(),
            nn.Linear(32, 32), nn.GELU(),
        )

        self.pair_embed = nn.Embedding(len(PAIRS), cfg["pair_embed_dim"])

        trunk_in = d + 32 + cfg["pair_embed_dim"]  # 176
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, cfg["trunk_hidden"]),  # 176→256
            nn.GELU(),
            nn.LayerNorm(cfg["trunk_hidden"]),
            nn.Linear(cfg["trunk_hidden"], cfg["trunk_out"]),  # 256→128
            nn.GELU(),
        )

        # ── Actor head: logits for Categorical distribution ──
        self.actor_head = nn.Sequential(
            nn.Linear(cfg["trunk_out"], 64),
            nn.GELU(),
            nn.Linear(64, self.n_actions),
        )

        # ── Critic head: scalar V(s) ──
        self.critic_head = nn.Sequential(
            nn.Linear(cfg["trunk_out"], 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def encode(self, market_state, position_info, pair_ids):
        """Encode state into trunk features (identical to DQN).

        Args:
            market_state: (B, lookback, n_features)
            position_info: (B, position_dim)
            pair_ids: (B,) int tensor

        Returns:
            trunk_features: (B, trunk_out=128)
        """
        projected = self.feature_proj(market_state)
        state_enc = self.encoder(projected)
        pos_emb = self.position_embed(position_info)
        pair_emb = self.pair_embed(pair_ids)
        combined = torch.cat([state_enc, pos_emb, pair_emb], dim=-1)
        return self.trunk(combined)

    def get_policy_and_value(self, market_state, position_info, pair_ids,
                             action_mask=None):
        """Get action distribution and state value.

        Args:
            market_state: (B, lookback, n_features)
            position_info: (B, position_dim)
            pair_ids: (B,) long
            action_mask: (B, 4) bool, True = valid action

        Returns:
            dist: Categorical distribution over actions
            value: (B,) state values
        """
        trunk_feat = self.encode(market_state, position_info, pair_ids)
        logits = self.actor_head(trunk_feat)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e8)

        dist = Categorical(logits=logits)
        value = self.critic_head(trunk_feat).squeeze(-1)
        return dist, value

    @torch.no_grad()
    def get_action(self, market_state, position_info, pair_id,
                   action_mask=None, deterministic=False):
        """Select action for a single observation.

        Args:
            market_state: numpy (lookback, n_features) or (1, lookback, n_features)
            position_info: numpy (position_dim,) or (1, position_dim)
            pair_id: int
            action_mask: numpy (4,) bool or None
            deterministic: if True, use argmax (greedy)

        Returns:
            action: int in {0,1,2,3}
            log_prob: float
            value: float
            probs: numpy (4,) action probabilities
        """
        device = next(self.parameters()).device

        if market_state.ndim == 2:
            market_state = market_state[np.newaxis]
        if position_info.ndim == 1:
            position_info = position_info[np.newaxis]

        ms_t = torch.tensor(market_state, dtype=torch.float32, device=device)
        pi_t = torch.tensor(position_info, dtype=torch.float32, device=device)
        pid_t = torch.tensor([pair_id], dtype=torch.long, device=device)

        mask_t = None
        if action_mask is not None:
            mask_t = torch.tensor(action_mask, dtype=torch.bool, device=device)
            if mask_t.ndim == 1:
                mask_t = mask_t.unsqueeze(0)

        dist, value = self.get_policy_and_value(ms_t, pi_t, pid_t,
                                                 action_mask=mask_t)

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        probs = dist.probs.cpu().numpy().flatten()

        return (int(action.item()), float(log_prob.item()),
                float(value.item()), probs)

    def evaluate_actions(self, market_state, position_info, pair_ids,
                         actions, action_masks=None):
        """Evaluate log_prob, value, entropy for batched actions (PPO update).

        Args:
            market_state: (B, lookback, n_features)
            position_info: (B, position_dim)
            pair_ids: (B,) long
            actions: (B,) long — actions taken
            action_masks: (B, 4) bool or None

        Returns:
            log_probs: (B,)
            values: (B,)
            entropy: (B,)
        """
        dist, value = self.get_policy_and_value(
            market_state, position_info, pair_ids, action_mask=action_masks,
        )
        return dist.log_prob(actions), value, dist.entropy()

    def get_classification_logits(self, market_state, position_info, pair_ids):
        """For HOA pre-training — same interface as DQN agent."""
        trunk_feat = self.encode(market_state, position_info, pair_ids)
        return self.actor_head(trunk_feat)

    def save_agent(self, path=None):
        path = path or os.path.join(MODEL_DIR, "ppo_agent.pt")
        if not path.endswith(".pt"):
            path = path + ".pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_agent(self, path=None):
        path = path or os.path.join(MODEL_DIR, "ppo_agent.pt")
        if not path.endswith(".pt"):
            path = path + ".pt"
        if os.path.exists(path):
            saved_state = torch.load(path, map_location="cpu", weights_only=True)
            self.load_state_dict(saved_state, strict=False)
            return True
        return False
