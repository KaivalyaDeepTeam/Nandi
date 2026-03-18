"""
SPINAgent — Stochastic Path Intelligence Network (V7).

~55K param model grounded in Financial Mathematics:
  - CausalConvEncoder: depthwise-separable causal convolutions (~11K params)
  - RegimeGate: learned market condition adaptation (~1.6K params)
  - Separate Entry/Exit heads: because entry and exit are different skills
  - PPO-compatible interface: same get_policy_and_value / evaluate_actions API

Architecture:
    market_state (B, 120, 65) -> feature_proj(65->48) -> CausalConvEncoder -> (B, 48)
                                                                  |
                                               RegimeGate -----> (B, 48)
                                                                  |
    position_info (B, 12) -> embed(12->24) --+                    |
    pair_id (B,) -> embed(8,8) -------------+                    |
                                              +-> concat(80) --> trunk(80->128->64)
                                                                  |
                    +-----------------------------+---------------+
                    v                             v               v
              Entry Head                    Exit Head        Value Head
            (when flat only)            (when in trade)
            64->32->3 (H/L/S)           64->32->2 (H/C)     64->32->1
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from nandi.config import SPIN_CONFIG, MODEL_DIR, PAIRS

# Action constants
HOLD = 0
LONG = 1
SHORT = 2
CLOSE = 3


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-separable causal convolution block."""

    def __init__(self, channels, kernel_size, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding

        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation,
            groups=channels, bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=True)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.causal_trim = padding

    def forward(self, x):
        """x: (B, T, C) -> (B, T, C)"""
        # Conv1d expects (B, C, T)
        residual = x
        h = x.transpose(1, 2)
        h = self.depthwise(h)
        # Trim future (causal)
        if self.causal_trim > 0:
            h = h[:, :, :-self.causal_trim]
        h = self.pointwise(h)
        h = h.transpose(1, 2)  # back to (B, T, C)
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)
        return h + residual


class CausalConvEncoder(nn.Module):
    """3-block depthwise-separable causal conv encoder (~11K params).

    Replaces 544K-param MSFAN. Receptive field ~22 bars.
    Path signatures handle long-range dependencies.
    """

    def __init__(self, d_model, kernels, dilations, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            DepthwiseSeparableConv1d(d_model, k, d, dropout)
            for k, d in zip(kernels, dilations)
        ])

    def forward(self, x):
        """x: (B, T, d_model) -> (B, d_model) [last timestep]"""
        for block in self.blocks:
            x = block(x)
        return x[:, -1, :]  # take last timestep


class RegimeGate(nn.Module):
    """Learned market condition adaptation (~1.6K params).

    Bottleneck MLP: d_model -> bottleneck -> d_model with Sigmoid.
    Modulates encoder output element-wise.
    """

    def __init__(self, d_model, bottleneck):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: (B, d_model) -> (B, d_model)"""
        return x * self.gate(x)


class SPINAgent(nn.Module):
    """SPIN: Stochastic Path Intelligence Network.

    ~55K params for 65 features. PPO-compatible interface.
    Separate entry/exit heads unified to 4-action logits.
    """

    ACTIONS = {"HOLD": 0, "LONG": 1, "SHORT": 2, "CLOSE": 3}
    N_ACTIONS = 4

    def __init__(self, n_features, spin_config=None):
        super().__init__()
        cfg = spin_config or SPIN_CONFIG
        d = cfg["d_model"]

        self.n_features = n_features
        self.d_model = d
        self.n_actions = 4
        self.trunk_out = cfg["trunk_out"]

        # ── Feature projection ──
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

        # ── CausalConvEncoder ──
        self.encoder = CausalConvEncoder(
            d_model=d,
            kernels=cfg["conv_kernels"],
            dilations=cfg["conv_dilations"],
            dropout=cfg["conv_dropout"],
        )

        # ── RegimeGate ──
        self.regime_gate = RegimeGate(d, cfg["regime_bottleneck"])

        # ── Position embedding (12 -> 24) ──
        self.position_embed = nn.Sequential(
            nn.Linear(cfg["position_dim"], cfg["position_embed_dim"]),
            nn.GELU(),
        )

        # ── Pair embedding ──
        self.pair_embed = nn.Embedding(len(PAIRS), cfg["pair_embed_dim"])

        # ── Trunk ──
        trunk_in = d + cfg["position_embed_dim"] + cfg["pair_embed_dim"]
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, cfg["trunk_hidden"]),
            nn.GELU(),
            nn.LayerNorm(cfg["trunk_hidden"]),
            nn.Dropout(0.12),
            nn.Linear(cfg["trunk_hidden"], cfg["trunk_out"]),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        # ── Entry head (flat -> HOLD/LONG/SHORT) ──
        self.entry_head = nn.Sequential(
            nn.Linear(cfg["trunk_out"], cfg["head_hidden"]),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(cfg["head_hidden"], cfg["n_entry_actions"]),
        )

        # ── Exit head (in trade -> HOLD/CLOSE) ──
        self.exit_head = nn.Sequential(
            nn.Linear(cfg["trunk_out"], cfg["head_hidden"]),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(cfg["head_hidden"], cfg["n_exit_actions"]),
        )

        # ── Value head ──
        self.value_head = nn.Sequential(
            nn.Linear(cfg["trunk_out"], cfg["head_hidden"]),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(cfg["head_hidden"], 1),
        )

        # ── HOA classification head (4 actions, for pre-training) ──
        self.hoa_head = nn.Sequential(
            nn.Linear(cfg["trunk_out"], cfg["head_hidden"]),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(cfg["head_hidden"], cfg["n_hoa_actions"]),
        )

    def encode(self, market_state, position_info, pair_ids):
        """Encode state into trunk features.

        Args:
            market_state: (B, lookback, n_features)
            position_info: (B, position_dim)
            pair_ids: (B,) int tensor

        Returns:
            trunk_features: (B, trunk_out)
        """
        projected = self.feature_proj(market_state)
        state_enc = self.encoder(projected)
        state_enc = self.regime_gate(state_enc)
        pos_emb = self.position_embed(position_info)
        pair_emb = self.pair_embed(pair_ids)
        combined = torch.cat([state_enc, pos_emb, pair_emb], dim=-1)
        return self.trunk(combined)

    def get_policy_and_value(self, market_state, position_info, pair_ids,
                             action_mask=None):
        """Get action distribution and state value.

        Uses position_info[:, 0] to dispatch between entry/exit heads.
        Flat (pos_state == 0) -> entry_head fills logits[0,1,2], logits[3] = -inf
        In trade (pos_state != 0) -> exit_head fills logits[0] and logits[3],
                                      logits[1,2] = -inf

        Args:
            market_state: (B, lookback, n_features)
            position_info: (B, position_dim)
            pair_ids: (B,) long
            action_mask: (B, 4) bool, True = valid action

        Returns:
            dist: Categorical distribution over 4 actions
            value: (B,) state values
        """
        trunk_feat = self.encode(market_state, position_info, pair_ids)

        # Determine position state from position_info[:, 0]
        pos_state = position_info[:, 0]  # (B,) — -1, 0, or +1
        is_flat = (pos_state.abs() < 0.5)  # (B,) bool

        # Compute both heads
        entry_logits = self.entry_head(trunk_feat)  # (B, 3): H, L, S
        exit_logits = self.exit_head(trunk_feat)    # (B, 2): H, C

        # Build unified 4-action logits
        B = trunk_feat.shape[0]
        device = trunk_feat.device
        logits = torch.full((B, 4), -1e8, device=device)

        # Flat positions: entry head -> HOLD(0), LONG(1), SHORT(2)
        flat_idx = is_flat.nonzero(as_tuple=True)[0]
        if len(flat_idx) > 0:
            logits[flat_idx, 0] = entry_logits[flat_idx, 0]  # HOLD
            logits[flat_idx, 1] = entry_logits[flat_idx, 1]  # LONG
            logits[flat_idx, 2] = entry_logits[flat_idx, 2]  # SHORT

        # In-trade positions: exit head -> HOLD(0), CLOSE(3)
        trade_idx = (~is_flat).nonzero(as_tuple=True)[0]
        if len(trade_idx) > 0:
            logits[trade_idx, 0] = exit_logits[trade_idx, 0]  # HOLD
            logits[trade_idx, 3] = exit_logits[trade_idx, 1]  # CLOSE

        # Apply action mask on top
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e8)

        dist = Categorical(logits=logits)
        value = self.value_head(trunk_feat).squeeze(-1)
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
            deterministic: if True, use argmax

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
            actions: (B,) long
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
        """For HOA pre-training — uses dedicated HOA head."""
        trunk_feat = self.encode(market_state, position_info, pair_ids)
        return self.hoa_head(trunk_feat)

    def save_agent(self, path=None):
        path = path or os.path.join(MODEL_DIR, "spin_agent.pt")
        if not path.endswith(".pt"):
            path = path + ".pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_agent(self, path=None):
        path = path or os.path.join(MODEL_DIR, "spin_agent.pt")
        if not path.endswith(".pt"):
            path = path + ".pt"
        if os.path.exists(path):
            saved_state = torch.load(path, map_location="cpu", weights_only=True)
            self.load_state_dict(saved_state, strict=False)
            return True
        return False
