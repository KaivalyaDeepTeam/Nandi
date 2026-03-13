"""
NandiDQNAgent — Dueling IQN-DQN with pair embedding and action masking.

Discrete action space: {HOLD=0, LONG=1, SHORT=2, CLOSE=3}
Architecture:
    market_state → feature_proj → MSFAN → (B, 128)
    position_info → position_embed → (B, 32)
    pair_id → pair_embed → (B, 16)
    concat → trunk → dueling (value + advantage) → Q(s, a) per action
    IQN extension: cosine τ-embedding → per-quantile Q-values
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nandi.config import ENCODER_CONFIG, DQN_CONFIG, MODEL_DIR, PAIRS
from nandi.models.msfan import MultiScaleEncoder


# ═══════════════════════════════════════════════════════════════════════
# NoisyLinear — learned exploration (replaces ε-greedy)
# ═══════════════════════════════════════════════════════════════════════

class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet layer (Fortunato et al., 2018)."""

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))

    @staticmethod
    def _factorized_noise(size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = self._factorized_noise(self.in_features)
        eps_out = self._factorized_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ═══════════════════════════════════════════════════════════════════════
# Quantile Embedding — reused concept from AEGIS
# ═══════════════════════════════════════════════════════════════════════

class QuantileEmbedding(nn.Module):
    """IQN cosine basis embedding: τ ∈ [0,1] → feature vector."""

    def __init__(self, embedding_dim=64, output_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.register_buffer(
            "basis_indices",
            torch.arange(1, embedding_dim + 1, dtype=torch.float32),
        )
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, tau):
        """tau: (B, n_tau) → (B, n_tau, output_dim)"""
        tau_expanded = tau.unsqueeze(-1)
        cos_features = torch.cos(math.pi * tau_expanded * self.basis_indices)
        return self.linear(cos_features)


# ═══════════════════════════════════════════════════════════════════════
# NandiDQNAgent — Dueling IQN-DQN
# ═══════════════════════════════════════════════════════════════════════

class NandiDQNAgent(nn.Module):
    """Dueling IQN-DQN with pair embedding and NoisyNet exploration.

    Architecture:
        market_state → feature_proj → MSFAN encoder → (B, 128)
        position_info → position_embed → (B, 32)
        pair_id → pair_embed → (B, 16)
        concat(128+32+16=176) → trunk(176→256→128)
        IQN: trunk_out ⊙ τ_embed → dueling streams → Q(s,a,τ)
        Mean over τ → expected Q(s,a)
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
        self.n_actions = cfg["n_actions"]
        self.n_tau = cfg["n_tau"]
        self.trunk_out = cfg["trunk_out"]  # 128

        # ── Feature projection + encoder (shared with AEGIS) ──
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, d), nn.GELU(), nn.LayerNorm(d),
        )
        self.encoder = MultiScaleEncoder(enc_cfg)

        # ── Position embedding: 6-dim → 32 ──
        self.position_embed = nn.Sequential(
            nn.Linear(cfg["position_dim"], 32), nn.GELU(),
            nn.Linear(32, 32), nn.GELU(),
        )

        # ── Pair embedding: pair_id → 16 ──
        self.pair_embed = nn.Embedding(len(PAIRS), cfg["pair_embed_dim"])

        # ── Trunk: 128+32+16=176 → 256 → 128 ──
        trunk_in = d + 32 + cfg["pair_embed_dim"]  # 176
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, cfg["trunk_hidden"]),  # 176→256
            nn.GELU(),
            nn.LayerNorm(cfg["trunk_hidden"]),
            nn.Linear(cfg["trunk_hidden"], cfg["trunk_out"]),  # 256→128
            nn.GELU(),
        )

        # ── IQN quantile embedding ──
        self.quantile_embed = QuantileEmbedding(
            embedding_dim=cfg["cosine_dim"],
            output_dim=cfg["trunk_out"],
        )

        # ── Dueling streams (NoisyLinear for exploration) ──
        sh = cfg["stream_hidden"]  # 64
        self.value_stream = nn.Sequential(
            NoisyLinear(cfg["trunk_out"], sh, cfg["noisy_sigma"]),
            nn.GELU(),
            NoisyLinear(sh, 1, cfg["noisy_sigma"]),
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(cfg["trunk_out"], sh, cfg["noisy_sigma"]),
            nn.GELU(),
            NoisyLinear(sh, self.n_actions, cfg["noisy_sigma"]),
        )

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def encode(self, market_state, position_info, pair_ids):
        """Encode state into trunk features.

        Args:
            market_state: (B, lookback, n_features)
            position_info: (B, 6)
            pair_ids: (B,) int tensor

        Returns:
            trunk_features: (B, trunk_out=128)
        """
        projected = self.feature_proj(market_state)
        state_enc = self.encoder(projected)  # (B, 128)
        pos_emb = self.position_embed(position_info)  # (B, 32)
        pair_emb = self.pair_embed(pair_ids)  # (B, 16)
        combined = torch.cat([state_enc, pos_emb, pair_emb], dim=-1)  # (B, 176)
        return self.trunk(combined)  # (B, 128)

    def forward(self, market_state, position_info, pair_ids, tau=None,
                n_tau=None, action_mask=None):
        """Full forward pass → per-quantile Q-values.

        Args:
            market_state: (B, lookback, n_features)
            position_info: (B, 6)
            pair_ids: (B,) int tensor
            tau: (B, n_tau) quantile levels, or None to sample
            n_tau: override number of quantiles (default self.n_tau)
            action_mask: (B, 4) bool tensor, True = valid action

        Returns:
            q_quantiles: (B, n_tau, n_actions) — per-quantile Q-values
            tau: (B, n_tau) — quantile levels used
        """
        B = market_state.shape[0]
        n = n_tau or self.n_tau

        if tau is None:
            tau = torch.rand(B, n, device=market_state.device)

        trunk_feat = self.encode(market_state, position_info, pair_ids)  # (B, 128)

        # IQN: Hadamard product of trunk features with quantile embedding
        tau_emb = self.quantile_embed(tau)  # (B, n_tau, 128)
        # Broadcast trunk: (B, 1, 128) * (B, n_tau, 128) → (B, n_tau, 128)
        combined = trunk_feat.unsqueeze(1) * tau_emb

        # Dueling: V + (A - mean(A))
        # Reshape for stream processing: (B*n_tau, 128)
        flat = combined.reshape(B * n, self.trunk_out)
        V = self.value_stream(flat)  # (B*n_tau, 1)
        A = self.advantage_stream(flat)  # (B*n_tau, n_actions)
        A_centered = A - A.mean(dim=-1, keepdim=True)
        Q = V + A_centered
        q_quantiles = Q.reshape(B, n, self.n_actions)  # (B, n_tau, n_actions)

        # Action masking: set invalid actions to very negative Q
        if action_mask is not None:
            # action_mask: (B, 4), True = valid
            # Expand to (B, 1, 4) for broadcasting with (B, n_tau, 4)
            invalid = ~action_mask.unsqueeze(1)
            q_quantiles = q_quantiles.masked_fill(invalid, -1e8)

        return q_quantiles, tau

    def get_q_values(self, market_state, position_info, pair_ids,
                     n_tau=None, action_mask=None):
        """Get expected Q-values (mean over quantiles).

        Returns:
            q_values: (B, n_actions) — expected Q per action
        """
        q_quantiles, _ = self.forward(
            market_state, position_info, pair_ids,
            n_tau=n_tau, action_mask=action_mask,
        )
        return q_quantiles.mean(dim=1)  # (B, n_actions)

    @torch.no_grad()
    def get_action(self, market_state, position_info, pair_id,
                   action_mask=None, deterministic=True):
        """Select action for a single observation.

        Args:
            market_state: numpy (lookback, n_features) or (1, lookback, n_features)
            position_info: numpy (6,) or (1, 6)
            pair_id: int
            action_mask: numpy (4,) bool or None
            deterministic: if True use greedy (NoisyNet provides exploration)

        Returns:
            action: int in {0,1,2,3}
            q_values: numpy (4,) Q-values for all actions
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

        q_values = self.get_q_values(ms_t, pi_t, pid_t, action_mask=mask_t)
        q_np = q_values.cpu().numpy().flatten()

        action = int(q_values.argmax(dim=-1).item())
        return action, q_np

    def get_classification_logits(self, market_state, position_info, pair_ids):
        """Get raw logits for HOA pre-training (no IQN, no noise).

        Uses trunk features directly through advantage stream mean path.
        Returns logits suitable for cross-entropy loss.
        """
        trunk_feat = self.encode(market_state, position_info, pair_ids)

        # Use advantage stream for classification (value irrelevant for classification)
        # Run through NoisyLinear in eval mode (no noise for supervised training)
        flat = trunk_feat
        A = self.advantage_stream(flat)  # (B, n_actions)
        return A

    def save_agent(self, path=None):
        path = path or os.path.join(MODEL_DIR, "dqn_agent.pt")
        if not path.endswith(".pt"):
            path = path + ".pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_agent(self, path=None):
        path = path or os.path.join(MODEL_DIR, "dqn_agent.pt")
        if not path.endswith(".pt"):
            path = path + ".pt"
        if os.path.exists(path):
            saved_state = torch.load(path, map_location="cpu", weights_only=True)
            self.load_state_dict(saved_state, strict=False)
            return True
        return False
