"""
AEGIS — Adaptive Edge-Gated Intelligent Scalper.

A novel RL algorithm designed for forex scalping that combines 6 innovations
no existing algorithm has together:

1. DISTRIBUTIONAL TWIN CRITICS (IQN): Learn the full return distribution,
   not just the expected value. Enables reasoning about tail risk.

2. CVaR POLICY OPTIMIZATION: Optimize for the worst α% of outcomes,
   not the average. Naturally conservative — protects capital.

3. EDGE GATE: A separate learned network that estimates P(profitable trade | state).
   Multiplicatively gates the policy — when confidence is low, position is zero.
   This is NOT entropy (which encourages exploration). This is ABSTENTION
   (which prevents trading when the model doesn't have an edge).

4. REGIME-CONDITIONAL POLICY VIA VAE: A variational autoencoder compresses
   market state into a latent regime vector. Policy is conditioned on this —
   automatically adapts to trending/mean-reverting/volatile without labels.

5. ASYMMETRIC CRITIC LOSS: Overestimating returns is MORE dangerous than
   underestimating. Penalize critic overestimation 2x harder.

6. CONVICTION-BASED SIZING: Position size = edge_score × policy_output.
   High conviction → full position. Low conviction → zero.

The philosophy: "Take many small, high-probability edges. Skip everything else."
"""

import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nandi.config import ENCODER_CONFIG, MODEL_DIR
from nandi.models.msfan import MultiScaleEncoder
from nandi.models.tft import TemporalFusionTransformer
from nandi.models.ssm import SelectiveSSM


# ═══════════════════════════════════════════════════════════════════════
# Component 1: Regime VAE — learns latent market regime without labels
# ═══════════════════════════════════════════════════════════════════════

class RegimeVAE(nn.Module):
    """Variational regime encoder.

    Maps market state encoding to a low-dimensional latent space that
    captures the current market regime (trending, mean-reverting, volatile,
    calm, etc.) WITHOUT requiring regime labels.

    The KL divergence regularization ensures the latent space is structured:
    similar market conditions map to nearby points.
    """

    def __init__(self, input_dim=128, regime_dim=8):
        super().__init__()
        self.regime_dim = regime_dim

        # Encoder: state → (μ, log_var)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.mu_head = nn.Linear(64, regime_dim)
        self.logvar_head = nn.Linear(64, regime_dim)

    def forward(self, state_encoding):
        """
        Args:
            state_encoding: (batch, input_dim) from market encoder

        Returns:
            mu: (batch, regime_dim)
            log_var: (batch, regime_dim) — log variance for reparameterization
            z: (batch, regime_dim) — sampled latent regime
        """
        h = self.encoder(state_encoding)
        mu = self.mu_head(h)
        log_var = torch.clamp(self.logvar_head(h), -4.0, 2.0)

        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu  # deterministic at inference

        return mu, log_var, z

    @staticmethod
    def kl_loss(mu, log_var):
        """KL divergence from N(mu, sigma²) to N(0, I)."""
        return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


# ═══════════════════════════════════════════════════════════════════════
# Component 2: Edge Gate — learns WHEN to trade (abstention mechanism)
# ═══════════════════════════════════════════════════════════════════════

class EdgeGate(nn.Module):
    """Learned edge detector — the heart of AEGIS.

    Outputs a scalar in [0, 1] representing the model's confidence that
    there's a profitable trading opportunity in the current state.

    Final position = edge_score × policy_output

    When edge_score ≈ 0: agent stays flat (no edge detected)
    When edge_score ≈ 1: agent takes full position (high conviction)

    This is fundamentally different from:
    - SAC's entropy (encourages exploration, not abstention)
    - Uncertainty thresholding (binary, not smooth)
    - Action masking (discrete, not continuous)

    The gate is trained with binary cross-entropy on whether the
    subsequent trade was profitable — it learns to recognize patterns
    that precede winning trades.
    """

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, combined_state):
        """
        Args:
            combined_state: (batch, input_dim) — cat(state_enc, pos_emb, regime_z)

        Returns:
            edge_score: (batch, 1) in [0, 1]
        """
        return torch.sigmoid(self.net(combined_state))


# ═══════════════════════════════════════════════════════════════════════
# Component 3: IQN Quantile Embedding — for distributional critic
# ═══════════════════════════════════════════════════════════════════════

class QuantileEmbedding(nn.Module):
    """Implicit Quantile Network (IQN) quantile embedding.

    Maps quantile values τ ∈ [0, 1] to feature vectors using cosine basis:
        φ(τ) = ReLU(Linear([cos(πτ), cos(2πτ), ..., cos(nπτ)]))

    This allows the critic to output different Q-values for different
    parts of the return distribution — essential for CVaR optimization.
    """

    def __init__(self, embedding_dim=64, output_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Cosine basis indices: [1, 2, ..., embedding_dim]
        self.register_buffer(
            "basis_indices",
            torch.arange(1, embedding_dim + 1, dtype=torch.float32)
        )
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, tau):
        """
        Args:
            tau: (batch, n_quantiles) — quantile values in [0, 1]

        Returns:
            (batch, n_quantiles, output_dim) — quantile feature vectors
        """
        # tau: (batch, n_quantiles) → (batch, n_quantiles, 1)
        tau_expanded = tau.unsqueeze(-1)
        # Cosine features: cos(π * i * τ) for i = 1..embedding_dim
        # (batch, n_quantiles, embedding_dim)
        cos_features = torch.cos(math.pi * tau_expanded * self.basis_indices)
        return self.linear(cos_features)


# ═══════════════════════════════════════════════════════════════════════
# Component 4: Distributional Twin Critic — learns full return distribution
# ═══════════════════════════════════════════════════════════════════════

class DistributionalCritic(nn.Module):
    """IQN-style distributional critic.

    Instead of learning Q(s,a) = E[return], learns the full QUANTILE FUNCTION:
        Z(s, a, τ) = the τ-th quantile of the return distribution

    This means:
    - Z(s, a, 0.1) = "what's the return in the worst 10% of outcomes?"
    - Z(s, a, 0.5) = "what's the median return?"
    - Z(s, a, 0.9) = "what's the return in the best 10% of outcomes?"

    CVaR_α = average of Z(s, a, τ) for τ ∈ [0, α]
    = "expected return given we're in the worst α% of outcomes"

    By optimizing CVaR instead of E[return], the policy becomes
    naturally conservative — it avoids actions with bad tail risk
    even if their average return is good.
    """

    def __init__(self, state_dim, action_dim=1, hidden_dim=128,
                 cosine_embedding_dim=64, n_quantiles=32):
        super().__init__()
        self.n_quantiles = n_quantiles

        # State-action encoder
        self.state_action_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Quantile embedding
        self.quantile_embed = QuantileEmbedding(
            embedding_dim=cosine_embedding_dim,
            output_dim=hidden_dim,
        )

        # Output head: combined → quantile values
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state_encoding, action, tau=None):
        """
        Args:
            state_encoding: (batch, state_dim) — cat(enc, pos_emb, regime_z)
            action: (batch, 1) — continuous action
            tau: (batch, n_quantiles) — quantile values, or None to sample

        Returns:
            quantile_values: (batch, n_quantiles) — Z(s, a, τ) for each τ
            tau: (batch, n_quantiles) — the τ values used
        """
        batch_size = state_encoding.shape[0]
        if tau is None:
            tau = torch.rand(batch_size, self.n_quantiles,
                             device=state_encoding.device)

        # State-action features: (batch, hidden_dim)
        sa = torch.cat([state_encoding, action], dim=-1)
        sa_embed = self.state_action_net(sa)

        # Quantile features: (batch, n_quantiles, hidden_dim)
        tau_embed = self.quantile_embed(tau)

        # Hadamard product: broadcast sa_embed across quantiles
        # (batch, 1, hidden_dim) * (batch, n_quantiles, hidden_dim)
        combined = sa_embed.unsqueeze(1) * tau_embed

        # Output: (batch, n_quantiles, 1) → (batch, n_quantiles)
        quantile_values = self.output_net(combined).squeeze(-1)

        return quantile_values, tau


class DistributionalTwinCritic(nn.Module):
    """Twin distributional critics for stability (min-Q trick on quantiles)."""

    def __init__(self, state_dim, action_dim=1, hidden_dim=128,
                 cosine_embedding_dim=64, n_quantiles=32):
        super().__init__()
        self.z1 = DistributionalCritic(
            state_dim, action_dim, hidden_dim, cosine_embedding_dim, n_quantiles
        )
        self.z2 = DistributionalCritic(
            state_dim, action_dim, hidden_dim, cosine_embedding_dim, n_quantiles
        )

    def forward(self, state_encoding, action, tau=None):
        """Returns quantile values from both critics."""
        z1, tau1 = self.z1(state_encoding, action, tau)
        z2, tau2 = self.z2(state_encoding, action, tau)
        return z1, z2, tau1

    def z1_forward(self, state_encoding, action, tau=None):
        """Forward through Q1 only (for actor update)."""
        return self.z1(state_encoding, action, tau)


# ═══════════════════════════════════════════════════════════════════════
# AEGIS Agent — orchestrates all components
# ═══════════════════════════════════════════════════════════════════════

class AEGISAgent(nn.Module):
    """Adaptive Edge-Gated Intelligent Scalper.

    Combines: encoder + regime VAE + edge gate + policy.
    The distributional critics are maintained separately by the trainer.

    Compatible with existing NandiAgent interface:
    - get_action(market_state, position_info) → (action, log_prob, value, uncertainty)
    - save_agent(path) / load_agent(path)
    """

    ENCODERS = {
        "msfan": MultiScaleEncoder,
        "tft": TemporalFusionTransformer,
        "ssm": SelectiveSSM,
    }

    def __init__(self, n_features, encoder_config=None, encoder_type="msfan",
                 regime_dim=8, cvar_alpha=0.25, use_text_encoder=False):
        super().__init__()
        config = encoder_config or ENCODER_CONFIG
        d = config["d_model"]  # 128
        self.d_model = d
        self.regime_dim = regime_dim
        self.cvar_alpha = cvar_alpha
        self.use_text_encoder = use_text_encoder

        # ── Shared Feature Extraction (same as NandiAgent) ──
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, d), nn.GELU(), nn.LayerNorm(d),
        )
        EncoderClass = self.ENCODERS.get(encoder_type, MultiScaleEncoder)
        self.encoder = EncoderClass(config)
        self.position_embed = nn.Sequential(
            nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(),
        )

        # ── AEGIS-Specific Components ──
        # Combined state dim: state_encoding(128) + pos_embedding(32) + regime_z(8)
        combined_dim = d + 32 + regime_dim

        # Regime VAE: captures market regime without labels
        self.regime_vae = RegimeVAE(input_dim=d, regime_dim=regime_dim)

        # ── Text Encoder (optional — Innovation #7: News Understanding) ──
        # When enabled, AEGIS reads actual news headlines via cross-attention.
        # When disabled or no headlines provided, works exactly as before.
        self.text_encoder = None
        self.market_news_attn = None
        if use_text_encoder:
            from nandi.models.text_encoder import FinancialTextEncoder, MarketNewsAttention
            self.text_encoder = FinancialTextEncoder(output_dim=d)
            self.market_news_attn = MarketNewsAttention(
                market_dim=d, news_dim=d, n_heads=4,
            )

        # Edge Gate: learns WHEN to trade
        self.edge_gate = EdgeGate(input_dim=combined_dim, hidden_dim=128)

        # Policy: regime-conditional actor
        self.policy_trunk = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.actor_mean = nn.Linear(128, 1)
        self.log_std_head = nn.Sequential(
            nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1),
        )

    def encode_state(self, market_state, position_info,
                     headlines=None, text_embeddings=None):
        """Full encoding pipeline: features → encoder → (text fusion) → regime VAE.

        Args:
            market_state: (batch, lookback, n_features)
            position_info: (batch, 4)
            headlines: optional list of headline strings (live mode)
            text_embeddings: optional pre-computed text embeddings (training mode)
                            shape: (batch, d_model)

        Returns:
            state_enc: (batch, d_model) — market state encoding (fused with text if available)
            pos_emb: (batch, 32) — position embedding
            regime_mu: (batch, regime_dim)
            regime_log_var: (batch, regime_dim)
            regime_z: (batch, regime_dim) — sampled regime latent
            combined: (batch, d_model + 32 + regime_dim) — full state
        """
        projected = self.feature_proj(market_state)
        state_enc = self.encoder(projected)

        # ── Text fusion via cross-attention ──
        # Market state attends to news headlines → enriched state
        if self.text_encoder is not None and self.market_news_attn is not None:
            text_emb = None

            if text_embeddings is not None:
                # Training mode: use pre-computed embeddings
                text_emb = text_embeddings
            elif headlines is not None:
                # Live mode: encode headlines on-the-fly
                text_emb, _ = self.text_encoder(headlines)

            if text_emb is not None:
                # Ensure batch dim matches
                if text_emb.shape[0] != state_enc.shape[0]:
                    text_emb = text_emb.expand(state_enc.shape[0], -1)
                # Cross-attention: market attends to news
                state_enc = self.market_news_attn(state_enc, text_emb)

        pos_emb = self.position_embed(position_info)
        regime_mu, regime_log_var, regime_z = self.regime_vae(state_enc)

        combined = torch.cat([state_enc, pos_emb, regime_z], dim=-1)
        return state_enc, pos_emb, regime_mu, regime_log_var, regime_z, combined

    def forward(self, market_state, position_info,
                headlines=None, text_embeddings=None):
        """Forward pass compatible with NandiAgent interface.

        Returns:
            pre_tanh_mean: (batch, 1) — policy mean before tanh
            action_std: (batch, 1) — exploration std
            value: (batch, 1) — placeholder (CVaR computed by trainer)
        """
        _, _, _, _, _, combined = self.encode_state(
            market_state, position_info,
            headlines=headlines, text_embeddings=text_embeddings,
        )
        h = self.policy_trunk(combined)
        pre_tanh_mean = self.actor_mean(h)
        log_std = torch.clamp(self.log_std_head(h), -2.0, 0.5)
        action_std = torch.exp(log_std)
        # Value placeholder — real CVaR comes from distributional critic
        value = torch.zeros_like(pre_tanh_mean)
        return pre_tanh_mean, action_std, value

    def get_policy_and_edge(self, market_state, position_info,
                            headlines=None, text_embeddings=None):
        """Full AEGIS forward: policy + edge gate + regime.

        Returns:
            pre_tanh_mean, action_std, edge_score, regime_mu, regime_log_var, combined
        """
        (state_enc, pos_emb, regime_mu, regime_log_var,
         regime_z, combined) = self.encode_state(
            market_state, position_info,
            headlines=headlines, text_embeddings=text_embeddings,
        )

        h = self.policy_trunk(combined)
        pre_tanh_mean = self.actor_mean(h)
        log_std = torch.clamp(self.log_std_head(h), -2.0, 0.5)
        action_std = torch.exp(log_std)
        edge_score = self.edge_gate(combined)

        return pre_tanh_mean, action_std, edge_score, regime_mu, regime_log_var, combined

    @staticmethod
    def _tanh_log_prob(pre_tanh_action, action, mean, std):
        """Log probability with tanh squashing correction."""
        gaussian_log_prob = -0.5 * (
            ((pre_tanh_action - mean) / (std + 1e-8)) ** 2
            + 2.0 * torch.log(std + 1e-8)
            + math.log(2.0 * math.pi)
        )
        gaussian_log_prob = gaussian_log_prob.sum(dim=-1)
        log_det_jacobian = torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        return gaussian_log_prob - log_det_jacobian

    @torch.no_grad()
    def get_action(self, market_state, position_info, deterministic=False,
                   headlines=None, text_embeddings=None):
        """Sample action from AEGIS policy.

        Args:
            market_state: numpy array or tensor
            position_info: numpy array or tensor
            deterministic: if True, use mean action (no noise)
            headlines: optional list of headline strings (live mode with text)
            text_embeddings: optional pre-computed text embeddings tensor

        Returns:
            (action, log_prob, value, edge_score)

        The action is GATED by edge_score:
            final_action = edge_score × raw_action

        Compatible with existing NandiAgent interface:
        - 4th return value is edge_score (replaces uncertainty)
        - Can be used as uncertainty: low edge = high uncertainty
        """
        device = next(self.parameters()).device

        if market_state.ndim == 2:
            market_state = market_state[np.newaxis]
        if position_info.ndim == 1:
            position_info = position_info[np.newaxis]

        ms_t = torch.tensor(market_state, dtype=torch.float32, device=device)
        pi_t = torch.tensor(position_info, dtype=torch.float32, device=device)

        if deterministic:
            self.eval()
        else:
            self.train()

        (pre_tanh_mean, std, edge_score,
         regime_mu, regime_log_var, combined) = self.get_policy_and_edge(
            ms_t, pi_t, headlines=headlines, text_embeddings=text_embeddings
        )

        if deterministic:
            pre_tanh_action = pre_tanh_mean
        else:
            pre_tanh_action = pre_tanh_mean + torch.randn_like(pre_tanh_mean) * std

        raw_action = torch.tanh(pre_tanh_action)
        log_prob = self._tanh_log_prob(
            pre_tanh_action, raw_action, pre_tanh_mean, std
        )

        # ── THE KEY INNOVATION: Edge-gated action ──
        # Position size is proportional to model's confidence
        edge = edge_score.squeeze(-1)
        final_action = raw_action.squeeze(-1) * edge

        return (
            final_action.cpu().numpy().flatten()[0],
            log_prob.cpu().numpy().flatten()[0],
            0.0,  # placeholder for CVaR value
            edge.cpu().numpy().flatten()[0],  # edge_score as "uncertainty" slot
        )

    def get_raw_action(self, market_state, position_info):
        """Get raw (un-gated) action for training critic targets.

        Returns:
            raw_action: (batch, 1) — tanh-squashed
            log_prob: (batch,) — log probability
            pre_tanh_action: (batch, 1) — before squashing
        """
        pre_tanh_mean, std, _, _, _, _ = self.get_policy_and_edge(
            market_state, position_info
        )
        noise = torch.randn_like(pre_tanh_mean) * std
        pre_tanh_action = pre_tanh_mean + noise
        raw_action = torch.tanh(pre_tanh_action)
        log_prob = self._tanh_log_prob(
            pre_tanh_action, raw_action, pre_tanh_mean, std
        )
        return raw_action, log_prob, pre_tanh_action

    def evaluate_actions(self, market_states, position_infos, actions):
        """Evaluate actions for compatibility with PPO-style callers.

        Returns (log_probs, values, entropy) — values are zeros (use critic).
        """
        pre_tanh_mean, std, _ = self(market_states, position_infos)
        pre_tanh_action = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        log_probs = self._tanh_log_prob(pre_tanh_action, actions, pre_tanh_mean, std)

        gaussian_entropy = 0.5 * (1.0 + torch.log(2.0 * math.pi * std ** 2 + 1e-8))
        entropy = gaussian_entropy.sum(dim=-1)

        values = torch.zeros(market_states.shape[0], device=market_states.device)
        return log_probs, values, entropy

    def save_agent(self, path=None):
        path = path or os.path.join(MODEL_DIR, "aegis_agent.pt")
        if not path.endswith(".pt"):
            path = path + ".pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save without FinBERT weights (they're loaded from HuggingFace)
        # Only save our trainable layers: projection, attention, keyword encoder
        state = {}
        for k, v in self.state_dict().items():
            if "text_encoder._finbert" not in k:
                state[k] = v
        torch.save(state, path)

    def load_agent(self, path=None):
        path = path or os.path.join(MODEL_DIR, "aegis_agent.pt")
        if not path.endswith(".pt"):
            path = path + ".pt"
        if os.path.exists(path):
            saved_state = torch.load(path, map_location="cpu", weights_only=True)
            # Load with strict=False to handle missing FinBERT weights
            # (FinBERT is loaded separately via text_encoder.load_finbert())
            self.load_state_dict(saved_state, strict=False)
            return True
        return False
