"""
Nandi PPO Actor-Critic Agent with MSFAN encoder.

Outputs continuous position [-1, 1] with learned exploration noise,
state value estimate, and epistemic uncertainty for trade gating.
"""

import os

import numpy as np
import torch
import torch.nn as nn

from nandi.config import ENCODER_CONFIG, MODEL_DIR
from nandi.models.msfan import MultiScaleEncoder
from nandi.models.tft import TemporalFusionTransformer
from nandi.models.ssm import SelectiveSSM


class NandiAgent(nn.Module):
    """PPO Actor-Critic agent with pluggable encoder (MSFAN, TFT, or SSM)."""

    ENCODERS = {
        "msfan": MultiScaleEncoder,
        "tft": TemporalFusionTransformer,
        "ssm": SelectiveSSM,
    }

    def __init__(self, n_features, encoder_config=None, encoder_type="msfan"):
        super().__init__()
        config = encoder_config or ENCODER_CONFIG
        d = config["d_model"]

        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, d), nn.GELU(), nn.LayerNorm(d),
        )
        EncoderClass = self.ENCODERS.get(encoder_type, MultiScaleEncoder)
        self.encoder = EncoderClass(config)
        self.position_embed = nn.Sequential(
            nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(d + 32, d), nn.GELU(), nn.LayerNorm(d),
            nn.Linear(d, d // 2), nn.GELU(),
        )
        self.actor_mean = nn.Linear(d // 2, 1)
        # State-dependent log_std instead of fixed parameter
        self.log_std_head = nn.Sequential(
            nn.Linear(d // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.critic_head = nn.Linear(d // 2, 1)

    def forward(self, market_state, position_info):
        """
        Args:
            market_state: (batch, lookback, n_features)
            position_info: (batch, 4)
        Returns:
            action_mean (pre-tanh), action_std, value
        """
        projected = self.feature_proj(market_state)
        encoded = self.encoder(projected)
        pos_emb = self.position_embed(position_info)
        combined = torch.cat([encoded, pos_emb], dim=-1)
        h = self.trunk(combined)

        # Return pre-tanh mean for proper squashing correction
        pre_tanh_mean = self.actor_mean(h)
        log_std = torch.clamp(self.log_std_head(h), -2.0, 0.5)
        action_std = torch.exp(log_std)
        value = self.critic_head(h)
        return pre_tanh_mean, action_std, value

    @staticmethod
    def _tanh_log_prob(pre_tanh_action, action, mean, std):
        """Compute log_prob with tanh squashing correction (log-det-Jacobian)."""
        # Gaussian log prob in pre-tanh space
        gaussian_log_prob = -0.5 * (
            ((pre_tanh_action - mean) / (std + 1e-8)) ** 2
            + 2.0 * torch.log(std + 1e-8)
            + np.log(2.0 * np.pi)
        )
        gaussian_log_prob = gaussian_log_prob.sum(dim=-1)
        # tanh squashing correction: log|det(d_tanh/d_pre_tanh)| = log(1 - tanh^2)
        log_det_jacobian = torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        return gaussian_log_prob - log_det_jacobian

    @torch.no_grad()
    def get_action(self, market_state, position_info, deterministic=False):
        """Sample action from policy. Returns: (action, log_prob, value, uncertainty)"""
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

        pre_tanh_mean, std, value = self(ms_t, pi_t)

        if deterministic:
            pre_tanh_action = pre_tanh_mean
        else:
            pre_tanh_action = pre_tanh_mean + torch.randn_like(pre_tanh_mean) * std

        action = torch.tanh(pre_tanh_action)
        log_prob = self._tanh_log_prob(pre_tanh_action, action, pre_tanh_mean, std)

        unc = 0.0
        if deterministic:
            self.train()
            preds = []
            for _ in range(5):
                m, _, _ = self(ms_t, pi_t)
                preds.append(torch.tanh(m).cpu().numpy())
            unc = float(np.std(preds))
            self.eval()

        return (
            action.cpu().numpy().flatten()[0],
            log_prob.cpu().numpy().flatten()[0],
            value.cpu().numpy().flatten()[0],
            unc,
        )

    def evaluate_actions(self, market_states, position_infos, actions):
        """Evaluate actions for PPO update. Returns: (log_probs, values, entropy)"""
        pre_tanh_mean, std, values = self(market_states, position_infos)

        # Invert tanh to get pre_tanh_action for log_prob computation
        pre_tanh_action = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        log_probs = self._tanh_log_prob(pre_tanh_action, actions, pre_tanh_mean, std)

        # Entropy of squashed Gaussian (approximate: Gaussian entropy + E[log|det J|])
        gaussian_entropy = 0.5 * (1.0 + torch.log(2.0 * np.pi * std ** 2 + 1e-8))
        gaussian_entropy = gaussian_entropy.sum(dim=-1)
        # Approximate tanh correction to entropy using current mean
        tanh_mean = torch.tanh(pre_tanh_mean)
        log_det_correction = torch.log(1 - tanh_mean.pow(2) + 1e-6).sum(dim=-1)
        entropy = gaussian_entropy + log_det_correction  # subtract because of sign

        return log_probs, values.squeeze(-1), entropy

    def get_uncertainty(self, market_state, position_info, n_samples=10):
        """Monte Carlo dropout uncertainty estimation."""
        device = next(self.parameters()).device

        if market_state.ndim == 2:
            market_state = market_state[np.newaxis]
        if position_info.ndim == 1:
            position_info = position_info[np.newaxis]

        ms_t = torch.tensor(market_state, dtype=torch.float32, device=device)
        pi_t = torch.tensor(position_info, dtype=torch.float32, device=device)

        self.train()
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                mean, _, _ = self(ms_t, pi_t)
                predictions.append(torch.tanh(mean).cpu().numpy())

        return float(np.std(predictions))

    def save_agent(self, path=None):
        path = path or os.path.join(MODEL_DIR, "nandi_agent.pt")
        if not path.endswith(".pt"):
            path = path + ".pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_agent(self, path=None):
        path = path or os.path.join(MODEL_DIR, "nandi_agent.pt")
        if not path.endswith(".pt"):
            path = path + ".pt"
        if os.path.exists(path):
            self.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )
            return True
        return False
