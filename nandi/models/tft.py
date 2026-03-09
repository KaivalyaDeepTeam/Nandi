"""Temporal Fusion Transformer (TFT) encoder for RL state encoding.

Bryan Lim et al. (2019). Simplified for RL: no multi-horizon forecasting,
just encodes a (batch, seq_len, n_features) window into a (batch, d_model) vector.

Key advantage: Variable Selection Network learns which features matter,
making the model interpretable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.fc(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    def __init__(self, d_model, d_hidden=None, dropout=0.1):
        super().__init__()
        d_hidden = d_hidden or d_model
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.gate = GatedLinearUnit(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        h = self.gate(h)
        return self.norm(x + h)


class VariableSelectionNetwork(nn.Module):
    """Learns importance weights for each input feature."""
    def __init__(self, n_features, d_model, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        # Per-feature transformation
        self.feature_transforms = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])
        # Selection weights
        self.selection = nn.Sequential(
            nn.Linear(n_features * d_model, n_features),
            nn.Softmax(dim=-1),
        )
        self.grn = GatedResidualNetwork(d_model, dropout=dropout)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        batch, seq_len, _ = x.shape

        # Transform each feature independently
        transformed = []
        for i in range(self.n_features):
            feat = x[:, :, i:i+1]  # (batch, seq_len, 1)
            transformed.append(self.feature_transforms[i](feat))  # (batch, seq_len, d_model)

        # Stack: (batch, seq_len, n_features, d_model)
        stacked = torch.stack(transformed, dim=2)

        # Compute selection weights
        flat = stacked.reshape(batch * seq_len, self.n_features * self.d_model)
        weights = self.selection(flat)  # (batch*seq_len, n_features)
        weights = weights.reshape(batch, seq_len, self.n_features, 1)

        # Weighted sum
        selected = (stacked * weights).sum(dim=2)  # (batch, seq_len, d_model)
        return self.grn(selected), weights.squeeze(-1).mean(dim=1)  # also return feature importance


class TemporalFusionTransformer(nn.Module):
    """Simplified TFT encoder for RL state encoding."""

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        n_features = config.get("n_features", 45)
        d_model = config.get("d_model", 128)
        n_heads = config.get("n_heads", 4)
        dropout = config.get("dropout", 0.15)

        self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True, dropout=0)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.grn_post_attn = GatedResidualNetwork(d_model, dropout=dropout)
        self.output_dense = nn.Linear(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)

        self._feature_importance = None

    def forward(self, x):
        # x: (batch, seq_len, d_model) — already projected by feature_proj
        # But TFT needs raw features for VSN, so we handle both cases
        selected, importance = self.vsn(x) if x.shape[-1] != self.vsn.d_model else (x, None)
        self._feature_importance = importance

        lstm_out, _ = self.lstm(selected)
        attended, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        gated = self.grn_post_attn(attended + lstm_out)

        # Take last timestep
        output = gated[:, -1, :]
        return self.output_norm(F.gelu(self.output_dense(output)))

    def get_feature_importance(self):
        return self._feature_importance
