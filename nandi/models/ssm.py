"""Selective State Space Model (Mamba-style) encoder.

S4/Mamba-inspired architecture for efficient long-range temporal processing.
O(n) complexity instead of O(n^2) for transformers.

Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectiveSSMBlock(nn.Module):
    """Single Mamba-style SSM block with input-dependent selection."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, d_conv,
            padding=d_conv - 1, groups=d_inner
        )

        # SSM parameters (input-dependent via projection)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1)  # B, C, delta

        # Learnable state matrix A (log-space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # Skip connection
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        residual = x
        batch, seq_len, _ = x.shape

        # Split into main path and gate
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_main, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Local convolution
        x_conv = x_main.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # causal trim
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # SSM computation (selective scan)
        ssm_params = self.x_proj(x_conv)  # (B, L, 2*d_state + 1)
        B_param = ssm_params[:, :, :self.d_state]
        C_param = ssm_params[:, :, self.d_state:2*self.d_state]
        delta = F.softplus(ssm_params[:, :, -1:])  # (B, L, 1)

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Simplified selective scan (sequential for correctness)
        d_inner = x_conv.shape[-1]
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            dt = delta[:, t, :]  # (B, 1)
            B_t = B_param[:, t, :]  # (B, d_state)
            C_t = C_param[:, t, :]  # (B, d_state)
            x_t = x_conv[:, t, :]  # (B, d_inner)

            # State update: h = exp(A * dt) * h + dt * B * x
            dA = torch.exp(A.unsqueeze(0) * dt.unsqueeze(-1))  # (B, d_inner, d_state)
            dB = dt.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, 1, d_state) -> broadcast
            h = dA * h + dB * x_t.unsqueeze(-1)  # (B, d_inner, d_state)

            # Output: y = C * h + D * x
            y = (h * C_t.unsqueeze(1)).sum(-1) + self.D * x_t  # (B, d_inner)
            outputs.append(y)

        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)

        # Gate and project
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)

        return self.norm(y + residual)


class SelectiveSSM(nn.Module):
    """Multi-layer SSM encoder for RL state encoding."""

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        d_model = config.get("d_model", 128)
        n_layers = config.get("n_layers", 2)
        d_state = config.get("d_state", 16)
        dropout = config.get("dropout", 0.15)

        self.layers = nn.ModuleList([
            SelectiveSSMBlock(d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.output_dense = nn.Linear(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model) — projected features
        Returns:
            (batch, d_model) — pooled encoding
        """
        for layer in self.layers:
            x = layer(x)

        # Take last timestep
        output = x[:, -1, :]
        return self.output_norm(F.gelu(self.output_dense(output)))
