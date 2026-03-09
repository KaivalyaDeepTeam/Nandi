"""
Multi-Scale Fractal Attention Network (MSFAN) encoder.

Processes market data at multiple temporal scales using dilated causal
convolutions, then uses cross-scale multi-head attention to capture
inter-scale relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nandi.config import ENCODER_CONFIG


class CausalConvBlock(nn.Module):
    """Dilated causal convolution block with residual connection."""

    def __init__(self, d_model, kernel_size, dilation_rate, dropout=0.1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation_rate
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation_rate)
        self.conv2 = nn.Conv1d(d_model, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (batch, time, d_model) -> (batch, time, d_model)"""
        residual = x
        h = x.transpose(1, 2)
        h = F.pad(h, (self.pad, 0))
        h = self.conv1(h)
        h = h.transpose(1, 2)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = h.transpose(1, 2)
        h = self.conv2(h)
        h = h.transpose(1, 2)
        h = self.norm2(h)
        return F.gelu(h + residual)


class MultiScaleEncoder(nn.Module):
    """MSFAN: processes market data at 3 temporal scales with cross-scale attention."""

    def __init__(self, config=None):
        super().__init__()
        config = config or ENCODER_CONFIG
        d = config["d_model"]

        self.scale_blocks = nn.ModuleList()
        for i in range(config["n_scales"]):
            block = CausalConvBlock(
                d, config["kernel_sizes"][i],
                config["dilations"][i],
                config["dropout"],
            )
            self.scale_blocks.append(block)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=config["n_heads"],
            dropout=config["dropout"], batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d)
        self.output_dense = nn.Linear(d, d)
        self.output_norm = nn.LayerNorm(d)

    def forward(self, x):
        """x: (batch, lookback, d_model) -> (batch, d_model)"""
        scale_outputs = []
        for block in self.scale_blocks:
            h = block(x)
            scale_outputs.append(h[:, -1:, :])

        multi_scale = torch.cat(scale_outputs, dim=1)
        attended, _ = self.cross_attn(multi_scale, multi_scale, multi_scale)
        attended = self.cross_norm(attended + multi_scale)
        pooled = attended.mean(dim=1)
        return self.output_norm(F.gelu(self.output_dense(pooled)))
