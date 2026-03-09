"""Graph Neural Network for cross-pair relationship modeling.

Models currency pairs as a graph where:
- Nodes = pairs (each with MSFAN/TFT encoding)
- Edges = correlation strength between pairs
- Message passing captures cross-pair dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Single graph attention layer (GAT-style)."""

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_edge = nn.Linear(1, n_heads)  # edge weight (correlation) to attention bias

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, adjacency):
        """
        Args:
            node_features: (batch, n_nodes, d_model) — per-pair encodings
            adjacency: (batch, n_nodes, n_nodes) — correlation matrix
        """
        B, N, D = node_features.shape

        Q = self.W_q(node_features).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(node_features).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(node_features).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Add edge bias from correlation matrix
        edge_bias = self.W_edge(adjacency.unsqueeze(-1))  # (B, N, N, n_heads)
        edge_bias = edge_bias.permute(0, 3, 1, 2)  # (B, n_heads, N, N)
        scores = scores + edge_bias

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, n_heads, N, d_head)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        return self.norm(node_features + self.dropout(out))


class CurrencyGraphNet(nn.Module):
    """GNN for cross-pair relationship modeling."""

    def __init__(self, n_pairs=8, d_model=128, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_pairs = n_pairs
        self.layers = nn.ModuleList([
            GraphAttentionLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, node_embeddings, adjacency_matrix):
        """
        Args:
            node_embeddings: (batch, n_pairs, d_model) from per-pair encoders
            adjacency_matrix: (batch, n_pairs, n_pairs) correlation matrix
        Returns:
            (batch, n_pairs, d_model) context-enriched pair embeddings
        """
        x = node_embeddings
        for layer in self.layers:
            x = layer(x, adjacency_matrix)
        return self.output_norm(x)
