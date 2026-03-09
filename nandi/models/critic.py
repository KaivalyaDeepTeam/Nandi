"""Twin critic networks for SAC and TD3 algorithms."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    """Q-function: state_encoding + position_info + action -> Q-value."""

    def __init__(self, state_dim=128, pos_dim=32, action_dim=1, hidden_dim=128):
        super().__init__()
        input_dim = state_dim + pos_dim + action_dim
        self.q_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state_encoding, pos_embedding, action):
        """
        Args:
            state_encoding: (batch, state_dim) from MSFAN encoder
            pos_embedding: (batch, pos_dim) from position embed
            action: (batch, 1) continuous action
        """
        x = torch.cat([state_encoding, pos_embedding, action], dim=-1)
        return self.q_net(x)


class TwinCritic(nn.Module):
    """Twin Q-networks for min-Q trick (SAC/TD3)."""

    def __init__(self, state_dim=128, pos_dim=32, action_dim=1, hidden_dim=128):
        super().__init__()
        self.q1 = CriticNetwork(state_dim, pos_dim, action_dim, hidden_dim)
        self.q2 = CriticNetwork(state_dim, pos_dim, action_dim, hidden_dim)

    def forward(self, state_encoding, pos_embedding, action):
        return self.q1(state_encoding, pos_embedding, action), \
               self.q2(state_encoding, pos_embedding, action)

    def q1_forward(self, state_encoding, pos_embedding, action):
        return self.q1(state_encoding, pos_embedding, action)
