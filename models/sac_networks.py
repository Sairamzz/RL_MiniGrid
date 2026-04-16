import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscretePolicyNet(nn.Module):
    def __init__(self, obs_dim=148, hidden_dim=256, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits


class DiscreteQNet(nn.Module):
    def __init__(self, obs_dim=148, hidden_dim=256, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        q_values = self.net(x)
        return q_values