import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticLSTM(nn.Module):
    def __init__(self, obs_dim = 148, hidden_dim = 128, action_dim = 3, lstm_hidden = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.lstm = nn.LSTMCell(hidden_dim, lstm_hidden) # Recurrent layer

        self.actor_head = nn.Linear(lstm_hidden, action_dim)
        self.critic_head = nn.Linear(lstm_hidden, 1)

        self.lstm_hidden_size = lstm_hidden
        
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.orthogonal_(param, gain = 1.0)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor):
        enc = self.encoder(x)
        hx, cx = self.lstm(enc, (hx, cx))
        logits = self.actor_head(hx)
        value = self.critic_head(hx).squeeze(-1)
        return logits, value, hx, cx

    def init_hidden(self, batch_size=1, device='cpu'):
        z = torch.zeros(batch_size, self.lstm_hidden_size, device=device)
        return z, z.clone()