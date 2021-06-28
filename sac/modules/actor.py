import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_log_std=-20, max_log_std=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        mu = self.mu_head(x)
        log_std_head = self.log_std_head(x)
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        
        return mu, log_std_head