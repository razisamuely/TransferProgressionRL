import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

class Actor(nn.Module):
    def __init__(self, n_state, max_n_action, n_action, hidden1, hidden2, action_space=None):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(n_state, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, max_n_action)
        )
        
        self.mu_layer = nn.Linear(1, 1)
        self.sigma_layer = nn.Linear(1, 1)
        
        self.max_n_action = max_n_action
        self.n_action = n_action
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        self.distribution = torch.distributions.Normal

    def _apply_gradient_mask(self, x):
        gradient_mask = torch.zeros_like(x)
        mask_indices = slice(None, self.n_action)
        
        if x.dim() == 1:
            gradient_mask[mask_indices] = 1
        else:
            gradient_mask[:, mask_indices] = 1
            
        return x * gradient_mask.detach()

    def _get_discrete_distribution(self, x):
        action_probs = F.softmax(x, dim=-1)
        valid_indices = slice(None, self.n_action)
        
        if x.dim() == 1:
            valid_probs = action_probs[valid_indices]
        else:
            valid_probs = action_probs[:, valid_indices]
            
        valid_probs = valid_probs / valid_probs.sum(dim=-1, keepdim=True)
        return torch.distributions.Categorical(probs=valid_probs)

    def _get_continuous_distribution(self, x):
        if x.dim() == 1:
            mu = self.mu_layer(x[:1])
            sigma = self.sigma_layer(x[1:2])
        else:
            mu = self.mu_layer(x[:,:1])
            sigma = self.sigma_layer(x[:,1:2])

        mu_final = 2 * torch.tanh(mu)
        sigma_final = F.softplus(sigma) + 1e-5
        
        return self.distribution(mu_final, sigma_final)

    def forward(self, state):
        x = self.base(state)
        
        if self.max_n_action > self.n_action:
            x = self._apply_gradient_mask(x)
            dist = (self._get_discrete_distribution(x) if self.is_discrete 
                   else self._get_continuous_distribution(x))
        else:
            dist = self._get_discrete_distribution(x)
                
        return dist

class Critic(nn.Module):
    def __init__(self, n_state, hidden1, hidden2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def forward(self, state):
        return self.net(state)