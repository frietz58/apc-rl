import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dims=[256, 256]):
        super().__init__()

        # Create the MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        for layer in self.hidden_layers:
            x = F.tanh(layer(x))
        x = self.output_layer(x)
        return x
    
    
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256], action_space=None, extra_logits=0):
        super().__init__()

        # MLP layers
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            prev_dim = h

        self.hidden_layers = nn.ModuleList(layers)
        self.fc_mean = nn.Linear(prev_dim, output_dim)
        self.fc_logstd = nn.Linear(prev_dim, output_dim)

        # Action scaling (optional)
        if action_space is not None:
            scale = np.concatenate([
                (action_space.high - action_space.low) / 2.0,
                np.ones(extra_logits)
            ])
            bias = np.concatenate([
                (action_space.high + action_space.low) / 2.0,
                np.zeros(extra_logits)
            ])

            self.register_buffer("action_scale", torch.tensor(scale, dtype=torch.float32))
            self.register_buffer("action_bias", torch.tensor(bias, dtype=torch.float32))
        else:
            self.register_buffer("action_scale", torch.ones(output_dim))
            self.register_buffer("action_bias", torch.zeros(output_dim))

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.tanh(layer(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    @staticmethod
    def atanh(x):
        return 0.5 * torch.log((1 + x) / (1 - x + 1e-6) + 1e-6)

    def compute_log_prob(self, obs, action_unsquashed):
        y_t = (action_unsquashed - self.action_bias) / self.action_scale
        y_t = torch.clamp(y_t, -1 + 1e-6, 1 - 1e-6)
        x_t = self.atanh(y_t)

        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return log_prob
    

class CategoricalActor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256]):
        super().__init__()

        # MLP layers
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            prev_dim = h

        self.hidden_layers = nn.ModuleList(layers)
        self.fc_logits = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.tanh(layer(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        logits = self(x)
        policy_dist = torch.distributions.Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs

