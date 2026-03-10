import torch
import torch.nn as nn

class ConditionalAffineCoupling(nn.Module):
    def __init__(self, dim, context_dim, hidden_dim=64, device=torch.device("cpu")):
        super().__init__()
        self.dim = dim
        self.split_idx = dim // 2

        input_dim = self.split_idx + context_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * (dim - self.split_idx))  # scale + shift
        )
        self.net.to(device)

        # optional...
        self.rescale = nn.utils.weight_norm(Rescale(dim - self.split_idx)).to(device)

    def forward(self, x, context):
        x1, x2 = x[:, :self.split_idx], x[:, self.split_idx:]
        # concatenate x1 and context
        h = self.net(torch.cat([x1, context], dim=1))
        s, t = h.chunk(2, dim=1)
        s = torch.tanh(s)
        s = self.rescale(s)
        z1 = x1
        z2 = x2 * torch.exp(s) + t
        z = torch.cat([z1, z2], dim=1)
        log_det = s.sum(dim=1)
        return z, log_det

    def inverse(self, z, context):
        z1, z2 = z[:, :self.split_idx], z[:, self.split_idx:]
        h = self.net(torch.cat([z1, context], dim=1))
        s, t = h.chunk(2, dim=1)
        s = torch.tanh(s)
        s = self.rescale(s)
        x1 = z1
        x2 = (z2 - t) * torch.exp(-s)
        x = torch.cat([x1, x2], dim=1)
        return x

    def log_abs_det_jacobian(self, x, context):
        x1 = x[:, :self.split_idx]
        h = self.net(torch.cat([x1, context], dim=1))
        s, _ = h.chunk(2, dim=1)
        s = torch.tanh(s)
        s = self.rescale(s)
        return s.sum(dim=1)


class Rescale(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))  # now 1D
    def forward(self, x):
        return self.weight * x


class ConditionalFlip(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, context):
        return x.flip(dims=[1]), self.log_abs_det_jacobian(x, context)

    def inverse(self, z, context):
        return z.flip(dims=[1])

    def log_abs_det_jacobian(self, x, context):
        return torch.zeros(x.shape[0])


class TanhTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, context=None):
        z = torch.tanh(x)
        log_det = self.log_abs_det_jacobian(x, context)
        return z, log_det

    def inverse(self, z, context=None):
        # Clamp input to avoid NaNs from arctanh at |z| = 1
        eps = 1e-6
        z = torch.clamp(z, -1 + eps, 1 - eps)
        x = 0.5 * torch.log1p(2 * z / (1 - z))
        return x

    def log_abs_det_jacobian(self, x, context=None):
        epsilon = 1e-6
        tanh_x = torch.tanh(x)
        log_det = torch.log(1 - tanh_x ** 2 + epsilon)
        return log_det.sum(dim=1)

