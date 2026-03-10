import torch
import torch.nn as nn


class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, transforms, context_dim, base_dist=None, data_dim=2, device=torch.device("cpu")):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.context_dim = context_dim
        self.data_dim = data_dim
        self.device = device

        self.base_dist = base_dist or torch.distributions.MultivariateNormal(
            loc=torch.zeros(data_dim),
            covariance_matrix=torch.eye(data_dim)
        )

        self.check_log_det()


        for t in self.transforms:
            t.to(self.device)

    def real_to_latent(self, x, context):  # x → z = f(x, context)
        x = x.to(self.device)
        context = context.to(self.device)

        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for transform in self.transforms:
            z, log_det = transform(z, context)
            log_det_total -= log_det.to(device=x.device)
        return z, log_det_total

    def latent_to_real(self, z, context):  # z → x = f⁻¹(z, context)
        z = z.to(self.device)
        context = context.to(self.device)

        x = z
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        for transform in reversed(self.transforms):
            x = transform.inverse(x, context)
            log_det = transform.log_abs_det_jacobian(x, context)
            log_det_total += log_det.to(self.device)
        return x, log_det_total

    # aliases for consistent with other flows and codebase...
    forward = real_to_latent
    inverse = latent_to_real

    def log_prob(self, x, context):
        x = x.to(self.device)
        context = context.to(self.device)

        z, log_det = self.forward(x, context)  # push given sample to latent space
        base_log_prob = self.base_dist.log_prob(z)  # evaluate latent base log prob

        if isinstance(self.base_dist, torch.distributions.multivariate_normal.MultivariateNormal):
            pass
        elif isinstance(self.base_dist, torch.distributions.uniform.Uniform):
            base_log_prob = base_log_prob.sum(dim=1)

        return base_log_prob - log_det

    def sample(self, num_samples, context):
        # context should be shape (num_samples, context_dim)
        z = self.base_dist.sample((num_samples,))
        x, _ = self.inverse(z, context)
        return x.to(self.device)

    def check_log_det(self):
        x = torch.randn(10, self.data_dim, device=self.device)
        context = torch.randn(10, self.context_dim, device=self.device)
        z, log_det_1 = self.real_to_latent(x, context)
        x_recon, log_det_2 = self.latent_to_real(z, context)

        # If signs are handled correctly:
        assert torch.allclose(x, x_recon, atol=1e-5)
        assert torch.allclose(log_det_1, -log_det_2, atol=1e-5)
