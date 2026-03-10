import torch

from .transforms import ConditionalFlip, ConditionalAffineCoupling
from .conditional_nflow import ConditionalNormalizingFlow


class IdentityFlow:
    base_dist = None
    @staticmethod
    def latent_to_real(action, context):
        return action, None

    @staticmethod
    def real_to_latent(action, context):
        return action, None


def make_flow(
        action_dim,
        state_dim,
        num_layers=10,
        hidden_dim=128,
        base_dist_scale=0.2,  # 1.0 for maze
        device=torch.device("cpu")
):
    transforms = []
    for _ in range(num_layers):
        transforms.append(ConditionalAffineCoupling(
            dim=action_dim, context_dim=state_dim, hidden_dim=hidden_dim, device=device))
        transforms.append(ConditionalFlip())

    base_dist = torch.distributions.MultivariateNormal(
        loc=torch.zeros(action_dim, device=device),
        covariance_matrix=torch.eye(
            action_dim, device=device) * base_dist_scale,
    )

    flow = ConditionalNormalizingFlow(
        transforms=transforms, data_dim=action_dim, context_dim=state_dim, base_dist=base_dist, device=device)
    return flow
