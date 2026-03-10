import os
import math
import gymnasium as gym
import numpy as np
from pyparsing import ABC, abstractmethod
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from .networks import Actor, SoftQNetwork, CategoricalActor
from .buffer import ReplayBuffer


class SAC(ABC):
    def __init__(
        self,
        envs,
        device,
        autotune_alpha=False,
        alpha=0.01,
        q_lr=1e-3,
        policy_lr=3e-4,
        buffer_size=int(1e6),
        batch_size=256,
        gamma=0.99,
        target_tau=0.005,
    ):
        self.envs = envs
        self.device = device

        # Hyperparameters
        self.autotune_alpha = autotune_alpha
        self.alpha = alpha
        self.q_lr = q_lr
        self.policy_lr = policy_lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_tau = target_tau

        # Build networks specific to action space
        self._build_networks()

        # Target nets start as copies
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # Optimizers
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.q_lr
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.policy_lr)

        # Entropy temperature
        if self.autotune_alpha:
            self.target_entropy = self._default_target_entropy()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.q_lr)
        else:
            self.alpha = alpha
            self.log_alpha = None
            self.a_optimizer = None
            self.target_entropy = None

        # Replay buffer
        self.envs.single_observation_space.dtype = np.float32
        self.rb = self._build_replay_buffer()

    # ---------- hooks to specialize in subclasses ----------

    @abstractmethod
    def _build_networks(self):
        pass

    @abstractmethod
    def _build_replay_buffer(self):
        pass

    @abstractmethod
    def _default_target_entropy(self):
        pass

    @abstractmethod
    def get_action(self, obs):
        """Return actions as numpy (or tensor) matching env vectorization."""
        pass

    @abstractmethod
    def train_step(self, data):
        """Run one training step; return metrics tuple for logging."""
        pass

    # ---------- shared utilities ----------
    def buffer_add(self, obs, next_obs, actions, rewards, terminations, infos):
        self.rb.add(obs, next_obs, actions, rewards, terminations, infos)

    def buffer_add_one(self, obs, next_obs, actions, rewards, terminations, infos):
        self.rb.add_one(obs, next_obs, actions, rewards, terminations, infos)

    def train(self):
        data = self.rb.sample(self.batch_size)
        return self.train_step(data)

    def _soft_update_(self, online: nn.Module, target: nn.Module, tau: float):
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.data.lerp_(p.data, tau)

    def _update_targets_(self):
        self._soft_update_(self.qf1, self.qf1_target, self.target_tau)
        self._soft_update_(self.qf2, self.qf2_target, self.target_tau)

    def save_models(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{save_dir}/actor.pt")
        torch.save(self.qf1.state_dict(), f"{save_dir}/qf1.pt")
        torch.save(self.qf2.state_dict(), f"{save_dir}/qf2.pt")
        torch.save(self.qf1_target.state_dict(), f"{save_dir}/qf1_target.pt")
        torch.save(self.qf2_target.state_dict(), f"{save_dir}/qf2_target.pt")
        if self.autotune_alpha and self.log_alpha is not None:
            torch.save(self.log_alpha, f"{save_dir}/log_alpha.pt")


class SACAgentContinuous(SAC):
    def __init__(
        self,
        envs,
        device,
        *,
        autotune_alpha=False,
        alpha=0.01,
        q_lr=1e-3,
        policy_lr=3e-4,
        buffer_size=int(1e6),
        batch_size=256,
        gamma=0.99,
        target_tau=0.005,
        actor_hidden=[512, 256],
        q_hidden=[512, 256],
        il_dataset_sample_fun=None,
        il_coef=None,
        il_use_q_filter=False,
    ):
        self._actor_hidden = actor_hidden
        self._q_hidden = q_hidden
        super().__init__(
            envs, device, autotune_alpha, alpha, q_lr, policy_lr,
            buffer_size, batch_size, gamma, target_tau
        )

        self.il_dataset_sample_fun = il_dataset_sample_fun  # Optional, for IL regularization
        self.il_coef = il_coef  # Coefficient for IL regularization, if applicable
        self.il_use_q_filter = il_use_q_filter

    # required overrides
    def _build_networks(self):
        obs_dim = self.envs.single_observation_space.shape[0]
        act_dim = self.envs.single_action_space.shape[0]

        self.actor = Actor(
            input_dim=obs_dim,
            output_dim=act_dim,
            action_space=self.envs.single_action_space,
            hidden_dims=self._actor_hidden
        ).to(self.device)

        self.qf1 = SoftQNetwork(
            input_dim=obs_dim + act_dim,
            hidden_dims=self._q_hidden
        ).to(self.device)
        self.qf2 = SoftQNetwork(
            input_dim=obs_dim + act_dim,
            hidden_dims=self._q_hidden
        ).to(self.device)
        self.qf1_target = SoftQNetwork(
            input_dim=obs_dim + act_dim,
            hidden_dims=self._q_hidden
        ).to(self.device)
        self.qf2_target = SoftQNetwork(
            input_dim=obs_dim + act_dim,
            hidden_dims=self._q_hidden
        ).to(self.device)

    def _build_replay_buffer(self):
        return ReplayBuffer(
            self.buffer_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            self.device,
            n_envs=self.envs.num_envs,
            handle_timeout_termination=False,
        )

    def _default_target_entropy(self):
        # standard SAC heuristic: -|A|
        act_dim = int(np.prod(self.envs.single_action_space.shape))
        return -float(act_dim)
    
    def _log_prior_continuous(self, states, actions):
        return torch.zeros(actions.shape[0], device=actions.device)

    def get_action(self, obs):
        actions, _, _ = self.actor.get_action(torch.as_tensor(obs, device=self.device))
        return actions

    @staticmethod
    def atanh(x):
        return 0.5 * torch.log((1 + x) / (1 - x + 1e-6) + 1e-6)

    def _log_prob_unsquashed(self, obs, action_unsquashed):
        # 1. Normalize to [-1, 1] range (undo rescaling)
        y_t = (action_unsquashed - self.actor.action_bias) / self.actor.action_scale

        # Clip to avoid numerical issues outside the domain of atanh
        eps = 1e-6
        y_t = torch.clamp(y_t, -1 + eps, 1 - eps)

        # 2. Undo tanh
        x_t = self.atanh(y_t)

        # 3. Get policy parameters
        mean, log_std = self.actor(obs)
        std = log_std.exp()

        # 4. Evaluate log-prob under Normal
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(x_t)

        # 5. Adjust log-prob for tanh squashing and rescaling
        log_prob -= torch.log(self.actor.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return log_prob

    def train_step(self, data):
        # ----- Q update -----
        with torch.no_grad():
            next_a, next_logp, _ = self.actor.get_action(data.next_observations)
            logp_prior_next = self._log_prior_continuous(data.next_observations, next_a)  # [B]
            q1_t = self.qf1_target(data.next_observations, next_a)
            q2_t = self.qf2_target(data.next_observations, next_a)
            min_q_t = torch.min(q1_t, q2_t).view(-1)
            next_v = min_q_t - self.alpha * (next_logp.view(-1) - logp_prior_next)
            target = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * next_v

        q1 = self.qf1(data.observations, data.actions).view(-1)
        q2 = self.qf2(data.observations, data.actions).view(-1)
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ----- actor update -----
        pi, logp, _ = self.actor.get_action(data.observations)
        logp_prior = self._log_prior_continuous(data.observations, pi)  # [B]
        q1_pi = self.qf1(data.observations, pi)
        q2_pi = self.qf2(data.observations, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * (logp.view(-1) - logp_prior) - min_q_pi).mean()

        if self.il_dataset_sample_fun is not None:
            il_states, il_actions = self.il_dataset_sample_fun(data.observations.shape[0])

            if self.il_use_q_filter:
                with torch.no_grad():
                    q1_il = self.qf1(il_states.to(self.device), il_actions.to(self.device)).view(-1)
                    q2_il = self.qf2(il_states.to(self.device), il_actions.to(self.device)).view(-1)
                    min_q_il = torch.min(q1_il, q2_il)

                    action_il_states, _, _ = self.actor.get_action(il_states.to(self.device))
                    q1_il_pi = self.qf1(il_states.to(self.device), action_il_states).view(-1)
                    q2_il_pi = self.qf2(il_states.to(self.device), action_il_states).view(-1)
                    min_q_il_pi = torch.min(q1_il_pi, q2_il_pi)

                    # Filter: only use IL samples where Q under expert action > Q under current policy action
                    filter_mask = (min_q_il < min_q_il_pi).float()
            else:
                filter_mask = torch.zeros(il_states.shape[0], device=self.device)

            il_log_probs = self._log_prob_unsquashed(il_states.to(self.device), il_actions.to(self.device))

            il_coef = torch.ones_like(il_log_probs)
            il_coef -= filter_mask.unsqueeze(-1)  # zero out weight for IL (s,a) sample based on Q-filter (demo action has lower Q-value than sampled action)
            il_coef *= self.il_coef

            actor_loss -= (il_coef * il_log_probs).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------- temperature (optional) ----------
        if self.autotune_alpha:
            with torch.no_grad():
                _, logp_det, _ = self.actor.get_action(data.observations)
                logp_det = logp_det.view(-1)
                kl_term = logp_det

            alpha_loss = (-self.log_alpha.exp() * (kl_term + self.target_entropy)).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = None

        # ----- targets -----
        self._update_targets_()

        return (
            q1, q2, q1_loss, q2_loss, q_loss, actor_loss, self.alpha, alpha_loss
        )
        

class SACAgentCategorical(SAC):
    def __init__(
        self,
        envs,
        device,
        *,
        discrete_action_space,
        autotune_alpha=False,
        alpha=0.01,
        q_lr=1e-3,
        policy_lr=3e-4,
        buffer_size=int(1e6),
        batch_size=256,
        gamma=0.99,
        target_tau=0.005,
        target_entropy=None,
        actor_hidden=[512, 256],
        q_hidden=[512, 256],
    ):
        self.discrete_action_space = discrete_action_space
        self._actor_hidden = actor_hidden
        self._q_hidden = q_hidden
        super().__init__(
            envs, device, autotune_alpha, alpha, q_lr, policy_lr,
            buffer_size, batch_size, gamma, target_tau
        )

    # required overrides
    def _build_networks(self):
        obs_dim = self.envs.single_observation_space.shape[0]
        nA = self.discrete_action_space.n

        self.actor = CategoricalActor(
            input_dim=obs_dim,
            output_dim=nA,
            hidden_dims=self._actor_hidden
        ).to(self.device)

        self.qf1 = SoftQNetwork(
            input_dim=obs_dim,
            output_dim=nA,
            hidden_dims=self._q_hidden
        ).to(self.device)
        self.qf2 = SoftQNetwork(
            input_dim=obs_dim,
            output_dim=nA,
            hidden_dims=self._q_hidden
        ).to(self.device)
        self.qf1_target = SoftQNetwork(
            input_dim=obs_dim,
            output_dim=nA,
            hidden_dims=self._q_hidden
        ).to(self.device)
        self.qf2_target = SoftQNetwork(
            input_dim=obs_dim,
            output_dim=nA,
            hidden_dims=self._q_hidden
        ).to(self.device)

    def _build_replay_buffer(self):
        return ReplayBuffer(
            self.buffer_size,
            self.envs.single_observation_space,
            self.discrete_action_space,
            self.device,
            n_envs=self.envs.num_envs,
            handle_timeout_termination=False,
        )

    def _default_target_entropy(self):
        # For categorical SAC, a common choice is -log(|A|)
        nA = self.discrete_action_space.n
        return -math.log(float(nA))
    
    def _log_prior_discrete(self, states):
        return torch.zeros(
            (states.shape[0], self.discrete_action_space.n), device=states.device
        )

    def get_action(self, obs):
        actions, _, _ = self.actor.get_action(torch.as_tensor(obs, device=self.device))
        return actions

    def train_step(self, data):
        # ---------- Q update with KL-control backup ----------
        with torch.no_grad():
            _, next_log_pi, next_probs = self.actor.get_action(data.next_observations)  # [B,A], [B,A]
            q1_t = self.qf1_target(data.next_observations)  # [B,A]
            q2_t = self.qf2_target(data.next_observations)  # [B,A]
            min_q_t = torch.min(q1_t, q2_t)

            log_prior_next = self._log_prior_discrete(data.next_observations)  # [B,A]
            # E_a [ minQ - alpha * (log_pi - log_prior) ]
            term = min_q_t - self.alpha * (next_log_pi - log_prior_next)
            next_v = (next_probs * term).sum(dim=1)
            target = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * next_v

        q1 = self.qf1(data.observations)  # [B,A]
        q2 = self.qf2(data.observations)  # [B,A]
        q1_a = q1.gather(1, data.actions.long()).view(-1)
        q2_a = q2.gather(1, data.actions.long()).view(-1)
        q1_loss = F.mse_loss(q1_a, target)
        q2_loss = F.mse_loss(q2_a, target)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ---------- actor update with KL(pi || prior) ----------
        _, log_pi, probs = self.actor.get_action(data.observations)  # [B,A]
        with torch.no_grad():
            q1_vals = self.qf1(data.observations)
            q2_vals = self.qf2(data.observations)
            min_q = torch.min(q1_vals, q2_vals)
        log_prior = self._log_prior_discrete(data.observations)  # [B,A]

        # E_a [ alpha*(log_pi - log_prior) - minQ ]
        actor_loss = (probs * (self.alpha * (log_pi - log_prior) - min_q)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------- temperature (optional) ----------
        if self.autotune_alpha:
            # E_a [(log_pi - log_prior) + target_entropy]
            _, log_pi, probs = self.actor.get_action(data.observations)  # [B,A]
            log_prior = self._log_prior_discrete(data.observations)  # [B,A]
            kl_term = (probs.detach() * (log_pi - log_prior + self.target_entropy)).sum(dim=1).mean()
            alpha_loss = (-self.log_alpha.exp() * kl_term)
            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = None

        self._update_targets_()

        return (q1_a, q2_a, q1_loss, q2_loss, q_loss, actor_loss, self.alpha, alpha_loss)