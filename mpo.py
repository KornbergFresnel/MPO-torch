from copy import deepcopy
from turtle import forward
from typing import Any, Dict, List, Type, Optional, Union

import warnings
import torch
import gym
import tianshou
import numpy as np
import scipy

from torch.nn import functional as F
from scipy.optimize import minimize

from tianshou.data import Batch, ReplayBuffer
from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.policy import BasePolicy
from tianshou.utils.net.continuous import Actor as ContinuousActor
from tianshou.utils.net.discrete import Actor as DiscreteActor


def _dual(target_q: np.ndarray, eta: float, epsilon: float):
    """Dual function of the non-parametric variational

    g(eta) = eta * dual_constraint + eta \sum \log(\sum\exp(Q(s,a)/eta))
    """

    max_q = np.max(target_q, -1)
    new_eta = (
        eta * epsilon
        + np.mean(max_q)
        + eta
        * np.mean(np.log(np.mean(np.exp((target_q - max_q[:, None]) / eta), axis=1)))
    )
    return new_eta


def get_dist_fn(discrete: bool):
    if discrete:
        return torch.distributions.Categorical
    else:

        def fn(logits):
            return torch.distributions.MultivariateNormal(
                loc=logits[:, 0], scale_tril=logits[:, 1]
            )

        return fn


def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)


def bt(m):
    return m.transpose(dim0=-2, dim1=-1)


def gaussian_kl(mu_i, mu, Ai, A):
    """
    decoupled KL between two multivariate gaussian distribution
    C_mu = KL(f(x|mu_i,sigma_i)||f(x|mu,sigma_i))
    C_sigma = KL(f(x|mu_i,sigma_i)||f(x|mu_i,sigma))
    :param mu_i: (B, n)
    :param mu: (B, n)
    :param Ai: (B, n, n)
    :param A: (B, n, n)
    :return: C_mu, C_sigma: scalar
        mean and covariance terms of the KL
    :return: mean of determinanats of sigma_i, sigma
    """
    n = A.size(-1)
    mu_i = mu_i.unsqueeze(-1)  # (B, n, 1)
    mu = mu.unsqueeze(-1)  # (B, n, 1)
    sigma_i = Ai @ bt(Ai)  # (B, n, n)
    sigma = A @ bt(A)  # (B, n, n)
    sigma_i_det = sigma_i.det()  # (B,)
    sigma_det = sigma.det()  # (B,)
    sigma_i_det = torch.clamp_min(sigma_i_det, 1e-6)
    sigma_det = torch.clamp_min(sigma_det, 1e-6)
    sigma_i_inv = sigma_i.inverse()  # (B, n, n)
    sigma_inv = sigma.inverse()  # (B, n, n)

    inner_mu = (
        (mu - mu_i).transpose(-2, -1) @ sigma_i_inv @ (mu - mu_i)
    ).squeeze()  # (B,)
    inner_sigma = (
        torch.log(sigma_det / sigma_i_det) - n + btr(sigma_inv @ sigma_i)
    )  # (B,)
    C_mu = 0.5 * torch.mean(inner_mu)
    C_sigma = 0.5 * torch.mean(inner_sigma)
    return C_mu, C_sigma, torch.mean(sigma_i_det), torch.mean(sigma_det)


class MPOPolicy(BasePolicy):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        actor: Optional[torch.nn.Module],
        actor_optim: Optional[torch.optim.Optimizer],
        critic: Optional[torch.nn.Module],
        critic_optim: Optional[torch.optim.Optimizer],
        eta: float = 0.4,
        epsilon: float = 0.1,
        alpha: float = 1.0,
        tau: float = 0.005,
        gamma: float = 0.99,
        actor_grad_norm: float = 5.0,
        critic_grad_norm: float = 5.0,
        critic_loss_type: str = "mse",
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        reward_normalization: bool = False,
        estimation_step: int = 1,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs,
        )

        if actor is not None and actor_optim is not None:
            self.actor: torch.nn.Module = actor
            self.actor_old = deepcopy(actor)
            self.actor_old.eval()
            self.actor_optim: torch.optim.Optimizer = actor_optim
        if critic is not None and critic_optim is not None:
            self.critic: torch.nn.Module = critic
            self.critic_old = deepcopy(critic)
            self.critic_old.eval()
            self.critic_optim: torch.optim.Optimizer = critic_optim

        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        self.tau = tau

        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"
        self._gamma = gamma

        self._noise = exploration_noise
        self._rew_norm = reward_normalization
        self._n_step = estimation_step

        # initialize Lagrange multiplier
        self._eta = eta
        self._eta_kl = 0.0
        self._epsilon = epsilon
        self._alpha = alpha
        self._lagrange_it = None
        self._actor_grad_norm = actor_grad_norm
        self._critic_grad_norm = critic_grad_norm
        self._discrete_act = isinstance(actor, DiscreteActor)
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._norm_critic_loss = (
            torch.nn.MSELoss() if critic_loss_type == "mse" else torch.nn.SmoothL1Loss()
        )
        self._mstep_iter_num = 10
        self.dist_fn = get_dist_fn(self._discrete_act)

    def set_exp_noise(self, noise: Optional[BaseNoise]):
        self._noise = noise

    def train(self, mode: bool = True) -> "MPOPolicy":
        """Switch to trainining mode.

        Args:
            mode (bool, optional): Enable training mode or not.. Defaults to True.

        Returns:
            MPOPolicy: MPO policy instance.
        """

        self.actor(mode)
        self.critic(mode)
        return self

    def sync_weight(self) -> None:
        self.soft_update(self.actor_old, self.actor, self.tau)
        self.soft_update(self.critic_old, self.critic, self.tau)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        target_q = self.critic_old(
            batch.obs_next, self(batch, model="actor_old", input="obs_next").act
        )
        # also save target q
        batch["target_q"] = target_q
        return target_q

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        with torch.no_grad():
            policy = self.actor(batch.obs)
            action_values = self.critic(batch.obs)
            state_values = (policy * action_values).sum(1, keepdim=True)
            next_policy = self.actor(batch.obs_next)
            next_action_values = self.critic(batch.obs_next)
            next_state_values = (next_policy * next_action_values).sum(1, keepdim=True)
            log_prob = torch.distributions.Categorical(probs=policy).log_prob(
                batch.actions
            )

        # TODO(ming): reshape rewar
        rho = self._alpha * (batch.old_log_prob - log_prob)
        if isinstance(batch.rew, torch.Tensor):
            old_rew_np = batch.rew.cpu().numpy()
        else:
            old_rew_np = batch.rew.copy()

        batch.rew = batch.rew - rho
        returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            next_state_values,
            state_values,
            gamma=self._gamma,
            gae_lambda=0.95,
        )

        # recover rew
        batch.rew = old_rew_np

        batch["state_values"] = state_values
        batch["next_state_values"] = next_state_values
        batch["returns"] = returns
        batch["advantages"] = advantages

        batch.to_torch()

        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[Dict, Batch, np.ndarray]] = None,
        model: str = "actor",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        model = getattr(self, model)
        obs = batch[input]
        logits, hidden = model(obs, state=state, info=batch.info)

        if not self.training:
            # arg max
            if self._discrete_act:
                actions = F.softmax(logits, dim=-1).argmax(-1)
            else:
                actions = logits[0]  # get means as actions
        else:
            actions = self.dist_fn(logits).rsample()

        return Batch(act=actions, state=hidden)

    def critic_update(self, batch: Batch, particle_num: int = 64):
        batch_size = batch.obs.size(0)
        with torch.no_grad():
            logits, _ = self.actor_old(batch.obs_next)
            policy: torch.distributions.Distribution = self.dist_fn(logits)
            sampled_next_actions = policy.sample((particle_num,)).transpose(
                0, 1
            )  # (batch_size, sample_num, action_dim)
            expaned_next_states = batch.obs_next[:, None, :].expand(
                -1, particle_num, -1
            )  # (batch_size, sample_num, obs_dim)

            # get expected Q vaue from target critic
            next_q_values = self.critic_old(
                expaned_next_states.reshape(-1, self._obs_dim)
            )
            next_state_values = next_q_values.gather(
                -1, sampled_next_actions.reshape(-1, self._act_dim).long()
            )
            next_state_values = next_state_values.reshape(batch_size, -1).mean(-1)

            y = batch.rew + self._gamma * next_state_values

        self.critic_optim.zero_grad()
        q_values = self.critic(batch.obs).squeeze()
        loss = self._norm_critic_loss(q_values, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.criti.parameters(), self._critic_grad_norm)
        self.critic_optim.step()
        return loss, y

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        batch_size = batch.obs.size(0)

        # compute critic loss
        particle_num = 64
        critic_loss, q_label = self.critic_update(batch, particle_num=particle_num)
        mean_est_q = q_label.abs().mean()

        # E-step for policy improvement
        with torch.no_grad():
            reference_logits, _ = self.actor_old(batch.obs)
            reference_policy = self.dist_fn(reference_logits)
            sampled_actions = reference_policy.sample(
                (particle_num,)
            )  # (K, batch_size, act_dim)
            expanded_states = batch.obs[None, ...].expand(
                particle_num, batch_size, -1
            )  # (K, batch_size, obs_dim)
            target_q = self.critic_old(
                expanded_states.reshape(-1, self._obs_dim)
            )  # (K*batch_size, act_dim)
            target_q = target_q.gather(
                -1, sampled_actions.reshape(-1, self._act_dim).long()
            ).reshape(
                particle_num, batch_size
            )  # (K, batch_size)
            target_q_np = target_q.cpu().transpose(0, 1).numpy()  # (batch_size, K)

        self._eta = minimize(
            _dual,
            target_q_np,
            self._eta,
            self._epsilon,
            method="SLSQP",
            bounds=[(1e-6, None)],
        ).x[0]

        # M-step: update actor based on Lagrangian
        average_actor_loss = 0.0
        # normalize q
        norm_target_q = F.softmax(target_q / self._eta, dim=0)  # (K, batch_size)
        for _ in range(self._mstep_iter_num):
            logits, _ = self.actor(batch.obs)
            policy = self.dist_fn(logits)
            mle_loss = torch.mean(
                norm_target_q
                * policy.expand((particle_num, batch_size)).log_prob(sampled_actions)
            )  # (K, batch_size)
            kl_to_ref_policy = torch.distributions.kl.kl_divergence(
                policy.probs, reference_policy.probs
            )

            # Update lagrange multipliers by gradient descent
            self._eta_kl -= (
                self._alpha * (self._eta_kl - kl_to_ref_policy).detach().item()
            )
            self._eta_kl = max(self._eta_kl, 0.0)

            self.actor_optim.zero_grad()

            actor_loss = -(mle_loss + self._eta_kl * (self._epsilon - kl_to_ref_policy))
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)

            self.actor_optim.step()
            average_actor_loss += actor_loss.item()

        self.sync_weight()

        return {
            "loss/actor": average_actor_loss,
            "loss/critic": critic_loss.item(),
            "est/q": mean_est_q.item(),
        }

    def exploration_noise(
        self, act: Union[np.ndarray, Batch], batch: Batch
    ) -> Union[np.ndarray, Batch]:
        if self._noise is None:
            return act
        if isinstance(act, np.ndarray):
            return act + self._noise(act.shape)
        warnings.warn("Cannot add exploration noise to non-numpy_array action.")
        return act
