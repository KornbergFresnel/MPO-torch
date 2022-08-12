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


def _dual(
    behavior_policy: np.ndarray, target_q: np.ndarray, eta: float, epsilon: float
):
    """Dual function of the non-parametric variational

    g(eta) = eta * dual_constraint + eta \sum \log(\sum\exp(Q(s,a)/eta))
    """

    max_q = np.max(target_q, 0)
    new_eta = (
        eta * epsilon
        + np.mean(max_q)
        + eta
        * np.mean(
            np.log(np.sum(behavior_policy * np.exp((target_q - max_q) / eta), axis=0))
        )
    )
    return new_eta


class MPOPolicy(BasePolicy):
    def __init__(
        self,
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
        actions, hidden = model(obs, state=state, info=batch.info)
        return Batch(act=actions, state=hidden)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        batch_size = batch.obs.size(0)

        # compute critic loss
        self.critic_optim.zero_grad()
        critic_loss = F.mse_loss(self.critic(batch.obs), batch.returns)

        # build behavior/ref policies
        sample_actions = (
            torch.arange(self.action_dim)[..., None]
            .expand(self.action_dim, batch_size)
            .to(self.device)
        )
        # behavior_policy = self.actor_old(batch.obs)
        # behavior_dist = torch.distributions.Categorical(probs=behavior_policy)
        # behavior_action_prob = behavior_policy.expand((self.action_dim, batch_size)).log_prob(sample_actions).exp()
        target_q = batch.target_q.detach()
        target_q = target_q.transpose(0, 1)
        ref_policy = torch.softmax(target_q / self._eta, dim=0)

        # E-step
        self._eta = minimize(
            _dual,
            ref_policy.cpu().numpy(),
            target_q.cpu().numpy(),
            self._eta,
            self._epsilon,
            method="SLSQP",
            bounds=[(1e-6, None)],
        )

        # M-step: update actor based on Lagrangian
        average_actor_loss = 0.0
        for _ in range(self.lagrange_it):
            policy = self.actor(batch.obs)
            dist = torch.distributions.Categorical(probs=policy)
            mle_loss = torch.mean(
                ref_policy
                * dist.expand((self.action_space.n, batch_size)).log_prob(
                    sample_actions
                )
            )
            kl_to_ref_policy = torch.distributions.kl.kl_divergence(policy, ref_policy)

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

        return {"loss/actor": average_actor_loss, "loss/critic": critic_loss.item()}

    def exploration_noise(
        self, act: Union[np.ndarray, Batch], batch: Batch
    ) -> Union[np.ndarray, Batch]:
        if self._noise is None:
            return act
        if isinstance(act, np.ndarray):
            return act + self._noise(act.shape)
        warnings.warn("Cannot add exploration noise to non-numpy_array action.")
        return act
