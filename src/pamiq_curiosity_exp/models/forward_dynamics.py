"""Defines forward dynamics models."""

from typing import override

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution

from .components.deterministic_normal import FCDeterministicNormalHead
from .components.multi_discretes import MultiEmbeddings
from .components.stacked_features import LerpStackedFeatures, ToStackedFeatures
from .components.stacked_hidden_state import StackedHiddenState
from .utils import ActionInfo, ObsInfo


class StackedHiddenFD(nn.Module):
    """Forward dynamics using StackedHiddenState model variants for core
    model."""

    @override
    def __init__(
        self,
        obs_info: ObsInfo,
        action_info: ActionInfo,
        dim: int,
        core_model: StackedHiddenState,
    ) -> None:
        """Initialize the forward dynamics model.

        Sets up the neural network components for predicting next observations
        from current observations and actions using a stacked hidden state model.

        Args:
            obs_info: Configuration for observation processing.
            action_info: Configuration for action processing.
            dim: Hidden dimension size for the core model input projection.
            core_model: The main stacked hidden state model that processes the
                concatenated observation-action features.
        """
        super().__init__()
        self.obs_flatten = LerpStackedFeatures(
            obs_info.dim, obs_info.dim_hidden, obs_info.num_tokens
        )
        self.action_flatten = MultiEmbeddings(
            action_info.choices, action_info.dim, do_flatten=True
        )
        self.obs_action_projection = nn.Linear(
            obs_info.dim_hidden + action_info.dim * len(action_info.choices), dim
        )
        self.core_model = core_model
        self.obs_hat_dist_head = nn.Sequential(
            ToStackedFeatures(dim, obs_info.dim, obs_info.num_tokens),
            FCDeterministicNormalHead(obs_info.dim, obs_info.dim),
        )

    def _flatten_obs_action(self, obs: Tensor, action: Tensor) -> Tensor:
        """Flatten and concat observation and action."""
        obs_flat = self.obs_flatten(obs)
        action_flat = self.action_flatten(action)
        return self.obs_action_projection(torch.cat((obs_flat, action_flat), dim=-1))

    @override
    def forward(
        self, obs: Tensor, action: Tensor, hidden: Tensor
    ) -> tuple[Distribution, Tensor]:
        """Forward pass to predict next observation distribution.

        Args:
            obs: Current observation tensor. shape is (*batch, len, num_token, obs_dim)
            action: Action tensor. shape is (*batch, len, num_token, action_chocies)
            hidden: Hidden state from previous timestep. shape is (*batch, depth, dim)

        Returns:
            A tuple containing:
                - Distribution representing predicted next observation.
                - Updated hidden state tensor for use in next prediction.
        """
        x = self._flatten_obs_action(obs, action)
        x, next_hidden = self.core_model(x, hidden)
        obs_hat_dist = self.obs_hat_dist_head(x)
        return obs_hat_dist, next_hidden

    @override
    def __call__(
        self, obs: Tensor, action: Tensor, hidden: Tensor
    ) -> tuple[Distribution, Tensor]:
        """Override __call__ with proper type annotations.

        See forward() method for full documentation.
        """
        return super().__call__(obs, action, hidden)

    def forward_with_no_len(
        self, obs: Tensor, action: Tensor, hidden: Tensor
    ) -> tuple[Distribution, Tensor]:
        """Forward with data which has no len dim. (for inference procedure.)

        Args:
            obs: Current observation tensor. shape is (*batch, num_token, obs_dim)
            action: Action tensor. shape is (*batch, num_token, action_chocies)
            hidden: Hidden state from previous timestep. shape is (*batch, depth, dim)

        Returns:
            A tuple containing:
                - Distribution representing predicted next observation.
                - Updated hidden state tensor for use in next prediction.
        """
        x = self._flatten_obs_action(obs, action)  # (*batch, dim)
        x, next_hidden = self.core_model.forward_with_no_len(x, hidden)
        return self.obs_hat_dist_head(x), next_hidden
