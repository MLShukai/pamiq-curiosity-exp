"""Defines forward-dynamics policy models."""

from abc import ABC, abstractmethod
from typing import override

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution

from .components.fc_scalar_head import FCScalarHead
from .components.multi_discretes import FCMultiCategoricalHead, MultiEmbeddings
from .components.stacked_features import LerpStackedFeatures, ToStackedFeatures
from .components.stacked_hidden_state import StackedHiddenState
from .utils import ActionInfo, ObsInfo


class HiddenStateFDPiV(ABC, nn.Module):
    """Abstract base class for forward-dynamics policy-value models with hidden
    state.

    Defines the interface for models that compute observation
    prediction, policy distributions and value estimates while
    maintaining internal hidden state.
    """

    @override
    @abstractmethod
    def forward(
        self,
        obs: Tensor,
        action: Tensor,
        upper_action: Tensor | None = None,
        hidden: Tensor | None = None,
        *,
        no_len: bool = False,
    ) -> tuple[Tensor, Distribution, Tensor, Tensor]:
        """Compute observation prediction, policy distribution and value from
        observation.

        Args:
            observation: Input observation tensor.
            hidden: Optional hidden state from previous timestep.
            upper_action: Optional hierarchical action from upper-level policy.
            no_len: If True, processes inputs without sequence dimension.

        Returns:
            Tuple of (next_observation_prediction, policy_distribution, value_estimate, updated_hidden_state).
        """
        pass

    @override
    def __call__(
        self,
        observation: Tensor,
        action: Tensor,
        upper_action: Tensor | None = None,
        hidden: Tensor | None = None,
        *,
        no_len: bool = False,
    ) -> tuple[Tensor, Distribution, Tensor, Tensor]:
        """Call method with proper type annotations.

        See forward() for documentation.
        """
        return super().__call__(
            observation, action, upper_action, hidden, no_len=no_len
        )


class StackedHiddenFDPiV(HiddenStateFDPiV):
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
        """Initialize the forward-dynamics policy-value model.

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
        self.obs_hat_head = ToStackedFeatures(dim, obs_info.dim, obs_info.num_tokens)
        self.policy_head = FCMultiCategoricalHead(dim, action_info.choices)
        self.value_head = FCScalarHead(dim, squeeze_scalar_dim=True)

    def _flatten_obs_action(self, obs: Tensor, action: Tensor) -> Tensor:
        """Flatten and concat observation and action."""
        obs_flat = self.obs_flatten(obs)
        action_flat = self.action_flatten(action)
        return self.obs_action_projection(torch.cat((obs_flat, action_flat), dim=-1))

    @override
    def forward(
        self,
        obs: Tensor,
        action: Tensor,
        upper_action: Tensor | None = None,
        hidden: Tensor | None = None,
        *,
        no_len: bool = False,
    ) -> tuple[Tensor, Distribution, Tensor, Tensor]:
        """Forward pass to predict next observation prediction, policy
        distribution, and value estimate.

        Args:
            obs: Current observation tensor. shape is (*batch, len, num_token, obs_dim)
            action: Action tensor. shape is (*batch, len, num_token, action_choices)
            upper_action: Not used in this implementation.
            hidden: Optional hidden state from previous timestep. shape is (*batch, depth, dim).
                If None, the hidden state is initialized to zeros

        Returns:
            A tuple containing:
                - Tensor representing predicted next observation.
                - Distribution representing the policy over actions.
                - Tensor representing the value estimate.
                - Updated hidden state tensor for use in next prediction.
        """
        x = self._flatten_obs_action(obs, action)
        x, next_hidden = self.core_model(x, hidden, no_len=no_len)
        obs_hat = self.obs_hat_head(x)
        action_dist = self.policy_head(x)
        value = self.value_head(x)
        return obs_hat, action_dist, value, next_hidden

    def forward_with_no_len(
        self,
        obs: Tensor,
        action: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Distribution, Tensor, Tensor]:
        """Forward with data which has no len dim. (for inference procedure.)

        Args:
            obs: Current observation tensor. shape is (*batch, num_token, obs_dim)
            action: Action tensor. shape is (*batch, num_token, action_choices)
            hidden: Optional hidden state from previous timestep. shape is (*batch, depth, dim).
                If None, the hidden state is initialized to zeros

        Returns:
            A tuple containing:
                - Tensor representing predicted next observation.
                - Distribution representing the policy over actions.
                - Tensor representing the value estimate.
                - Updated hidden state tensor for use in next prediction.
        """
        x = self._flatten_obs_action(obs, action)  # (*batch, dim)
        x, next_hidden = self.core_model(x, hidden, no_len=True)
        return (
            self.obs_hat_head(x),
            self.policy_head(x),
            self.value_head(x),
            next_hidden,
        )
