"""Defines policy models."""

from typing import override

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution

from .components.fc_scalar_head import FCScalarHead
from .components.multi_discretes import FCMultiCategoricalHead
from .components.stacked_features import LerpStackedFeatures
from .components.stacked_hidden_state import StackedHiddenState
from .utils import ObsInfo


class StackedHiddenPiV(nn.Module):
    """Module with shared models for policy (pi) and value (V) functions.

    Using StackedHiddenState as core model.
    """

    def __init__(
        self,
        obs_info: ObsInfo,
        action_choices: list[int],
        dim: int,
        core_model: StackedHiddenState,
    ) -> None:
        """Initialize the policy-value model.

        Sets up neural network components for computing both policy distributions
        and value estimates from observations using a shared core model.

        Args:
            obs_info: Configuration for observation processing including dimensions,
                hidden dimensions, and number of tokens.
            action_choices: List specifying the number of choices for each discrete
                action dimension.
            dim: Hidden dimension size for the core model and policy/value heads.
            core_model: The main stacked hidden state model that processes the
                flattened observation features.
        """
        super().__init__()

        self.obs_flatten = LerpStackedFeatures(obs_info.dim, dim, obs_info.num_tokens)
        self.core_model = core_model
        self.policy_head = FCMultiCategoricalHead(dim, action_choices)
        self.value_head = FCScalarHead(dim, squeeze_scalar_dim=True)

    @override
    def forward(
        self, observation: Tensor, hidden: Tensor
    ) -> tuple[Distribution, Tensor, Tensor]:
        """Process observation and compute policy and value outputs.

        Args:
            observation: Input observation tensor (*batch, len, num_token, obs_dim)
            hidden: Hidden state tensor from previous timestep (*batch, depth, dim)

        Returns:
            A tuple containing:
                - Distribution representing the policy (action probabilities)
                - Tensor containing the estimated state value
                - Updated hidden state tensor for use in next forward pass
        """
        obs_flat = self.obs_flatten(observation)
        x, hidden_out = self.core_model(obs_flat, hidden)
        return self.policy_head(x), self.value_head(x), hidden_out

    @override
    def __call__(
        self, observation: Tensor, hidden: Tensor
    ) -> tuple[Distribution, Tensor, Tensor]:
        """Override __call__ with proper type annotations."""
        return super().__call__(observation, hidden)
