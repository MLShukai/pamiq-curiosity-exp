"""Defines policy models."""

from abc import ABC, abstractmethod
from typing import override

import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution

from .components.fc_scalar_head import FCScalarHead
from .components.multi_discretes import FCMultiCategoricalHead
from .components.stacked_features import LerpStackedFeatures
from .components.stacked_hidden_state import StackedHiddenState
from .utils import ObsInfo


class HiddenStatePiV(ABC, nn.Module):
    """Abstract base class for policy-value models with hidden state.

    Defines the interface for models that compute both policy
    distributions and value estimates while maintaining internal hidden
    state.
    """

    @override
    @abstractmethod
    def forward(
        self,
        observation: Tensor,
        hidden: Tensor | None = None,
        upper_action: Tensor | None = None,
        *,
        no_len: bool = False,
    ) -> tuple[Distribution, Tensor, Tensor]:
        """Compute policy distribution and value from observation.

        Args:
            observation: Input observation tensor.
            hidden: Optional hidden state from previous timestep.
            upper_action: Optional hierarchical action from upper-level policy.
            no_len: If True, processes inputs without sequence dimension.

        Returns:
            Tuple of (policy_distribution, value_estimate, updated_hidden_state).
        """
        pass

    @override
    def __call__(
        self,
        observation: Tensor,
        hidden: Tensor | None = None,
        upper_action: Tensor | None = None,
        *,
        no_len: bool = False,
    ) -> tuple[Distribution, Tensor, Tensor]:
        """Call method with proper type annotations.

        See forward() for documentation.
        """
        return super().__call__(observation, hidden, upper_action, no_len=no_len)


class StackedHiddenPiV(HiddenStatePiV):
    """Policy-value model using StackedHiddenState core."""

    def __init__(
        self,
        obs_info: ObsInfo,
        action_choices: list[int],
        dim: int,
        core_model: StackedHiddenState,
    ) -> None:
        """Initialize policy-value model with shared core.

        Args:
            obs_info: Observation configuration.
            action_choices: Number of choices per discrete action.
            dim: Hidden dimension size.
            core_model: Stacked hidden state model for processing.
        """
        super().__init__()

        self.obs_flatten = LerpStackedFeatures(obs_info.dim, dim, obs_info.num_tokens)
        self.core_model = core_model
        self.policy_head = FCMultiCategoricalHead(dim, action_choices)
        self.value_head = FCScalarHead(dim, squeeze_scalar_dim=True)

    @override
    def forward(
        self,
        observation: Tensor,
        hidden: Tensor | None = None,
        upper_action: Tensor | None = None,
        *,
        no_len: bool = False,
    ) -> tuple[Distribution, Tensor, Tensor]:
        """Compute policy and value from observation.

        Args:
            observation: Shape (*batch, len, num_token, obs_dim).
            hidden: Shape (*batch, depth, dim) or None.
            upper_action: Not used in this implementation.
            no_len: Process without sequence dimension if True.

        Returns:
            Tuple of (policy_distribution, value, updated_hidden).
        """
        obs_flat = self.obs_flatten(observation)
        x, hidden_out = self.core_model(obs_flat, hidden, no_len=no_len)
        return self.policy_head(x), self.value_head(x), hidden_out

    def forward_with_no_len(
        self, observation: Tensor, hidden: Tensor | None = None
    ) -> tuple[Distribution, Tensor, Tensor]:
        """Single-step inference without sequence dimension.

        Args:
            observation: Shape (*batch, num_token, obs_dim).
            hidden: Shape (*batch, depth, dim) or None.

        Returns:
            Tuple of (policy_distribution, value, updated_hidden).
        """
        obs_flat = self.obs_flatten(observation)
        x, hidden_out = self.core_model(obs_flat, hidden, no_len=True)
        return self.policy_head(x), self.value_head(x), hidden_out
