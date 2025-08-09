from __future__ import annotations

from collections.abc import Callable
from typing import override

import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution

from .components import (
    FCMultiCategoricalHead,
    FCScalarHead,
    LerpStackedFeatures,
    StackedHiddenState,
)
from .utils import ObsInfo


class LatentPiVFramework(nn.Module):
    """Policy-Value framework with separated Encoder and Generator components.

    Attributes:
        encoder: Module that encodes observations into latent representations.
        generator: Module that generates policy distributions and value estimates from latent representations.
    """

    def __init__(
        self,
        encoder: Encoder,
        generator: Generator,
    ) -> None:
        """Initialize the LatentPiVFramework.

        Args:
            encoder: Encoder module for processing observations.
            generator: Generator module for producing policy and value outputs.
        """
        super().__init__()

        self.encoder = encoder
        self.generator = generator

    @override
    def forward(
        self,
        obs: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Distribution, Tensor, Tensor]:
        """Forward pass through the framework.

        Args:
            obs: Observation tensor. Shape depends on encoder configuration.
            hidden: Optional hidden state for the encoder. Shape is (*batch, depth, dim).

        Returns:
            A tuple containing:
                - Policy distribution from the generator.
                - Value estimate from the generator.
                - Updated hidden state from the encoder.
        """
        x, hidden = self.encoder(obs, hidden)
        policy_dist, value = self.generator(x)
        return policy_dist, value, hidden


class Encoder(nn.Module):
    """Encoder module for processing observations into latent
    representations."""

    def __init__(
        self,
        obs_info: ObsInfo | int,
        core_model_dim: int,
        core_model: StackedHiddenState,
        out_dim: int | None = None,
    ) -> None:
        """Initialize the Encoder.

        Args:
            obs_info: Observation configuration. If ObsInfo, uses lerp-based feature
                extraction. If int, treats as direct observation dimension.
            core_model_dim: Input dimension for the core model.
            core_model: Stacked hidden state model for processing features.
            out_dim: Optional output dimension. If None, uses core_model output dimension.
        """
        super().__init__()

        # Setup observation projection.
        self.obs_flatten = nn.Identity()
        if isinstance(obs_info, ObsInfo):
            self.obs_flatten = LerpStackedFeatures(
                obs_info.dim, obs_info.dim_hidden, obs_info.num_tokens
            )
            obs_dim = obs_info.dim_hidden
        else:
            obs_dim = obs_info

        self.obs_proj = nn.Linear(obs_dim, core_model_dim)
        self.core_model = core_model

        self.out_proj = nn.Identity()
        if out_dim is not None:
            self.out_proj = nn.Linear(core_model_dim, out_dim)

    @override
    def forward(
        self,
        obs: Tensor,
        hidden: Tensor | None = None,
        *,
        no_len: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Encode observations into latent representations.

        Args:
            obs: Observation tensor. Shape is (*batch, len, num_tokens, obs_dim) if ObsInfo
                was provided, otherwise (*batch, len, obs_dim).
            hidden: Optional hidden state from previous timestep. Shape is (*batch, depth, dim).
                If None, the core model will initialize its hidden state.
            no_len: If True, processes inputs as single-step without sequence length dimension.

        Returns:
            A tuple containing:
                - Latent representation tensor of shape (*batch, len, out_dim).
                - Updated hidden state tensor of shape (*batch, depth, dim).
        """
        x = self.obs_flatten(obs)
        x = self.obs_proj(x)
        x, hidden = self.core_model.forward(x, hidden, no_len=no_len)
        return self.out_proj(x), hidden

    @override
    def __call__(
        self,
        obs: Tensor,
        hidden: Tensor | None = None,
        *,
        no_len: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Override __call__ with proper type annotations.

        See forward() method for full documentation.
        """
        return super().__call__(obs, hidden, no_len=no_len)


class Generator(nn.Module):
    """Generator module for producing policy distributions and value estimates
    from latent representations."""

    def __init__(
        self,
        latent_dim: int,
        action_choices: list[int],
        core_model: nn.Module | None = None,
        core_model_dim: int | None = None,
    ) -> None:
        """Initialize the Generator.

        Args:
            latent_dim: Dimension of input latent representations.
            action_choices: List specifying the number of choices for each discrete
                action dimension.
            core_model: Optional core model for processing latents. If None,
                uses identity transformation.
            core_model_dim: Hidden dimension for core model. If provided,
                adds input projection from latent_dim to core_model_dim.
        """
        super().__init__()

        self.input_proj = nn.Identity()
        if core_model_dim is not None:
            self.input_proj = nn.Linear(latent_dim, core_model_dim)

        if core_model is None:
            core_model = nn.Identity()
        self.core_model = core_model

        head_dim_in = latent_dim if core_model_dim is None else core_model_dim
        self.policy_head = FCMultiCategoricalHead(head_dim_in, action_choices)
        self.value_head = FCScalarHead(head_dim_in, squeeze_scalar_dim=True)

    @override
    def forward(self, latent: Tensor) -> tuple[Distribution, Tensor]:
        """Generate policy distributions and value estimates from latent
        representations.

        Args:
            latent: Latent representation tensor. Shape is (*batch, len, latent_dim)
                for sequential inputs or (*batch, latent_dim) for single-step.

        Returns:
            A tuple containing:
                - Distribution representing the policy (action probabilities).
                - Tensor containing the estimated state value.
        """
        x = self.input_proj(latent)
        x = self.core_model(x)
        return self.policy_head(x), self.value_head(x)

    __call__: Callable[[Tensor], tuple[Distribution, Tensor]]
