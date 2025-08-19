from __future__ import annotations

from collections.abc import Callable
from typing import override

import torch
import torch.nn as nn
from pamiq_core.torch import TorchTrainingModel
from torch import Tensor
from torch.distributions import Distribution

from .components import (
    FCDeterministicNormalHead,
    FCMultiCategoricalHead,
    FCScalarHead,
    LerpStackedFeatures,
    StackedHiddenState,
)
from .policy import HiddenStatePiV
from .utils import ActionInfo, ObsInfo


class LatentPiVFramework(HiddenStatePiV):
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
        observation: Tensor,
        hidden: Tensor | None = None,
        upper_action: Tensor | None = None,
        *,
        no_len: bool = False,
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
        x, hidden = self.encoder(observation, hidden, upper_action, no_len=no_len)
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
        upper_action_dim: int | None = None,
    ) -> None:
        """Initialize the Encoder.

        Args:
            obs_info: Observation configuration. If ObsInfo, uses lerp-based feature
                extraction. If int, treats as direct observation dimension.
            core_model_dim: Input dimension for the core model.
            core_model: Stacked hidden state model for processing features.
            out_dim: Optional output dimension. If None, uses core_model output dimension.
            upper_action_dim: Optional dimension for upper-level policy actions. If provided,
                enables hierarchical policy by concatenating upper-level actions with observations.
        """
        super().__init__()

        # Setup observation projection.
        self.obs_flatten = nn.Identity()
        obs_action_dim = 0
        if isinstance(obs_info, ObsInfo):
            self.obs_flatten = LerpStackedFeatures(
                obs_info.dim, obs_info.dim_hidden, obs_info.num_tokens
            )
            obs_action_dim += obs_info.dim_hidden
        else:
            obs_action_dim += obs_info

        # Setup upper action projection
        if upper_action_dim is not None:
            obs_action_dim += upper_action_dim

        self.obs_action_proj = nn.Linear(obs_action_dim, core_model_dim)
        self.core_model = core_model

        self.out_proj = nn.Identity()
        if out_dim is not None:
            self.out_proj = nn.Linear(core_model_dim, out_dim)

    @override
    def forward(
        self,
        obs: Tensor,
        hidden: Tensor | None = None,
        upper_action: Tensor | None = None,
        *,
        no_len: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Encode observations into latent representations.

        Args:
            obs: Observation tensor. Shape is (*batch, len, num_tokens, obs_dim) if ObsInfo
                was provided, otherwise (*batch, len, obs_dim).
            hidden: Optional hidden state from previous timestep. Shape is (*batch, depth, dim).
                If None, the core model will initialize its hidden state.
            upper_action: Optional upper-level policy action tensor for hierarchical policies.
                Shape is (*batch, len, upper_action_dim). If provided, concatenated with
                observations before processing.
            no_len: If True, processes inputs as single-step without sequence length dimension.

        Returns:
            A tuple containing:
                - Latent representation tensor of shape (*batch, len, out_dim).
                - Updated hidden state tensor of shape (*batch, depth, dim).
        """
        x = self.obs_flatten(obs)
        if upper_action is not None:
            x = torch.cat([x, upper_action], dim=-1)
        x = self.obs_action_proj(x)
        x, hidden = self.core_model.forward(x, hidden, no_len=no_len)
        return self.out_proj(x), hidden

    @override
    def __call__(
        self,
        obs: Tensor,
        hidden: Tensor | None = None,
        upper_action: Tensor | None = None,
        *,
        no_len: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Override __call__ with proper type annotations.

        See forward() method for full documentation.
        """
        return super().__call__(obs, hidden, upper_action, no_len=no_len)


class Generator(nn.Module):
    """Generator module for producing policy distributions and value estimates
    from latent representations."""

    def __init__(
        self,
        latent_dim: int,
        action_info: ActionInfo | int,
        core_model: nn.Module | None = None,
        core_model_dim: int | None = None,
    ) -> None:
        """Initialize the Generator.

        Args:
            latent_dim: Dimension of input latent representations.
            action_info: Action configuration specifying action structure.
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
        if isinstance(action_info, ActionInfo):
            self.policy_head = FCMultiCategoricalHead(head_dim_in, action_info.choices)
        else:
            self.policy_head = FCDeterministicNormalHead(head_dim_in, action_info)
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


def create_hierarchical(
    obs_info: ObsInfo,
    action_info: ActionInfo,
    action_dims: list[int],
    latent_obs_dims: list[int],
    latent_action_dims: list[int],
    core_model_dims: list[int],
    core_models: list[StackedHiddenState],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> list[TorchTrainingModel[LatentPiVFramework]]:
    """Creates hierarchical LatentPiV models."""
    if not (
        len(latent_obs_dims)
        == len(action_dims)
        == len(latent_action_dims)
        == len(core_model_dims)
        == len(core_models)
    ):
        raise ValueError("All dimension lists must have the same length.")

    num_layers = len(core_models)
    models = []
    for i in range(num_layers):
        obs_cfg = obs_info if i == 0 else latent_obs_dims[i - 1]
        act_cfg = action_info if i == 0 else latent_action_dims[i - 1]
        upper_action_dim = action_dims[i + 1] if (i + 1) < num_layers else None
        models.append(
            TorchTrainingModel(
                LatentPiVFramework(
                    encoder=Encoder(
                        obs_info=obs_cfg,
                        core_model_dim=core_model_dims[i],
                        core_model=core_models[i],
                        out_dim=latent_action_dims[i],
                        upper_action_dim=upper_action_dim,
                    ),
                    generator=Generator(
                        latent_dim=latent_action_dims[i],
                        action_info=act_cfg,
                    ),
                ),
                device=device,
                dtype=dtype,
            )
        )
    return models
