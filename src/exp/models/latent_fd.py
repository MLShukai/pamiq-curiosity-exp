from __future__ import annotations

from collections.abc import Callable
from typing import override

import torch
import torch.nn as nn
from pamiq_core.torch import TorchTrainingModel
from torch import Tensor

from .components import (
    LerpStackedFeatures,
    MultiEmbeddings,
    StackedHiddenState,
    ToStackedFeatures,
)
from .forward_dynamics import HiddenStateFD
from .utils import ActionInfo, ObsInfo


class LatentFDFramework(HiddenStateFD):
    """Forward Dynamics framework with separated Encoder and Predictor
    components.

    Attributes:
        encoder: Module that encodes observation-action pairs into latent representations.
        predictor: Module that predicts next observations from latent representations.
    """

    def __init__(
        self,
        encoder: Encoder,
        predictor: Predictor,
    ) -> None:
        """Initialize the LatentFDFramework.

        Args:
            encoder: Encoder module for processing observation-action pairs.
            predictor: Predictor module for generating next observation predictions.
        """
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor

    @override
    def forward(
        self,
        obs: Tensor,
        action: Tensor,
        hidden: Tensor | None = None,
        *,
        no_len: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through the framework.

        Args:
            obs: Observation tensor. Shape depends on encoder configuration.
            action: Action tensor. Shape depends on encoder configuration.
            hidden: Optional hidden state for the encoder. Shape is (*batch, depth, dim).

        Returns:
            A tuple containing:
                - Predicted next observation from the predictor.
                - Updated hidden state from the encoder.
        """
        x, hidden = self.encoder(obs, action, hidden, no_len=no_len)
        return self.predictor(x), hidden


class Encoder(nn.Module):
    """Encoder module for processing observation-action pairs into latent
    representations."""

    def __init__(
        self,
        obs_info: ObsInfo | int,
        action_info: ActionInfo | int,
        core_model_dim: int,
        core_model: StackedHiddenState,
        out_dim: int | None = None,
    ) -> None:
        """Initialize the Encoder.

        Args:
            obs_info: Observation configuration. If ObsInfo, uses lerp-based feature
                extraction. If int, treats as direct observation dimension.
            action_info: Action configuration. If ActionInfo, uses multi-discrete
                embeddings. If int, treats as direct action dimension.
            core_model_dim: Input dimension for the core model.
            core_model: Stacked hidden state model for processing concatenated features.
            out_dim: Optional output dimension. If None, uses core_model output dimension.
        """
        super().__init__()

        # Setup observation and action projection.
        obs_action_dim = 0

        self.obs_flatten = nn.Identity()
        if isinstance(obs_info, ObsInfo):
            self.obs_flatten = LerpStackedFeatures(
                obs_info.dim, obs_info.dim_hidden, obs_info.num_tokens
            )
            obs_action_dim += obs_info.dim_hidden
        else:
            obs_action_dim += obs_info

        self.action_flatten = nn.Identity()
        if isinstance(action_info, ActionInfo):
            self.action_flatten = MultiEmbeddings(
                action_info.choices, action_info.dim, do_flatten=True
            )
            obs_action_dim += len(action_info.choices) * action_info.dim
        else:
            obs_action_dim += action_info

        self.obs_action_proj = nn.Linear(obs_action_dim, core_model_dim)

        self.core_model = core_model

        self.out_proj = nn.Identity()
        if out_dim is not None and core_model_dim != out_dim:
            self.out_proj = nn.Linear(core_model_dim, out_dim)

    def _flatten_obs_action(self, obs: Tensor, action: Tensor) -> Tensor:
        """Flatten and concatenate observation and action tensors.

        Args:
            obs: Observation tensor to be flattened.
            action: Action tensor to be flattened.

        Returns:
            Concatenated and projected tensor ready for core model processing.
        """
        obs_flat = self.obs_flatten(obs)
        action_flat = self.action_flatten(action)
        return self.obs_action_proj(torch.cat((obs_flat, action_flat), dim=-1))

    @override
    def forward(
        self,
        obs: Tensor,
        action: Tensor,
        hidden: Tensor | None = None,
        *,
        no_len: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Encode observation-action pairs into latent representations.

        Args:
            obs: Observation tensor. Shape is (*batch, len, num_tokens, obs_dim) if ObsInfo
                was provided, otherwise (*batch, len, obs_dim).
            action: Action tensor. Shape is (*batch, len, action_choices) if ActionInfo
                was provided, otherwise (*batch, len, action_dim).
            hidden: Optional hidden state from previous timestep. Shape is (*batch, depth, dim).
                If None, the core model will initialize its hidden state.
            no_len: If True, processes inputs as single-step without sequence length dimension.

        Returns:
            A tuple containing:
                - Latent representation tensor of shape (*batch, len, out_dim).
                - Updated hidden state tensor of shape (*batch, depth, dim).
        """
        x = self._flatten_obs_action(obs, action)
        x, hidden = self.core_model.forward(x, hidden, no_len=no_len)
        return self.out_proj(x), hidden

    @override
    def __call__(
        self,
        obs: Tensor,
        action: Tensor,
        hidden: Tensor | None = None,
        *,
        no_len: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Override __call__ with proper type annotations.

        See forward() method for full documentation.
        """
        return super().__call__(obs, action, hidden, no_len=no_len)


class Predictor(nn.Module):
    """Predictor module for generating observation predictions from latent
    representations."""

    def __init__(
        self,
        latent_dim: int,
        obs_info: ObsInfo | int,
        core_model: nn.Module | None = None,
        core_model_dim: int | None = None,
    ) -> None:
        """Initialize the Predictor.

        Args:
            latent_dim: Dimension of input latent representations.
            obs_info: Observation configuration for output. If ObsInfo, uses
                stacked features output. If int, uses direct linear projection.
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

        obs_dim_in = latent_dim if core_model_dim is None else core_model_dim
        if isinstance(obs_info, ObsInfo):
            self.obs_hat_head = ToStackedFeatures(
                obs_dim_in, obs_info.dim, obs_info.num_tokens
            )
        else:
            self.obs_hat_head = nn.Linear(obs_dim_in, obs_info)

    @override
    def forward(self, latent: Tensor) -> Tensor:
        """Generate observation predictions from latent representations.

        Args:
            latent: Latent representation tensor. Shape is (*batch, len, latent_dim)
                for sequential inputs or (*batch, latent_dim) for single-step.

        Returns:
            Predicted observations. Shape depends on obs_info configuration:
                - If ObsInfo: (*batch, len, num_tokens, obs_dim)
                - If int: (*batch, len, obs_dim)
        """
        x = self.input_proj(latent)
        x = self.core_model(x)
        return self.obs_hat_head(x)

    __call__: Callable[[Tensor], Tensor]


def create_hierarchical(
    obs_info: ObsInfo,
    latent_action_dims: list[int],
    latent_obs_dims: list[int],
    core_model_dims: list[int],
    core_models: list[StackedHiddenState],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> list[TorchTrainingModel[LatentFDFramework]]:
    """Creates hierarchical LatentFD models."""
    from .components.qlstm import FFNSwiGLU

    if not (
        len(latent_action_dims)
        == len(latent_obs_dims)
        == len(core_model_dims)
        == len(core_models)
    ):
        raise ValueError("All dimension lists must have the same length.")

    num_layers = len(core_models)
    models = []
    for i in range(num_layers):
        obs_cfg = obs_info if i == 0 else latent_obs_dims[i - 1]
        models.append(
            TorchTrainingModel(
                LatentFDFramework(
                    encoder=Encoder(
                        obs_info=obs_cfg,
                        action_info=latent_action_dims[i],
                        core_model_dim=core_model_dims[i],
                        core_model=core_models[i],
                        out_dim=latent_obs_dims[i],
                    ),
                    predictor=Predictor(
                        latent_dim=latent_obs_dims[i],
                        obs_info=obs_cfg,
                        core_model=FFNSwiGLU(
                            dim=core_model_dims[i], dim_ff_hidden=core_model_dims[i] * 4
                        ),
                        core_model_dim=core_model_dims[i],
                    ),
                ),
                device=device,
                dtype=dtype,
            )
        )
    return models
