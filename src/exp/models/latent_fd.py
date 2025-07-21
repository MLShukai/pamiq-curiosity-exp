"""Forward Dynamics construct with Latent Encoder and Predictor."""

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


class Encoder(nn.Module):
    """Latent encoder that processes observations and actions into latent
    representations.

    This encoder combines observation and action inputs, processes them
    through a stacked hidden state model, and outputs latent
    representations suitable for forward dynamics prediction.
    """

    @override
    def __init__(
        self,
        obs_info: ObsInfo | int,
        action_info: ActionInfo | int,
        core_model: StackedHiddenState,
        core_model_dim: int,
        embed_dim: int | None = None,
    ) -> None:
        """Initialize the latent encoder.

        Args:
            obs_info: Configuration for observation processing. If ObsInfo, sets up
                lerp-based feature extraction. If int, uses direct dimension.
            action_info: Configuration for action processing. If ActionInfo, sets up
                multi-discrete embeddings. If int, uses direct dimension.
            core_model: The stacked hidden state model for processing concatenated features.
            core_model_dim: Input dimension for the core model.
            embed_dim: Optional output embedding dimension. If None, uses core_model output dim.
        """
        super().__init__()

        self.obs_flatten = None
        dim_in = 0
        if isinstance(obs_info, ObsInfo):
            self.obs_flatten = LerpStackedFeatures(
                obs_info.dim, obs_info.dim_hidden, obs_info.num_tokens
            )
            dim_in += obs_info.dim_hidden
        else:
            dim_in += obs_info

        self.action_flatten = None
        if isinstance(action_info, ActionInfo):
            self.action_flatten = MultiEmbeddings(
                action_info.choices, action_info.dim, do_flatten=True
            )
            dim_in += len(action_info.choices) * action_info.dim
        else:
            dim_in += action_info

        self.obs_action_proj = nn.Linear(dim_in, core_model_dim)

        self.core_model = core_model

        self.out_proj = None
        if embed_dim is not None and embed_dim != core_model_dim:
            self.out_proj = nn.Linear(core_model_dim, embed_dim)

    def _flatten_obs_action(self, obs: Tensor, action: Tensor) -> Tensor:
        """Flatten and concat observation and action."""
        if self.obs_flatten:
            obs = self.obs_flatten(obs)
        if self.action_flatten:
            action = self.action_flatten(action)
        return self.obs_action_proj(torch.cat((obs, action), dim=-1))

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
            no_len: If True, calls core_model.forward_with_no_len for single-step processing.

        Returns:
            A tuple containing:
                - Latent representation tensor of shape (*batch, len, embed_dim).
                - Updated hidden state tensor of shape (*batch, depth, dim).
        """
        x = self._flatten_obs_action(obs, action)
        x, next_hidden = self.core_model(x, hidden, no_len=no_len)
        if self.out_proj:
            x = self.out_proj(x)
        return x, next_hidden

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

    def forward_with_no_len(
        self,
        obs: Tensor,
        action: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass for single-step encoding without length dimension.

        Convenience method for inference where inputs don't have a sequence length dimension.

        Args:
            obs: Observation tensor. Shape is (*batch, num_tokens, obs_dim) if ObsInfo
                was provided, otherwise (*batch, obs_dim).
            action: Action tensor. Shape is (*batch, action_choices) if ActionInfo
                was provided, otherwise (*batch, action_dim).
            hidden: Optional hidden state from previous timestep. Shape is (*batch, depth, dim).

        Returns:
            A tuple containing:
                - Latent representation tensor of shape (*batch, embed_dim).
                - Updated hidden state tensor of shape (*batch, depth, dim).
        """
        return self.forward(obs, action, hidden, no_len=True)


class Predictor(nn.Module):
    """Latent predictor that generates observation distributions from latent
    representations.

    This predictor takes latent representations from the encoder and
    predicts the distribution of next observations using a stacked
    hidden state model followed by a deterministic normal distribution
    head.
    """

    @override
    def __init__(
        self,
        obs_info: ObsInfo | int,
        core_model: StackedHiddenState,
        core_model_dim: int,
        embed_dim: int | None = None,
    ) -> None:
        """Initialize the latent predictor.

        Args:
            obs_info: Configuration for observation prediction. If ObsInfo, sets up
                stacked features output. If int, uses direct dimension.
            core_model: The stacked hidden state model for processing latent representations.
            core_model_dim: Hidden dimension of the core model.
            embed_dim: Optional input embedding dimension. If provided and different from
                core_model_dim, adds an input projection layer.
        """
        super().__init__()

        self.input_proj = None
        if embed_dim is not None:
            self.input_proj = nn.Linear(embed_dim, core_model_dim)

        self.core_model = core_model

        if isinstance(obs_info, ObsInfo):
            self.obs_hat_dist_head = nn.Sequential(
                ToStackedFeatures(core_model_dim, obs_info.dim, obs_info.num_tokens),
                FCDeterministicNormalHead(obs_info.dim, obs_info.dim),
            )
        else:
            self.obs_hat_dist_head = FCDeterministicNormalHead(obs_info, obs_info)

    @override
    def forward(
        self, latent: Tensor, hidden: Tensor | None = None, *, no_len: bool = False
    ) -> tuple[Distribution, Tensor]:
        """Predict observation distribution from latent representation.

        Args:
            latent: Latent representation tensor. Shape is (*batch, len, embed_dim).
            hidden: Optional hidden state from previous timestep. Shape is (*batch, depth, dim).
                If None, the core model will initialize its hidden state.
            no_len: If True, calls core_model.forward_with_no_len for single-step processing.

        Returns:
            A tuple containing:
                - Distribution representing predicted observations.
                - Updated hidden state tensor of shape (*batch, depth, dim).
        """
        if self.input_proj:
            latent = self.input_proj(latent)

        x, next_hidden = self.core_model(latent, hidden, no_len=no_len)
        obs_hat = self.obs_hat_dist_head(x)
        return obs_hat, next_hidden

    @override
    def __call__(
        self, latent: Tensor, hidden: Tensor | None = None, *, no_len: bool = False
    ) -> tuple[Distribution, Tensor]:
        return super().__call__(latent, hidden, no_len=no_len)

    def forward_with_no_len(
        self,
        latent: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Distribution, Tensor]:
        """Forward pass for single-step prediction without length dimension.

        Convenience method for inference where inputs don't have a sequence length dimension.

        Args:
            latent: Latent representation tensor. Shape is (*batch, embed_dim).
            hidden: Optional hidden state from previous timestep. Shape is (*batch, depth, dim).

        Returns:
            A tuple containing:
                - Distribution representing predicted observations.
                - Updated hidden state tensor of shape (*batch, depth, dim).
        """
        return self.forward(latent, hidden, no_len=True)
