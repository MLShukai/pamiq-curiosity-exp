"""Policy construct with Latent Encoder and Predictor."""

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


class LatentPolicy(nn.Module):
    """Policy model that outputs latent representations and predicts next
    observations from those representations.

    This model combines an encoder that processes observations and
    actions into latent representations, and a predictor that generates
    the distribution of next observations based on those latent
    representations.
    """

    @override
    def __init__(
        self,
        obs_upper_action_flatten_head: nn.Module,
        encoder: StackedHiddenState,
        predictor: StackedHiddenState,
        action_dist_head: nn.Module,
        value_head: nn.Module,
    ) -> None:
        """Initialize the LatentPolicy model.

        Args:
            obs_action_flatten_head: The head that processes observations and actions
            encoder: The stacked hidden state model for encoding observation-action pairs.
            predictor: The stacked hidden state model for predicting next observations.
            obs_predict_head: The head that generates the distribution of next observations
                from latent representations.
        """
        super().__init__()
        self.obs_upper_action_flatten_head = obs_upper_action_flatten_head
        self.encoder = encoder
        self.predictor = predictor
        self.action_dist_head = action_dist_head
        self.value_head = value_head

    @override
    def forward(
        self,
        obs: Tensor,
        upper_action: Tensor,
        hidden_encoder: Tensor | None = None,
        hidden_predictor: Tensor | None = None,
        no_len: bool = False,
    ) -> tuple[Distribution, Tensor, Tensor, Tensor, Tensor]:
        """Process observation and upper action, compute latent representation,
        and predict next observation distribution.

        Args:
            obs: Input observation tensor (*batch, len, num_token, obs_dim).
            upper_action: Upper action tensor (*batch, len, num_token, action_dim).
            hidden_encoder: Optional hidden state for the encoder (*batch, depth, dim).
            hidden_predictor: Optional hidden state for the predictor (*batch, depth, dim).
            no_len: If True, ignores sequence length dimension.

        Returns:
            A tuple containing:
                - Distribution representing predicted observations.
                - Value estimates from the value head.
                - Latent representation from the encoder.
                - Next hidden state for the predictor.
                - Next hidden state for the encoder.
        """
        latent = self.obs_upper_action_flatten_head(obs, upper_action)

        latent_encoder, next_hidden_encoder = self.encoder(
            latent, hidden_encoder, no_len=no_len
        )
        latent_predictor, next_hidden_predictor = self.predictor(
            latent_encoder, hidden_predictor, no_len=no_len
        )

        action_dist = self.action_dist_head(latent_predictor)
        value = self.value_head(latent_predictor)

        return (
            action_dist,
            value,
            latent_encoder,
            next_hidden_predictor,
            next_hidden_encoder,
        )

    @override
    def __call__(
        self,
        obs: Tensor,
        upper_action: Tensor,
        hidden_encoder: Tensor | None = None,
        hidden_predictor: Tensor | None = None,
        no_len: bool = False,
    ) -> tuple[Distribution, Tensor, Tensor, Tensor, Tensor]:
        """Override __call__ with proper type annotations.

        See forward() method for full documentation.
        """
        return super().__call__(
            obs, upper_action, hidden_encoder, hidden_predictor, no_len=no_len
        )

    def forward_with_no_len(
        self,
        obs: Tensor,
        upper_action: Tensor,
        hidden_encoder: Tensor | None = None,
        hidden_predictor: Tensor | None = None,
    ) -> tuple[Distribution, Tensor, Tensor, Tensor, Tensor]:
        """Forward with data which has no len dimension.

        Args:
            obs: Input observation tensor (*batch, num_token, obs_dim).
            upper_action: Upper action tensor (*batch, num_token, action_dim).
            hidden_encoder: Optional hidden state for the encoder (*batch, depth, dim).
            hidden_predictor: Optional hidden state for the predictor (*batch, depth, dim).

        Returns:
            A tuple containing:
                - Distribution representing predicted observations.
                - Value estimates from the value head.
                - Latent representation from the encoder.
                - Next hidden state for the predictor.
                - Next hidden state for the encoder.
        """
        return self.forward(
            obs, upper_action, hidden_encoder, hidden_predictor, no_len=True
        )


class ObsUpperActionFlattenHead(nn.Module):
    """Head that processes observations and actions into latent
    representations.

    This head combines observation and action inputs, processes them
    through a stacked hidden state model, and outputs latent
    representations suitable for forward dynamics prediction.
    """

    @override
    def __init__(
        self,
        obs_info: ObsInfo,
        action_dim: int,
        output_dim: int,
    ) -> None:
        """Initialize the observation-action head.

        Args:
            obs_info: Configuration for observation processing.
            action_dim: Dimension of the action input.
            output_dim: Dimension of the output latent representation.
        """
        super().__init__()
        self.action_dim = action_dim
        self.obs_flatten = LerpStackedFeatures(
            obs_info.dim, obs_info.dim_hidden, obs_info.num_tokens
        )
        self.obs_action_proj = nn.Linear(
            obs_info.dim_hidden + action_dim,
            output_dim,
        )

    @override
    def forward(self, obs: Tensor, upper_action: Tensor | None) -> Tensor:
        """Process observations and actions into latent representations.

        Args:
            obs: Observation tensor. Shape is (*batch, len, num_tokens, obs_dim).
            action: Action tensor. Shape is (*batch, len, action_choices).

        Returns:
            Latent representation tensor of shape (*batch, len, output_dim).
        """
        obs = self.obs_flatten(obs)
        if upper_action is None:
            upper_action = torch.zeros(
                (*obs.shape[:-1], self.action_dim), dtype=obs.dtype, device=obs.device
            )
        return self.obs_action_proj(torch.cat((obs, upper_action), dim=-1))

    @override
    def __call__(self, obs: Tensor, action: Tensor) -> Tensor:
        """Override __call__ with proper type annotations.

        See forward() method for full documentation.
        """
        return super().__call__(obs, action)


class LatentObsUpperActionFlattenHead(nn.Module):
    """Head that processes observations and actions into latent
    representations.

    This head combines observation and action inputs, processes them
    through a stacked hidden state model, and outputs latent
    representations suitable for forward dynamics prediction.
    """

    @override
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        output_dim: int,
    ) -> None:
        """Initialize the observation-action head.

        Args:
            obs_info: Configuration for observation processing.
            action_dim: Dimension of the action input.
            output_dim: Dimension of the output latent representation.
        """
        super().__init__()
        self.action_dim = action_dim
        self.obs_action_proj = nn.Linear(
            obs_dim + action_dim,
            output_dim,
        )

    @override
    def forward(self, obs: Tensor, upper_action: Tensor | None) -> Tensor:
        """Process observations and actions into latent representations.

        Args:
            obs: Observation tensor. Shape is (*batch, len, obs_dim).
            action: Action tensor. Shape is (*batch, len, action_dim).

        Returns:
            Latent representation tensor of shape (*batch, len, output_dim).
        """
        if upper_action is None:
            upper_action = torch.zeros(
                (*obs.shape[:-1], self.action_dim), dtype=obs.dtype, device=obs.device
            )
        return self.obs_action_proj(torch.cat((obs, upper_action), dim=-1))

    @override
    def __call__(self, obs: Tensor, action: Tensor) -> Tensor:
        """Override __call__ with proper type annotations.

        See forward() method for full documentation.
        """
        return super().__call__(obs, action)
