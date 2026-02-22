"""Defines forward-dynamics policy models."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size, Tensor
from torch.distributions import Distribution
from torch.distributions.independent import Independent

from exp.agents.curiosity.ttt import Action, Hidden

from .components.beta import FCBetaHead
from .components.fc_scalar_head import FCScalarHead
from .components.identity import IdentityHead
from .components.multi_discretes import FCMultiCategoricalHead, MultiEmbeddings
from .components.multi_distributions import MultiDistributions
from .components.stacked_features import LerpStackedFeatures, ToStackedFeatures
from .components.stacked_hidden_state import (
    StackedHiddenState,
    StackedTTT,
)
from .components.vit_encoder import VitEncoder
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
        self.dim = dim

    def _flatten_obs_action(self, obs: Tensor, action: Tensor | None) -> Tensor:
        """Flatten and concat observation and action."""
        obs_flat = self.obs_flatten(obs)
        if action is None:
            return obs_flat.new_zeros((*obs_flat.shape[:-1], self.dim))
        else:
            action_flat = self.action_flatten(action)
            return self.obs_action_projection(
                torch.cat((obs_flat, action_flat), dim=-1)
            )

    @override
    def forward(
        self,
        obs: Tensor,
        action: Tensor | None,
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
        action: Tensor | None,
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


class TTTFDPiV(nn.Module):
    """Forward dynamics using StackedHiddenState model variants for core
    model."""

    @override
    def __init__(
        self,
        obs_dim_hidden: int,
        action_info: ActionInfo,
        external_action_dim: int,
        attention_dim: int,
        surprisal_dim: int,
        value_dim: int,
        internal_action_dim: int,
        body_state_dim: int,
        dim: int,
        obs_encoder: nn.Module,
        obs_time_mixer: StackedHiddenState,
        core_model: StackedTTT,
        surprisal_shape: Iterable[int],
    ) -> None:
        """Initialize the forward-dynamics policy-value model.

        Sets up the neural network components for predicting next observations
        from current observations and actions using a stacked hidden state model.

        Args:
            obs_info: Configuration for observation processing.
            obs_dim_hidden: Hidden dimension size for observation processing.
            action_info: Configuration for action processing.
            internal_action_dim: Dimension of the internal action space.
            internal_state_dim: Dimension of the internal state space.
            dim: Hidden dimension size for the core model input projection.
            obs_encoder: Encoder module for processing observations.
            core_model: The main stacked hidden state model that processes the
                concatenated observation-action features.
        """
        super().__init__()
        self.obs_flatten = obs_encoder
        self.obs_time_mixer = obs_time_mixer

        self.action_flatten = nn.Sequential(
            MultiEmbeddings(action_info.choices, action_info.dim, do_flatten=True),
            nn.Linear(action_info.dim * len(action_info.choices), external_action_dim),
        )
        self.dim = dim

        self.obs_dim_hidden = obs_dim_hidden
        self.external_action_dim = external_action_dim
        self.value_dim = value_dim

        self.attention_dim = attention_dim
        self.surprisal_dim = surprisal_dim
        self.internal_action_dim = internal_action_dim
        self.body_state_dim = body_state_dim
        self.internal_state_dim = (
            value_dim
            + attention_dim
            + surprisal_dim
            + internal_action_dim
            + body_state_dim
        )

        embedding_dim = obs_dim_hidden + external_action_dim + self.internal_state_dim
        self.embedding_dim = embedding_dim

        self.attention_projection = nn.Parameter(
            torch.randn(embedding_dim, attention_dim)
        )

        self.obs_projection = nn.Parameter(torch.randn(dim, obs_dim_hidden))
        self.external_action_projection = nn.Parameter(
            torch.randn(dim, self.external_action_dim)
        )
        self.value_projection = nn.Parameter(torch.randn(dim, value_dim))
        self.attention_projection = nn.Parameter(torch.randn(dim, attention_dim))
        self.surprisal_projection = nn.Parameter(torch.randn(dim, surprisal_dim))
        self.internal_action_projection = nn.Parameter(
            torch.randn(dim, internal_action_dim)
        )
        self.body_state_projection = nn.Parameter(torch.randn(dim, body_state_dim))

        self.attention_coef_logit_projection = nn.Parameter(
            torch.randn(embedding_dim, attention_dim)
        )
        self.surprisal_coef_logit_projection = nn.Parameter(
            torch.randn(*surprisal_shape, surprisal_dim)
        )

        self.external_action_head = FCMultiCategoricalHead(
            external_action_dim, action_info.choices
        )
        self.internal_state_head = IdentityHead(self.internal_state_dim)

        self.core_model = core_model

        self.value_head = FCScalarHead(value_dim, squeeze_scalar_dim=True)

    @override
    def forward(
        self,
        obs: Tensor,
        external_action: Tensor | None,
        internal_action: Tensor | None,
        body_state: Tensor | None,
        hidden: Hidden | None = None,
        *,
        no_len: bool = False,
    ) -> tuple[
        Tensor,
        Tensor,
        MultiDistributions,
        Tensor,
        Hidden,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """Forward pass to predict next observation prediction, policy
        distribution, value estimate, and surprisal.

        Args:
            obs: Current observation tensor. shape is (*batch, len, num_token, obs_dim)
            external_action: External action tensor. shape is (*batch, len, num_token, action_choices)
            internal_action: Internal action tensor. shape is (*batch, len, dim)
            hidden: Optional hidden state from previous timestep. shape is (*batch, depth, dim).
                If None, the hidden state is initialized to zeros

        Returns:
            A tuple containing:
                - Tensor representing predicted next observation.
                - Distribution representing the policy over actions.
                - Tensor representing the value estimate.
                - Updated hidden state tensor for use in next prediction.
                - Tensor representing the surprisal.
                - Tensor representing the shallow surprisal.
                - Tensor representing the deep surprisal.
        """
        hidden_time = None if hidden is None else hidden["time"]
        hidden_ttt = None if hidden is None else hidden["ttt"]

        if obs.ndim == 5:  # (*batch, len, channels, height, width)
            batch_size, seq_len = obs.shape[:2]
            obs = obs.view(batch_size * seq_len, *obs.shape[2:])
            obs_proj = self.obs_flatten(obs)
            obs_proj = obs_proj.view(batch_size, seq_len, *obs_proj.shape[1:])
        else:
            obs_proj = self.obs_flatten(obs)
        obs_emb, next_hidden_time = self.obs_time_mixer(
            obs_proj, hidden_time, no_len=no_len
        )

        external_action_emb = (
            self.action_flatten(external_action)
            if external_action is not None
            else obs_emb.new_zeros((*obs_emb.shape[:-1], self.external_action_dim))
        )
        if internal_action is not None and body_state is not None:
            if len(internal_action.shape) != len(body_state.shape):
                body_state = body_state.unsqueeze(-1)
            internal_state = torch.cat([internal_action, body_state], dim=-1)
        else:
            internal_state = obs_emb.new_zeros(
                (*obs_emb.shape[:-1], self.internal_state_dim)
            )

        attention_coef_logit = torch.einsum(
            "...i,ji->...j",
            F.softplus(
                internal_state[
                    ..., self.value_dim : self.value_dim + self.attention_dim
                ]
            ),
            self.attention_coef_logit_projection,
        )
        surprisal_coef_logit = torch.einsum(
            "...i,dhji->...dhj",
            F.softplus(
                internal_state[
                    ...,
                    self.value_dim + self.attention_dim : self.value_dim
                    + self.attention_dim
                    + self.surprisal_dim,
                ]
            ),
            self.surprisal_coef_logit_projection,
        )

        emb = torch.cat((obs_emb, external_action_emb, internal_state), dim=-1)
        emb_attention = emb * F.softmax(
            attention_coef_logit, dim=-1
        )  # Apply attention to the embedding
        emb_projection = torch.cat(
            [
                self.obs_projection,
                self.external_action_projection,
                self.value_projection,
                self.attention_projection,
                self.surprisal_projection,
                self.internal_action_projection,
                self.body_state_projection,
            ],
            dim=-1,
        )
        emb_core = torch.einsum("...i,ji->...j", emb_attention, emb_projection)

        emb_core_next, next_hidden_ttt, surprisal = self.core_model(
            emb_core, hidden_ttt, no_len=no_len
        )
        obs_hat = F.layer_norm(
            torch.einsum("...i,ij->...j", emb_core_next, self.obs_projection),
            self.obs_projection.shape[1:],
        )
        external_action_dist = self.external_action_head(
            F.layer_norm(
                torch.einsum(
                    "...i,ij->...j", emb_core_next, self.external_action_projection
                ),
                self.external_action_projection.shape[1:],
            )
        )
        latent_value = F.layer_norm(
            torch.einsum("...i,ij->...j", emb_core_next, self.value_projection),
            self.value_projection.shape[1:],
        )
        value = self.value_head(latent_value)

        next_attention = F.layer_norm(
            torch.einsum("...i,ij->...j", emb_core_next, self.attention_projection),
            self.attention_projection.shape[1:],
        )
        next_surprisal = F.layer_norm(
            torch.einsum("...i,ij->...j", emb_core_next, self.surprisal_projection),
            self.surprisal_projection.shape[1:],
        )
        next_internal_action = F.layer_norm(
            torch.einsum(
                "...i,ij->...j",
                emb_core_next,
                self.internal_action_projection,
            ),
            self.internal_action_projection.shape[1:],
        )

        internal_action_dist = Independent(
            self.internal_state_head(
                torch.cat(
                    [
                        latent_value,
                        next_attention,
                        next_surprisal,
                        next_internal_action,
                    ],
                    dim=-1,
                )
            ),
            1,
        )
        action_dist = MultiDistributions(external_action_dist, internal_action_dist)
        next_hidden = {
            "time": next_hidden_time,
            "ttt": next_hidden_ttt,
        }
        shallow_surprisal = (
            surprisal.detach()
            * F.softmax(surprisal_coef_logit.flatten(), dim=-1).view(*surprisal.shape)
        ).sum(dim=(-3, -2, -1))
        deep_surprisal = (
            surprisal.detach()
            * F.softmax(-surprisal_coef_logit.flatten(), dim=-1).view(*surprisal.shape)
        ).sum(dim=(-3, -2, -1))
        return (
            obs_emb,
            obs_hat,
            action_dist,
            value,
            next_hidden,
            surprisal,
            shallow_surprisal,
            deep_surprisal,
        )

    def forward_with_no_len(
        self,
        obs: Tensor,
        external_action: Tensor | None,
        internal_state: Tensor | None,
        body_state: Tensor | None,
        hidden: Hidden | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
        MultiDistributions,
        Tensor,
        Hidden,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """Forward with data which has no len dim. (for inference procedure.)

        Args:
            obs: Current observation tensor. shape is (*batch, num_token, obs_dim)
            external_action: Action tensor. shape is (*batch, num_token, action_choices)
            internal_state: Internal state tensor. shape is (*batch, dim)
            hidden: Optional hidden state from previous timestep. shape is (*batch, depth, dim).
                If None, the hidden state is initialized to zeros

        Returns:
            A tuple containing:
                - Tensor representing predicted next observation.
                - Distribution representing the policy over actions.
                - Tensor representing the value estimate.
                - Updated hidden state tensor for use in next prediction.
                - Tensor representing the surprisal.
                - Tensor representing the shallow surprisal.
                - Tensor representing the deep surprisal.
        """
        return self.forward(
            obs, external_action, internal_state, body_state, hidden, no_len=True
        )
