"""Utility tools for model definitions."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m: nn.Module, init_std: float) -> None:
    """Initialize the weights with truncated normal distribution and zeros for
    biases.

    Args:
        m: Module to initialize.
        init_std: Standard deviation for the truncated normal initialization.
    """
    match m:
        case nn.Linear() | nn.Conv2d() | nn.ConvTranspose2d():
            nn.init.trunc_normal_(m.weight, std=init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        case nn.LayerNorm():
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        case _:
            pass


@dataclass
class ObsInfo:
    """Configuration for observation processing in forward dynamcis and policy.

    Attributes:
        dim: Input dimension of the observation.
        dim_hidden: Hidden dimension after feature transformation.
        num_tokens: Number of tokens of observation.
    """

    dim: int
    dim_hidden: int
    num_tokens: int


@dataclass
class ActionInfo:
    """Configuration for action processing in forward dynamics.

    Attributes:
        choices: List specifying the number of choices for each discrete action dimension.
        dim: Embedding dimension for each action component.
    """

    choices: list[int]
    dim: int


def layer_norm(x: torch.Tensor) -> torch.Tensor:
    """Applies layer normalization to final dim."""
    return F.layer_norm(x, x.shape[-1:])
