from typing import Optional, override

import torch
import torch.nn as nn
from torch import Tensor

from .qlstm import RMSNorm


class VitEncoder(nn.Module):
    """Vision Transformer (ViT) encoder module for processing input tokens."""

    @override
    def __init__(
        self,
        dim_in: int,
        dim: int,
        dim_hidden: int,
        num_layers: int,
        num_heads: int,
        num_tokens: int,
    ) -> None:
        """Initialize the ViT encoder module.

        Args:
            dim: Input feature dimension.
            dim_hidden: Hidden dimension for the feedforward network.
            num_layers: Number of transformer encoder layers.
            num_tokens: Number of tokens to produce in the stack.
        """
        super().__init__()
        self.fc_in = nn.Linear(dim_in, dim)
        self.positional_embedding = nn.Parameter(torch.randn(num_tokens + 1, dim))
        self.encoder_layer_list = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=dim_hidden,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.last_norm = RMSNorm(dim)
        self.dim = dim

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Encode input features using ViT-like transformer layers.

        Args:
            x: Input tensor of shape (*, num_tokens, dim) where * can be
                any number of batch dimensions.
        Returns:
            Encoded tensor of shape (*, dim).
        """
        no_batch = len(x.shape) == 2
        if no_batch:
            x = x.unsqueeze(0)
            no_batch = True
        batch_shape = x.shape[:-2]
        tokens, dim = x.shape[-2:]
        x = x.reshape(-1, tokens, dim)
        x = self.fc_in(x)
        x = torch.cat(
            [torch.zeros(x.size(0), 1, self.dim, device=x.device), x], dim=1
        )  # Add CLS token
        x = x + self.positional_embedding.unsqueeze(0)
        for layer in self.encoder_layer_list:
            x = layer(x)
        x = x[:, 0, :]  # Take the representation of CLS token
        x = self.last_norm(x)
        x = x.reshape(*batch_shape, -1)
        if no_batch:
            x = x.squeeze(0)
        return x
