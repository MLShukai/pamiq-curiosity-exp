from collections.abc import Callable
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .qlstm import FFNSwiGLU, RMSNorm, scan
from .stacked_hidden_state import StackedHiddenState


class QGRULayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.fc_forget = nn.Linear(dim, dim)
        self.fc_input = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.fc_out = nn.Linear(dim, dim)

    __call__: Callable[[Tensor, Tensor | None], tuple[Tensor, Tensor]]

    @override
    def forward(self, x: Tensor, hidden: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Apply the QGRU layer.

        Args:
            x: The input tensor of shape (batch, len, dim).
            hidden: Optional hidden state tensor of shape (batch, dim_hidden). If None, initialized to zeros.
        Returns:
            The output tensor of shape (batch, len, dim) and the new hidden state tensor of shape (batch, len, dim_hidden).
        """

        batch, len, dim = x.shape

        if hidden is None:
            hidden = torch.zeros(batch, dim, device=x.device, dtype=x.dtype)

        remember = (
            F.sigmoid(self.fc_forget(x))
            * torch.linspace(0.0, 1.0, dim, device=x.device)[None, None, :]
        )
        forget = 1 - remember

        input = self.tanh(self.fc_input(x))
        h_inner_chunk = (
            scan(
                forget.transpose(2, 1).reshape(batch * dim, len),
                (input * remember).transpose(2, 1).reshape(batch * dim, len),
            )
            .reshape(batch, dim, len)
            .transpose(2, 1)
        )

        h = torch.addcmul(h_inner_chunk, hidden[:, None, :], forget.cumprod(1))
        y = self.fc_out(h)

        return y, h


class QGRUBlock(nn.Module):
    """QGRU Block, which consists of a QGRU layer and a feed forward
    network."""

    def __init__(self, dim: int, dim_ff_hidden: int, dropout: float):
        """Initialize the QGRU block.

        Args:
            dim: The number of features in the input.
            dim_ff_hidden: The number of features in the hidden layer.
            dropout: The dropout rate.
        """
        super().__init__()
        self.qgru = QGRULayer(dim)
        self.ffn = FFNSwiGLU(dim, dim_ff_hidden)
        self.norm_qgru = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @override
    def forward(self, x: Tensor, hidden: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Apply the QGRU block.

        Args:
            x: The input tensor of shape (batch, len, dim).
            hidden: Optional hidden state tensor of shape (batch, len, dim_hidden). If None, initialized to zeros.
        Returns:
            The output tensor of shape (batch, len, dim) and the new hidden state tensor of shape (batch, len, dim_hidden).
        """
        x_ = x
        x = self.norm_qgru(x)
        x, hidden = self.qgru(x, hidden)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_

        return x, hidden


class QGRU(StackedHiddenState):
    """QGRU, which is a stack of QGRU blocks."""

    def __init__(self, depth: int, dim: int, dim_ff_hidden: int, dropout: float):
        """Initialize the QGRU.

        Args:
            depth: The number of QGRU blocks.
            dim: The number of features in the input.
            dim_ff_hidden: The number of features in the hidden layer.
            dropout: The dropout rate.
        """
        super().__init__(
            nn.ModuleList(
                [QGRUBlock(dim, dim_ff_hidden, dropout) for _ in range(depth)]
            )
        )


def create_multiple(
    depth: int,
    dim_list: list[int],
    dim_ff_hidden_scale: float,
    dropout: float,
) -> nn.ModuleList:
    """Create multiple QGRU blocks.

    Args:
        depth: The number of QGRU blocks.
        dim_list: A list of dimensions for each QGRU block.
        dim_ff_hidden_scale: The scaling factor for the hidden layer dimensions.
        dropout: The dropout rate.

    Returns:
        A ModuleList containing the QGRU blocks.
    """
    dim_ff_hidden = [int(dim * dim_ff_hidden_scale) for dim in dim_list]
    return nn.ModuleList(
        [QGRU(depth, dim, dim_ff_hidden[i], dropout) for i, dim in enumerate(dim_list)]
    )
