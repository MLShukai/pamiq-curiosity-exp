from collections.abc import Callable
from typing import override

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .qlstm import FFNSwiGLU, RMSNorm
from .stacked_hidden_state import StackedTTT


def silu_backward(x):
    return F.silu(x) + F.sigmoid(x) * (1 - F.silu(x))


class MultiHeadMLPTTTLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_hidden: int,
        num_head: int,
        base_lr: tuple[float, float],
    ):
        super().__init__()
        assert dim % num_head == 0, "dim must be divisible by num_head"
        assert dim_hidden % num_head == 0, "dim_hidden must be divisible by num_head"
        self.dim_hidden = dim_hidden
        self.num_head = num_head
        self.log_base_lr_1 = nn.Parameter(
            torch.linspace(np.log(base_lr[0]), np.log(base_lr[1]), num_head)
        )
        self.log_base_lr_2 = nn.Parameter(
            torch.linspace(np.log(base_lr[0]), np.log(base_lr[1]), num_head)
        )
        self.fc_lr_1 = nn.Linear(dim, num_head)
        self.fc_weight_decay_1 = nn.Linear(dim, num_head)
        self.fc_lr_2 = nn.Linear(dim, num_head)
        self.fc_weight_decay_2 = nn.Linear(dim, num_head)
        self.fc_query = nn.Linear(dim, dim)
        self.fc_key = nn.Linear(dim, dim)
        self.fc_value = nn.Linear(dim, dim)
        self.fc_out = nn.Linear(dim, dim)

    __call__: Callable[
        [Tensor, dict[str, Tensor]], tuple[Tensor, dict[str, Tensor], Tensor]
    ]

    @override
    def forward(
        self, x: Tensor, hidden: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        batch, length, dim = x.shape
        num_head = self.num_head
        head_dim = dim // num_head
        W1_prev = hidden["W1"]  # (batch, num_head, head_dim_hidden, head_dim)
        W2_prev = hidden["W2"]  # (batch, num_head, head_dim, head_dim_hidden)
        query = (
            self.fc_query(x).view(batch, length, num_head, head_dim).transpose(2, 1)
        )  # (batch, num_head, length, head_dim)
        key = (
            self.fc_key(x).view(batch, length, num_head, head_dim).transpose(2, 1)
        )  # (batch, num_head, length, head_dim)
        value = (
            self.fc_value(x).view(batch, length, num_head, head_dim).transpose(2, 1)
        )  # (batch, num_head, length, head_dim)

        lr_1 = torch.exp(self.log_base_lr_1)[None, :, None] * F.sigmoid(
            self.fc_lr_1(x)
        ).transpose(2, 1)  # (batch, num_head, length)
        log_weight_decay_1 = torch.log(
            1
            - torch.exp(self.log_base_lr_1)[None, :, None]
            * F.sigmoid(self.fc_weight_decay_1(x)).transpose(2, 1)
        )  # (batch, num_head, length)
        weight_decay_cross_chunk_1 = torch.exp(
            torch.cumsum(log_weight_decay_1, dim=2)
        )  # (batch, num_head, length)
        weight_decay_inner_chunk_1 = torch.exp(
            torch.cumsum(
                einops.repeat(log_weight_decay_1, "b n l -> b n m l", m=length).triu(1),
                dim=3,
            )
        ).triu()  # (batch, num_head, length, length)

        lr_2 = torch.exp(self.log_base_lr_2)[None, :, None] * F.sigmoid(
            self.fc_lr_2(x)
        ).transpose(2, 1)  # (batch, num_head, length)
        log_weight_decay_2 = torch.log(
            1
            - torch.exp(self.log_base_lr_2)[None, :, None]
            * F.sigmoid(self.fc_weight_decay_2(x)).transpose(2, 1)
        )  # (batch, num_head, length)
        weight_decay_cross_chunk_2 = torch.exp(
            torch.cumsum(log_weight_decay_2, dim=2)
        )  # (batch, num_head, length)
        weight_decay_inner_chunk_2 = torch.exp(
            torch.cumsum(
                einops.repeat(log_weight_decay_2, "b n l -> b n m l", m=length).triu(1),
                dim=3,
            )
        ).triu()  # (batch, num_head, length, length)

        X1 = key  # (batch, num_head, length, head_dim)
        Z1 = torch.einsum(
            "b n h d, b n l d -> b n l h", W1_prev, X1
        )  # (batch, num_head, length, head_dim_hidden)
        X2 = F.silu(Z1)  # (batch, num_head, length, head_dim_hidden)
        Z2 = torch.einsum(
            "b n d h, b n l h -> b n l d", W2_prev, X2
        )  # (batch, num_head, length, head_dim)

        surprisal = 0.5 * ((Z2 - value) ** 2).mean(dim=-1)  # (batch, num_head, length)

        grad_Z2 = (Z2 - value) / head_dim  # (batch, num_head, length, head_dim)

        grad_X2 = torch.einsum(
            "b n d h, b n l d -> b n l h", W2_prev, grad_Z2
        )  # (batch, num_head, length, head_dim_hidden)
        grad_Z1 = (
            silu_backward(Z1) * grad_X2
        )  # (batch, num_head, length, head_dim_hidden)
        # grad_X1 = torch.einsum("b n h d, b n l h -> b n l d", W1_prev, grad_Z1) # (batch, num_head, length, head_dim)
        X1_ = query
        X1X1_ = torch.einsum(
            "b n l d, b n m d -> b n l m", X1, X1_
        )  # (batch, num_head, length, length)
        mask_X1X1_ = (
            X1X1_ * weight_decay_inner_chunk_1
        )  # (batch, num_head, length, length)
        Z1__inner_chunk = -torch.einsum(
            "b n l h, b n l, b n l m -> b n m h", grad_Z1, lr_1, mask_X1X1_
        )  # (batch, num_head, length, head_dim_hidden)
        Z1__cross_chunk = torch.einsum(
            "b n h d, b n l d, b n l -> b n l h",
            W1_prev,
            X1_,
            weight_decay_cross_chunk_1,
        )  # (batch, num_head, length, head_dim_hidden)
        Z1_ = (
            Z1__inner_chunk + Z1__cross_chunk
        )  # (batch, num_head, length, head_dim_hidden)
        W1_next_inner_chunk = -torch.einsum(
            "b n l h, b n l, b n l d -> b n h d",
            grad_Z1,
            lr_1 * weight_decay_inner_chunk_1[:, :, :, -1],
            X1,
        )  # (batch, num_head, head_dim_hidden, head_dim)
        W1_next_cross_chunk = (
            W1_prev * weight_decay_cross_chunk_1[:, :, -1][:, :, None, None]
        )  # (batch, num_head, head_dim_hidden, head_dim)
        W1_next = (
            W1_next_inner_chunk + W1_next_cross_chunk
        )  # (batch, num_head, head_dim_hidden, head_dim)
        X2_ = F.silu(Z1_)  # (batch, num_head, length, head_dim_hidden)
        X2X2_ = torch.einsum(
            "b n l h, b n m h -> b n l m", X2, X2_
        )  # (batch, num_head, length, length)
        mask_X2X2_ = (
            X2X2_ * weight_decay_inner_chunk_2
        )  # (batch, num_head, length, length)
        Z2__inner_chunk = -torch.einsum(
            "b n l d, b n l, b n l m -> b n m d", grad_Z2, lr_2, mask_X2X2_
        )  # (batch, num_head, length, head_dim_hidden)
        Z2__cross_chunk = torch.einsum(
            "b n d h, b n l h, b n l -> b n l d",
            W2_prev,
            X2_,
            weight_decay_cross_chunk_2,
        )  # (batch, num_head, length, head_dim)
        Z2_ = Z2__inner_chunk + Z2__cross_chunk  # (batch, num_head, length, head_dim)
        W2_next_inner_chunk = -torch.einsum(
            "b n l d, b n l, b n l h -> b n d h",
            grad_Z2,
            lr_2 * weight_decay_inner_chunk_2[:, :, :, -1],
            X2,
        )  # (batch, num_head, head_dim, head_dim_hidden)
        W2_next_cross_chunk = (
            W2_prev * weight_decay_cross_chunk_2[:, :, -1][:, :, None, None]
        )  # (batch, num_head, head_dim, head_dim_hidden)
        W2_next = (
            W2_next_inner_chunk + W2_next_cross_chunk
        )  # (batch, num_head, head_dim, head_dim_hidden)
        hidden_next = {"W1": W1_next, "W2": W2_next}
        return (
            self.fc_out(Z2_.transpose(2, 1).reshape(batch, length, dim)),
            hidden_next,
            surprisal.transpose(-2, -1),
        )


class ChunkwiseTTT(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_hidden: int,
        num_head: int,
        base_lr: tuple[float, float],
        chunk_size: int,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.memory = MultiHeadMLPTTTLayer(dim, dim_hidden, num_head, base_lr)
        self.num_head = num_head
        self.head_dim = dim // num_head
        self.head_dim_hidden = dim_hidden // num_head

    __call__: Callable[
        [Tensor, dict[str, Tensor] | None], tuple[Tensor, dict[str, Tensor], Tensor]
    ]

    @override
    def forward(
        self, x: Tensor, hidden: dict[str, Tensor] | None
    ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        batch, length, dim = x.shape

        if hidden is None:
            W1 = (
                torch.randn(
                    (batch, self.num_head, self.head_dim_hidden, self.head_dim),
                    device=x.device,
                    dtype=x.dtype,
                )
                * self.head_dim**-0.5
            )
            W2 = (
                torch.randn(
                    (batch, self.num_head, self.head_dim, self.head_dim_hidden),
                    device=x.device,
                    dtype=x.dtype,
                )
                * self.head_dim_hidden**-0.5
            )
            hidden = {"W1": W1, "W2": W2}
        else:
            hidden = {k: v.detach() for k, v in hidden.items()}

        input_chunks = x.split(self.chunk_size, dim=1)
        output_chunks = []
        surprisal_chunks = []
        for input_chunk in input_chunks:
            output_chunk, hidden, surprisal = self.memory(input_chunk, hidden)
            output_chunks.append(output_chunk)
            surprisal_chunks.append(surprisal)
        return (
            torch.cat(output_chunks, dim=1),
            hidden,
            torch.cat(surprisal_chunks, dim=1),
        )


class TTTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_ff_hidden: int,
        num_head: int,
        base_lr: tuple[float, float],
        chunk_size: int,
        dropout: float,
    ):
        super().__init__()
        self.memory = ChunkwiseTTT(dim, dim_ff_hidden, num_head, base_lr, chunk_size)
        self.ffn = FFNSwiGLU(dim, dim_ff_hidden)
        self.norm_memory = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @override
    def forward(
        self, x: Tensor, hidden: dict[str, Tensor] | None
    ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        x_ = x
        x = self.norm_memory(x)
        x, hidden, surprisal = self.memory(x, hidden)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_

        return x, hidden, surprisal


class TTT(StackedTTT):
    """TTT model using Multi-Head MLP TTT layers."""

    def __init__(
        self,
        depth: int,
        dim: int,
        dim_ff_hidden: int,
        num_head: int,
        base_lr: tuple[float, float],
        chunk_size: int,
        dropout: float,
    ):
        super().__init__(
            nn.ModuleList(
                [
                    TTTBlock(
                        dim,
                        dim_ff_hidden,
                        num_head,
                        base_lr,
                        chunk_size,
                        dropout,
                    )
                    for _ in range(depth)
                ]
            )
        )
