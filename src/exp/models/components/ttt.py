from collections.abc import Callable
from typing import override

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .qlstm import FFNSwiGLU, RMSNorm
from .stacked_hidden_state import StackedHiddenStateSingleHidden


def silu_backward(x):
    return F.silu(x) + F.sigmoid(x) * (1 - F.silu(x))


class MultiHeadMLPTTTLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_hidden: int,
        num_head: int,
        base_lr: float,
        base_weight_decay: float,
    ):
        super().__init__()
        assert dim % num_head == 0, "dim must be divisible by num_head"
        assert dim_hidden % num_head == 0, "dim_hidden must be divisible by num_head"
        self.dim_hidden = dim_hidden
        self.num_head = num_head
        self.log_base_lr = nn.Parameter(torch.ones(num_head) * np.log(base_lr))
        self.fc_lr = nn.Linear(dim, num_head)
        self.log_base_weight_decay = nn.Parameter(
            torch.ones(num_head) * np.log(base_weight_decay)
        )
        self.fc_weight_decay = nn.Linear(dim, num_head)
        self.fc_query = nn.Linear(dim, dim)
        self.fc_key = nn.Linear(dim, dim)
        self.fc_value = nn.Linear(dim, dim)
        self.fc_out = nn.Linear(dim, dim)

    __call__: Callable[[Tensor, dict[str, Tensor]], tuple[Tensor, dict[str, Tensor]]]

    @override
    def forward(
        self, x: Tensor, hidden: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
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
        lr = torch.exp(self.log_base_lr)[None, :, None] * F.sigmoid(
            self.fc_lr(x)
        ).transpose(2, 1)  # (batch, num_head, length)
        log_weight_decay = torch.log(
            1
            - torch.exp(self.log_base_weight_decay)[None, :, None]
            * F.sigmoid(self.fc_weight_decay(x)).transpose(2, 1)
        )  # (batch, num_head, length)
        weight_decay_cross_chunk = torch.exp(
            torch.cumsum(log_weight_decay, dim=2)
        )  # (batch, num_head, length)
        weight_decay_inner_chunk = torch.exp(
            torch.cumsum(
                einops.repeat(log_weight_decay, "b n l -> b n m l", m=length).triu(1),
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
        grad_Z2 = Z2 - value  # (batch, num_head, length, head_dim)
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
            X1X1_ * weight_decay_inner_chunk
        )  # (batch, num_head, length, length)
        Z1__inner_chunk = -torch.einsum(
            "b n l h, b n l, b n l m -> b n m h", grad_Z1, lr, mask_X1X1_
        )  # (batch, num_head, length, head_dim_hidden)
        Z1__cross_chunk = torch.einsum(
            "b n h d, b n l d, b n l -> b n l h", W1_prev, X1_, weight_decay_cross_chunk
        )  # (batch, num_head, length, head_dim_hidden)
        Z1_ = (
            Z1__inner_chunk + Z1__cross_chunk
        )  # (batch, num_head, length, head_dim_hidden)
        W1_next_inner_chunk = -torch.einsum(
            "b n l h, b n l, b n l d -> b n h d",
            grad_Z1,
            lr * weight_decay_inner_chunk[:, :, :, -1],
            X1,
        )  # (batch, num_head, head_dim_hidden, head_dim)
        W1_next_cross_chunk = (
            W1_prev * weight_decay_cross_chunk[:, :, -1][:, :, None, None]
        )  # (batch, num_head, head_dim_hidden, head_dim)
        W1_next = (
            W1_next_inner_chunk + W1_next_cross_chunk
        )  # (batch, num_head, head_dim_hidden, head_dim)
        X2_ = F.silu(Z1_)  # (batch, num_head, length, head_dim_hidden)
        X2X2_ = torch.einsum(
            "b n l h, b n m h -> b n l m", X2, X2_
        )  # (batch, num_head, length, length)
        mask_X2X2_ = (
            X2X2_ * weight_decay_inner_chunk
        )  # (batch, num_head, length, length)
        Z2__inner_chunk = -torch.einsum(
            "b n l d, b n l, b n l m -> b n m d", grad_Z2, lr, mask_X2X2_
        )  # (batch, num_head, length, head_dim_hidden)
        Z2__cross_chunk = torch.einsum(
            "b n d h, b n l h, b n l -> b n l d", W2_prev, X2_, weight_decay_cross_chunk
        )  # (batch, num_head, length, head_dim)
        Z2_ = Z2__inner_chunk + Z2__cross_chunk  # (batch, num_head, length, head_dim)
        W2_next_inner_chunk = -torch.einsum(
            "b n l d, b n l, b n l h -> b n d h",
            grad_Z2,
            lr * weight_decay_inner_chunk[:, :, :, -1],
            X2,
        )  # (batch, num_head, head_dim, head_dim_hidden)
        W2_next_cross_chunk = (
            W2_prev * weight_decay_cross_chunk[:, :, -1][:, :, None, None]
        )  # (batch, num_head, head_dim, head_dim_hidden)
        W2_next = (
            W2_next_inner_chunk + W2_next_cross_chunk
        )  # (batch, num_head, head_dim, head_dim_hidden)
        hidden_next = {"W1": W1_next, "W2": W2_next}
        return self.fc_out(Z2_.transpose(2, 1).reshape(batch, length, dim)), hidden_next


class ChunkwiseTTT(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_hidden: int,
        num_head: int,
        base_lr: float,
        base_weight_decay: float,
        chunk_size: int,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.memory = MultiHeadMLPTTTLayer(
            dim, dim_hidden, num_head, base_lr, base_weight_decay
        )
        self.num_head = num_head
        self.head_dim = dim // num_head
        self.head_dim_hidden = dim_hidden // num_head

    __call__: Callable[
        [Tensor, dict[str, Tensor] | None], tuple[Tensor, dict[str, Tensor]]
    ]

    @override
    def forward(
        self, x: Tensor, hidden: dict[str, Tensor] | None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch, length, dim = x.shape

        if hidden is None:
            W1 = torch.zeros(
                (batch, self.num_head, self.head_dim_hidden, self.head_dim),
                device=x.device,
                dtype=x.dtype,
            )
            W2 = torch.zeros(
                (batch, self.num_head, self.head_dim, self.head_dim_hidden),
                device=x.device,
                dtype=x.dtype,
            )
            hidden = {"W1": W1, "W2": W2}
        else:
            hidden = {k: v.detach() for k, v in hidden.items()}

        input_chunks = x.split(self.chunk_size, dim=1)
        output_chunks = []
        for input_chunk in input_chunks:
            output_chunk, hidden = self.memory(input_chunk, hidden)
            output_chunks.append(output_chunk)

        return torch.cat(output_chunks, dim=1), hidden


class TTTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_ff_hidden: int,
        num_head: int,
        base_lr: float,
        base_weight_decay: float,
        chunk_size: int,
        dropout: float,
    ):
        super().__init__()
        self.memory = ChunkwiseTTT(
            dim, dim_ff_hidden, num_head, base_lr, base_weight_decay, chunk_size
        )
        self.ffn = FFNSwiGLU(dim, dim_ff_hidden)
        self.norm_memory = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @override
    def forward(
        self, x: Tensor, hidden: dict[str, Tensor] | None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        x_ = x
        x = self.norm_memory(x)
        x, hidden = self.memory(x, hidden)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_

        return x, hidden


class TTT(StackedHiddenStateSingleHidden):
    """TTT model using Multi-Head MLP TTT layers."""

    def __init__(
        self,
        depth: int,
        dim: int,
        dim_ff_hidden: int,
        num_head: int,
        base_lr: float,
        base_weight_decay: float,
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
                        base_weight_decay,
                        chunk_size,
                        dropout,
                    )
                    for _ in range(depth)
                ]
            )
        )
