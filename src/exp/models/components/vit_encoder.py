from typing import Optional, override

import torch
import torch.nn as nn
from torch import Tensor

from .qlstm import FFNSwiGLU, RMSNorm


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
            dim_in: Input feature dimension.
            dim: Input feature dimension.
            dim_hidden: Hidden dimension for the feedforward network.
            num_layers: Number of transformer encoder layers.
            num_heads: Number of attention heads.
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


class MultiHeadAttentionLayer(nn.Module):
    """Multi-head attention layer module."""

    @override
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_hidden: int,
    ) -> None:
        """Initialize the Multi-head attention layer module.

        Args:
            dim: Input feature dimension.
            num_heads: Number of attention heads.
            dim_hidden: Hidden dimension for the feedforward network.
        """
        super().__init__()
        self.fc_q = nn.Linear(dim, dim, bias=False)
        self.fc_k = nn.Linear(dim, dim, bias=False)
        self.fc_v = nn.Linear(dim, dim, bias=False)
        self.dim = dim
        self.num_heads = num_heads

    @override
    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        """Apply the multi-head attention layer.

        Args:
            x: The input tensor of shape (batch, len, dim).
            memory: The memory tensor from the encoder of shape (batch, mem_len, dim).
        Returns:
            The output tensor of shape (batch, len, dim).
        """
        q = self.fc_q(x).view(
            x.size(0), x.size(1), self.num_heads, self.dim // self.num_heads
        )
        k = self.fc_k(memory).view(
            memory.size(0), memory.size(1), self.num_heads, self.dim // self.num_heads
        )
        v = self.fc_v(memory).view(
            memory.size(0), memory.size(1), self.num_heads, self.dim // self.num_heads
        )
        attn_scores = (
            torch.einsum("bqhd,bkhd->bhqk", q, k) / (self.dim // self.num_heads) ** 0.5
        )
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = (
            torch.einsum("bhqk,bkhd->bqhd", attn_probs, v)
            .contiguous()
            .view(x.size(0), x.size(1), self.dim)
        )

        return attn_output


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer module."""

    @override
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_hidden: int,
    ) -> None:
        """Initialize the Transformer decoder layer module.

        Args:
            dim: Input feature dimension.
            num_heads: Number of attention heads.
            dim_hidden: Hidden dimension for the feedforward network.
        """
        super().__init__()
        self.norm_mha1 = RMSNorm(dim)
        self.self_mha_layer = MultiHeadAttentionLayer(
            dim=dim,
            num_heads=num_heads,
            dim_hidden=dim_hidden,
        )
        self.norm_mha2 = RMSNorm(dim)
        self.cross_mha_layer = MultiHeadAttentionLayer(
            dim=dim,
            num_heads=num_heads,
            dim_hidden=dim_hidden,
        )
        self.norm_ffn = RMSNorm(dim)
        self.ffn = FFNSwiGLU(dim, dim_hidden)

    @override
    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        """Apply the Transformer decoder layer.

        Args:
            x: The input tensor of shape (batch, len, dim).
            memory: The memory tensor from the encoder of shape (batch, mem_len, dim).
        Returns:
            The output tensor of shape (batch, len, dim).
        """
        x_ = x
        x = self.norm_mha1(x)
        x = self.self_mha_layer(x, x)
        x = x + x_

        x_ = x
        x = self.norm_mha2(x)
        x = self.cross_mha_layer(x, memory)
        x = x + x_

        x_ = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = x + x_

        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer module."""

    @override
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_hidden: int,
    ) -> None:
        """Initialize the Transformer encoder layer module.

        Args:
            dim: Input feature dimension.
            num_heads: Number of attention heads.
            dim_hidden: Hidden dimension for the feedforward network.
        """
        super().__init__()
        self.norm_mha = RMSNorm(dim)
        self.mha_layer = MultiHeadAttentionLayer(
            dim=dim,
            num_heads=num_heads,
            dim_hidden=dim_hidden,
        )
        self.norm_ffn = RMSNorm(dim)
        self.ffn = FFNSwiGLU(dim, dim_hidden)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply the Transformer encoder layer.

        Args:
            x: The input tensor of shape (batch, len, dim).
        Returns:
            The output tensor of shape (batch, len, dim).
        """
        x_ = x
        x = self.norm_mha(x)
        x = self.mha_layer(x, x)
        x = x + x_

        x_ = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = x + x_

        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder module for processing input tokens."""

    @override
    def __init__(
        self,
        dim_in: int,
        dim: int,
        dim_hidden: int,
        num_enc_layers: int,
        num_dec_layers: int,
        num_heads: int,
        num_tokens_enc: int,
        num_tokens_dec: int,
        positional_embedding_std: float = 0.02,
    ) -> None:
        """Initialize the Transformer encoder module.

        Args:
            dim_in: Input feature dimension.
            dim: Input feature dimension.
            dim_hidden: Hidden dimension for the feedforward network.
            num_enc_layers: Number of transformer encoder layers.
            num_dec_layers: Number of transformer decoder layers.
            num_heads: Number of attention heads.
            num_tokens_enc: Number of tokens to produce in the encoder stack.
            num_tokens_dec: Number of tokens to produce in the decoder stack.
        """
        super().__init__()
        self.fc_in = nn.Linear(dim_in, dim)
        self.positional_embedding_enc = nn.Parameter(
            torch.randn(num_tokens_enc, dim) * positional_embedding_std
        )
        self.positional_embedding_dec = nn.Parameter(
            torch.randn(num_tokens_dec, dim) * positional_embedding_std
        )
        self.encoder_layer_list = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim=dim,
                    num_heads=num_heads,
                    dim_hidden=dim_hidden,
                )
                for _ in range(num_enc_layers)
            ]
        )
        self.decoder_layer_list = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    dim=dim,
                    num_heads=num_heads,
                    dim_hidden=dim_hidden,
                )
                for _ in range(num_dec_layers)
            ]
        )
        self.enc_last_norm = RMSNorm(dim)
        self.dec_last_norm = RMSNorm(dim)
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
        x = x + self.positional_embedding_enc.unsqueeze(0)
        for layer in self.encoder_layer_list:
            x = layer(x)
        y = self.positional_embedding_dec.unsqueeze(0).repeat(x.size(0), 1, 1)
        for layer in self.decoder_layer_list:
            y = layer(y, x)
        y = y[:, 0, :]  # Take the representation of CLS token
        y = self.dec_last_norm(y)
        y = y.reshape(*batch_shape, -1)
        if no_batch:
            y = y.squeeze(0)
        return y
