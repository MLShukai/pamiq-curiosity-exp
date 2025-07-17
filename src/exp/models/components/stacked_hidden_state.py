from typing import override

import torch
import torch.nn as nn
from torch import Tensor


class StackedHiddenState(nn.Module):
    """Stacked hidden state for a sequence of modules.

    This module takes a sequence of modules and applies them to the
    input tensor and the hidden state tensor. It stacks the hidden
    states from each module and returns the output tensor and the
    stacked hidden state tensor.
    """

    def __init__(self, module_list: nn.ModuleList):
        """Initialize the StackedHiddenState module.

        Args:
            module_list: A list of modules to apply to the input tensor and the hidden state tensor.
        """
        super().__init__()
        self.module_list = module_list

    @override
    def forward(
        self,
        x: Tensor,
        hidden_stack: Tensor | None = None,
        *,
        no_len: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Apply the stacked hidden state module.

        Args:
            x: The input tensor of shape (*batch, len, dim) or (len, dim) or (dim).
            hidden_stack: Optional hidden state tensor of shape (*batch, depth, dim) or (depth, dim).
                If None, the hidden state is initialized to zeros.
            no_len: If True, performs single-step processing.

        Returns:
            The output tensor of shape (*batch, len, dim) or (len, dim) or (dim).
            The stacked hidden state tensor of shape (*batch, depth, len, dim) or (depth, len, dim) or (*batch, depth, dim) or (depth, dim).
        """
        if no_len:
            x = x.unsqueeze(-2)
        no_batch = x.ndim < 3
        if no_batch:
            x = x.unsqueeze(0)
            if hidden_stack is not None:
                hidden_stack = hidden_stack.unsqueeze(0)

        if hidden_stack is not None and x.shape[:-2] != hidden_stack.shape[:-2]:
            raise ValueError("Batch shape mismatch between x and hidden_stack")
        if hidden_stack is not None and x.size(-1) != hidden_stack.size(-1):
            raise ValueError("Feature dim mismatch between x and hidden_stack")

        batch_shape = x.shape[:-2]
        x = x.reshape(-1, *x.shape[len(batch_shape) :])
        if hidden_stack is not None:
            hidden_stack = hidden_stack.reshape(
                -1, *hidden_stack.shape[len(batch_shape) :]
            )

        hidden_out_list = []
        for i, module in enumerate(self.module_list):
            hidden = hidden_stack[:, i, :] if hidden_stack is not None else None
            x, hidden_out = module(x, hidden)
            hidden_out_list.append(hidden_out)

        hidden_out_stack = torch.stack(hidden_out_list).transpose(1, 0)

        x = x.view(*batch_shape, *x.shape[1:])
        hidden_out_stack = hidden_out_stack.view(
            *batch_shape, *hidden_out_stack.shape[1:]
        )

        if no_batch:
            x = x.squeeze(0)
            hidden_out_stack = hidden_out_stack.squeeze(0)

        if no_len:
            x = x.squeeze(-2)
            hidden_out_stack = hidden_out_stack.squeeze(-2)

        return x, hidden_out_stack

    def forward_with_no_len(
        self,
        x: Tensor,
        hidden_stack: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply the stacked hidden state module with data has no len dim.

        Args:
            x: The input tensor of shape (*batch, dim) or (dim, )
            hidden_stack: Optional hidden state tensor of shape (*batch, depth, dim) or (depth, dim).
                If None, the hidden state is initialized to zeros.
        Returns:
            The output tensor of shape (*batch, dim) or (dim, )
            The stacked hidden state tensor of shape (*batch, depth, dim) or (depth, dim).
        """
        return self.forward(x, hidden_stack, no_len=True)
