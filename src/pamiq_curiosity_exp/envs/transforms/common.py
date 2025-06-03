import functools
from collections.abc import Callable, Iterable
from typing import Any

import torch

type AnyFunction = Callable[..., Any]


def compose(
    *fns: AnyFunction, functions: Iterable[AnyFunction] | None = None
) -> AnyFunction:
    """Compose multiple functions into a single function.

    Creates a new function that applies the given functions in sequence from left to right.
    For functions f1, f2, f3, the composition applies as f3(f2(f1(x))).

    Args:
        *fns: Variable number of functions to compose.
        functions: Optional iterable of additional functions to include in composition.

    Returns:
        A composed function that applies all input functions in sequence.

    Examples:
        >>> add_one = lambda x: x + 1
        >>> multiply_two = lambda x: x * 2
        >>> composed = compose(add_one, multiply_two)
        >>> composed(5)  # (5 + 1) * 2 = 12
        12
    """
    if functions is not None:
        fns = (*fns, *functions)
    return lambda x: functools.reduce(lambda result, f: f(result), fns, x)


class Standardize:
    """Standardize input by removing mean and dividing by standard deviation.

    This module performs standardization (zero mean, unit variance) on
    the input tensor.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize the Standardize transform.

        Args:
            eps: Small value to add to standard deviation to avoid division by zero.
        """
        super().__init__()
        self.eps = eps

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """Apply standardization to the input tensor.

        Args:
            input: Input tensor to standardize.

        Returns:
            Standardized tensor with zero mean and unit variance.
        """
        if input.numel() <= 1:  # Can not compute std
            return torch.zeros_like(input)

        return (input - input.mean()) / (input.std() + self.eps)


class ToDtype:
    """Convert tensor to specified data type.

    This transform converts input tensors to the specified dtype. If no
    dtype is provided, it uses PyTorch's default dtype.
    """

    def __init__(self, dtype: torch.dtype | None = None) -> None:
        """Initialize the ToDtype transform.

        Args:
            dtype: Target data type. If None, uses PyTorch's default dtype.
        """
        if dtype is None:
            dtype = torch.get_default_dtype()
        self.dtype = dtype

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """Convert input tensor to the target data type.

        Args:
            input: Input tensor to convert.

        Returns:
            Tensor converted to the target dtype.
        """
        return input.type(self.dtype)


class ToDevice:
    """Move tensor to specified device.

    This transform moves input tensors to the specified device. If no
    device is provided, it uses PyTorch's default device.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        """Initialize the ToDevice transform.

        Args:
            device: Target device. If None, uses PyTorch's default device.
        """
        if device is None:
            device = torch.get_default_device()
        self.device = device

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """Move input tensor to the target device.

        Args:
            input: Input tensor to move.

        Returns:
            Tensor moved to the target device.
        """
        return input.to(self.device)
