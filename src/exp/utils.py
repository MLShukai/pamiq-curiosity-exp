import copy

import torch

type size_2d = int | tuple[int, int]


def size_2d_to_int_tuple(size: size_2d) -> tuple[int, int]:
    """Convert `size_2d` type to int tuple."""
    return (size, size) if isinstance(size, int) else (size[0], size[1])


def average_exponentially(sequence: torch.Tensor) -> torch.Tensor:
    """Averages a sequence tensor using exponential decay weighting.

    Computes a weighted average where weights increase exponentially for later
    elements in the sequence, giving more importance to recent values. The decay
    rate is automatically determined based on sequence length.

    The decay factor is calculated as e^(-1/N) where N is the sequence length,
    resulting in approximately 63% weight retention between consecutive steps.

    Args:
        sequence: Input tensor of shape (sequence_length, *), where * represents
            any number of additional dimensions.

    Returns:
        Tensor with the same shape as input except the first dimension is removed,
        containing the exponentially weighted average across the sequence dimension.

    ValueError:
        If input tensor is scalar tensor.
    """
    if sequence.ndim == 0:
        raise ValueError("Input sequence dimension must be larger than 1d!")
    decay = torch.e ** (-1 / sequence.size(0))

    decay_factors = decay ** torch.arange(
        len(sequence), device=sequence.device, dtype=sequence.dtype
    )
    for _ in range(sequence.ndim - 1):
        decay_factors.unsqueeze_(-1)

    equi_series_sum = (1 - decay ** len(sequence)) / (1 - decay)

    return torch.sum(sequence * decay_factors, dim=0) / equi_series_sum


def replicate[T](obj: T, num: int) -> list[T]:
    """Create multiple deep copies of an object.

    Args:
        obj: The object to replicate.
        num: Number of copies to create.

    Returns:
        List containing num deep copies of the object.

    Raises:
        ValueError: If num is negative.
    """
    if num < 0:
        raise ValueError(f"Number of copies must be non-negative, got {num}")
    return [copy.deepcopy(obj) for _ in range(num)]
