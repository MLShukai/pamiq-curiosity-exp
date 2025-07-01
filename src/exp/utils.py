from collections.abc import Mapping, Sequence
from typing import Any

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


def flatten_config(
    config: Mapping[str, Any] | Sequence[Any],
    parent_key: str = "",
    separator: str = ".",
) -> dict[str, Any]:
    """Flatten nested configuration mappings and sequences into a single-level
    dictionary.

    Recursively processes nested dictionaries and sequences to create a flat
    dictionary with keys joined by the specified separator.

    Args:
        config: The configuration to flatten. Can be a mapping (dict) or sequence (list/tuple).
        parent_key: The parent key prefix for nested values.
        separator: The separator to use when joining keys.

    Returns:
        A flattened dictionary with all nested values converted for MLflow compatibility.
    """
    flat_cfg: dict[str, Any] = {}

    match config:
        case Mapping():
            iterator = config.items()
        case Sequence():
            iterator = enumerate(config)

    for key, value in iterator:
        new_key = f"{parent_key}{separator}{key}" if parent_key else str(key)
        if not isinstance(value, str) and isinstance(value, Mapping | Sequence):
            flat_cfg.update(flatten_config(value, new_key, separator))
        else:
            flat_cfg[new_key] = convert_value_for_mlflow(value)

    return flat_cfg


def convert_value_for_mlflow(value: Any) -> str | int | float | bool:
    """Convert values to MLflow-compatible types.

    MLflow logging supports str, int, float, and bool types. Other types
    are converted to string representation.

    Args:
        value: The value to convert.

    Returns:
        The value if it's already MLflow-compatible, otherwise its string representation.
    """
    if isinstance(value, str | int | float | bool):
        return value
    return str(value)
