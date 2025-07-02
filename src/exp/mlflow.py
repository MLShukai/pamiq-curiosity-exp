from collections.abc import Mapping, Sequence
from typing import Any


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
