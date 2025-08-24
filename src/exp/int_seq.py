"""Functions for creating numerical integer sequences."""


def same(value: int, num: int) -> list[int]:
    """Generate sequence with repeated value.

    Args:
        value: Value to repeat.
        num: Number of elements.

    Returns:
        List with repeated value.
    """
    return [value] * num


def geometric(init: float, ratio: float, num: int) -> list[int]:
    """Generate geometric sequence.

    Args:
        init: Initial value.
        ratio: Common ratio between consecutive elements.
        num: Number of elements.

    Returns:
        List with geometric progression.
    """
    return [int(init * ratio**i) for i in range(num)]


def arithmetic(start: int, diff: int, num: int) -> list[int]:
    """Generate arithmetic sequence.

    Args:
        start: First element value.
        diff: Difference between consecutive elements.
        num: Number of elements.

    Returns:
        List with arithmetic progression.
    """
    return [start + (diff * i) for i in range(num)]
