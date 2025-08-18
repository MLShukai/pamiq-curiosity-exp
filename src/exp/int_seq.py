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


def exponential(base: int, power: float, num: int, shift: int = 0) -> list[int]:
    """Generate exponential sequence.

    Args:
        base: Base for exponential calculation.
        power: Power multiplier for each step.
        num: Number of elements.
        shift: Value added to each element.

    Returns:
        List with exponential progression.
    """
    return [shift + int(base ** (power * i)) for i in range(num)]


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
