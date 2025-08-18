"""Tests for int_seq module."""

import pytest

from src.exp.int_seq import arithmetic, exponential, same


class TestSame:
    """Tests for same function."""

    @pytest.mark.parametrize(
        "value,num,expected",
        [
            (5, 3, [5, 5, 5]),
            (0, 4, [0, 0, 0, 0]),
            (-2, 2, [-2, -2]),
            (1, 0, []),
            (100, 1, [100]),
        ],
    )
    def test_same_basic_cases(self, value: int, num: int, expected: list[int]) -> None:
        """Test same function with various inputs."""
        result = same(value, num)
        assert result == expected


class TestExponential:
    """Tests for exponential function."""

    @pytest.mark.parametrize(
        "base,power,num,shift,expected",
        [
            (2, 1.0, 4, 0, [1, 2, 4, 8]),
            (3, 0.5, 3, 0, [1, 1, 3]),  # int(3^0.5) = int(1.732) = 1
            (2, 2.0, 3, 5, [6, 9, 21]),  # [5+1, 5+4, 5+16]
            (10, 0.0, 3, 0, [1, 1, 1]),  # 10^0 = 1 for all
            (2, 1.0, 0, 0, []),
        ],
    )
    def test_exponential_basic_cases(
        self, base: int, power: float, num: int, shift: int, expected: list[int]
    ) -> None:
        """Test exponential function with various parameters."""
        result = exponential(base, power, num, shift)
        assert result == expected

    def test_exponential_default_shift(self) -> None:
        """Test exponential function with default shift parameter."""
        result = exponential(2, 1.0, 3)
        assert result == [1, 2, 4]


class TestArithmetic:
    """Tests for arithmetic function."""

    @pytest.mark.parametrize(
        "start,diff,num,expected",
        [
            (0, 1, 5, [0, 1, 2, 3, 4]),
            (10, 2, 4, [10, 12, 14, 16]),
            (5, -1, 3, [5, 4, 3]),
            (0, 0, 3, [0, 0, 0]),
            (100, 10, 0, []),
            (-5, 3, 4, [-5, -2, 1, 4]),
        ],
    )
    def test_arithmetic_basic_cases(
        self, start: int, diff: int, num: int, expected: list[int]
    ) -> None:
        """Test arithmetic function with various progressions."""
        result = arithmetic(start, diff, num)
        assert result == expected
