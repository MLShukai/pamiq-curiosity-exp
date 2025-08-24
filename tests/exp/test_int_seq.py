"""Tests for int_seq module."""

import pytest

from src.exp.int_seq import arithmetic, geometric, same


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


class TestGeometric:
    """Tests for geometric function."""

    @pytest.mark.parametrize(
        "init,ratio,num,expected",
        [
            (1.0, 2.0, 4, [1, 2, 4, 8]),
            (10.0, 0.5, 4, [10, 5, 2, 1]),
            (3.0, 3.0, 3, [3, 9, 27]),
            (100.0, 0.1, 3, [100, 10, 1]),
            (2.0, 1.0, 3, [2, 2, 2]),  # ratio=1 gives constant sequence
            (5.0, 2.0, 0, []),
            (0.0, 2.0, 3, [0, 0, 0]),  # init=0 gives all zeros
        ],
    )
    def test_geometric_basic_cases(
        self, init: float, ratio: float, num: int, expected: list[int]
    ) -> None:
        """Test geometric function with various parameters."""
        result = geometric(init, ratio, num)
        assert result == expected

    def test_geometric_with_fractional_results(self) -> None:
        """Test geometric function with values that get truncated."""
        result = geometric(10.0, 0.3, 4)
        # 10, 3, 0.9, 0.27 -> [10, 3, 0, 0] when converted to int
        assert result == [10, 3, 0, 0]


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
