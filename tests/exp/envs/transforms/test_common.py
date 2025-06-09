import pytest
import torch

from exp.envs.transforms.common import (
    Standardize,
    ToDevice,
    ToDtype,
    compose,
)
from tests.helpers import parametrize_device


class TestCompose:
    def test_single_function(self):
        """Test composition with a single function."""

        def add_one(x):
            return x + 1

        composed = compose(add_one)

        assert composed(5) == 6
        assert composed(0) == 1

    def test_multiple_functions(self):
        """Test composition with multiple functions."""

        def add_one(x):
            return x + 1

        def multiply_two(x):
            return x * 2

        def subtract_three(x):
            return x - 3

        composed = compose(add_one, multiply_two, subtract_three)
        result = composed(5)  # ((5 + 1) * 2) - 3 = 9

        assert result == 9

    def test_function_application_order(self):
        """Test that functions are applied left to right."""

        def append_a(x):
            return x + "a"

        def append_b(x):
            return x + "b"

        def append_c(x):
            return x + "c"

        composed = compose(append_a, append_b, append_c)
        result = composed("start")

        assert result == "startabc"

    def test_empty_composition(self):
        """Test composition with no functions returns identity."""
        composed = compose()

        assert composed(42) == 42
        assert composed("test") == "test"
        assert composed([1, 2, 3]) == [1, 2, 3]

    def test_functions_parameter(self):
        """Test using the functions parameter."""

        def add_one(x):
            return x + 1

        def multiply_two(x):
            return x * 2

        def subtract_one(x):
            return x - 1

        def multiply_three(x):
            return x * 3

        functions_list = [subtract_one, multiply_three]

        composed = compose(add_one, multiply_two, functions=functions_list)
        result = composed(5)  # (((5 + 1) * 2) - 1) * 3 = 33

        assert result == 33


class TestStandardize:
    @pytest.mark.parametrize(
        "input_shape",
        [(100,), (3, 32, 32), (5, 10, 10), (2, 3, 64, 64)],
    )
    def test_standardization(self, input_shape):
        transform = Standardize()
        input_tensor = torch.randn(input_shape) * 5 + 3
        output = transform(input_tensor)

        assert output.mean().item() == pytest.approx(0.0, abs=1e-6)
        assert output.std().item() == pytest.approx(1.0, abs=1e-6)

    def test_constant_tensor(self):
        transform = Standardize(eps=1e-8)
        input_tensor = torch.ones(10, 10) * 5.0
        output = transform(input_tensor)

        assert torch.allclose(output, torch.zeros_like(output))

    def test_single_value_tensor(self):
        transform = Standardize()
        input_tensor = torch.tensor([42.0])
        output = transform(input_tensor)

        assert output.shape == (1,)
        assert output.item() == pytest.approx(0.0)

    def test_empty_tensor(self):
        transform = Standardize()
        input_tensor = torch.tensor([])
        output = transform(input_tensor)

        assert output.shape == (0,)


class TestToDtype:
    @pytest.mark.parametrize(
        "target_dtype",
        [torch.float32, torch.float64, torch.bfloat16],
    )
    def test_default_dtype(self, target_dtype):
        """Test conversion using PyTorch's default dtype."""
        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(target_dtype)
            transform = ToDtype()
            input_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
            output = transform(input_tensor)

            assert output.dtype == torch.get_default_dtype()
            assert torch.equal(output, input_tensor.type(torch.get_default_dtype()))
        finally:
            torch.set_default_dtype(default_dtype)

    @pytest.mark.parametrize(
        "target_dtype",
        [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool],
    )
    def test_explicit_dtype(self, target_dtype):
        """Test conversion to explicit dtypes."""
        transform = ToDtype(target_dtype)
        input_tensor = torch.randn(3, 4)
        output = transform(input_tensor)

        assert output.dtype == target_dtype
        assert output.shape == input_tensor.shape


class TestToDevice:
    @parametrize_device
    def test_default_device(self, device):
        """Test moving to PyTorch's default device."""
        default_device = torch.get_default_device()
        try:
            torch.set_default_device(device)
            transform = ToDevice()
            input_tensor = torch.tensor([1, 2, 3])
            output = transform(input_tensor)

            assert output.device == torch.get_default_device()
        finally:
            torch.set_default_device(default_device)

    @parametrize_device
    def test_explicit_device(self, device):
        """Test moving to CPU device explicitly."""
        transform = ToDevice(device)
        input_tensor = torch.tensor([1, 2, 3])
        output = transform(input_tensor)

        assert output.device == device
