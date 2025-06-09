import pytest
import torch

from exp.envs.transforms.image import (
    ResizeAndCenterCrop,
)


class TestResizeAndCenterCrop:
    @pytest.mark.parametrize(
        "input_shape,target_size,expected_shape",
        [
            ((3, 100, 200), (50, 50), (3, 50, 50)),
            ((3, 200, 100), (50, 50), (3, 50, 50)),
            ((1, 3, 400, 300), (100, 100), (1, 3, 100, 100)),
            # Aspect ratio < 1
            ((3, 200, 200), (50, 100), (3, 50, 100)),
            # Aspect ratio > 1
            ((3, 300, 300), (100, 50), (3, 100, 50)),
        ],
    )
    def test_output_shape(self, input_shape, target_size, expected_shape):
        transform = ResizeAndCenterCrop(target_size)
        input_tensor = torch.randn(input_shape)
        output = transform(input_tensor)
        assert output.shape == expected_shape

    @pytest.mark.parametrize(
        "input_shape,error_message",
        [
            ((10,), "Input tensor must have at least 3 dimensions, got 1"),
            ((3, 0, 100), r"Input image dimensions must be non-zero, got \(0, 100\)"),
            ((3, 100, 0), r"Input image dimensions must be non-zero, got \(100, 0\)"),
        ],
    )
    def test_invalid_input_errors(self, input_shape, error_message):
        transform = ResizeAndCenterCrop((50, 50))
        input_tensor = torch.randn(input_shape)
        with pytest.raises(ValueError, match=error_message):
            transform(input_tensor)

    def test_content_preservation(self):
        transform = ResizeAndCenterCrop((50, 50))
        input_tensor = torch.zeros(3, 100, 100)
        input_tensor[:, 40:60, 40:60] = 1.0

        output = transform(input_tensor)
        center_mean = output[:, 20:30, 20:30].mean()
        assert center_mean > 0.9
