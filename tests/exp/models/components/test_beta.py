import pytest
import torch
from torch.distributions import Beta

from exp.models.components.beta import (
    FCBetaHead,
)


class TestFCBetaHead:
    def test_forward(self):
        """Test the forward pass returns a Normal distribution with expected
        shape."""
        layer = FCBetaHead(10, 20)
        out = layer(torch.randn(10))

        assert isinstance(out, Beta)
        assert out.sample().shape == (20,)

        assert layer(torch.randn(1, 2, 3, 10)).sample().shape == (1, 2, 3, 20)
