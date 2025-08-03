import pytest
import torch
from torch.distributions import Normal

from exp.models.components.normal import (
    FCNormalHead,
)


class TestFCNormalHead:
    def test_forward(self):
        """Test the forward pass returns a Normal distribution with expected
        shape."""
        layer = FCNormalHead(10, 20)
        out = layer(torch.randn(10))

        assert isinstance(out, Normal)
        assert out.sample().shape == (20,)

        assert layer(torch.randn(1, 2, 3, 10)).sample().shape == (1, 2, 3, 20)

    def test_squeeze_feature_dim(self):
        """Test the squeeze_feature_dim parameter works correctly."""
        with pytest.raises(ValueError):
            # out_features must be 1 when squeeze_feature_dim=True
            FCNormalHead(10, 2, squeeze_feature_dim=True)

        # squeeze_feature_dim default is False
        FCNormalHead(10, 2)

        net = FCNormalHead(10, 1, squeeze_feature_dim=True)
        x = torch.randn(10)
        out = net(x)
        assert out.sample().shape == ()
