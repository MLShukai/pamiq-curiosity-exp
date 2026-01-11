import math
from typing import override

import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal


class FCNormalHead(nn.Module):
    """The layer which returns the normal distribution.

    This module applies a linear transformation to the input features
    and returns a Normal distribution with the transformed values as the
    mean and a standard deviation.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        std_min: float = 1e-6,
        squeeze_feature_dim: bool = False,
    ) -> None:
        """Initialize the FCDeterministicNormalHead layer.

        Args:
            dim_in: Number of input features.
            dim_out: Number of output features representing the dimensionality of the distribution mean and standard deviation.
            std_min: Minimum standard deviation value for all dimensions of the distribution.
            squeeze_feature_dim: If True, removes the last dimension of the output.
                Only valid when dim_out=1. Useful when the output should not have
                a trailing singleton dimension.

        Raises:
            ValueError: If squeeze_feature_dim is True but dim_out is not 1.
        """
        super().__init__()

        if squeeze_feature_dim and dim_out != 1:
            raise ValueError("Can not squeeze feature dimension!")

        self.fc = nn.Linear(dim_in, dim_out)
        self.std = nn.Linear(dim_in, dim_out)
        self.softplus = nn.Softplus()
        self.std_min = std_min
        self.squeeze_feature_dim = squeeze_feature_dim

    @override
    def forward(self, x: Tensor) -> Normal:
        """Compute the normal distribution.

        Args:
            x: Input tensor of shape [..., dim_in].

        Returns:
            A Normal distribution with mean and standard deviation from the linear transformation
        """
        mean: Tensor = self.fc(x)
        std: Tensor = self.softplus(self.std(x)) + self.std_min
        if self.squeeze_feature_dim:
            mean = mean.squeeze(-1)
            std = std.squeeze(-1)

        return Normal(mean, std)
