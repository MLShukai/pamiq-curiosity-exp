from collections.abc import Iterable
from typing import override

import torch.nn as nn
from torch import Size, Tensor
from torch.distributions import Beta, constraints
from torch.types import _size


class BetaMinusOneToOne(Beta):
    """Beta distribution scaled to the range [-1, 1].

    This class extends the standard Beta distribution to map its output
    from the range [0, 1] to [-1, 1]. This is useful in scenarios where
    actions or outputs need to be within the range of -1 to 1.
    """

    support = constraints.interval(-1.0, 1.0)

    @override
    def sample(self, sample_shape: _size = ()) -> Tensor:
        raw_sample = super().rsample(sample_shape)
        return raw_sample * 2.0 - 1.0

    @override
    def rsample(self, sample_shape: _size = ()) -> Tensor:
        raw_sample = super().rsample(sample_shape)
        return raw_sample * 2.0 - 1.0

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        # Transform value from [-1, 1] to [0, 1]
        transformed_value = (value + 1.0) / 2.0
        # Compute log probability using the base Beta distribution
        log_prob = super().log_prob(transformed_value)
        return log_prob


class FCBetaMOTOHead(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
    ):
        """Initialize the FCBetaHead layer.

        Args:
            dim_in: Number of input features.
            dim_out: Number of output features representing the dimensionality of the distribution parameters.
        """
        super().__init__()

        self.fc_alpha = nn.Linear(dim_in, dim_out)
        self.fc_beta = nn.Linear(dim_in, dim_out)
        self.softplus = nn.Softplus()

    @override
    def forward(self, x: Tensor) -> BetaMinusOneToOne:
        """Compute the beta distribution.

        Args:
            x: Input tensor of shape [..., dim_in].

        Returns:
            A Beta distribution with alpha and beta parameters from the linear transformation
        """
        alpha: Tensor = self.softplus(self.fc_alpha(x)) + 1.0
        beta: Tensor = self.softplus(self.fc_beta(x)) + 1.0
        return BetaMinusOneToOne(concentration1=alpha, concentration0=beta)
