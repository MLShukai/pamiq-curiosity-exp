from collections.abc import Iterable
from typing import override

import torch.nn as nn
from torch import Size, Tensor
from torch.distributions import Beta, constraints
from torch.types import _size


class FCBetaHead(nn.Module):
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
    def forward(self, x: Tensor) -> Beta:
        """Compute the beta distribution.

        Args:
            x: Input tensor of shape [..., dim_in].

        Returns:
            A Beta distribution with alpha and beta parameters from the linear transformation
        """
        alpha: Tensor = self.softplus(self.fc_alpha(x)) + 1.0
        beta: Tensor = self.softplus(self.fc_beta(x)) + 1.0
        return Beta(concentration1=alpha, concentration0=beta)
