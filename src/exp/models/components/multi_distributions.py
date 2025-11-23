from collections.abc import Iterable

import torch
from torch.distributions import Distribution


class MultiDistributions:
    """Collection of multiple independent distributions."""

    def __init__(self, *distributions: Distribution):
        """Constructs Multi Distributions from a collection of distributions.

        Args:
            distributions: A collection of distributions, where each distribution may
                have a different number of categories but must share the same batch shape.
        Raises:
            ValueError: If the collection is empty or if the batch shapes don't match.
        """
        if len(distributions) == 0:
            raise ValueError("Input distributions collection is empty.")
        first_dist = distributions[0]
        if not all(first_dist.batch_shape == d.batch_shape for d in distributions):
            raise ValueError("All batch shapes must be same.")
        self.dists = distributions

    @property
    def batch_shape(self) -> torch.Size:
        """Returns the batch shape of the distributions.

        Returns:
            The batch shape of the distributions.
        """
        return self.dists[0].batch_shape

    def sample(self, *sample_shapes: torch.Size) -> Iterable[torch.Tensor]:
        """Sample from each distribution.

        Args:
            sample_shapes: Shape of the samples to draw for each distribution.
        Returns:
            List of tensors of sampled actions for each distribution.
        """
        if not sample_shapes:
            sample_shapes = tuple(torch.Size() for _ in self.dists)
        elif len(sample_shapes) != len(self.dists):
            raise ValueError(
                f"Expected {len(self.dists)} sample shapes, but got {len(sample_shapes)}."
            )
        return [
            d.sample(sample_shape) for d, sample_shape in zip(self.dists, sample_shapes)
        ]

    def log_prob(self, *values: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions for each distribution.

        Args:
            value: List of tensors of actions for each distribution.
        Returns:
            Tensor of log probabilities with shape (*,).
        """
        if len(values) != len(self.dists):
            raise ValueError(
                f"Expected {len(self.dists)} values, but got {len(values)}."
            )
        return torch.stack(
            [d.log_prob(v) for d, v in zip(self.dists, values)], dim=-1
        ).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Compute entropy for each distribution.

        Returns:
            Tensor of entropies with batch shape
        """
        return torch.stack([d.entropy() for d in self.dists], dim=-1).sum(dim=-1)
