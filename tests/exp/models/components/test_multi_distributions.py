import pytest
import torch
from torch.distributions import Categorical, Normal
from torch.distributions.distribution import Distribution

from exp.models.components.multi_distributions import (
    MultiDistribusions,
)


class TestMultiDistributions:
    @pytest.fixture
    def distributions(self) -> list[Distribution]:
        batch_size = 8
        return [
            Categorical(logits=torch.zeros(batch_size, 3)),
            Normal(loc=torch.zeros(batch_size), scale=torch.ones(batch_size)),
        ]

    @pytest.fixture
    def multi_distributions(self, distributions) -> MultiDistribusions:
        return MultiDistribusions(distributions)

    def test_init(self, multi_distributions: MultiDistribusions):
        assert multi_distributions.batch_shape == (8,)

    def test_sample(self, multi_distributions: MultiDistribusions):
        samples = multi_distributions.sample([torch.Size(()), torch.Size(())])
        samples = list(samples)
        assert len(samples) == 2
        assert samples[0].shape == (8,)
        assert samples[1].shape == (8,)

    def test_log_prob(self, multi_distributions: MultiDistribusions):
        samples = multi_distributions.sample([torch.Size(()), torch.Size(())])
        log_prob = multi_distributions.log_prob(samples)
        assert log_prob.shape == (8,)

    def test_entropy(self, multi_distributions: MultiDistribusions):
        entropy = multi_distributions.entropy()
        assert entropy.shape == (8,)
