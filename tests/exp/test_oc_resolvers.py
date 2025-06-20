import pytest
import torch
from omegaconf import OmegaConf

from exp.oc_resolvers import register_custom_resolvers


@pytest.mark.parametrize(
    "key,value,expected",
    [
        ("eval", "${python.eval: 1 + 2 * 3 / 4}", 2.5),
        ("device", "${torch.device: cuda:0}", torch.device("cuda:0")),
        ("dtype", "${torch.dtype: complex64}", torch.complex64),
    ],
)
def test_resolovers(key, value, expected):
    register_custom_resolvers()

    assert OmegaConf.create({key: value})[key] == expected
