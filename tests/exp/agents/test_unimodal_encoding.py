import pytest
import torch
from pamiq_core.testing import (
    connect_components,
    create_mock_buffer,
    create_mock_models,
)
from pytest_mock import MockerFixture

from exp.agents.unimodal_encoding import UnimodalEncodingAgent
from exp.data import BufferName, DataKey
from exp.models import ModelName


class TestUnimodalEncodingAgent:
    """Tests for the UnimodalEncodingAgent class."""

    @pytest.fixture
    def models(self):
        training_model, _ = create_mock_models()

        return {ModelName.IMAGE_JEPA_TARGET_ENCODER: training_model}

    @pytest.fixture
    def buffers(self):
        return {BufferName.IMAGE: create_mock_buffer()}

    def test_initilization(self, models, buffers):
        """Test initialization of the agent."""

        agent = UnimodalEncodingAgent()

        components = connect_components(agent, buffers=buffers, models=models)

        assert (
            agent.encoder
            is components.inference_models[ModelName.IMAGE_JEPA_TARGET_ENCODER]
        )
        assert agent.collector is components.data_collectors[BufferName.IMAGE]

    @pytest.mark.parametrize(
        "input_shape,expected_output_shape",
        [
            ((3, 32, 32), (16, 1, 1)),
            ((2, 1600), (8, 1)),
        ],
    )
    def test_step(
        self, input_shape, expected_output_shape, models, buffers, mocker: MockerFixture
    ):
        """Test that the agent correctly encodes observations and collects
        data."""
        agent = UnimodalEncodingAgent()

        components = connect_components(agent, buffers=buffers, models=models)
        components.inference_models[
            ModelName.IMAGE_JEPA_TARGET_ENCODER
        ].return_value = torch.ones(  # pyright: ignore[reportAttributeAccessIssue]
            expected_output_shape
        )

        spy_collect = mocker.spy(
            components.data_collectors[BufferName.IMAGE], "collect"
        )

        observation = torch.randn(input_shape)
        output = agent.step(observation)

        agent.encoder.assert_called_once_with(observation)  # pyright: ignore[reportAttributeAccessIssue]
        assert output.shape == expected_output_shape

        spy_collect.assert_called_once()
        call_args = spy_collect.call_args[0][0]
        assert DataKey.OBSERVATION in call_args
        assert torch.equal(call_args[DataKey.OBSERVATION], observation)
