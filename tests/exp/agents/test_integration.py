from collections.abc import Mapping

import pytest
import torch
from pamiq_core import Agent
from pytest_mock import MockerFixture

from exp.agents.integration import IntegratedCuriosityFramework
from exp.agents.unimodal_encoding import UnimodalEncodingAgent


class TestIntegratedCuriosityFramework:
    """Tests for the IntegratedCuriosityFramework class."""

    @pytest.fixture
    def mock_unimodal_agent(self, mocker: MockerFixture) -> UnimodalEncodingAgent:
        """Create mock unimodal encoding agents."""
        image_agent = mocker.Mock(spec=UnimodalEncodingAgent)
        image_agent.step.return_value = torch.ones(16, 256)

        return image_agent

    @pytest.fixture
    def mock_curiosity_agent(self, mocker: MockerFixture) -> Agent:
        """Create mock curiosity agent."""
        curiosity_agent = mocker.Mock(spec=Agent)
        curiosity_agent.step.return_value = torch.ones(4)
        return curiosity_agent

    @pytest.fixture
    def framework(
        self, mock_unimodal_agent, mock_curiosity_agent
    ) -> IntegratedCuriosityFramework:
        """Create the integrated framework with mock agents."""
        return IntegratedCuriosityFramework(
            mock_curiosity_agent, unimodal_encoding=mock_unimodal_agent
        )

    def test_step(self, framework, mock_unimodal_agent, mock_curiosity_agent):
        """Test step method correctly processes observations through all
        agents."""
        # Create test observations
        obs = torch.randn(3, 224, 224)

        # Call step
        action = framework.step(obs)

        # Verify each unimodal agent was called with correct observation
        mock_unimodal_agent.step.assert_called_once()
        assert torch.equal(mock_unimodal_agent.step.call_args[0][0], obs)
        # Verify curiosity agent was called with temporal encoding
        mock_curiosity_agent.step.assert_called_once_with(
            mock_unimodal_agent.step.return_value
        )

        # Verify final output
        assert torch.equal(action, mock_curiosity_agent.step.return_value)
