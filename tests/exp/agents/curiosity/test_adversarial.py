import pytest
import torch
from pamiq_core.testing import (
    connect_components,
    create_mock_buffer,
    create_mock_models,
)
from pytest_mock import MockerFixture
from torch.distributions import Normal

from exp.agents.curiosity.adversarial import AdversarialCuriosityAgent
from exp.data import BufferName, DataKey
from exp.models import ModelName

# Constants
OBSERVATION_DIM = 16
ACTION_DIM = 4
HIDDEN_DIM = 32
DEPTH = 2


class TestAdversarialCuriosityAgent:
    """Tests for the AdversarialCuriosityAgent class."""

    @pytest.fixture
    def models(self):
        forward_dynamics_model, _ = create_mock_models()
        policy_value_model, _ = create_mock_models()

        # Mock forward dynamics model behavior
        obs_dist = Normal(
            torch.zeros(3, OBSERVATION_DIM), torch.ones(3, OBSERVATION_DIM)
        )
        hidden = torch.zeros(3, DEPTH, HIDDEN_DIM)
        forward_dynamics_model.inference_model.return_value = (obs_dist, hidden)

        # Mock policy value model behavior
        action_dist = Normal(torch.zeros(ACTION_DIM), torch.ones(ACTION_DIM))
        value = torch.tensor(0.5)
        policy_hidden = torch.zeros(DEPTH, HIDDEN_DIM)
        policy_value_model.inference_model.return_value = (
            action_dist,
            value,
            policy_hidden,
        )

        return {
            ModelName.FORWARD_DYNAMICS: forward_dynamics_model,
            ModelName.POLICY_VALUE: policy_value_model,
        }

    @pytest.fixture
    def buffers(self):
        return {
            BufferName.FORWARD_DYNAMICS: create_mock_buffer(),
            BufferName.POLICY: create_mock_buffer(),
        }

    @pytest.fixture
    def mock_mlflow(self, mocker: MockerFixture):
        return mocker.patch("exp.agents.curiosity.adversarial.mlflow")

    @pytest.fixture
    def agent(self, models, buffers, mock_mlflow):
        agent = AdversarialCuriosityAgent(
            max_imagination_steps=3,
            log_every_n_steps=5,
        )

        connect_components(agent, buffers=buffers, models=models)
        return agent

    def test_initialization(self):
        """Test agent initialization."""
        agent = AdversarialCuriosityAgent(
            max_imagination_steps=2,
            log_every_n_steps=10,
        )

        assert agent.head_forward_dynamics_hidden_state is None
        assert agent.policy_hidden_state is None
        assert agent.max_imagination_steps == 2
        assert agent.global_step == 0

    def test_invalid_imagination_steps(self):
        """Test that agent raises error for invalid max_imagination_steps."""
        with pytest.raises(ValueError, match="`max_imagination_steps` must be >= 1"):
            AdversarialCuriosityAgent(
                max_imagination_steps=0,
            )

    def test_setup_step_teardown(
        self, agent: AdversarialCuriosityAgent, mocker: MockerFixture
    ):
        """Test the main interaction loop of the agent."""
        agent.setup()
        assert agent.initial_step

        observation = torch.randn(OBSERVATION_DIM)

        spy_fd_collect = mocker.spy(agent.collector_forward_dynamics, "collect")
        spy_policy_collect = mocker.spy(agent.collector_policy, "collect")

        # First step - no reward calculation
        action = agent.step(observation)
        assert not agent.initial_step
        assert action.shape == (ACTION_DIM,)
        assert agent.global_step == 1

        # Second step - should calculate reward
        action = agent.step(observation)
        assert agent.global_step == 2

        # Verify data collection
        assert spy_fd_collect.call_count == 2
        assert spy_policy_collect.call_count == 1  # Only called after first step

        # Check collected data keys
        fd_data = spy_fd_collect.call_args_list[-1][0][0]
        assert DataKey.OBSERVATION in fd_data
        assert DataKey.ACTION in fd_data
        assert DataKey.HIDDEN in fd_data
        # HIDDEN key is only present after first step when hidden is not None

        policy_data = spy_policy_collect.call_args_list[-1][0][0]
        assert DataKey.OBSERVATION in policy_data
        assert DataKey.ACTION in policy_data
        assert DataKey.ACTION_LOG_PROB in policy_data
        assert DataKey.HIDDEN in policy_data
        assert DataKey.VALUE in policy_data
        assert DataKey.REWARD in policy_data

    def test_logging(self, agent: AdversarialCuriosityAgent, mock_mlflow):
        """Test metrics logging."""
        agent.setup()
        observation = torch.randn(OBSERVATION_DIM)

        # Step multiple times to trigger logging
        for _ in range(6):
            agent.step(observation)

        # Should log on step 5 (log_every_n_steps=5)
        mock_mlflow.log_metrics.assert_called()
        call_args = mock_mlflow.log_metrics.call_args[0][0]
        assert "curiosity-agent/reward" in call_args
        assert "curiosity-agent/value" in call_args

    def test_save_and_load_state(self, agent: AdversarialCuriosityAgent, tmp_path):
        """Test state saving and loading functionality."""
        agent.global_step = 42
        agent.head_forward_dynamics_hidden_state = torch.randn(DEPTH, HIDDEN_DIM)
        agent.policy_hidden_state = torch.randn(DEPTH, HIDDEN_DIM)

        # Save state
        save_path = tmp_path / "agent_state"
        agent.save_state(save_path)

        assert (save_path / "head_forward_dynamics_hidden_state.pt").exists()
        assert (save_path / "policy_hidden_state.pt").exists()
        assert (save_path / "global_step").exists()

        # Create new agent and load state
        new_agent = AdversarialCuriosityAgent()

        new_agent.load_state(save_path)

        assert new_agent.head_forward_dynamics_hidden_state is not None
        assert torch.equal(
            new_agent.head_forward_dynamics_hidden_state,
            agent.head_forward_dynamics_hidden_state,
        )
        assert new_agent.policy_hidden_state is not None
        assert torch.equal(new_agent.policy_hidden_state, agent.policy_hidden_state)
        assert new_agent.global_step == 42

    def test_save_and_load_state_with_none_hidden(
        self, agent: AdversarialCuriosityAgent, tmp_path
    ):
        """Test state saving and loading when hidden states are None."""
        agent.global_step = 100
        agent.head_forward_dynamics_hidden_state = None
        agent.policy_hidden_state = None

        # Save state
        save_path = tmp_path / "agent_state_none"
        agent.save_state(save_path)

        assert not (save_path / "head_forward_dynamics_hidden_state.pt").exists()
        assert not (save_path / "policy_hidden_state.pt").exists()
        assert (save_path / "global_step").exists()

        # Create new agent and load state
        new_agent = AdversarialCuriosityAgent()
        new_agent.load_state(save_path)

        assert new_agent.head_forward_dynamics_hidden_state is None
        assert new_agent.policy_hidden_state is None
        assert new_agent.global_step == 100
