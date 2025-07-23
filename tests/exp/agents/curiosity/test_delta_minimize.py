import pytest
import torch
from pamiq_core.testing import (
    connect_components,
    create_mock_buffer,
    create_mock_models,
)
from pytest_mock import MockerFixture
from torch.distributions import Normal

from exp.agents.curiosity.delta_minimize import DeltaMinimizeAgent
from exp.data import BufferName, DataKey
from exp.models import ModelName

# Constants
OBSERVATION_DIM = 16
ACTION_DIM = 4
HIDDEN_DIM = 32
DEPTH = 2


class TestDeltaMinimizeAgent:
    """Tests for the DeltaMinimizeAgent class."""

    @pytest.fixture
    def models(self):
        forward_dynamics_model, _ = create_mock_models()
        policy_value_model, _ = create_mock_models()

        # Mock forward dynamics model behavior
        obs_hat = torch.zeros(3, OBSERVATION_DIM)
        hidden = torch.zeros(3, DEPTH, HIDDEN_DIM)
        forward_dynamics_model.inference_model.return_value = (obs_hat, hidden)

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
    def mock_aim_run(self, mocker: MockerFixture):
        return mocker.patch("exp.agents.curiosity.delta_minimize.get_global_run")

    @pytest.fixture
    def agent(self, models, buffers, mock_aim_run):
        agent = DeltaMinimizeAgent(
            max_imagination_steps=3,
            log_every_n_steps=5,
        )

        connect_components(agent, buffers=buffers, models=models)
        return agent

    def test_initialization(self):
        """Test agent initialization."""
        agent = DeltaMinimizeAgent(
            max_imagination_steps=2,
            log_every_n_steps=10,
        )

        assert agent.head_forward_dynamics_hidden_state is None
        assert agent.policy_hidden_state is None
        assert agent.max_imagination_steps == 2
        assert agent.global_step == 0
        assert agent.previous_error is None

    def test_invalid_imagination_steps(self):
        """Test that agent raises error for invalid max_imagination_steps."""
        with pytest.raises(ValueError, match="`max_imagination_steps` must be >= 1"):
            DeltaMinimizeAgent(
                max_imagination_steps=0,
            )

    def test_setup_step_teardown(
        self, agent: DeltaMinimizeAgent, mocker: MockerFixture
    ):
        """Test the main interaction loop of the agent."""
        agent.setup()
        assert agent.initial_step
        assert agent.previous_error is None

        observation = torch.randn(OBSERVATION_DIM)

        spy_fd_collect = mocker.spy(agent.collector_forward_dynamics, "collect")
        spy_policy_collect = mocker.spy(agent.collector_policy, "collect")

        # First step - no reward calculation (should set reward to 0)
        action = agent.step(observation)
        assert not agent.initial_step
        assert action.shape == (ACTION_DIM,)
        assert agent.global_step == 1

        # Second step - should calculate learning progress reward
        action = agent.step(observation)
        assert agent.global_step == 2
        assert agent.previous_error is not None

        # Third step - should have learning progress reward calculation
        action = agent.step(observation)
        assert agent.global_step == 3

        # Verify data collection
        assert spy_fd_collect.call_count == 2  # Called after first and second step
        assert spy_policy_collect.call_count == 1  # Called after second step
        fd_data_prev = spy_fd_collect.call_args_list[-1][0][0]
        policy_data_prev = spy_policy_collect.call_args_list[-1][0][0]

        action = agent.step(observation)
        # Check collected data keys
        fd_data = spy_fd_collect.call_args_list[-1][0][0]
        assert fd_data is not fd_data_prev
        fd_data = spy_fd_collect.call_args_list[-1][0][0]
        assert DataKey.OBSERVATION in fd_data
        assert DataKey.ACTION in fd_data
        assert DataKey.HIDDEN in fd_data

        policy_data = spy_policy_collect.call_args_list[-1][0][0]
        policy_data = spy_policy_collect.call_args_list[-1][0][0]
        assert policy_data is not policy_data_prev
        assert DataKey.OBSERVATION in policy_data
        assert DataKey.ACTION in policy_data
        assert DataKey.ACTION_LOG_PROB in policy_data
        assert DataKey.HIDDEN in policy_data
        assert DataKey.VALUE in policy_data
        assert DataKey.REWARD in policy_data

    def test_logging(self, agent: DeltaMinimizeAgent, mock_aim_run):
        """Test metrics logging."""
        # Create a mock run object
        mock_run = mock_aim_run.return_value

        agent.setup()
        observation = torch.randn(OBSERVATION_DIM)

        # Step multiple times to trigger logging
        for _ in range(6):
            agent.step(observation)

        # Should log on step 5 (log_every_n_steps=5)
        mock_run.track.assert_called()

        # Verify that the correct metrics were tracked
        # Get all track calls
        track_calls = mock_run.track.call_args_list

        # Extract metric names from all calls
        tracked_metrics = {call[1]["name"] for call in track_calls}

        # Verify expected metrics were tracked
        assert "reward" in tracked_metrics
        assert "value" in tracked_metrics
        assert "error" in tracked_metrics

    def test_save_and_load_state(self, agent: DeltaMinimizeAgent, tmp_path):
        """Test state saving and loading functionality."""
        agent.global_step = 42
        agent.head_forward_dynamics_hidden_state = torch.randn(DEPTH, HIDDEN_DIM)
        agent.policy_hidden_state = torch.randn(DEPTH, HIDDEN_DIM)
        agent.previous_error = 1.23

        # Save state
        save_path = tmp_path / "agent_state"
        agent.save_state(save_path)

        assert (save_path / "head_forward_dynamics_hidden_state.pt").exists()
        assert (save_path / "policy_hidden_state.pt").exists()
        assert (save_path / "global_step").exists()
        assert (save_path / "previous_error").exists()

        # Create new agent and load state
        new_agent = DeltaMinimizeAgent()

        new_agent.load_state(save_path)

        assert new_agent.head_forward_dynamics_hidden_state is not None
        assert torch.equal(
            new_agent.head_forward_dynamics_hidden_state,
            agent.head_forward_dynamics_hidden_state,
        )
        assert new_agent.policy_hidden_state is not None
        assert torch.equal(new_agent.policy_hidden_state, agent.policy_hidden_state)
        assert new_agent.global_step == 42
        assert new_agent.previous_error == 1.23

    def test_save_and_load_state_with_none_values(
        self, agent: DeltaMinimizeAgent, tmp_path
    ):
        """Test state saving and loading when some values are None."""
        agent.global_step = 100
        agent.head_forward_dynamics_hidden_state = None
        agent.policy_hidden_state = None
        agent.previous_error = None

        # Save state
        save_path = tmp_path / "agent_state_none"
        agent.save_state(save_path)

        assert not (save_path / "head_forward_dynamics_hidden_state.pt").exists()
        assert not (save_path / "policy_hidden_state.pt").exists()
        assert (save_path / "global_step").exists()
        assert not (save_path / "previous_error").exists()

        # Create new agent and load state
        new_agent = DeltaMinimizeAgent()
        new_agent.load_state(save_path)

        assert new_agent.head_forward_dynamics_hidden_state is None
        assert new_agent.policy_hidden_state is None
        assert new_agent.global_step == 100
        assert new_agent.previous_error is None
