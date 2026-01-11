import pytest
import torch
from pamiq_core.testing import (
    connect_components,
    create_mock_buffer,
    create_mock_models,
)
from pytest_mock import MockerFixture
from torch.distributions import Normal

from exp.agents.curiosity.unified import UnifiedAdversarialCuriosityAgent
from exp.data import BufferName, DataKey
from exp.models import ModelName

# Constants
OBSERVATION_DIM = 16
ACTION_DIM = 4
HIDDEN_DIM = 32
DEPTH = 2


class TestUnifiedAdversarialCuriosityAgent:
    """Tests for the UnifiedAdversarialCuriosityAgent class."""

    @pytest.fixture
    def models(self):
        fd_piv_model, _ = create_mock_models()

        # Mock FDPiV model behavior
        obs_hat = torch.zeros(3, OBSERVATION_DIM)
        action_dist = Normal(torch.zeros(ACTION_DIM), torch.ones(ACTION_DIM))
        value = torch.tensor(0.5)
        hidden = torch.zeros(3, DEPTH, HIDDEN_DIM)

        fd_piv_model.inference_model.return_value = (
            obs_hat,
            action_dist,
            value,
            hidden,
        )

        return {
            ModelName.FD_POLICY_VALUE: fd_piv_model,
        }

    @pytest.fixture
    def buffers(self):
        return {
            BufferName.FD_POLICY_VALUE: create_mock_buffer(),
        }

    @pytest.fixture
    def mock_aim_run(self, mocker: MockerFixture):
        return mocker.patch("exp.agents.curiosity.unified.get_global_run")

    @pytest.fixture
    def agent(self, models, buffers, mock_aim_run):
        agent = UnifiedAdversarialCuriosityAgent(
            max_imagination_steps=3,
            log_every_n_steps=5,
        )

        connect_components(agent, buffers=buffers, models=models)
        return agent

    def test_initialization(self):
        """Test agent initialization."""
        agent = UnifiedAdversarialCuriosityAgent(
            max_imagination_steps=2,
            log_every_n_steps=10,
        )

        assert agent.hidden_state is None
        assert agent.action is None
        assert agent.max_imagination_steps == 2
        assert agent.global_step == 0

    def test_invalid_imagination_steps(self):
        """Test that agent raises error for invalid max_imagination_steps."""
        with pytest.raises(ValueError, match="`max_imagination_steps` must be >= 1"):
            UnifiedAdversarialCuriosityAgent(
                max_imagination_steps=0,
            )

    def test_setup_step_teardown(
        self, agent: UnifiedAdversarialCuriosityAgent, mocker: MockerFixture
    ):
        """Test the main interaction loop of the agent."""
        agent.setup()

        observation = torch.randn(OBSERVATION_DIM)

        spy_fd_piv_collect = mocker.spy(agent.collector_fd_piv, "collect")

        # First step - no reward calculation
        action = agent.step(observation)
        assert action.shape == (ACTION_DIM,)
        assert agent.global_step == 1

        # Second step - should calculate reward
        action = agent.step(observation)
        assert agent.global_step == 2
        # Verify data collection
        assert spy_fd_piv_collect.call_count == 1

        # Third step
        action = agent.step(observation)
        assert agent.global_step == 3
        assert spy_fd_piv_collect.call_count == 2
        fd_data_prev = spy_fd_piv_collect.call_args_list[-1][0][0]

        action = agent.step(observation)
        # Check collected data keys
        fd_data = spy_fd_piv_collect.call_args_list[-1][0][0]
        assert fd_data is not fd_data_prev
        assert DataKey.OBSERVATION in fd_data
        assert DataKey.ACTION in fd_data
        assert DataKey.HIDDEN in fd_data
        assert DataKey.ACTION_LOG_PROB in fd_data
        assert DataKey.VALUE in fd_data
        assert DataKey.REWARD in fd_data

    def test_logging(self, agent: UnifiedAdversarialCuriosityAgent, mock_aim_run):
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

    def test_save_and_load_state(
        self, agent: UnifiedAdversarialCuriosityAgent, tmp_path
    ):
        """Test state saving and loading functionality."""
        agent.global_step = 42
        agent.hidden_state = torch.randn(DEPTH, HIDDEN_DIM)
        agent.action = torch.randn(ACTION_DIM)
        agent.obs_hat = torch.randn(OBSERVATION_DIM)

        # Save state
        save_path = tmp_path / "agent_state"
        agent.save_state(save_path)

        assert (save_path / "hidden_state.pt").exists()
        assert (save_path / "action.pt").exists()
        assert (save_path / "obs_hat.pt").exists()
        assert (save_path / "global_step").exists()

        # Create new agent and load state
        new_agent = UnifiedAdversarialCuriosityAgent()

        new_agent.load_state(save_path)

        assert new_agent.hidden_state is not None
        assert torch.equal(
            new_agent.hidden_state,
            agent.hidden_state,
        )
        assert new_agent.action is not None
        assert torch.equal(new_agent.action, agent.action)
        assert new_agent.global_step == 42

    def test_save_and_load_state_with_none_hidden(
        self, agent: UnifiedAdversarialCuriosityAgent, tmp_path
    ):
        """Test state saving and loading when hidden states are None."""
        agent.global_step = 100
        agent.hidden_state = None
        agent.action = None

        # Save state
        save_path = tmp_path / "agent_state_none"
        agent.save_state(save_path)

        assert not (save_path / "hidden_state.pt").exists()
        assert not (save_path / "action.pt").exists()
        assert not (save_path / "obs_hat.pt").exists()
        assert (save_path / "global_step").exists()

        # Create new agent and load state
        new_agent = UnifiedAdversarialCuriosityAgent()
        new_agent.load_state(save_path)

        assert new_agent.hidden_state is None
        assert new_agent.action is None
        assert new_agent.obs_hat is None
        assert new_agent.global_step == 100
