import pytest
import torch
from pamiq_core.testing import (
    connect_components,
    create_mock_buffer,
    create_mock_models,
)
from pytest_mock import MockerFixture
from torch.distributions import Normal

from exp.agents.curiosity.ttt import TTTCuriosityAgent
from exp.data import BufferName, DataKey
from exp.models import ModelName
from exp.models.components.multi_distributions import MultiDistributions

# Constants
OBSERVATION_DIM = 16
ACTION_DIM = 4
HIDDEN_DIM = 32
DEPTH = 2


class TestTTTCuriosityAgent:
    """Tests for the TTTCuriosityAgent class."""

    @pytest.fixture
    def models(self):
        fd_piv_model, _ = create_mock_models()

        # Mock FDPiV model behavior
        obs_hat = torch.zeros(3, OBSERVATION_DIM)
        obs_emb = torch.zeros(3, OBSERVATION_DIM)
        action_dist = MultiDistributions(
            Normal(torch.zeros(ACTION_DIM), torch.ones(ACTION_DIM)),
            Normal(torch.zeros(ACTION_DIM), torch.ones(ACTION_DIM)),
        )
        value = torch.tensor(0.5)
        hidden = {
            "time": torch.randn(DEPTH, HIDDEN_DIM),
            "ttt": [{"test": torch.randn(DEPTH, HIDDEN_DIM)}],
        }
        surprisal = torch.randn(DEPTH, 4, 6)

        fd_piv_model.inference_model.return_value = (
            obs_hat,
            obs_emb,
            action_dist,
            value,
            hidden,
            surprisal,
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
        return mocker.patch("exp.agents.curiosity.ttt.get_global_run")

    @pytest.fixture
    def agent(self, models, buffers, mock_aim_run):
        agent = TTTCuriosityAgent(
            log_every_n_steps=5,
        )

        connect_components(agent, buffers=buffers, models=models)
        return agent

    def test_initialization(self):
        """Test agent initialization."""
        agent = TTTCuriosityAgent(
            log_every_n_steps=10,
        )

        assert agent.hidden_state is None
        assert agent.external_action is None
        assert agent.internal_action is None
        assert agent.global_step == 0

    def test_setup_step_teardown(self, agent: TTTCuriosityAgent, mocker: MockerFixture):
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
        assert spy_fd_piv_collect.call_count == 0

        # Third step
        action = agent.step(observation)
        assert agent.global_step == 3
        assert spy_fd_piv_collect.call_count == 1
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

    def test_logging(self, agent: TTTCuriosityAgent, mock_aim_run):
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

    def test_save_and_load_state(self, agent: TTTCuriosityAgent, tmp_path):
        """Test state saving and loading functionality."""
        agent.global_step = 42
        agent.hidden_state = {
            "time": torch.randn(1),
            "ttt": [{"test": torch.randn(DEPTH, HIDDEN_DIM)}],
        }
        agent.external_action = torch.randn(ACTION_DIM)
        agent.internal_action = torch.randn(ACTION_DIM)
        agent.fast_surprisal_ema_decay = torch.randn(1)
        agent.slow_surprisal_ema_decay = torch.randn(1)
        agent.fast_surprisal_ema = torch.randn(1)
        agent.slow_surprisal_ema = torch.randn(1)

        # Save state
        save_path = tmp_path / "agent_state"
        agent.save_state(save_path)

        assert (save_path / "hidden_state.pt").exists()
        assert (save_path / "external_action.pt").exists()
        assert (save_path / "internal_action.pt").exists()
        assert (save_path / "fast_surprisal_ema_decay.pt").exists()
        assert (save_path / "slow_surprisal_ema_decay.pt").exists()
        assert (save_path / "fast_surprisal_ema.pt").exists()
        assert (save_path / "slow_surprisal_ema.pt").exists()
        assert (save_path / "global_step").exists()

        # Create new agent and load state
        new_agent = TTTCuriosityAgent()

        new_agent.load_state(save_path)

        assert new_agent.hidden_state is not None
        agent_hidden_qgru, agent_hidden_ttt = (
            agent.hidden_state["time"]
            if isinstance(agent.hidden_state["time"], torch.Tensor)
            else torch.tensor(0),
            agent.hidden_state["ttt"]
            if isinstance(agent.hidden_state["ttt"], list)
            else [],
        )
        new_hidden_qgru, new_hidden_ttt = (
            new_agent.hidden_state["time"]
            if isinstance(new_agent.hidden_state["time"], torch.Tensor)
            else torch.tensor(0),
            new_agent.hidden_state["ttt"]
            if isinstance(new_agent.hidden_state["ttt"], list)
            else [],
        )
        assert torch.equal(new_hidden_qgru, agent_hidden_qgru)
        assert all(
            torch.equal(new_layer[key], old_layer[key])
            for new_layer, old_layer in zip(new_hidden_ttt, agent_hidden_ttt)
            for key in new_layer
        )
        assert new_agent.external_action is not None
        assert torch.equal(new_agent.external_action, agent.external_action)
        assert new_agent.internal_action is not None
        assert torch.equal(new_agent.internal_action, agent.internal_action)
        assert new_agent.global_step == 42
        assert new_agent.fast_surprisal_ema_decay is not None
        assert torch.equal(
            new_agent.fast_surprisal_ema_decay, agent.fast_surprisal_ema_decay
        )
        assert new_agent.slow_surprisal_ema_decay is not None
        assert torch.equal(
            new_agent.slow_surprisal_ema_decay, agent.slow_surprisal_ema_decay
        )
        assert new_agent.fast_surprisal_ema is not None
        assert torch.equal(new_agent.fast_surprisal_ema, agent.fast_surprisal_ema)
        assert new_agent.slow_surprisal_ema is not None
        assert torch.equal(new_agent.slow_surprisal_ema, agent.slow_surprisal_ema)

    def test_save_and_load_state_with_none_hidden(
        self, agent: TTTCuriosityAgent, tmp_path
    ):
        """Test state saving and loading when hidden states are None."""
        agent.global_step = 100
        agent.hidden_state = None
        agent.external_action = None
        agent.internal_action = None
        agent.fast_surprisal_ema_decay = None
        agent.slow_surprisal_ema_decay = None
        agent.fast_surprisal_ema = None
        agent.slow_surprisal_ema = None
        # Save state
        save_path = tmp_path / "agent_state_none"
        agent.save_state(save_path)

        assert not (save_path / "hidden_state.pt").exists()
        assert not (save_path / "external_action.pt").exists()
        assert not (save_path / "internal_action.pt").exists()
        assert not (save_path / "fast_surprisal_ema_decay.pt").exists()
        assert not (save_path / "slow_surprisal_ema_decay.pt").exists()
        assert not (save_path / "fast_surprisal_ema.pt").exists()
        assert not (save_path / "slow_surprisal_ema.pt").exists()
        assert (save_path / "global_step").exists()

        # Create new agent and load state
        new_agent = TTTCuriosityAgent()
        new_agent.load_state(save_path)

        assert new_agent.hidden_state is None
        assert new_agent.external_action is None
        assert new_agent.internal_action is None
        assert new_agent.fast_surprisal_ema_decay is None
        assert new_agent.slow_surprisal_ema_decay is None
        assert new_agent.fast_surprisal_ema is None
        assert new_agent.slow_surprisal_ema is None
        assert new_agent.global_step == 100
