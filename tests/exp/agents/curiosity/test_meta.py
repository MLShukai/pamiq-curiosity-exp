import pytest
import torch
from pamiq_core.testing import (
    connect_components,
    create_mock_buffer,
    create_mock_models,
)
from pytest_mock import MockerFixture
from torch.distributions import Normal

from exp.agents.curiosity.meta import MetaCuriosityAgent
from exp.data import BufferName, DataKey
from exp.models import ModelName

# Constants
OBSERVATION_DIM = 16
ACTION_DIM = 4
HIDDEN_DIM = 32
DEPTH = 2
NUM_LEVELS = 3


class TestMetaCuriosityAgent:
    """Tests for the MetaCuriosityAgent class."""

    @pytest.fixture
    def models(self):
        # Create forward dynamics models for each meta level
        models = {}
        for i in range(NUM_LEVELS):
            fd_model, _ = create_mock_models()
            pred_obs = torch.randn(OBSERVATION_DIM)
            hidden = torch.randn(DEPTH, HIDDEN_DIM)
            fd_model.inference_model.return_value = (pred_obs, hidden)
            models[ModelName.FORWARD_DYNAMICS + str(i)] = fd_model

        # Create policy value model
        policy_value_model, _ = create_mock_models()
        action_dist = Normal(torch.zeros(ACTION_DIM), torch.ones(ACTION_DIM))
        value = torch.tensor(0.5)
        policy_hidden = torch.randn(DEPTH, HIDDEN_DIM)
        policy_value_model.inference_model.return_value = (
            action_dist,
            value,
            policy_hidden,
        )
        models[ModelName.POLICY_VALUE] = policy_value_model

        return models

    @pytest.fixture
    def buffers(self):
        buffers = {str(BufferName.POLICY): create_mock_buffer()}
        for i in range(NUM_LEVELS):
            buffers[BufferName.FORWARD_DYNAMICS + str(i)] = create_mock_buffer()
        return buffers

    @pytest.fixture
    def mock_aim_run(self, mocker: MockerFixture):
        return mocker.patch("exp.agents.curiosity.meta.get_global_run")

    @pytest.fixture
    def agent(self, models, buffers, mock_aim_run):
        agent = MetaCuriosityAgent(
            num_meta_levels=NUM_LEVELS,
            surprisal_coefficients_method="minimize_to_maximize",
            log_every_n_steps=5,
        )
        connect_components(agent, buffers=buffers, models=models)
        return agent

    def test_invalid_num_meta_levels(self):
        """Test that agent raises error for invalid num_meta_levels."""
        with pytest.raises(ValueError, match="`num_meta_levels` must be >= 1"):
            MetaCuriosityAgent(num_meta_levels=0)

        with pytest.raises(ValueError, match="`num_meta_levels` must be >= 1"):
            MetaCuriosityAgent(num_meta_levels=-1)

    def test_surprisal_coefficients_methods(self):
        """Test different surprisal coefficient generation methods."""
        # Test minimize_to_maximize (default)
        agent = MetaCuriosityAgent(
            num_meta_levels=3, surprisal_coefficients_method="minimize_to_maximize"
        )
        assert len(agent.surprisal_coefficients) == 3
        assert agent.surprisal_coefficients[0] == pytest.approx(-1.0)
        assert agent.surprisal_coefficients[1] == pytest.approx(0.0)
        assert agent.surprisal_coefficients[2] == pytest.approx(1.0)

        # Test minimize
        agent = MetaCuriosityAgent(
            num_meta_levels=3, surprisal_coefficients_method="minimize"
        )
        assert len(agent.surprisal_coefficients) == 3
        assert all(coef == -1.0 for coef in agent.surprisal_coefficients)

        # Test maximize
        agent = MetaCuriosityAgent(
            num_meta_levels=3, surprisal_coefficients_method="maximize"
        )
        assert len(agent.surprisal_coefficients) == 3
        assert all(coef == 1.0 for coef in agent.surprisal_coefficients)

    def test_setup_step_teardown(
        self, agent: MetaCuriosityAgent, mocker: MockerFixture
    ):
        """Test the main interaction loop of the agent."""
        agent.setup()

        # Verify initial state
        assert agent.policy_hidden_state is None
        assert len(agent.forward_dynamics_hiddens) == NUM_LEVELS
        assert all(h is None for h in agent.forward_dynamics_hiddens)
        assert agent.predicted_obses is None
        assert isinstance(agent.step_data_policy, dict)
        assert len(agent.step_data_fd) == NUM_LEVELS

        observation = torch.randn(OBSERVATION_DIM)

        # Create spies for data collectors
        spy_policy_collect = mocker.spy(agent.collector_policy, "collect")
        spy_fd_collects = []
        for i in range(NUM_LEVELS):
            spy = mocker.spy(agent.collectors_fd[i], "collect")
            spy_fd_collects.append(spy)

        # First step - no reward calculation since no previous predictions
        action = agent.step(observation)
        assert action.shape == (ACTION_DIM,)
        assert agent.global_step == 1
        assert agent.predicted_obses is not None
        assert len(agent.predicted_obses) == NUM_LEVELS

        # Data should not be collected on first step (no previous hidden states)
        assert spy_policy_collect.call_count == 0
        for spy in spy_fd_collects:
            assert spy.call_count == 0

        # Second step - should calculate meta rewards
        observation2 = torch.randn(OBSERVATION_DIM)
        agent.step(observation2)
        assert agent.global_step == 2

        # Data collection happens on the NEXT step after hidden states are available
        # So no collection yet
        assert spy_policy_collect.call_count == 0
        for spy in spy_fd_collects:
            assert spy.call_count == 0

        # Third step - now data should be collected from previous step
        observation3 = torch.randn(OBSERVATION_DIM)
        agent.step(observation3)
        assert agent.global_step == 3

        # Should have collected data for forward dynamics
        for i, spy in enumerate(spy_fd_collects):
            assert spy.call_count == 1
            fd_data = spy.call_args[0][0]
            assert DataKey.OBSERVATION in fd_data
            assert DataKey.ACTION in fd_data
            assert DataKey.HIDDEN in fd_data
            assert DataKey.TARGET in fd_data

        # Policy data should be collected
        assert spy_policy_collect.call_count == 1
        policy_data = spy_policy_collect.call_args[0][0]
        assert DataKey.OBSERVATION in policy_data
        assert DataKey.ACTION in policy_data
        assert DataKey.ACTION_LOG_PROB in policy_data
        assert DataKey.VALUE in policy_data
        assert DataKey.REWARD in policy_data
        assert DataKey.HIDDEN in policy_data

        # Check that coefficients are set
        assert len(agent.surprisal_coefficients) == NUM_LEVELS

    def test_logging(self, agent: MetaCuriosityAgent, mock_aim_run):
        """Test metrics logging at specified intervals."""
        mock_run = mock_aim_run.return_value

        agent.setup()
        observation = torch.randn(OBSERVATION_DIM)

        # Step multiple times to trigger logging
        for _ in range(6):
            agent.step(observation)

        # Should log on step 5 (log_every_n_steps=5)
        mock_run.track.assert_called()

        # Extract tracked metrics
        track_calls = mock_run.track.call_args_list
        tracked_metrics = {call[1]["name"] for call in track_calls}

        # Verify meta rewards are tracked
        for i in range(NUM_LEVELS):
            assert f"curiosity-agent/reward_{i}" in tracked_metrics
        assert "curiosity-agent/reward_sum" in tracked_metrics

        # Verify context includes curiosity type
        for call in track_calls:
            assert call[1]["context"]["curiosity_type"] == "meta"

    def test_save_and_load_state(self, agent: MetaCuriosityAgent, tmp_path):
        """Test state saving and loading functionality."""
        agent.setup()

        # Step to set up state
        observation = torch.randn(OBSERVATION_DIM)
        agent.step(observation)
        agent.step(observation)

        # Set up some state after stepping
        agent.global_step = 42  # Reset to specific value after steps
        agent.policy_hidden_state = torch.randn(DEPTH, HIDDEN_DIM)

        # Set forward dynamics hidden states
        for i in range(NUM_LEVELS):
            agent.forward_dynamics_hiddens[i] = torch.randn(DEPTH, HIDDEN_DIM)

        # Save state
        save_path = tmp_path / "agent_state"
        agent.save_state(save_path)

        # Verify files exist
        assert (save_path / "policy_hidden_state.pt").exists()
        assert (save_path / "global_step").exists()
        for i in range(NUM_LEVELS):
            assert (save_path / f"forward_dynamics_hidden_{i}.pt").exists()

        # Create new agent and load state
        new_agent = MetaCuriosityAgent(
            num_meta_levels=NUM_LEVELS,
            surprisal_coefficients_method="minimize_to_maximize",
        )
        new_agent.load_state(save_path)

        # Verify state was loaded correctly
        assert new_agent.global_step == 42
        assert new_agent.policy_hidden_state is not None
        assert torch.allclose(new_agent.policy_hidden_state, agent.policy_hidden_state)

        for i in range(NUM_LEVELS):
            assert torch.allclose(
                new_agent.forward_dynamics_hiddens[i],
                agent.forward_dynamics_hiddens[i],
            )
