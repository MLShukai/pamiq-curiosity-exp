import pytest
import torch
from pamiq_core.testing import (
    connect_components,
    create_mock_buffer,
    create_mock_models,
)
from pytest_mock import MockerFixture
from torch.distributions import Normal

from exp.agents.curiosity.hierarchical import HierarchicalCuriosityAgent
from exp.data import BufferName, DataKey
from exp.models import ModelName

# Constants
OBSERVATION_DIM = 16
ACTION_DIM = 16
HIDDEN_DIM = 32
DEPTH = 2


class TestHierarchicalCuriosityAgent:
    """Tests for the HierarchicalCuriosityAgent class."""

    @pytest.fixture
    def models(self):
        forward_dynamics_model_0, _ = create_mock_models()
        forward_dynamics_model_1, _ = create_mock_models()
        policy_value_model_0, _ = create_mock_models()
        policy_value_model_1, _ = create_mock_models()

        # Mock forward dynamics model behavior
        obs_dist = Normal(
            torch.zeros(3, OBSERVATION_DIM), torch.ones(3, OBSERVATION_DIM)
        )
        hidden = torch.zeros(3, DEPTH, HIDDEN_DIM)
        latent = torch.zeros(3, OBSERVATION_DIM)
        forward_dynamics_model_0.inference_model.return_value = (
            obs_dist,
            latent,
            hidden,
        )
        forward_dynamics_model_1.inference_model.return_value = (
            obs_dist,
            latent,
            hidden,
        )

        # Mock policy value model behavior
        action_dist = Normal(torch.zeros(ACTION_DIM), torch.ones(ACTION_DIM))
        value = torch.tensor(0.5)
        latent = torch.zeros(ACTION_DIM)
        policy_hidden = torch.zeros(DEPTH, HIDDEN_DIM)
        policy_value_model_0.inference_model.return_value = (
            action_dist,
            value,
            latent,
            policy_hidden,
        )
        policy_value_model_1.inference_model.return_value = (
            action_dist,
            value,
            latent,
            policy_hidden,
        )

        return {
            ModelName.FORWARD_DYNAMICS + str(0): forward_dynamics_model_0,
            ModelName.FORWARD_DYNAMICS + str(1): forward_dynamics_model_1,
            ModelName.POLICY_VALUE + str(0): policy_value_model_0,
            ModelName.POLICY_VALUE + str(1): policy_value_model_1,
        }

    @pytest.fixture
    def buffers(self):
        return {
            BufferName.FORWARD_DYNAMICS + str(0): create_mock_buffer(),
            BufferName.FORWARD_DYNAMICS + str(1): create_mock_buffer(),
            BufferName.POLICY + str(0): create_mock_buffer(),
            BufferName.POLICY + str(1): create_mock_buffer(),
        }

    @pytest.fixture
    def mock_aim_run(self, mocker: MockerFixture):
        return mocker.patch("exp.agents.curiosity.hierarchical.get_global_run")

    @pytest.fixture
    def agent(self, models, buffers, mock_aim_run):
        agent = HierarchicalCuriosityAgent(
            num_hierarchical_levels=2,
            prev_latent_action_list_init=[ACTION_DIM, ACTION_DIM],
            prev_action_list_init=[ACTION_DIM, ACTION_DIM],
            log_every_n_steps=5,
        )

        connect_components(agent, buffers=buffers, models=models)
        return agent

    def test_initialization(self):
        """Test agent initialization."""
        agent = HierarchicalCuriosityAgent(
            num_hierarchical_levels=2,
            prev_latent_action_list_init=[ACTION_DIM, ACTION_DIM],
            prev_action_list_init=[ACTION_DIM, ACTION_DIM],
            log_every_n_steps=10,
        )

        assert agent.num_hierarchical_levels == 2
        assert agent.global_step == 0

    def test_invalid_imagination_steps(self):
        """Test that agent raises error for invalid num_hierarchical_level."""
        with pytest.raises(ValueError, match="`num_hierarchical_levels` must be >= 1"):
            HierarchicalCuriosityAgent(
                num_hierarchical_levels=0,
                prev_latent_action_list_init=[ACTION_DIM, ACTION_DIM],
                prev_action_list_init=[ACTION_DIM, ACTION_DIM],
            )

    def test_setup_step_teardown(
        self, agent: HierarchicalCuriosityAgent, mocker: MockerFixture
    ):
        """Test the main interaction loop of the agent."""
        agent.setup()

        observation = torch.randn(OBSERVATION_DIM)

        spy_fd_collect_0 = mocker.spy(
            agent.collector_forward_dynamics_list[0], "collect"
        )
        spy_fd_collect_1 = mocker.spy(
            agent.collector_forward_dynamics_list[1], "collect"
        )
        spy_policy_collect_0 = mocker.spy(agent.collector_policy_list[0], "collect")
        spy_policy_collect_1 = mocker.spy(agent.collector_policy_list[1], "collect")

        # First step - no reward calculation
        action = agent.step(observation)
        assert action.shape == (ACTION_DIM,)
        assert agent.global_step == 1

        # Verify data collection
        assert spy_fd_collect_0.call_count == 0
        assert spy_fd_collect_1.call_count == 0
        assert spy_policy_collect_0.call_count == 0
        assert spy_policy_collect_1.call_count == 0

        # Second step - should calculate reward
        action = agent.step(observation)
        assert agent.global_step == 2

        # Verify data collection
        assert spy_fd_collect_0.call_count == 1
        assert spy_fd_collect_1.call_count == 1
        assert spy_policy_collect_0.call_count == 1
        assert spy_policy_collect_1.call_count == 1
        ##############################
        # Third step
        action = agent.step(observation)
        assert agent.global_step == 3
        assert spy_fd_collect_0.call_count == 2
        assert spy_policy_collect_0.call_count == 2
        fd_data_prev = spy_fd_collect_0.call_args_list[-1][0][0]
        policy_data_prev = spy_policy_collect_0.call_args_list[-1][0][0]

        action = agent.step(observation)
        # Check collected data keys
        fd_data = spy_fd_collect_0.call_args_list[-1][0][0]
        assert fd_data is not fd_data_prev
        assert DataKey.OBSERVATION in fd_data
        assert DataKey.LATENT_ACTION in fd_data
        assert DataKey.HIDDEN in fd_data

        policy_data = spy_policy_collect_0.call_args_list[-1][0][0]
        assert policy_data is not policy_data_prev
        assert DataKey.OBSERVATION in policy_data
        assert DataKey.ACTION in policy_data
        assert DataKey.ACTION_LOG_PROB in policy_data
        assert DataKey.HIDDEN in policy_data
        assert DataKey.VALUE in policy_data
        assert DataKey.REWARD in policy_data
        assert DataKey.HIDDEN in policy_data

    def test_logging(self, agent: HierarchicalCuriosityAgent, mock_aim_run):
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
        assert "curiosity-agent/reward0" in tracked_metrics
        assert "curiosity-agent/reward1" in tracked_metrics
        assert "curiosity-agent/value0" in tracked_metrics
        assert "curiosity-agent/value1" in tracked_metrics

    def test_save_and_load_state(self, agent: HierarchicalCuriosityAgent, tmp_path):
        """Test state saving and loading functionality."""
        agent.global_step = 42
        agent.prev_action_list = [
            torch.zeros(ACTION_DIM),
            torch.zeros(ACTION_DIM),
        ]
        agent.prev_fd_hidden_list = [
            torch.zeros(DEPTH, HIDDEN_DIM),
            torch.zeros(DEPTH, HIDDEN_DIM),
        ]
        agent.prev_policy_hidden_list = [
            torch.zeros(DEPTH, HIDDEN_DIM),
            torch.zeros(DEPTH, HIDDEN_DIM),
        ]
        agent.prev_latent_action_list = [
            torch.zeros(ACTION_DIM),
            torch.zeros(ACTION_DIM),
        ]
        agent.self_reward_list = [
            torch.zeros(1),
        ]
        agent.surprisal_coefficient_vector_list = [
            torch.zeros(OBSERVATION_DIM),
            torch.zeros(OBSERVATION_DIM),
        ]

        # Save state
        save_path = tmp_path / "agent_state"
        agent.save_state(save_path)

        assert (save_path / "global_step").exists()
        assert (save_path / "hierarchical_curiosity_agent_state.pt").exists()

        # Create new agent and load state
        new_agent = HierarchicalCuriosityAgent(
            num_hierarchical_levels=2,
            prev_latent_action_list_init=[ACTION_DIM, ACTION_DIM],
            prev_action_list_init=[ACTION_DIM, ACTION_DIM],
        )

        new_agent.load_state(save_path)

        assert new_agent.global_step == 42
        for prev_action, new_prev_action in zip(
            agent.prev_action_list, new_agent.prev_action_list
        ):
            assert prev_action is not None
            assert new_prev_action is not None
            assert torch.equal(prev_action, new_prev_action)
        for prev_fd_hidden, new_prev_fd_hidden in zip(
            agent.prev_fd_hidden_list, new_agent.prev_fd_hidden_list
        ):
            assert prev_fd_hidden is not None
            assert new_prev_fd_hidden is not None
            assert torch.equal(prev_fd_hidden, new_prev_fd_hidden)
        for prev_policy_hidden, new_prev_policy_hidden in zip(
            agent.prev_policy_hidden_list, new_agent.prev_policy_hidden_list
        ):
            assert prev_policy_hidden is not None
            assert new_prev_policy_hidden is not None
            assert torch.equal(prev_policy_hidden, new_prev_policy_hidden)
        for prev_observation, new_prev_observation in zip(
            agent.prev_latent_action_list, new_agent.prev_latent_action_list
        ):
            assert prev_observation is not None
            assert new_prev_observation is not None
            assert torch.equal(prev_observation, new_prev_observation)
        for self_reward, new_self_reward in zip(
            agent.self_reward_list, new_agent.self_reward_list
        ):
            assert self_reward is not None
            assert new_self_reward is not None
            assert torch.equal(self_reward, new_self_reward)
        for surprisal_coefficient, new_surprisal_coefficient in zip(
            agent.surprisal_coefficient_vector_list,
            new_agent.surprisal_coefficient_vector_list,
        ):
            assert surprisal_coefficient is not None
            assert new_surprisal_coefficient is not None
            assert torch.equal(surprisal_coefficient, new_surprisal_coefficient)
