import pytest
import torch
from pamiq_core.testing import (
    connect_components,
    create_mock_buffer,
    create_mock_models,
)
from pytest_mock import MockerFixture
from torch.distributions import Normal

from exp.agents.curiosity.hierarchical import (
    HierarchicalCuriosityAgent,
    LayerCuriosityAgent,
    LayerInput,
    LayerOutput,
)
from exp.data import BufferName, DataKey
from exp.models import ModelName

OBSERVATION_DIM = 16
LATENT_OBSERVATION_DIM = 32
ACTION_DIM = 4
HIDDEN_DIM = 32
DEPTH = 2
MODEL_BUFFER_SUFFIX = "_test"
MODEL_BUFFER_SUFFIX_1 = "_test_1"
MODEL_BUFFER_SUFFIX_2 = "_test_2"


class TestLayerCuriosityAgent:
    """Test suite for LayerCuriosityAgent."""

    @pytest.fixture
    def forward_dynamics(self):
        model, _ = create_mock_models()
        obs_hat = torch.zeros(OBSERVATION_DIM)
        latent_obs = torch.zeros(LATENT_OBSERVATION_DIM)
        hidden = torch.zeros(DEPTH, HIDDEN_DIM)
        model.inference_model.return_value = (obs_hat, latent_obs, hidden)
        return model

    @pytest.fixture
    def policy_value(self):
        model, _ = create_mock_models()
        action_dist = Normal(torch.zeros(ACTION_DIM), torch.ones(ACTION_DIM))
        value = torch.tensor(0.5)
        policy_hidden = torch.zeros(DEPTH, HIDDEN_DIM)
        model.inference_model.return_value = (action_dist, value, policy_hidden)
        return model

    @pytest.fixture
    def models(self, forward_dynamics, policy_value):
        return {
            ModelName.FORWARD_DYNAMICS + MODEL_BUFFER_SUFFIX: forward_dynamics,
            ModelName.POLICY_VALUE + MODEL_BUFFER_SUFFIX: policy_value,
        }

    @pytest.fixture
    def buffers(self):
        return {
            BufferName.FORWARD_DYNAMICS + MODEL_BUFFER_SUFFIX: create_mock_buffer(),
            BufferName.POLICY + MODEL_BUFFER_SUFFIX: create_mock_buffer(),
        }

    @pytest.fixture
    def agent(self, models, buffers):
        agent = LayerCuriosityAgent(
            model_buffer_suffix=MODEL_BUFFER_SUFFIX,
            reward_coef=1.0,
            reward_lerp_ratio=0.5,
            device=torch.device("cpu"),
        )
        connect_components(agent, models=models, buffers=buffers)
        return agent

    def test_setup(self, agent: LayerCuriosityAgent):
        agent.setup()
        assert agent.step_data_fd == {}
        assert agent.step_data_policy == {}

    def test_setup_step(self, agent: LayerCuriosityAgent, mocker: MockerFixture):
        spy_fd_collect = mocker.spy(agent.fd_collector, "collect")
        spy_policy_collect = mocker.spy(agent.policy_collector, "collect")
        agent.setup()
        assert agent.obs_hat is None
        assert agent.policy_hidden is None
        assert agent.fd_hidden is None

        observation = LayerInput(
            observation=torch.zeros(OBSERVATION_DIM),
            upper_action=torch.zeros(ACTION_DIM),
            upper_reward=torch.tensor(0.0),
        )

        assert agent.obs_hat is None
        assert agent.policy_hidden is None
        assert agent.fd_hidden is None

        output = agent.step(observation)
        observation_from_lower, action, reward = (
            output.lower_observation,
            output.action,
            output.reward,
        )
        assert observation_from_lower.shape == (LATENT_OBSERVATION_DIM,)
        assert action.shape == (ACTION_DIM,)
        assert reward is None
        assert agent.obs_hat is not None and agent.obs_hat.shape == (OBSERVATION_DIM,)
        assert agent.policy_hidden is not None and agent.policy_hidden.shape == (
            DEPTH,
            HIDDEN_DIM,
        )
        assert agent.fd_hidden is not None and agent.fd_hidden.shape == (
            DEPTH,
            HIDDEN_DIM,
        )
        assert spy_fd_collect.call_count == 1
        assert spy_policy_collect.call_count == 0

        output = agent.step(observation)
        observation_from_lower, action, reward = (
            output.lower_observation,
            output.action,
            output.reward,
        )
        assert observation_from_lower.shape == (LATENT_OBSERVATION_DIM,)
        assert action.shape == (ACTION_DIM,)
        assert reward is not None and reward.shape == ()
        assert agent.obs_hat is not None and agent.obs_hat.shape == (OBSERVATION_DIM,)
        assert agent.policy_hidden is not None and agent.policy_hidden.shape == (
            DEPTH,
            HIDDEN_DIM,
        )
        assert agent.fd_hidden is not None and agent.fd_hidden.shape == (
            DEPTH,
            HIDDEN_DIM,
        )
        assert spy_fd_collect.call_count == 2
        assert spy_policy_collect.call_count == 0

        output = agent.step(observation)
        observation_from_lower, action, reward = (
            output.lower_observation,
            output.action,
            output.reward,
        )
        assert observation_from_lower.shape == (LATENT_OBSERVATION_DIM,)
        assert action.shape == (ACTION_DIM,)
        assert reward is not None and reward.shape == ()
        assert agent.obs_hat is not None and agent.obs_hat.shape == (OBSERVATION_DIM,)
        assert agent.policy_hidden is not None and agent.policy_hidden.shape == (
            DEPTH,
            HIDDEN_DIM,
        )
        assert agent.fd_hidden is not None and agent.fd_hidden.shape == (
            DEPTH,
            HIDDEN_DIM,
        )
        assert spy_fd_collect.call_count == 3
        assert spy_policy_collect.call_count == 1

    def test_save_and_load(self, agent: LayerCuriosityAgent, tmp_path):
        """Test saving and loading agent state."""
        agent.setup()
        observation = LayerInput(
            observation=torch.zeros(OBSERVATION_DIM),
            upper_action=None,
            upper_reward=None,
        )
        agent.step(observation)

        # Save state
        save_path = tmp_path / "agent_state"
        agent.save_state(save_path)

        # Load state
        new_agent = LayerCuriosityAgent(
            model_buffer_suffix=MODEL_BUFFER_SUFFIX,
            reward_coef=1.0,
            reward_lerp_ratio=0.5,
            device=torch.device("cpu"),
        )
        new_agent.load_state(save_path)

        assert (
            new_agent.obs_hat is not None
            and agent.obs_hat is not None
            and torch.equal(new_agent.obs_hat, agent.obs_hat)
        )
        assert (
            new_agent.policy_hidden is not None
            and agent.policy_hidden is not None
            and torch.equal(new_agent.policy_hidden, agent.policy_hidden)
        )
        assert (
            new_agent.fd_hidden is not None
            and agent.fd_hidden is not None
            and torch.equal(new_agent.fd_hidden, agent.fd_hidden)
        )


class TestHierarchicalCuriosityAgent:
    """Test suite for HierarchicalCuriosityAgent."""

    @pytest.fixture
    def forward_dynamics(self):
        model, _ = create_mock_models()
        obs_hat = torch.zeros(OBSERVATION_DIM)
        latent_obs = torch.zeros(OBSERVATION_DIM)
        hidden = torch.zeros(DEPTH, HIDDEN_DIM)
        model.inference_model.return_value = (obs_hat, latent_obs, hidden)
        return model

    @pytest.fixture
    def policy_value(self):
        model, _ = create_mock_models()
        action_dist = Normal(torch.zeros(OBSERVATION_DIM), torch.ones(OBSERVATION_DIM))
        value = torch.tensor(0.5)
        policy_hidden = torch.zeros(DEPTH, HIDDEN_DIM)
        model.inference_model.return_value = (action_dist, value, policy_hidden)
        return model

    @pytest.fixture
    def models(self, forward_dynamics, policy_value):
        return {
            ModelName.FORWARD_DYNAMICS + MODEL_BUFFER_SUFFIX_1: forward_dynamics,
            ModelName.POLICY_VALUE + MODEL_BUFFER_SUFFIX_1: policy_value,
            ModelName.FORWARD_DYNAMICS + MODEL_BUFFER_SUFFIX_2: forward_dynamics,
            ModelName.POLICY_VALUE + MODEL_BUFFER_SUFFIX_2: policy_value,
        }

    @pytest.fixture
    def buffers(self):
        return {
            BufferName.FORWARD_DYNAMICS + MODEL_BUFFER_SUFFIX_1: create_mock_buffer(),
            BufferName.POLICY + MODEL_BUFFER_SUFFIX_1: create_mock_buffer(),
            BufferName.FORWARD_DYNAMICS + MODEL_BUFFER_SUFFIX_2: create_mock_buffer(),
            BufferName.POLICY + MODEL_BUFFER_SUFFIX_2: create_mock_buffer(),
        }

    @pytest.fixture
    def hierarchical_agent(self, models, buffers):
        agent = HierarchicalCuriosityAgent(
            reward_coef_list=[1.0, 1.0],
            reward_lerp_ratio=0.5,
            model_key_list=[MODEL_BUFFER_SUFFIX_1, MODEL_BUFFER_SUFFIX_2],
            device_list=[None, None],
            layer_timescale_list=[1, 2],
        )
        connect_components(agent, models=models, buffers=buffers)
        return agent

    def test_setup(self, hierarchical_agent: HierarchicalCuriosityAgent):
        hierarchical_agent.setup()
        for agent in hierarchical_agent.layer_agent_dict.values():
            assert agent.step_data_fd == {}
            assert agent.step_data_policy == {}

    def test_setup_step(
        self, hierarchical_agent: HierarchicalCuriosityAgent, mocker: MockerFixture
    ):
        spy_fd_collect_1 = mocker.spy(
            hierarchical_agent.layer_agent_dict[MODEL_BUFFER_SUFFIX_1].fd_collector,
            "collect",
        )
        spy_fd_collect_2 = mocker.spy(
            hierarchical_agent.layer_agent_dict[MODEL_BUFFER_SUFFIX_2].fd_collector,
            "collect",
        )

        hierarchical_agent.setup()
        assert (
            hierarchical_agent.action_to_lower_list
            == [None] * hierarchical_agent.num_layers
        )
        assert (
            hierarchical_agent.reward_to_lower_list
            == [None] * hierarchical_agent.num_layers
        )
        assert (
            hierarchical_agent.observation_to_upper_list
            == [None] * hierarchical_agent.num_layers
        )

        observation = torch.zeros(OBSERVATION_DIM)

        action = hierarchical_agent.step(observation)
        assert action.shape == (OBSERVATION_DIM,)
        assert spy_fd_collect_1.call_count == 1
        assert spy_fd_collect_2.call_count == 1

        action = hierarchical_agent.step(observation)
        assert action.shape == (OBSERVATION_DIM,)
        assert spy_fd_collect_1.call_count == 2
        assert spy_fd_collect_2.call_count == 1

        action = hierarchical_agent.step(observation)
        assert action.shape == (OBSERVATION_DIM,)
        assert spy_fd_collect_1.call_count == 3
        assert spy_fd_collect_2.call_count == 2
