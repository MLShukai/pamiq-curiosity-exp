from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import torch
import torch.nn.functional as F
from pamiq_core import Agent
from torch import Tensor
from torch.distributions import Distribution

from exp.data import BufferName, DataKey
from exp.models import ModelName


@dataclass(frozen=True)
class LayerInput:
    observation: Tensor
    upper_action: Tensor | None
    upper_reward: Tensor | None


@dataclass(frozen=True)
class LayerOutput:
    lower_observation: Tensor
    action: Tensor
    reward: Tensor | None


STEP_DATA_POLICY_REQUIRED_KEYS = {
    DataKey.ACTION,
    DataKey.ACTION_LOG_PROB,
    DataKey.OBSERVATION,
    DataKey.UPPER_ACTION,
    DataKey.HIDDEN,
    DataKey.VALUE,
    DataKey.REWARD,
}

STEP_DATA_FD_REQUIRED_KEYS = {
    DataKey.OBSERVATION,
    DataKey.ACTION,
    DataKey.HIDDEN,
}


class LayerCuriosityAgent(Agent[LayerInput, LayerOutput]):
    """A Layer Curiosity Agent for hierarchical reinforcement learning.

    This layer takes observations and actions from an upper layer,
    computes rewards using a forward dynamics model, and collects data
    for policy and forward dynamics models.
    """

    def __init__(
        self,
        model_buffer_suffix: str,
        reward_coef: float,
        reward_lerp_ratio: float,
        device: torch.device | None = None,
    ) -> None:
        """Initializes the LayerCuriosityAgent with model buffer suffix, reward
        coefficient, and reward lerp ratio.

        Args:
            model_buffer_suffix: Suffix for the model buffer names.
            reward_coef: Coefficient for the reward computation.
            reward_lerp_ratio: Ratio for linear interpolation of rewards.
        """
        super().__init__()
        self.reward_coef = reward_coef
        self.reward_lerp_ratio = reward_lerp_ratio
        self.model_buffer_suffix = model_buffer_suffix
        self.obs_hat: Tensor | None = None
        self.policy_hidden: Tensor | None = None
        self.fd_hidden: Tensor | None = None
        self.device = device or torch.get_default_device()

    @override
    def on_data_collectors_attached(self) -> None:
        """Attaches data collectors for policy and forward dynamics buffers."""
        super().on_data_collectors_attached()
        self.policy_collector = self.get_data_collector(
            BufferName.POLICY + self.model_buffer_suffix
        )
        self.fd_collector = self.get_data_collector(
            BufferName.FORWARD_DYNAMICS + self.model_buffer_suffix
        )

    @override
    def on_inference_models_attached(self) -> None:
        """Attaches inference models for policy value and forward dynamics."""
        super().on_inference_models_attached()
        self.policy_value = self.get_inference_model(
            ModelName.POLICY_VALUE + self.model_buffer_suffix
        )
        self.forward_dynamics = self.get_inference_model(
            ModelName.FORWARD_DYNAMICS + self.model_buffer_suffix
        )

    @override
    def setup(self) -> None:
        """Sets up the agent by initializing step data dictionaries for policy
        and forward dynamics."""
        super().setup()
        self.step_data_fd = dict[str, Any]()
        self.step_data_policy = dict[str, Any]()

    @override
    def step(self, observation: LayerInput) -> LayerOutput:
        """Performs a step in the agent's operation, processing the observation
        and computing rewards and actions.

        Args:
            observation: Input containing observation, action from upper layer, and reward from upper layer.
        Returns:
            Output containing the observation from lower layer, action taken, and computed reward.
        """
        obs, upper_action, upper_reward = (
            observation.observation,
            observation.upper_action,
            observation.upper_reward,
        )

        # ================================================
        #                 Reward Computing
        # ================================================

        if self.obs_hat is not None:
            reward = F.mse_loss(obs, self.obs_hat) * self.reward_coef
            if upper_reward is not None:
                reward = (
                    self.reward_lerp_ratio * reward
                    + (1 - self.reward_lerp_ratio) * upper_reward
                )
            self.step_data_policy[DataKey.REWARD] = reward.cpu()
        else:
            reward = None

        # ================================================
        #                 Policy Step
        # ================================================

        if set(self.step_data_policy.keys()) == set(STEP_DATA_POLICY_REQUIRED_KEYS):
            self.policy_collector.collect(self.step_data_policy.copy())

        if self.policy_hidden is not None:
            self.step_data_policy[DataKey.HIDDEN] = self.policy_hidden.cpu()

        if upper_action is not None:
            self.step_data_policy[DataKey.UPPER_ACTION] = upper_action.cpu()

        action_dist: Distribution
        value: Tensor
        action_dist, value, self.policy_hidden = self.policy_value(
            obs, upper_action, self.policy_hidden
        )

        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)

        self.step_data_policy[DataKey.OBSERVATION] = obs.cpu()
        self.step_data_policy[DataKey.ACTION] = action.cpu()
        self.step_data_policy[DataKey.ACTION_LOG_PROB] = action_log_prob.cpu()
        self.step_data_policy[DataKey.VALUE] = value.cpu()

        # ================================================
        #             Forward Dynamics Step
        # ================================================

        if self.fd_hidden is not None:
            self.step_data_fd[DataKey.HIDDEN] = self.fd_hidden.cpu()

        self.obs_hat, latent_obs, self.fd_hidden = self.forward_dynamics(
            obs, action, self.fd_hidden
        )
        self.step_data_fd[DataKey.OBSERVATION] = obs.cpu()
        self.step_data_fd[DataKey.ACTION] = action.cpu()

        self.fd_collector.collect(self.step_data_fd.copy())

        # ================================================
        #                 Return Output
        # ================================================

        return LayerOutput(lower_observation=latent_obs, action=action, reward=reward)

    @override
    def save_state(self, path: Path):
        """Saves the agent's state, including the observation prediction,
        policy hidden state, and forward dynamics hidden state.

        Args:
            path: The path to save the state.
        """
        super().save_state(path)
        path.mkdir(exist_ok=True)
        torch.save(self.obs_hat, path / "obs_hat.pt")
        torch.save(self.policy_hidden, path / "policy_hidden.pt")
        torch.save(self.fd_hidden, path / "fd_hidden.pt")

    @override
    def load_state(self, path: Path):
        """Loads the agent's state, including the observation prediction,
        policy hidden state, and forward dynamics hidden state.

        Args:
            path: The path to load the state from.
        """
        super().load_state(path)
        self.obs_hat = torch.load(path / "obs_hat.pt", map_location=self.device)
        self.policy_hidden = torch.load(
            path / "policy_hidden.pt", map_location=self.device
        )
        self.fd_hidden = torch.load(path / "fd_hidden.pt", map_location=self.device)


class HierarchicalCuriosityAgent(Agent[Tensor, Tensor]):
    """Hierarchical Curiosity Agent that manages multiple LayerCuriosityAgents.

    This agent coordinates multiple layers of curiosity agents, each
    with its own time scale and action/reward propagation.
    """

    def __init__(
        self,
        layer_agent_dict: dict[str, LayerCuriosityAgent],
        layer_timescale: list[int],
    ) -> None:
        """Initializes the HierarchicalCuriosityAgent with layer agents and
        their respective time scales.

        Args:
            layer_agent_dict: Dictionary mapping layer names to LayerCuriosityAgent instances.
            layer_timescale: List of time scales for each layer agent.
        """
        super().__init__(layer_agent_dict)
        self.layer_agents = list(layer_agent_dict.values())
        self.num_layers = len(layer_agent_dict)

        if len(layer_agent_dict) != len(layer_timescale):
            raise ValueError("Layer agents and time scales must match in length.")
        self.layer_timescale = []
        timescale_cumprod = 1
        for timescale in layer_timescale:
            if timescale < 1:
                raise ValueError("Layer time scale must be >= 1.")
            timescale_cumprod *= timescale
            self.layer_timescale.append(timescale_cumprod)
        self.period = timescale_cumprod

        self.action_to_lower_list: list[Tensor | None] = [None] * self.num_layers
        self.reward_to_lower_list: list[Tensor | None] = [None] * self.num_layers
        self.observation_to_upper_list: list[Tensor | None] = [None] * self.num_layers

        self.counter = 0

    @override
    def setup(self) -> None:
        """Sets up the hierarchical agent by initializing each layer agent."""
        super().setup()
        for agent in self.layer_agents:
            agent.setup()

    @override
    def step(self, observation: Tensor) -> Tensor:
        """Performs a step through the hierarchical agent, propagating actions
        and rewards through each layer.

        Args:
            observation: The input observation for the bottom layer.
        """
        for i, agent in enumerate(self.layer_agents):
            if self.counter % self.layer_timescale[i] != 0:
                continue
            upper_action = (
                self.action_to_lower_list[i + 1]
                if i < self.num_layers - 1
                else self.observation_to_upper_list[i]
            )
            upper_reward = (
                self.reward_to_lower_list[i + 1] if i < self.num_layers - 1 else None
            )
            if i > 0 and self.observation_to_upper_list[i - 1] is not None:
                lower_observation = self.observation_to_upper_list[i - 1]
            else:
                lower_observation = observation
            if lower_observation is None:
                raise ValueError(
                    f"Lower observation for layer {i} is None, which is not allowed."
                )
            observation = lower_observation
            layer_input = LayerInput(
                observation=observation,
                upper_action=upper_action,
                upper_reward=upper_reward,
            )
            layer_output = agent.step(layer_input)
            self.action_to_lower_list[i] = layer_output.action
            self.reward_to_lower_list[i] = layer_output.reward
            self.observation_to_upper_list[i] = layer_output.lower_observation

        if self.action_to_lower_list[0] is None:
            raise ValueError(
                "Action to lower layer for the top layer is None, which is not allowed."
            )
        action = self.action_to_lower_list[0]
        self.counter = (self.counter + 1) % self.period
        return action
