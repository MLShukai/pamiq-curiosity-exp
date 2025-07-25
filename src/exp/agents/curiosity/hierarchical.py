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
    action_from_upper: Tensor | None
    reward_from_upper: Tensor | None


@dataclass(frozen=True)
class LayerOutput:
    observation_from_lower: Tensor
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
            observation.action_from_upper,
            observation.reward_from_upper,
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

        return LayerOutput(
            observation_from_lower=latent_obs, action=action, reward=reward
        )

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
