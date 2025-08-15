from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import torch
import torch.nn.functional as F
from pamiq_core.torch import TorchAgent
from torch import Tensor

from exp.data import BufferName, DataKey
from exp.models import ModelName
from exp.models.latent_fd_new import LatentFDFramework
from exp.models.latent_policy_new import LatentPiVFramework
from exp.models.utils import layer_norm


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


class LayerCuriosityAgent(TorchAgent[LayerInput, LayerOutput]):
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
        is_top: bool,
        device: torch.device | None = None,
    ) -> None:
        """Initializes the LayerCuriosityAgent with model buffer suffix, reward
        coefficient, and reward lerp ratio.

        Args:
            model_buffer_suffix: Suffix for the model buffer names.
            reward_coef: Coefficient for the reward computation.
            reward_lerp_ratio: Ratio for linear interpolation of rewards.
            is_top: Top layer or not.
        """
        super().__init__()
        self.reward_coef = reward_coef
        self.reward_lerp_ratio = reward_lerp_ratio
        self.model_buffer_suffix = model_buffer_suffix
        self.is_top = is_top

        self.obs_hat: Tensor | None = None
        self.policy_encoder_hidden: Tensor | None = None
        self.fd_encoder_hidden: Tensor | None = None
        self.device = device or torch.get_default_device()

        self.step_data_policy_required_keys = STEP_DATA_POLICY_REQUIRED_KEYS.copy()
        if is_top:
            self.step_data_policy_required_keys.discard(DataKey.UPPER_ACTION)

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
        self.policy_value = self.get_torch_inference_model(
            ModelName.POLICY_VALUE + self.model_buffer_suffix, LatentPiVFramework
        )
        self.forward_dynamics = self.get_torch_inference_model(
            ModelName.FORWARD_DYNAMICS + self.model_buffer_suffix, LatentFDFramework
        )

    @override
    def setup(self) -> None:
        """Sets up the agent by initializing step data dictionaries for policy
        and forward dynamics."""
        super().setup()
        self.step_data_fd = dict[str, Any]()
        self.step_data_policy = dict[str, Any]()

    @override
    @torch.inference_mode()
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

        if set(self.step_data_policy.keys()) == self.step_data_policy_required_keys:
            self.policy_collector.collect(self.step_data_policy.copy())

        if self.policy_encoder_hidden is not None:
            self.step_data_policy[DataKey.HIDDEN] = self.policy_encoder_hidden.cpu()

        if upper_action is not None and not self.is_top:
            self.step_data_policy[DataKey.UPPER_ACTION] = upper_action.cpu()

        with self.policy_value.unwrap() as m:
            latent_action, self.policy_encoder_hidden = m.encoder(
                obs, self.policy_encoder_hidden, upper_action, no_len=True
            )
            action_dist, value = m.generator(latent_action)

        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)

        self.step_data_policy[DataKey.OBSERVATION] = obs.cpu()
        self.step_data_policy[DataKey.ACTION] = action.cpu()
        self.step_data_policy[DataKey.ACTION_LOG_PROB] = action_log_prob.cpu()
        self.step_data_policy[DataKey.VALUE] = value.cpu()

        # ================================================
        #             Forward Dynamics Step
        # ================================================

        if self.fd_encoder_hidden is not None:
            self.step_data_fd[DataKey.HIDDEN] = self.fd_encoder_hidden.cpu()

        with self.forward_dynamics.unwrap() as m:
            latent_obs, self.fd_encoder_hidden = m.encoder(
                obs, latent_action, self.fd_encoder_hidden, no_len=True
            )
            self.obs_hat = m.predictor(latent_obs)

        self.step_data_fd[DataKey.OBSERVATION] = obs.cpu()
        self.step_data_fd[DataKey.ACTION] = latent_action.cpu()

        if set(self.step_data_fd.keys()) == STEP_DATA_FD_REQUIRED_KEYS:
            self.fd_collector.collect(self.step_data_fd.copy())

        # ================================================
        #                 Return Output
        # ================================================

        return LayerOutput(
            lower_observation=layer_norm(latent_obs), action=action, reward=reward
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
        if self.policy_encoder_hidden is not None:
            torch.save(self.policy_encoder_hidden, path / "policy_encoder_hidden.pt")
        if self.fd_encoder_hidden is not None:
            torch.save(self.fd_encoder_hidden, path / "fd_encoder_hidden.pt")

    @override
    def load_state(self, path: Path):
        """Loads the agent's state, including the observation prediction,
        policy hidden state, and forward dynamics hidden state.

        Args:
            path: The path to load the state from.
        """
        super().load_state(path)
        self.obs_hat = torch.load(path / "obs_hat.pt", map_location=self.device)
        if (p := path / "policy_encoder_hidden.pt").is_file():
            self.policy_encoder_hidden = torch.load(p, map_location=self.device)
        if (p := path / "fd_encoder_hidden.pt").is_file():
            self.fd_encoder_hidden = torch.load(p, map_location=self.device)


def create_reward_coef(method: str, num_layers: int) -> list[float]:
    """Creates a list of reward coefficients based on the specified method.

    Args:
        method: The method to use for creating reward coefficients.
        num_layers: The number of layers in the hierarchical agent.

    Returns:
        A list of reward coefficients for each layer.
    """
    match method:
        case "minimize_all":
            return [-1.0] * num_layers
        case "maximize_all":
            return [1.0] * num_layers
        case "minimize_lower_half":
            return [-1.0] * (num_layers // 2) + [1.0] * (num_layers - num_layers // 2)
        case "maximize_lower_half":
            return [1.0] * (num_layers // 2) + [-1.0] * (num_layers - num_layers // 2)
        case "lerp_min_max":
            if num_layers < 2:
                raise ValueError(
                    "Number of layers must be at least 2 for lerp_min_max."
                )
            return [-1.0 + (i / (num_layers - 1)) * 2.0 for i in range(num_layers)]
        case "lerp_max_min":
            if num_layers < 2:
                raise ValueError(
                    "Number of layers must be at least 2 for lerp_max_min."
                )
            return [1.0 - (i / (num_layers - 1)) * 2.0 for i in range(num_layers)]
        case _:
            raise ValueError(f"Unknown reward coefficient method: {method}")


def create_layer_timescale(
    method: str, num_layers: int, timescale_multiplier: int = 2
) -> list[int]:
    """Creates a list of layer timescales based on the specified method.

    Args:
        method: The method to use for creating layer timescales.
        num_layers: The number of layers in the hierarchical agent.
        timescale_multiplier: The multiplier for exponential growth.

    Returns:
        A list of layer timescales for each layer.
    """
    match method:
        case "exponential_growth":
            if num_layers < 1:
                raise ValueError(
                    "Number of layers must be at least 1 for exponential growth."
                )
            if timescale_multiplier < 1:
                raise ValueError(
                    "Timescale multiplier must be at least 1 for exponential growth."
                )
            # Generate timescales as powers of the multiplier
            # e.g., [1, 2, 4, 8] for num_layers = 4 and timescale_multiplier = 2
            return [timescale_multiplier**i for i in range(num_layers)]
        case _:
            raise ValueError(f"Unknown layer timescale method: {method}")


class HierarchicalCuriosityAgent(TorchAgent[Tensor, Tensor]):
    """Hierarchical Curiosity Agent that manages multiple LayerCuriosityAgents.

    This agent coordinates multiple layers of curiosity agents, each
    with its own time scale and action/reward propagation.
    """

    def __init__(
        self,
        reward_lerp_ratio: float,
        reward_coefficients: list[float],
        timescales: list[int],
        device_list: list[torch.device | None] | torch.device | None = None,
    ) -> None:
        """Initializes the HierarchicalCuriosityAgent with layer agents and
        their respective time scales.

        Args:
            layer_agent_dict: Dictionary mapping layer names to LayerCuriosityAgent instances.
            layer_timescale: List of time scales for each layer agent.
        """
        self.num_layers = len(reward_coefficients)
        model_key_list = [str(i) for i in range(self.num_layers)]
        self.layer_agent_dict: dict[str, LayerCuriosityAgent] = {}
        if not isinstance(device_list, Sequence):
            device_list = [device_list] * self.num_layers
        if not (
            len(device_list)
            == len(reward_coefficients)
            == len(timescales)
            == self.num_layers
        ):
            raise ValueError(
                "device_list, reward_coef_list, and timescale_list must have the same length as num_layers."
            )

        def is_top(index: int) -> bool:
            return (index + 1) == self.num_layers

        for i, (reward_coef, model_key, device) in enumerate(
            zip(reward_coefficients, model_key_list, device_list, strict=True)
        ):
            layer_agent = LayerCuriosityAgent(
                model_buffer_suffix=model_key,
                reward_coef=reward_coef,
                reward_lerp_ratio=reward_lerp_ratio,
                is_top=is_top(i),
                device=device,
            )
            self.layer_agent_dict[model_key] = layer_agent
        super().__init__(self.layer_agent_dict)

        self.timescales = timescales
        self.period = timescales[-1]  # The period of the last layer

        self.action_to_lower_list: list[Tensor | None] = [None] * (
            self.num_layers + 1
        )  # +1 for top (always None)
        self.reward_to_lower_list: list[Tensor | None] = [None] * (
            self.num_layers + 1
        )  # +1 for top (always None)

        self.counter = 0

    @override
    def step(self, observation: Tensor) -> Tensor:
        """Performs a step through the hierarchical agent, propagating actions
        and rewards through each layer.

        Args:
            observation: The input observation for the bottom layer.
        """
        for i, agent in enumerate(self.layer_agent_dict.values()):
            if self.counter % self.timescales[i] != 0:
                continue
            upper_action = self.action_to_lower_list[i + 1]
            upper_reward = self.reward_to_lower_list[i + 1]
            layer_input = LayerInput(
                observation=observation,
                upper_action=upper_action,
                upper_reward=upper_reward,
            )
            layer_output = agent.step(layer_input)
            self.action_to_lower_list[i] = layer_output.action
            self.reward_to_lower_list[i] = layer_output.reward
            observation = layer_output.lower_observation

        if self.action_to_lower_list[0] is None:
            raise ValueError(
                "Action to lower layer for the top layer is None, some program logic is broken."
            )
        action = self.action_to_lower_list[0]
        self.counter = (self.counter + 1) % self.period
        return action

    @override
    def save_state(self, path: Path) -> None:
        super().save_state(path)
        path.mkdir(exist_ok=True)

        torch.save(self.action_to_lower_list, path / "action_to_lower_list.pt")
        torch.save(self.reward_to_lower_list, path / "reward_to_lower_list.pt")
        (path / "counter").write_text(str(self.counter), encoding="utf-8")

    @override
    def load_state(self, path: Path) -> None:
        super().load_state(path)

        self.action_to_lower_list = torch.load(path / "action_to_lower_list.pt")
        self.reward_to_lower_list = torch.load(path / "reward_to_lower_list.pt")
        self.counter = int((path / "counter").read_text("utf-8"))
