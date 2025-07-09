from collections.abc import Callable
from pathlib import Path
from typing import override

import torch
from pamiq_core import Agent
from pamiq_core.utils.schedulers import StepIntervalScheduler
from torch import Tensor
from torch.distributions import Distribution

from exp.aim_utils import get_global_run
from exp.data import BufferName, DataKey
from exp.models import ModelName
from exp.utils import average_exponentially


class HierarchicalCuriosityAgent(Agent[Tensor, Tensor]):
    """A reinforcement learning agent that uses curiosity-driven exploration
    through forward dynamics prediction.

    This agent implements curiosity-driven exploration by predicting
    future observations and using prediction errors as intrinsic
    rewards. It maintains a forward dynamics model to predict future
    states and a policy-value network for action selection.
    """

    def __init__(
        self,
        num_hierarchical_levels: int,
        log_every_n_steps: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the AdversarialCuriosityAgent.

        Args:
            num_hierarchical_levels: Number of hierarchical levels for the agent.
            log_every_n_steps: Frequency of logging metrics to Aim. Defaults to 1.
            device: Device to run computations on. Defaults to None.
            dtype: Data type for tensors. Defaults to None.

        Raises:
            ValueError: If num_hierarchical_levels is less than 1.
        """
        super().__init__()
        if num_hierarchical_levels < 1:
            raise ValueError("`num_hierarchical_levels` must be >= 1")
        self.num_hierarchical_levels = num_hierarchical_levels

        self.prev_observation_list: list[None | Tensor] = [
            None
        ] * num_hierarchical_levels
        self.prev_action_list: list[None | Tensor] = [None] * num_hierarchical_levels
        self.prev_fd_hidden_list: list[None | Tensor] = [None] * num_hierarchical_levels
        self.prev_reward_vector_list: list[None | Tensor] = [
            None
        ] * num_hierarchical_levels
        self.prev_policy_hidden_list: list[None | Tensor] = [
            None
        ] * num_hierarchical_levels

        self.device = device
        self.dtype = dtype

        self.metrics: dict[str, float] = {}
        self.scheduler = StepIntervalScheduler(log_every_n_steps, self.log_metrics)

        self.global_step = 0

    @override
    def on_inference_models_attached(self) -> None:
        """Retrieve models when models are attached."""
        super().on_inference_models_attached()

        self.forward_dynamics_list = [
            self.get_inference_model(ModelName.FORWARD_DYNAMICS + str(i))
            for i in range(self.num_hierarchical_levels)
        ]
        self.policy_value_list = [
            self.get_inference_model(ModelName.POLICY_VALUE + str(i))
            for i in range(self.num_hierarchical_levels)
        ]

    @override
    def on_data_collectors_attached(self) -> None:
        """Retrieve data collectors when collectors are attached."""
        super().on_data_collectors_attached()
        self.collector_forward_dynamics_list = [
            self.get_data_collector(BufferName.FORWARD_DYNAMICS + str(i))
            for i in range(self.num_hierarchical_levels)
        ]
        self.collector_policy_list = [
            self.get_data_collector(BufferName.POLICY + str(i))
            for i in range(self.num_hierarchical_levels)
        ]

    @override
    def step(self, observation: Tensor) -> Tensor:
        """Execute the common step procedure for the curiosity-driven agent.

        Calculates intrinsic rewards from prediction errors, selects actions
        using the policy network, and predicts future states using the forward
        dynamics model.

        Args:
            observation: Current observation from the environment

        Returns:
            Selected action to be executed in the environment
        """
        observation = observation.to(
            device=self.device, dtype=self.dtype
        )  # convert type and send to device

        for i in range(self.num_hierarchical_levels):
            prev_observation = self.prev_observation_list[i]
            self.prev_observation_list[i] = observation
            prev_action = self.prev_action_list[i]
            prev_fd_hidden = self.prev_fd_hidden_list[i]

            if (
                prev_observation is not None
                and prev_action is not None
                and prev_fd_hidden is not None
            ):
                step_data_fd = {}
                step_data_fd[DataKey.OBSERVATION] = prev_observation.cpu()
                step_data_fd[DataKey.ACTION] = prev_action.cpu()
                step_data_fd[DataKey.HIDDEN] = prev_fd_hidden.cpu()
                self.collector_forward_dynamics_list[i].collect(step_data_fd.copy())

            forward_dynamics = self.forward_dynamics_list[i]
            obs_dist, fd_hidden = forward_dynamics(
                prev_observation, prev_action, prev_fd_hidden
            )
            self.prev_fd_hidden_list[i] = fd_hidden
            surprisal_vector = -obs_dist.log_prob(observation)
            reward_vector = (
                torch.where(
                    torch.rand(surprisal_vector.shape, device=self.device)
                    < i / (self.num_hierarchical_levels - 1),
                    1,
                    -1,
                )
                * surprisal_vector
            )
            self.prev_reward_vector_list[i] = reward_vector
            next_level_action = (
                self.prev_action_list[i + 1]
                if i + 1 < self.num_hierarchical_levels
                else surprisal_vector
            )
            policy_value = self.policy_value_list[i]
            prev_policy_hidden_state = self.prev_policy_hidden_list[i]
            action_dist, value, policy_hidden_state = policy_value(
                observation, next_level_action, prev_policy_hidden_state
            )
            action = action_dist.sample()
            self.prev_action_list[i] = action
            self.metrics["value" + str(i)] = value.item()
            self.prev_policy_hidden_list[i] = policy_hidden_state
            if next_level_action is not None and prev_policy_hidden_state is not None:
                step_data_policy = {}
                step_data_policy[DataKey.OBSERVATION] = observation.cpu()
                step_data_policy[DataKey.ACTION] = next_level_action.cpu()
                step_data_policy[DataKey.HIDDEN] = prev_policy_hidden_state.cpu()
                step_data_policy[DataKey.ACTION_LOG_PROB] = action_dist.log_prob(
                    action
                ).cpu()
                step_data_policy[DataKey.VALUE] = value.cpu()
                next_level_reward_vector = (
                    self.prev_reward_vector_list[i + 1]
                    if i + 1 < self.num_hierarchical_levels
                    else None
                )
                if next_level_reward_vector is not None:
                    reward = torch.cat((reward_vector, next_level_reward_vector)).mean()
                else:
                    reward = reward_vector.mean()
                self.metrics["reward" + str(i)] = reward.item()
                step_data_policy[DataKey.REWARD] = reward.cpu()
                self.collector_policy_list[i].collect(step_data_policy.copy())
            observation = surprisal_vector

        self.scheduler.update()
        self.global_step += 1
        return (
            self.prev_action_list[0]
            if self.prev_action_list[0] is not None
            else torch.zeros(0)
        )  # dummy

    def log_metrics(self) -> None:
        """Log collected metrics to Aim.

        Writes all metrics in the metrics dictionary to Aim with the
        current global step.
        """
        if run := get_global_run():
            for k, v in self.metrics.items():
                run.track(v, name=f"curiosity-agent/{k}", step=self.global_step)

    # ------ State Persistence ------

    @override
    def save_state(self, path: Path) -> None:
        """Save agent state to disk.

        Saves forward dynamics hidden state list, policy hidden state list, and temporary action, observation, reward, and global step counter.
        Hidden states can be None.

        Args:
            path: Directory path where to save the state
        """
        super().save_state(path)
        path.mkdir(exist_ok=True)

        (path / "global_step").write_text(str(self.global_step), "utf-8")
        torch.save(
            {
                "prev_observation_list": self.prev_observation_list,
                "prev_action_list": self.prev_action_list,
                "prev_fd_hidden_list": self.prev_fd_hidden_list,
                "prev_reward_vector_list": self.prev_reward_vector_list,
                "prev_policy_hidden_list": self.prev_policy_hidden_list,
            },
            path / "hierarchical_curiosity_agent_state.pt",
        )

    @override
    def load_state(self, path: Path) -> None:
        """Load agent state from disk.

        Restores forward dynamics hidden state, policy hidden state, and global step counter.
        Hidden states are set to None if the corresponding files don't exist.

        Args:
            path: Directory path from where to load the state
        """
        super().load_state(path)

        self.global_step = int((path / "global_step").read_text("utf-8"))
        prev_states = torch.load(
            path / "hierarchical_curiosity_agent_state.pt", map_location=self.device
        )
        self.prev_observation_list = prev_states["prev_observation_list"]
        self.prev_action_list = prev_states["prev_action_list"]
        self.prev_fd_hidden_list = prev_states["prev_fd_hidden_list"]
        self.prev_reward_vector_list = prev_states["prev_reward_vector_list"]
        self.prev_policy_hidden_list = prev_states["prev_policy_hidden_list"]
