from collections.abc import Callable
from pathlib import Path
from typing import override

import torch
import torch.nn.functional as F
from pamiq_core import Agent
from pamiq_core.utils.schedulers import StepIntervalScheduler
from torch import Tensor
from torch.distributions import Distribution

from exp.aim_utils import get_global_run
from exp.data import BufferName, DataKey
from exp.models import ModelName
from exp.utils import average_exponentially


class DeltaMinimizeAgent(Agent[Tensor, Tensor]):
    """A reinforcement learning agent that minimizes prediction error deltas.

    This agent implements curiosity-driven exploration by rewarding the reduction
    in prediction error differences over time. The reward is calculated as the
    negative delta of prediction errors: -(error_t - error_{t-1}), promoting
    actions that lead to continuous improvement in predictability.

    The agent focuses on minimizing the temporal derivative of prediction errors,
    encouraging behaviors that result in better forward dynamics learning.
    """

    def __init__(
        self,
        max_imagination_steps: int = 1,
        reward_average_method: Callable[[Tensor], Tensor] = average_exponentially,
        log_every_n_steps: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the DeltaMinimizeAgent.

        Args:
            max_imagination_steps: Maximum number of steps to imagine into the future. Must be >= 1.
            reward_average_method: Function to average prediction errors across imagination steps.
                Takes a tensor of errors (imagination_steps,) and returns a scalar error.
            log_every_n_steps: Frequency of logging metrics to Aim.
            device: Device to run computations on.
            dtype: Data type for tensors.

        Raises:
            ValueError: If max_imagination_steps is less than 1.
        """
        super().__init__()

        if max_imagination_steps < 1:
            raise ValueError(
                f"`max_imagination_steps` must be >= 1! Your input: {max_imagination_steps}"
            )

        self.head_forward_dynamics_hidden_state = None
        self.policy_hidden_state = None
        self.max_imagination_steps = max_imagination_steps
        self.reward_average_method = reward_average_method
        self.device = device
        self.dtype = dtype

        self.metrics: dict[str, float] = {}
        self.scheduler = StepIntervalScheduler(log_every_n_steps, self.log_metrics)

        self.global_step = 0
        self.previous_error: float | None = None

    @override
    def on_inference_models_attached(self) -> None:
        """Retrieve models when models are attached."""
        super().on_inference_models_attached()

        self.forward_dynamics = self.get_inference_model(ModelName.FORWARD_DYNAMICS)
        self.policy_value = self.get_inference_model(ModelName.POLICY_VALUE)

    @override
    def on_data_collectors_attached(self) -> None:
        """Retrieve data collectors when collectors are attached."""
        super().on_data_collectors_attached()
        self.collector_forward_dynamics = self.get_data_collector(
            BufferName.FORWARD_DYNAMICS
        )
        self.collector_policy = self.get_data_collector(BufferName.POLICY)

    # ------ INTERACTION PROCESS ------

    head_forward_dynamics_hidden_state: Tensor | None  # (depth, dim) or None
    policy_hidden_state: Tensor | None  # (depth, dim) or None
    obs_imaginations: Tensor  # (imaginations, dim)
    forward_dynamics_hidden_imaginations: (
        Tensor | None
    )  # (imaginations, depth, dim) or None
    step_data_policy: dict[str, Tensor]
    step_data_fd: dict[str, Tensor]

    @override
    def setup(self) -> None:
        """Initialize agent state.

        Resets step data collectors, imagination buffers, previous error
        tracker, and sets initial_step flag.
        """
        super().setup()
        self.step_data_fd, self.step_data_policy = {}, {}
        self.previous_error = None

        self.forward_dynamics_hidden_imaginations = None
        self.obs_imaginations = torch.empty(0, device=self.device, dtype=self.dtype)
        self.initial_step = True

    @override
    def step(self, observation: Tensor) -> Tensor:
        """Process observation and return action for environment interaction.

        Args:
            observation: Current observation from the environment

        Returns:
            Selected action to be executed in the environment
        """
        action = self._common_step(observation, self.initial_step)
        self.initial_step = False
        return action

    def _common_step(self, observation: Tensor, initial_step: bool) -> Tensor:
        """Execute the common step procedure for the delta minimize agent.

        Calculates delta minimization rewards from the reduction in prediction errors,
        selects actions using the policy network, and predicts future states using
        the forward dynamics model.

        Args:
            observation: Current observation from the environment
            initial_step: Whether this is the first step in an episode.
                When True, skips reward calculation and sets reward to 0.

        Returns:
            Selected action to be executed in the environment
        """
        observation = observation.to(
            device=self.device, dtype=self.dtype
        )  # convert type and send to device

        # ==============================================================================
        #                        Delta Minimization Reward Computation
        # ==============================================================================
        if not initial_step:
            target_obses = observation.expand_as(self.obs_imaginations)
            error_imaginations = (
                F.mse_loss(self.obs_imaginations, target_obses, reduction="none")
                .flatten(1)
                .mean(-1)
            )

            current_error = float(
                self.reward_average_method(error_imaginations).cpu().item()
            )

            if self.previous_error is None:
                reward = 0.0
            else:
                # Reward the reduction in prediction error (learning progress)
                reward = -(current_error - self.previous_error)

            self.metrics["reward"] = reward
            self.metrics["error"] = current_error

            self.step_data_policy[DataKey.REWARD] = torch.tensor(reward).cpu()
            if DataKey.HIDDEN in self.step_data_policy:
                self.collector_policy.collect(self.step_data_policy.copy())

            self.previous_error = current_error

        # ==============================================================================
        #                               Policy Process
        # ==============================================================================

        if self.policy_hidden_state is not None:
            self.step_data_policy[DataKey.HIDDEN] = (
                self.policy_hidden_state.cpu()
            )  # Store before update

        action_dist: Distribution
        value: Tensor
        action_dist, value, self.policy_hidden_state = self.policy_value(
            observation, self.policy_hidden_state
        )
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)

        # ==============================================================================
        #                           Forward Dynamics Process
        # ==============================================================================
        obs_imaginations = torch.cat(
            [observation[torch.newaxis], self.obs_imaginations]
        )[: self.max_imagination_steps]  # (imaginations, dim)
        if self.head_forward_dynamics_hidden_state is None:
            hidden_imaginations = None
        else:
            hidden_list = [self.head_forward_dynamics_hidden_state[torch.newaxis]]
            if self.forward_dynamics_hidden_imaginations is not None:
                hidden_list.append(self.forward_dynamics_hidden_imaginations)
            hidden_imaginations = torch.cat(hidden_list)[
                : self.max_imagination_steps
            ]  # (imaginations, depth, dim)

        obs_imaginations, hidden_imaginations = self.forward_dynamics(
            obs_imaginations,
            action.expand((len(obs_imaginations), *action.shape)),
            hidden_imaginations,
        )

        # ==============================================================================
        #                               Data Collection
        # ==============================================================================

        self.step_data_fd[DataKey.OBSERVATION] = self.step_data_policy[
            DataKey.OBSERVATION
        ] = observation.cpu()
        self.step_data_fd[DataKey.ACTION] = self.step_data_policy[DataKey.ACTION] = (
            action.cpu()
        )
        if self.head_forward_dynamics_hidden_state is not None:
            self.step_data_fd[DataKey.HIDDEN] = (
                self.head_forward_dynamics_hidden_state.cpu()
            )
            self.collector_forward_dynamics.collect(self.step_data_fd.copy())

        # Store for next loop
        self.step_data_policy[DataKey.ACTION_LOG_PROB] = action_log_prob.cpu()
        self.step_data_policy[DataKey.VALUE] = value.cpu()
        self.metrics["value"] = value.cpu().item()

        self.obs_imaginations = obs_imaginations
        self.forward_dynamics_hidden_imaginations = hidden_imaginations
        self.head_forward_dynamics_hidden_state = (
            hidden_imaginations[0] if hidden_imaginations is not None else None
        )

        self.scheduler.update()
        self.global_step += 1
        return action

    def log_metrics(self) -> None:
        """Log collected metrics to Aim.

        Writes all metrics in the metrics dictionary to Aim with the
        current global step.
        """
        if run := get_global_run():
            for k, v in self.metrics.items():
                run.track(v, name=f"delta-minimize-agent/{k}", step=self.global_step)

    # ------ State Persistence ------

    @override
    def save_state(self, path: Path) -> None:
        """Save agent state to disk.

        Saves forward dynamics hidden state, policy hidden state, global step counter,
        and previous error value. Hidden states can be None.

        Args:
            path: Directory path where to save the state
        """
        super().save_state(path)
        path.mkdir(exist_ok=True)

        if self.head_forward_dynamics_hidden_state is not None:
            torch.save(
                self.head_forward_dynamics_hidden_state,
                path / "head_forward_dynamics_hidden_state.pt",
            )
        if self.policy_hidden_state is not None:
            torch.save(self.policy_hidden_state, path / "policy_hidden_state.pt")

        (path / "global_step").write_text(str(self.global_step), "utf-8")

        if self.previous_error is not None:
            (path / "previous_error").write_text(str(self.previous_error), "utf-8")

    @override
    def load_state(self, path: Path) -> None:
        """Load agent state from disk.

        Restores forward dynamics hidden state, policy hidden state, global step counter,
        and previous error value. Hidden states are set to None if the corresponding
        files don't exist.

        Args:
            path: Directory path from where to load the state
        """
        super().load_state(path)

        fd_hidden_path = path / "head_forward_dynamics_hidden_state.pt"
        if fd_hidden_path.exists():
            self.head_forward_dynamics_hidden_state = torch.load(
                fd_hidden_path, map_location=self.device
            )

        policy_hidden_path = path / "policy_hidden_state.pt"
        if policy_hidden_path.exists():
            self.policy_hidden_state = torch.load(
                policy_hidden_path, map_location=self.device
            )

        self.global_step = int((path / "global_step").read_text("utf-8"))

        previous_error_path = path / "previous_error"
        if previous_error_path.exists():
            self.previous_error = float(previous_error_path.read_text("utf-8"))
