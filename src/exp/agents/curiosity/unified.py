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

STEP_DATA_REQUIRED_KEYS = {
    DataKey.ACTION,
    DataKey.ACTION_LOG_PROB,
    DataKey.OBSERVATION,
    DataKey.HIDDEN,
    DataKey.VALUE,
    DataKey.REWARD,
}


class UnifiedAdversarialCuriosityAgent(Agent[Tensor, Tensor]):
    """A reinforcement learning agent that uses curiosity-driven exploration
    through forward dynamics prediction.

    This agent implements curiosity-driven exploration by predicting
    future observations and using prediction errors as intrinsic
    rewards. It maintains a forward dynamics model to predict future
    states and a policy-value network for action selection.
    """

    def __init__(
        self,
        max_imagination_steps: int = 1,
        reward_average_method: Callable[[Tensor], Tensor] = average_exponentially,
        log_every_n_steps: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the AdversarialCuriosityAgent.

        Args:
            max_imagination_steps: Maximum number of steps to imagine into the future. Must be >= 1. Defaults to 1.
            reward_average_method: Function to average rewards across imagination steps.
                Takes a tensor of rewards (imagination_steps,) and returns a scalar reward. Defaults to average_exponentially.
            log_every_n_steps: Frequency of logging metrics to Aim. Defaults to 1.
            device: Device to run computations on. Defaults to None.
            dtype: Data type for tensors. Defaults to None.

        Raises:
            ValueError: If max_imagination_steps is less than 1.
        """
        super().__init__()

        if max_imagination_steps < 1:
            raise ValueError(
                f"`max_imagination_steps` must be >= 1! Your input: {max_imagination_steps}"
            )

        self.hidden_state = None
        self.action = None
        self.obs_hat = None
        self.max_imagination_steps = max_imagination_steps
        self.reward_average_method = reward_average_method
        self.device = device
        self.dtype = dtype

        self.metrics: dict[str, float] = {}
        self.scheduler = StepIntervalScheduler(log_every_n_steps, self.log_metrics)
        self.step_data_policy_required_keys = STEP_DATA_REQUIRED_KEYS.copy()

        self.global_step = 0

    @override
    def on_inference_models_attached(self) -> None:
        """Retrieve models when models are attached."""
        super().on_inference_models_attached()

        self.fd_piv = self.get_inference_model(ModelName.FD_POLICY_VALUE)

    @override
    def on_data_collectors_attached(self) -> None:
        """Retrieve data collectors when collectors are attached."""
        super().on_data_collectors_attached()
        self.collector_fd_piv = self.get_data_collector(BufferName.FD_POLICY_VALUE)

    # ------ INTERACTION PROCESS ------

    hidden_state: Tensor | None  # (depth, dim) or None
    action: Tensor | None  # (action_choices,) or None
    obs_hat: Tensor | None
    step_data_fd_piv: dict[str, Tensor]

    @override
    def setup(self) -> None:
        """Initialize agent state.

        Resets step data collectors, imagination buffers
        """
        super().setup()
        self.step_data_fd_piv = {}

        self.forward_dynamics_hidden_imaginations = None
        self.obs_imaginations = torch.empty(0, device=self.device, dtype=self.dtype)

    @override
    def step(self, observation: Tensor) -> Tensor:
        """Process observation and return action for environment interaction.

        Args:
            observation: Current observation from the environment

        Returns:
            Selected action to be executed in the environment
        """
        action = self._common_step(observation)
        return action

    def _common_step(self, observation: Tensor) -> Tensor:
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

        # ==============================================================================
        #                             Reward Computation
        # ==============================================================================
        if self.obs_hat is not None:
            reward_imaginations = (
                F.mse_loss(self.obs_hat, observation, reduction="none")
                .flatten(1)
                .mean(-1)
            )

            reward = self.reward_average_method(reward_imaginations)
            self.metrics["reward"] = reward.item()

            self.step_data_fd_piv[DataKey.REWARD] = reward.cpu()

        # ==============================================================================
        #                               Forward Dynamics and Policy Process
        # ==============================================================================

        if self.hidden_state is not None:
            self.step_data_fd_piv[DataKey.HIDDEN] = (
                self.hidden_state.cpu()
            )  # Store before update

        action_dist: Distribution
        value: Tensor
        self.obs_hat, action_dist, value, self.hidden_state = self.fd_piv(
            observation, self.action, hidden=self.hidden_state
        )
        self.action = action_dist.sample()
        action_log_prob = action_dist.log_prob(self.action)

        # ==============================================================================
        #                               Data Collection
        # ==============================================================================

        self.step_data_fd_piv[DataKey.OBSERVATION] = self.step_data_fd_piv[
            DataKey.OBSERVATION
        ] = observation.cpu()
        self.step_data_fd_piv[DataKey.ACTION] = self.step_data_fd_piv[
            DataKey.ACTION
        ] = self.action.cpu()

        if set(self.step_data_fd_piv.keys()) >= self.step_data_policy_required_keys:
            self.collector_fd_piv.collect(self.step_data_fd_piv.copy())

        # Store for next loop
        self.step_data_fd_piv[DataKey.ACTION_LOG_PROB] = action_log_prob.cpu()
        self.step_data_fd_piv[DataKey.VALUE] = value.cpu()
        self.metrics["value"] = value.cpu().item()

        self.scheduler.update()
        self.global_step += 1
        return self.action

    def log_metrics(self) -> None:
        """Log collected metrics to Aim.

        Writes all metrics in the metrics dictionary to Aim with the
        current global step.
        """
        if run := get_global_run():
            for k, v in self.metrics.items():
                run.track(
                    v,
                    name=k,
                    step=self.global_step,
                    context={"namespace": "agent", "curiosity_type": "adversarial"},
                )

    # ------ State Persistence ------

    @override
    def save_state(self, path: Path) -> None:
        """Save agent state to disk.

        Saves forward dynamics hidden state, policy hidden state, and global step counter.
        Hidden states can be None.

        Args:
            path: Directory path where to save the state
        """
        super().save_state(path)
        path.mkdir(exist_ok=True)

        if self.hidden_state is not None:
            torch.save(self.hidden_state, path / "hidden_state.pt")
        if self.action is not None:
            torch.save(self.action, path / "action.pt")
        if self.obs_hat is not None:
            torch.save(self.obs_hat, path / "obs_hat.pt")
        (path / "global_step").write_text(str(self.global_step), "utf-8")

    @override
    def load_state(self, path: Path) -> None:
        """Load agent state from disk.

        Restores forward dynamics hidden state, policy hidden state, and global step counter.
        Hidden states are set to None if the corresponding files don't exist.

        Args:
            path: Directory path from where to load the state
        """
        super().load_state(path)

        hidden_path = path / "hidden_state.pt"
        self.hidden_state = (
            torch.load(hidden_path, map_location=self.device)
            if hidden_path.exists()
            else None
        )
        action_path = path / "action.pt"
        self.action = (
            torch.load(action_path, map_location=self.device)
            if action_path.exists()
            else None
        )
        obs_hat_path = path / "obs_hat.pt"
        self.obs_hat = (
            torch.load(obs_hat_path, map_location=self.device)
            if obs_hat_path.exists()
            else None
        )

        self.global_step = int((path / "global_step").read_text("utf-8"))
