from pathlib import Path
from typing import Literal, override

import torch
from pamiq_core import Agent
from pamiq_core.utils.schedulers import StepIntervalScheduler
from torch import Tensor
from torch.distributions import Distribution

from exp.aim_utils import get_global_run
from exp.data import BufferName, DataKey
from exp.models import ModelName


class MetaCuriosityAgent(Agent[Tensor, Tensor]):
    """A reinforcement learning agent that uses meta-curiosity-driven
    exploration.

    This agent implements meta-curiosity-driven exploration by
    maintaining multiple forward dynamics models at different
    meta levels. Each level predicts observations at different
    abstraction levels, and prediction errors are combined with learned
    coefficients to generate intrinsic rewards. The agent uses a policy-
    value network for action selection. The meta aspect refers to
    curiosity about curiosity itself - the agent learns to modulate
    its own curiosity signals across different levels.
    """

    def __init__(
        self,
        num_meta_levels: int = 2,
        surprisal_coefficients_method: str
        | Literal[
            "minimize_to_maximize", "minimize", "maximize"
        ] = "minimize_to_maximize",
        log_every_n_steps: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the MetaCuriosityAgent.

        Args:
            num_meta_levels: Number of meta levels for forward dynamics models.
                Must be >= 1.
            surprisal_coefficients_method: Method to generate coefficients that weight
                prediction errors at each meta level. Options:
                - "minimize_to_maximize": Linear interpolation from -1 to 1
                - "minimize": All coefficients set to -1.0
                - "maximize": All coefficients set to 1.0
            log_every_n_steps: Frequency of logging metrics to Aim.
            device: Device to run computations on.
            dtype: Data type for tensors.
        Raises:
            ValueError: If num_meta_levels is less than 1.
        """
        super().__init__()

        if num_meta_levels < 1:
            raise ValueError(
                f"`num_meta_levels` must be >= 1! Your input: {num_meta_levels}"
            )

        match surprisal_coefficients_method:
            case "maximize":
                surprisal_coefficients = [1.0] * num_meta_levels
            case "minimize":
                surprisal_coefficients = [-1.0] * num_meta_levels
            case "minimize_to_maximize":
                surprisal_coefficients: list[float] = torch.linspace(
                    -1, 1, num_meta_levels
                ).tolist()
            case _:
                raise ValueError(
                    f"Unknown surprisal_coefficients_method: {surprisal_coefficients_method!r}"
                )

        self.num_meta_levels = num_meta_levels
        self.surprisal_coefficients = surprisal_coefficients

        self.metrics: dict[str, float] = {}
        self.scheduler = StepIntervalScheduler(log_every_n_steps, self.log_metrics)
        self.global_step = 0

        self.device = device
        self.dtype = dtype

        # Initialize hidden states (will be properly set in setup())
        self.policy_hidden_state: Tensor | None = None
        self.forward_dynamics_hiddens: list[Tensor | None] = [
            None
        ] * self.num_meta_levels

    @override
    def on_inference_models_attached(self) -> None:
        """Retrieve models when models are attached.

        Initializes forward dynamics models for each meta level and the
        policy-value network.
        """
        super().on_inference_models_attached()

        self.forward_dynamicses = [
            self.get_inference_model(ModelName.FORWARD_DYNAMICS + str(i))
            for i in range(self.num_meta_levels)
        ]

        self.policy_value = self.get_inference_model(ModelName.POLICY_VALUE)

    @override
    def on_data_collectors_attached(self) -> None:
        """Retrieve data collectors when collectors are attached.

        Initializes data collectors for policy and forward dynamics at
        each meta level.
        """
        super().on_data_collectors_attached()

        self.collector_policy = self.get_data_collector(BufferName.POLICY)
        self.collectors_fd = [
            self.get_data_collector(BufferName.FORWARD_DYNAMICS + str(i))
            for i in range(self.num_meta_levels)
        ]

    @override
    def setup(self) -> None:
        """Initialize agent state.

        Resets hidden states for policy and all forward dynamics models,
        clears predicted observations, and initializes step data
        collectors.
        """
        super().setup()
        self.predicted_obses: list[Tensor] | None = None

        self.step_data_policy: dict[str, Tensor] = {}
        self.step_data_fd = [dict[str, Tensor]() for _ in range(self.num_meta_levels)]

    @override
    def step(self, observation: Tensor) -> Tensor:
        """Process observation and return action for environment interaction.

        Computes intrinsic rewards from meta-level prediction errors,
        selects actions using the policy network, and updates forward
        dynamics predictions at all levels.

        Args:
            observation: Current observation from the environment.

        Returns:
            Selected action to be executed in the environment.
        """
        observation = observation.to(self.device, self.dtype)

        if self.predicted_obses is not None:
            target_obs = observation
            reward_sum = torch.tensor(0.0)
            for i, (pred_obs, coef) in enumerate(
                zip(
                    self.predicted_obses,
                    self.surprisal_coefficients,
                    strict=True,
                )
            ):
                # Store target observation
                self.step_data_fd[i][DataKey.TARGET] = target_obs

                # Collect forward dynamics step data.
                if DataKey.HIDDEN in (step_data_fd := self.step_data_fd[i]):
                    self.collectors_fd[i].collect(step_data_fd.copy())

                delta_obs = target_obs = target_obs - pred_obs
                reward = (delta_obs.square() * coef).mean().cpu()
                self.metrics[f"reward_{i}"] = reward.item()
                reward_sum += reward
            self.metrics["reward_sum"] = reward_sum.item()
            self.step_data_policy[DataKey.REWARD] = reward_sum
            if DataKey.HIDDEN in self.step_data_policy:
                self.collector_policy.collect(self.step_data_policy.copy())

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

        # Data Collection
        self.step_data_policy[DataKey.OBSERVATION] = observation.cpu()
        self.step_data_policy[DataKey.ACTION] = action.cpu()
        self.step_data_policy[DataKey.ACTION_LOG_PROB] = action_log_prob.cpu()
        self.step_data_policy[DataKey.VALUE] = value.cpu()
        self.metrics["value"] = value.cpu().item()

        # ==============================================================================
        #                           Forward Dynamics Process
        # ==============================================================================
        self.predicted_obses = []
        for i, fd in enumerate(self.forward_dynamicses):
            hidden = self.forward_dynamics_hiddens[i]
            if hidden is not None:
                self.step_data_fd[i][DataKey.HIDDEN] = hidden.cpu()
            self.step_data_fd[i].update(
                {
                    DataKey.OBSERVATION: observation.cpu(),
                    DataKey.ACTION: action.cpu(),
                }
            )
            pred_obs, hidden = fd(observation, action, hidden)
            self.forward_dynamics_hiddens[i] = hidden
            self.predicted_obses.append(pred_obs)

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
                run.track(
                    v,
                    name=k,
                    step=self.global_step,
                    context={"namespace": "agent", "curiosity_type": "meta"},
                )

    # ------ State Persistence ------

    @override
    def save_state(self, path: Path) -> None:
        """Save agent state to disk.

        Saves policy hidden state, forward dynamics hidden states for all levels,
        and global step counter.
        Hidden states can be None.

        Args:
            path: Directory path where to save the state.
        """
        super().save_state(path)
        path.mkdir(exist_ok=True)

        # Save policy hidden state
        if self.policy_hidden_state is not None:
            torch.save(self.policy_hidden_state, path / "policy_hidden_state.pt")

        # Save forward dynamics hidden states
        for i, hidden in enumerate(self.forward_dynamics_hiddens):
            if hidden is not None:
                torch.save(hidden, path / f"forward_dynamics_hidden_{i}.pt")

        # Save global step
        (path / "global_step").write_text(str(self.global_step), "utf-8")

    @override
    def load_state(self, path: Path) -> None:
        """Load agent state from disk.

        Restores policy hidden state, forward dynamics hidden states for all levels,
        and global step counter.
        Hidden states are set to None if the corresponding files don't exist.

        Args:
            path: Directory path from where to load the state.
        """
        super().load_state(path)

        # Load policy hidden state
        policy_hidden_path = path / "policy_hidden_state.pt"
        if policy_hidden_path.exists():
            self.policy_hidden_state = torch.load(
                policy_hidden_path, map_location=self.device
            )

        # Load forward dynamics hidden states
        self.forward_dynamics_hiddens = []
        for i in range(self.num_meta_levels):
            fd_hidden_path = path / f"forward_dynamics_hidden_{i}.pt"
            if fd_hidden_path.exists():
                hidden = torch.load(fd_hidden_path, map_location=self.device)
                self.forward_dynamics_hiddens.append(hidden)
            else:
                self.forward_dynamics_hiddens.append(None)

        # Load global step
        self.global_step = int((path / "global_step").read_text("utf-8"))
