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
from exp.models.components.multi_distributions import MultiDistributions

STEP_DATA_REQUIRED_KEYS = {
    DataKey.OBSERVATION,
    DataKey.TARGET,
    DataKey.HIDDEN,
    DataKey.PREVIOUS_ACTION,
    DataKey.ACTION,
    DataKey.ACTION_LOG_PROB,
    DataKey.INTERNAL_STATE,
    DataKey.VALUE,
    DataKey.REWARD,
}

Hidden_Time = Tensor
Hidden_TTT = list[dict[str, Tensor]]
Hidden = dict[str, Hidden_Time | Hidden_TTT]
Action = dict[str, Tensor]


class TTTCuriosityAgent(Agent[Tensor, Tensor]):
    """A reinforcement learning agent that uses curiosity-driven exploration
    through forward dynamics prediction.

    This agent implements curiosity-driven exploration by predicting
    future observations and using prediction errors as intrinsic
    rewards. It maintains a forward dynamics model to predict future
    states and a policy-value network for action selection.
    """

    def __init__(
        self,
        log_every_n_steps: int = 1,
        surprisal_mean_ema_decay: float = 0.9999,
        fatigue_decay: float = 0.99,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the AdversarialCuriosityAgent.

        Args:
            log_every_n_steps: Frequency of logging metrics to Aim. Defaults to 1.
            device: Device to run computations on. Defaults to None.
            dtype: Data type for tensors. Defaults to None.

        Raises:
            ValueError: If max_imagination_steps is less than 1.
        """
        super().__init__()

        self.hidden_state = None
        self.external_action = None
        self.internal_action = None
        self.surprisal_mean_ema_decay = surprisal_mean_ema_decay
        self.surprisal_mean_ema = None
        self.shallow_surprisal_mean_ema = None
        self.deep_surprisal_mean_ema = None
        self.surprisal_coef = None
        self.fatigue = None
        self.fatigue_decay = fatigue_decay
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

    hidden_state: Hidden | None
    external_action: Tensor | None  # (action_choices,) or None
    internal_action: Tensor | None  # (dim,) or None
    surprisal_mean_ema: Tensor | None
    shallow_surprisal_mean_ema: Tensor | None
    deep_surprisal_mean_ema: Tensor | None
    surprisal_coef: Tensor | None
    fatigue: Tensor | None
    obs_hat: Tensor | None
    step_data_fd_piv: dict[str, Tensor | Hidden | Action]

    @override
    def setup(self) -> None:
        """Initialize agent state.

        Resets step data collectors, imagination buffers
        """
        super().setup()
        self.step_data_fd_piv = {}

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

        if set(self.step_data_fd_piv.keys()) >= self.step_data_policy_required_keys:
            self.collector_fd_piv.collect(self.step_data_fd_piv.copy())

        # ==============================================================================
        #                               Forward Dynamics and Policy Process
        # ==============================================================================

        if self.hidden_state is not None:
            hidden_time: Hidden_Time = (
                self.hidden_state["time"].cpu()
                if isinstance(self.hidden_state["time"], Tensor)
                else torch.tensor(0)
            )
            hidden_ttt: Hidden_TTT = (
                [
                    {key: val.cpu() for key, val in layer.items()}
                    for layer in self.hidden_state["ttt"]
                ]
                if isinstance(self.hidden_state["ttt"], list)
                else []
            )
            self.step_data_fd_piv[DataKey.HIDDEN] = {
                "time": hidden_time,
                "ttt": hidden_ttt,
            }

            self.metrics["hidden_time_norm"] = torch.norm(
                self.hidden_state["time"]
            ).item()
            self.metrics["hidden_ttt_norm"] = sum(
                torch.norm(v).item()
                for layer in self.hidden_state["ttt"]
                for v in layer.values()
            )

        internal_state = (
            torch.clamp(self.fatigue, 0.0, 2.0) - 1.0
            if self.fatigue is not None
            else None
        )

        action_dist: MultiDistributions
        value: Tensor
        (
            obs_embedding,
            _,
            action_dist,
            value,
            self.hidden_state,
            surprisal,
        ) = self.fd_piv(
            observation,
            self.external_action,
            self.internal_action,
            internal_state,
            hidden=self.hidden_state,
        )
        if self.external_action is not None and self.internal_action is not None:
            self.step_data_fd_piv[DataKey.PREVIOUS_ACTION] = {
                "external_action": self.external_action.cpu(),
                "internal_action": self.internal_action.cpu(),
            }
        self.external_action, self.internal_action = action_dist.sample()
        action_log_prob = action_dist.log_prob(
            self.external_action, self.internal_action
        )
        if internal_state is not None:
            self.step_data_fd_piv[DataKey.INTERNAL_STATE] = internal_state.cpu()
        # ==============================================================================
        #                             Reward Computation
        # ==============================================================================
        if self.surprisal_coef is None:
            surprisal_depth, surprisal_num_head, surprisal_dim_per_head = (
                surprisal.shape
            )
            surprisal_rand = torch.rand_like(surprisal, device=self.device)
            surprisal_depth_factor = torch.linspace(
                0.0, 1.0, surprisal_depth, device=self.device
            )[:, None, None].expand(-1, surprisal_num_head, surprisal_dim_per_head)
            self.surprisal_coef = (surprisal_rand < surprisal_depth_factor).float()
        shallow_surprisal = surprisal * (1 - self.surprisal_coef)
        deep_surprisal = surprisal * self.surprisal_coef

        surprisal_mean: Tensor = surprisal.mean().detach()
        if self.surprisal_mean_ema is None:
            self.surprisal_mean_ema = surprisal_mean
        self.surprisal_mean_ema = (
            self.surprisal_mean_ema * self.surprisal_mean_ema_decay
            + surprisal_mean * (1 - self.surprisal_mean_ema_decay)
        )
        shallow_surprisal_mean: Tensor = shallow_surprisal.mean().detach()
        if self.shallow_surprisal_mean_ema is None:
            self.shallow_surprisal_mean_ema = shallow_surprisal_mean
        self.shallow_surprisal_mean_ema = (
            self.shallow_surprisal_mean_ema * self.surprisal_mean_ema_decay
            + shallow_surprisal_mean * (1 - self.surprisal_mean_ema_decay)
        )
        deep_surprisal_mean: Tensor = deep_surprisal.mean().detach()
        if self.deep_surprisal_mean_ema is None:
            self.deep_surprisal_mean_ema = deep_surprisal_mean
        self.deep_surprisal_mean_ema = (
            self.deep_surprisal_mean_ema * self.surprisal_mean_ema_decay
            + deep_surprisal_mean * (1 - self.surprisal_mean_ema_decay)
        )

        self.metrics["surprisal"] = surprisal_mean.item()
        self.metrics["shallow_surprisal"] = shallow_surprisal_mean.item()
        self.metrics["deep_surprisal"] = deep_surprisal_mean.item()

        normalized_surprisal_mean = surprisal_mean / self.surprisal_mean_ema
        normalized_shallow_surprisal = (
            shallow_surprisal / self.shallow_surprisal_mean_ema.item()
        )
        normalized_deep_surprisal = deep_surprisal / self.deep_surprisal_mean_ema.item()

        self.metrics["normalized_surprisal"] = (
            surprisal.mean().item() / self.surprisal_mean_ema.item()
        )
        self.metrics["normalized_shallow_surprisal"] = (
            normalized_shallow_surprisal.mean().item()
        )
        self.metrics["normalized_deep_surprisal"] = (
            normalized_deep_surprisal.mean().item()
        )

        if self.fatigue is None:
            self.fatigue = torch.zeros(1, dtype=self.dtype, device=self.device)
        self.fatigue = self.fatigue * self.fatigue_decay + normalized_surprisal_mean * (
            1 - self.fatigue_decay
        )

        reward = torch.tanh(
            (
                torch.lerp(
                    normalized_deep_surprisal,
                    -normalized_shallow_surprisal,
                    torch.clamp(self.fatigue, 0.0, 2.0) * 0.5,
                )
            ).mean()
        )

        self.metrics["reward"] = reward.item()
        self.metrics["fatigue"] = self.fatigue.item()

        self.step_data_fd_piv[DataKey.REWARD] = reward.cpu()

        # ==============================================================================
        #                               Data Collection
        # ==============================================================================

        self.step_data_fd_piv[DataKey.TARGET] = obs_embedding.cpu()
        self.step_data_fd_piv[DataKey.OBSERVATION] = observation.cpu()
        self.step_data_fd_piv[DataKey.ACTION] = {
            "external_action": self.external_action.cpu(),
            "internal_action": self.internal_action.cpu(),
        }

        # Store for next loop
        self.step_data_fd_piv[DataKey.ACTION_LOG_PROB] = action_log_prob.cpu()
        self.step_data_fd_piv[DataKey.VALUE] = value.cpu()
        self.metrics["value"] = value.cpu().item()

        self.scheduler.update()
        self.global_step += 1
        return self.external_action

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
                    context={"namespace": "agent", "curiosity_type": "deep-surprise"},
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
        if self.external_action is not None:
            torch.save(self.external_action, path / "external_action.pt")
        if self.internal_action is not None:
            torch.save(self.internal_action, path / "internal_action.pt")
        if self.fatigue is not None:
            torch.save(self.fatigue, path / "fatigue.pt")
        if self.surprisal_mean_ema is not None:
            torch.save(self.surprisal_mean_ema, path / "surprisal_mean_ema.pt")
        if self.shallow_surprisal_mean_ema is not None:
            torch.save(
                self.shallow_surprisal_mean_ema, path / "shallow_surprisal_mean_ema.pt"
            )
        if self.deep_surprisal_mean_ema is not None:
            torch.save(
                self.deep_surprisal_mean_ema, path / "deep_surprisal_mean_ema.pt"
            )
        if self.surprisal_coef is not None:
            torch.save(self.surprisal_coef, path / "surprisal_coef.pt")
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
        external_action_path = path / "external_action.pt"
        self.external_action = (
            torch.load(external_action_path, map_location=self.device)
            if external_action_path.exists()
            else None
        )
        internal_action_path = path / "internal_action.pt"
        self.internal_action = (
            torch.load(internal_action_path, map_location=self.device)
            if internal_action_path.exists()
            else None
        )
        fatigue_path = path / "fatigue.pt"
        self.fatigue = (
            torch.load(fatigue_path, map_location=self.device)
            if fatigue_path.exists()
            else None
        )
        surprisal_mean_ema_path = path / "surprisal_mean_ema.pt"
        self.surprisal_mean_ema = (
            torch.load(surprisal_mean_ema_path, map_location=self.device)
            if surprisal_mean_ema_path.exists()
            else None
        )
        shallow_surprisal_mean_ema_path = path / "shallow_surprisal_mean_ema.pt"
        self.shallow_surprisal_mean_ema = (
            torch.load(shallow_surprisal_mean_ema_path, map_location=self.device)
            if shallow_surprisal_mean_ema_path.exists()
            else None
        )
        deep_surprisal_mean_ema_path = path / "deep_surprisal_mean_ema.pt"
        self.deep_surprisal_mean_ema = (
            torch.load(deep_surprisal_mean_ema_path, map_location=self.device)
            if deep_surprisal_mean_ema_path.exists()
            else None
        )
        surprisal_coef_path = path / "surprisal_coef.pt"
        self.surprisal_coef = (
            torch.load(surprisal_coef_path, map_location=self.device)
            if surprisal_coef_path.exists()
            else None
        )
        self.global_step = int((path / "global_step").read_text("utf-8"))
