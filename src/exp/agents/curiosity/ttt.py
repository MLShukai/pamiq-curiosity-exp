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
    DataKey.CORE_EMB,
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
Hidden = dict[str, Hidden_Time | Hidden_TTT | Tensor]
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
        fast_surprisal_ema_decay_range: tuple[float, float] = (0.8, 0.9),
        slow_surprisal_ema_decay_range: tuple[float, float] = (0.98, 0.99),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the TTTCuriosityAgent.

        Args:
            log_every_n_steps: Frequency of logging metrics to Aim. Defaults to 1.
            device: Device to run computations on. Defaults to None.
            dtype: Data type for tensors. Defaults to None.

        Raises:
            ValueError: If max_imagination_steps is less than 1.
        """
        super().__init__()

        self.hidden_state = None
        self.core_emb = None
        self.external_action = None
        self.internal_action = None
        self.fast_surprisal_ema_decay_range = fast_surprisal_ema_decay_range
        self.slow_surprisal_ema_decay_range = slow_surprisal_ema_decay_range
        self.fast_surprisal_ema_decay = None
        self.slow_surprisal_ema_decay = None
        self.slow_surprisal_ema = None
        self.fast_surprisal_ema = None
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
    core_emb: Tensor | None
    external_action: Tensor | None  # (action_choices,) or None
    internal_action: Tensor | None  # (dim,) or None
    fast_surprisal_ema_decay: Tensor | None
    slow_surprisal_ema_decay: Tensor | None
    slow_surprisal_ema: Tensor | None
    fast_surprisal_ema: Tensor | None
    obs_hat: Tensor | None
    step_data_fd_piv: dict[str, Tensor | Hidden | Action]

    @override
    def setup(self) -> None:
        """Initialize agent state.

        Resets step data collectors, imagination buffers
        """
        super().setup()
        self.step_data_fd_piv = {}

    @property
    def fatigue(self) -> Tensor | None:
        """Calculate fatigue based on the normalized surprisal.

        Fatigue is defined as the hyperbolic tangent of the normalized surprisal,
        which is the ratio of the fast surprisal EMA to the slow surprisal EMA.
        This provides a measure of how surprising recent observations are compared
        to longer-term trends, which can be used to modulate exploration.

        Returns:
            A tensor representing fatigue, or None if EMAs are not initialized.
        """
        if self.fast_surprisal_ema is not None and self.slow_surprisal_ema is not None:
            return (self.fast_surprisal_ema + 1e-8) / (
                self.slow_surprisal_ema + 1e-8
            ) - 1
        return None

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
            F.tanh(self.fatigue).flatten() if self.fatigue is not None else None
        )
        if self.core_emb is not None:
            self.step_data_fd_piv[DataKey.CORE_EMB] = self.core_emb.cpu()
            self.metrics["core_emb_norm"] = torch.norm(self.core_emb).item()

        action_dist: MultiDistributions
        value: Tensor
        (
            obs_embedding,
            self.core_emb,
            _,
            action_dist,
            value,
            self.hidden_state,
            surprisal,
            surprisal_coef,
        ) = self.fd_piv(
            observation,
            self.core_emb,
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

        if not isinstance(surprisal, Tensor):
            raise ValueError("Surprisal must be a Tensor.")
        if self.fast_surprisal_ema_decay is None:
            self.fast_surprisal_ema_decay = (
                1
                - torch.lerp(
                    (
                        (1 - self.fast_surprisal_ema_decay_range[0])
                        * torch.ones_like(surprisal, device=surprisal.device)
                    ).log(),
                    (
                        (1 - self.fast_surprisal_ema_decay_range[1])
                        * torch.ones_like(surprisal, device=surprisal.device)
                    ).log(),
                    torch.rand_like(surprisal, device=surprisal.device),
                ).exp()
            )
        if not isinstance(self.fast_surprisal_ema_decay, Tensor):
            raise ValueError("fast_surprisal_ema_decay must be a Tensor.")
        if self.fast_surprisal_ema is None:
            self.fast_surprisal_ema = surprisal
        self.fast_surprisal_ema = torch.lerp(
            surprisal, self.fast_surprisal_ema, self.fast_surprisal_ema_decay
        )

        if self.slow_surprisal_ema_decay is None:
            self.slow_surprisal_ema_decay = (
                1
                - torch.lerp(
                    (
                        (1 - self.slow_surprisal_ema_decay_range[0])
                        * torch.ones_like(surprisal, device=surprisal.device)
                    ).log(),
                    (
                        (1 - self.slow_surprisal_ema_decay_range[1])
                        * torch.ones_like(surprisal, device=surprisal.device)
                    ).log(),
                    torch.rand_like(surprisal, device=surprisal.device),
                ).exp()
            )
        if not isinstance(self.slow_surprisal_ema_decay, Tensor):
            raise ValueError("slow_surprisal_ema_decay must be a Tensor.")
        if self.slow_surprisal_ema is None:
            self.slow_surprisal_ema = surprisal
        self.slow_surprisal_ema = torch.lerp(
            surprisal, self.slow_surprisal_ema, self.slow_surprisal_ema_decay
        )

        self.metrics["surprisal"] = surprisal.mean().item()

        self.metrics["normalized_surprisal"] = (
            (self.fatigue + 1).mean().item() if self.fatigue is not None else 1.0
        )

        reward = (
            # F.tanh((F.relu(-self.fatigue) * surprisal_coef).sum())
            F.tanh((self.fatigue * surprisal_coef).sum())
            if self.fatigue is not None
            else torch.zeros(1, device=surprisal.device)
        )

        self.metrics["reward"] = reward.item()

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
        if self.core_emb is not None:
            torch.save(self.core_emb, path / "core_emb.pt")
        if self.external_action is not None:
            torch.save(self.external_action, path / "external_action.pt")
        if self.internal_action is not None:
            torch.save(self.internal_action, path / "internal_action.pt")
        if self.fast_surprisal_ema_decay is not None:
            torch.save(
                self.fast_surprisal_ema_decay, path / "fast_surprisal_ema_decay.pt"
            )
        if self.slow_surprisal_ema_decay is not None:
            torch.save(
                self.slow_surprisal_ema_decay, path / "slow_surprisal_ema_decay.pt"
            )
        if self.fast_surprisal_ema is not None:
            torch.save(self.fast_surprisal_ema, path / "fast_surprisal_ema.pt")
        if self.slow_surprisal_ema is not None:
            torch.save(self.slow_surprisal_ema, path / "slow_surprisal_ema.pt")
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
        core_emb_path = path / "core_emb.pt"
        self.core_emb = (
            torch.load(core_emb_path, map_location=self.device)
            if core_emb_path.exists()
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
        fast_surprisal_ema_decay_path = path / "fast_surprisal_ema_decay.pt"
        self.fast_surprisal_ema_decay = (
            torch.load(fast_surprisal_ema_decay_path, map_location=self.device)
            if fast_surprisal_ema_decay_path.exists()
            else None
        )
        slow_surprisal_ema_decay_path = path / "slow_surprisal_ema_decay.pt"
        self.slow_surprisal_ema_decay = (
            torch.load(slow_surprisal_ema_decay_path, map_location=self.device)
            if slow_surprisal_ema_decay_path.exists()
            else None
        )
        fast_surprisal_ema_path = path / "fast_surprisal_ema.pt"
        self.fast_surprisal_ema = (
            torch.load(fast_surprisal_ema_path, map_location=self.device)
            if fast_surprisal_ema_path.exists()
            else None
        )
        slow_surprisal_ema_path = path / "slow_surprisal_ema.pt"
        self.slow_surprisal_ema = (
            torch.load(slow_surprisal_ema_path, map_location=self.device)
            if slow_surprisal_ema_path.exists()
            else None
        )
        self.global_step = int((path / "global_step").read_text("utf-8"))
