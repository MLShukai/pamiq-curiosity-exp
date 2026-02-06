from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, Self, cast, override

import torch
from pamiq_core import DataUser
from pamiq_core.data.impls import DictSequentialBuffer
from pamiq_core.torch import OptimizersSetup, TorchTrainer, get_device
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from exp.agents.curiosity.ttt import Action, Hidden
from exp.aim_utils import get_global_run
from exp.data import BufferName, DataKey
from exp.data.dict_intermittent_chunk_buffer import DictIntermittentChunkBuffer
from exp.models import ModelName
from exp.models.fd_policy import TTTFDPiV
from exp.trainers.sampler import RandomTimeSeriesSampler
from exp.utils import average_exponentially

from .ppo_policy import compute_advantage

OPTIMIZER_NAME = "optimizer"

type BatchType = tuple[
    Tensor,
    Tensor,
    Hidden,
    Action,
    Action,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]


class TTTFDPiVTrainer(TorchTrainer):
    """Trainer for forward dynamics and policy using Proximal Policy
    Optimization (PPO)."""

    @override
    def __init__(
        self,
        partial_optimizer: partial[Optimizer],
        gamma: float,
        gae_lambda: float = 0.95,
        max_epochs: int = 1,
        batch_size: int = 1,
        norm_advantage: bool = True,
        clip_coef: float = 0.1,
        external_action_entropy_coef: float = 0.0,
        external_action_entropy_coef_decay: float = 1.0,
        internal_action_entropy_coef: float = 0.0,
        internal_action_entropy_coef_decay: float = 1.0,
        target_delay_frames: int = 1,
        vfunc_coef: float = 0.5,
        grad_clip_norm: float = 10.0,
        model_name: str = ModelName.FD_POLICY_VALUE,
        data_user_name: str = BufferName.FD_POLICY_VALUE,
        log_prefix: str = "ttt",
        min_buffer_size: int = 1,
        min_new_data_count: int = 0,
    ) -> None:
        """Initialize the PPO Policy trainer.

        Args:
            partial_optimizer: Partially initialized optimizer lacking with model parameters.
            gamma: Discount factor for future rewards (0 <= gamma <= 1).
            gae_lambda: Lambda parameter for Generalized Advantage Estimation (GAE)
                to balance bias-variance tradeoff (0 <= gae_lambda <= 1).
            max_epochs: Maximum number of epochs to train per training session.
            norm_advantage: Whether to normalize advantages.
            clip_coef: Clipping coefficient for PPO.
            entropy_coef: Coefficient for entropy regularization.
            vfunc_coef: Coefficient for value function loss.
            data_user_name: Name of the data user providing training data.
            min_buffer_size: Minimum buffer size required before training starts.
            min_new_data_count: Minimum number of new data points required for training.
        """
        if not (0 <= gamma <= 1):
            raise ValueError(f"gamma must be in range [0, 1], got {gamma}")
        if not (0 <= gae_lambda <= 1):
            raise ValueError(f"gae_lambda must be in range [0, 1], got {gae_lambda}")
        super().__init__(data_user_name, min_buffer_size, min_new_data_count)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.grad_clip_norm = grad_clip_norm

        self.model_name = model_name
        self.data_user_name = data_user_name
        self.log_prefix = log_prefix
        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial(DataLoader, batch_size=batch_size)
        self.max_epochs = max_epochs
        self.norm_advantage = norm_advantage
        self.clip_coef = clip_coef
        self.external_action_entropy_coef = external_action_entropy_coef
        self.external_action_entropy_coef_decay = external_action_entropy_coef_decay
        self.internal_action_entropy_coef = internal_action_entropy_coef
        self.internal_action_entropy_coef_decay = internal_action_entropy_coef_decay
        self.target_delay_frames = target_delay_frames
        self.vfunc_coef = vfunc_coef
        self.global_step = 0

    @override
    def on_data_users_attached(self) -> None:
        """Set up data user references when they are attached to the
        trainer."""
        super().on_data_users_attached()
        self.data_user: DataUser[
            dict[str, list[list[Tensor]] | list[list[Action]] | list[Hidden]]
        ] = self.get_data_user(self.data_user_name)

    @override
    def on_training_models_attached(self) -> None:
        """Set up model references when they are attached to the trainer."""
        super().on_training_models_attached()
        self.fd_piv = self.get_torch_training_model(self.model_name, TTTFDPiV)

    @override
    def create_optimizers(self) -> OptimizersSetup:
        """Create optimizers for FD+PPO training.

        Returns:
            Dictionary mapping optimizer name to configured optimizer instance.
        """
        return {OPTIMIZER_NAME: self.partial_optimizer(self.fd_piv.model.parameters())}

    def training_step(self, batch: BatchType) -> dict[str, Tensor]:
        """Perform a single training step on a batch of data."""
        (
            observations,
            obs_embeddings,
            hiddens,
            previous_actions,
            actions,
            action_log_probs,
            internal_states,
            values,
            advantages,
            returns,
        ) = batch

        external_previous_actions = previous_actions["external_action"]
        internal_previous_actions = previous_actions["internal_action"]
        external_actions = actions["external_action"]
        internal_actions = actions["internal_action"]

        # Get new distributions and values
        _, obs_hat, new_dist, new_values, _, _, active_surprisal, stable_surprisal = (
            self.fd_piv.model(
                observations,
                external_previous_actions,
                internal_previous_actions,
                internal_states,
                hiddens,
            )
        )
        new_log_probs = new_dist.log_prob(external_actions, internal_actions)

        external_action_entropy, internal_action_entropy = new_dist.entropy_per_dist()

        # Calculate ratio for PPO
        log_ratio = new_log_probs - action_log_probs
        ratio = log_ratio.exp()

        # Calculate KL divergence and clip fraction
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clipfracs = ((ratio - 1.0).abs() > self.clip_coef).float().mean()

        # Normalize advantages
        if self.norm_advantage:
            advantages = advantages / (advantages.std() + 1e-8)

        # Adjust dimensions if needed
        if advantages.ndim < ratio.ndim:
            for _ in range(ratio.ndim - advantages.ndim):
                advantages = advantages.unsqueeze(-1)
        advantages = advantages.detach()  # Stop Gradient

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1 - self.clip_coef, 1 + self.clip_coef
        )
        pg_loss = torch.max(input=pg_loss1, other=pg_loss2).mean()

        # Value loss
        new_values = new_values.flatten()
        returns = returns.flatten()
        values = values.flatten()

        v_loss_unclipped = (new_values - returns) ** 2
        v_clipped = values + torch.clamp(
            new_values - values, -self.clip_coef, self.clip_coef
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()

        external_action_entropy_coef = torch.tensor(
            self.external_action_entropy_coef
            * self.external_action_entropy_coef_decay**self.global_step
        )
        external_action_entropy_loss = (
            external_action_entropy.mean() * external_action_entropy_coef
        )

        internal_action_entropy_coef = torch.tensor(
            self.internal_action_entropy_coef
            * self.internal_action_entropy_coef_decay**self.global_step
        )
        internal_action_entropy_loss = (
            internal_action_entropy.mean() * internal_action_entropy_coef
        )

        # Forward dynamics loss
        fd_loss = torch.nn.functional.mse_loss(obs_hat, obs_embeddings)

        # Total loss
        loss = (
            pg_loss
            + external_action_entropy_loss
            + internal_action_entropy_loss
            + v_loss * self.vfunc_coef
            + fd_loss
            - active_surprisal.sum(dim=(-2, -1)).mean()
            + stable_surprisal.sum(dim=(-2, -1)).mean()
        )

        return {
            "loss": loss,
            "policy_loss": pg_loss,
            "value_loss": v_loss,
            "fd_loss": fd_loss,
            "external_action_entropy": external_action_entropy.mean(),
            "external_action_entropy_coef": external_action_entropy_coef,
            "internal_action_entropy": internal_action_entropy.mean(),
            "internal_action_entropy_coef": internal_action_entropy_coef,
            "approx_kl": approx_kl,
            "clipfrac": clipfracs,
            "advantage_mean": advantages.mean(),
            "ratio_mean": ratio.mean(),
            "new_log_prob_mean": new_log_probs.mean(),
            "action_log_prob_mean": action_log_probs.mean(),
            "log_ratio_mean": log_ratio.mean(),
        }

    @override
    def train(self) -> None:
        """Execute PPO training process."""

        # Get dataset from data user
        data = self.data_user.get_data()

        chunk_keys = [
            DataKey.OBSERVATION,
            DataKey.TARGET,
            DataKey.ACTION_LOG_PROB,
            DataKey.INTERNAL_STATE,
            DataKey.REWARD,
            DataKey.VALUE,
            DataKey.PREVIOUS_ACTION,
        ]

        # tensors = {key: torch.stack(data[key][:-1]) for key in chunk_keys}
        chunks = {
            key: [
                torch.stack(
                    [t if isinstance(t, Tensor) else torch.zeros(1) for t in chunk]
                )
                for chunk in data[key]
            ]
            for key in chunk_keys
        }

        previous_actions_chunks = [
            {
                "external_action": torch.stack(
                    [
                        t["external_action"] if isinstance(t, dict) else torch.zeros(1)
                        for t in chunk
                    ]
                ),
                "internal_action": torch.stack(
                    [
                        t["internal_action"] if isinstance(t, dict) else torch.zeros(1)
                        for t in chunk
                    ]
                ),
            }
            for chunk in data[DataKey.PREVIOUS_ACTION]
        ]
        actions_chunks = [
            {
                "external_action": torch.stack(
                    [
                        t["external_action"] if isinstance(t, dict) else torch.zeros(1)
                        for t in chunk
                    ]
                ),
                "internal_action": torch.stack(
                    [
                        t["internal_action"] if isinstance(t, dict) else torch.zeros(1)
                        for t in chunk
                    ]
                ),
            }
            for chunk in data[DataKey.ACTION]
        ]

        # compute advantages and returns
        advantages_list = [
            compute_advantage(
                rewards=rewards,
                values=values,
                final_next_value=values[-1],
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
            for rewards, values in zip(chunks[DataKey.REWARD], chunks[DataKey.VALUE])
        ]
        returns_list = [
            advantages + values
            for advantages, values in zip(advantages_list, chunks[DataKey.VALUE])
        ]

        hidden_list = data[DataKey.HIDDEN]

        dataset_raw: list[BatchType] = [
            (
                observations,
                obs_embeddings,
                hiddens,
                previous_actions,
                actions,
                action_log_probs,
                internal_states,
                values,
                advantages,
                returns,
            )
            for observations, obs_embeddings, hiddens, previous_actions, actions, action_log_probs, internal_states, values, advantages, returns in zip(
                chunks[DataKey.OBSERVATION],
                chunks[DataKey.TARGET],
                hidden_list,
                previous_actions_chunks,
                actions_chunks,
                chunks[DataKey.ACTION_LOG_PROB],
                chunks[DataKey.INTERNAL_STATE],
                chunks[DataKey.VALUE],
                advantages_list,
                returns_list,
            )
        ]

        class TTTDataset(Dataset):
            def __init__(
                self,
                data: list[BatchType],
            ) -> None:
                self.data = data

            def __len__(self) -> int:
                return len(self.data)

            @override
            def __getitem__(self, index: int) -> BatchType:
                return self.data[index]

        dataset = TTTDataset(dataset_raw)
        # sampler = self.partial_sampler(dataset)
        dataloader = self.partial_dataloader(dataset=dataset, shuffle=True)
        device = get_device(self.fd_piv.model)

        for _ in range(self.max_epochs):
            batch: BatchType
            for batch in dataloader:
                self.optimizers[OPTIMIZER_NAME].zero_grad()

                data_list: list[Tensor | Hidden | Action | None] = [
                    d.to(device)
                    if isinstance(d, Tensor)
                    else (
                        {
                            k: v.to(device)
                            if isinstance(v, Tensor)
                            else [
                                {k1: v1.to(device) for k1, v1 in item.items()}
                                for item in v
                            ]
                            for k, v in d.items()
                        }
                        if isinstance(d, dict)
                        else d
                    )
                    for d in batch
                ]

                # Perform training step
                outputs = self.training_step(cast(BatchType, tuple(data_list)))
                loss = outputs["loss"]

                # Backward pass
                loss.backward()

                # Calculate gradient norm
                grad_norm = torch.cat(
                    [
                        p.grad.flatten()
                        for p in self.fd_piv.model.parameters()
                        if p.grad is not None
                    ]
                ).norm()

                param_norm = torch.cat(
                    [
                        p.flatten()
                        for p in self.fd_piv.model.parameters()
                        if p.grad is not None
                    ]
                ).norm()

                torch.nn.utils.clip_grad_norm_(
                    self.fd_piv.model.parameters(), max_norm=self.grad_clip_norm
                )

                self.optimizers[OPTIMIZER_NAME].step()

                # Logging
                metrics = {k: v.item() for k, v in outputs.items()}
                metrics["grad_norm"] = grad_norm.item()
                metrics["param_norm"] = param_norm.item()

                if run := get_global_run():
                    for tag, v in metrics.items():
                        value = v.item() if isinstance(v, torch.Tensor) else v
                        run.track(
                            value,
                            name=tag,
                            step=self.global_step,
                            context={
                                "namespace": "trainer",
                                "trainer_type": self.log_prefix,
                            },
                        )
                self.global_step += 1

    @override
    def save_state(self, path: Path) -> None:
        """Save trainer state to disk."""
        super().save_state(path)
        path.mkdir(exist_ok=True)
        (path / "global_step").write_text(str(self.global_step), "utf-8")

    @override
    def load_state(self, path: Path) -> None:
        """Load trainer state from disk."""
        super().load_state(path)
        self.global_step = int((path / "global_step").read_text("utf-8"))

    @staticmethod
    def create_buffer(
        max_size: int, get_interval: int, target_delay_frames: int = 1
    ) -> DictIntermittentChunkBuffer[Tensor | list[dict[str, Tensor]]]:
        """Create data buffer for this trainer."""
        intermittent_keys_first_add_steps: Mapping[str, int] = {
            DataKey.HIDDEN: 0,
        }
        chunk_keys_first_add_steps: Mapping[str, int] = {
            DataKey.OBSERVATION: 0,
            DataKey.TARGET: target_delay_frames,
            DataKey.PREVIOUS_ACTION: 0,
            DataKey.ACTION: 0,
            DataKey.ACTION_LOG_PROB: 0,
            DataKey.INTERNAL_STATE: 0,
            DataKey.REWARD: 0,
            DataKey.VALUE: 0,
        }

        return DictIntermittentChunkBuffer(
            intermittent_keys_first_add_steps,
            chunk_keys_first_add_steps,
            get_interval=get_interval,
            max_size=max_size,
        )
