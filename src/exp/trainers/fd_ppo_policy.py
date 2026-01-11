from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Self, cast, override

import torch
from pamiq_core import DataUser
from pamiq_core.data.impls import DictSequentialBuffer
from pamiq_core.torch import OptimizersSetup, TorchTrainer, get_device
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from exp.aim_utils import get_global_run
from exp.data import BufferName, DataKey
from exp.models import ModelName
from exp.models.fd_policy import HiddenStateFDPiV
from exp.trainers.sampler import RandomTimeSeriesSampler
from exp.utils import average_exponentially

from .ppo_policy import compute_advantage

OPTIMIZER_NAME = "optimizer"

type BatchType = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
]


class PPOHiddenStateFDPiVTrainer(TorchTrainer):
    """Trainer for forward dynamics and policy using Proximal Policy
    Optimization (PPO)."""

    @override
    def __init__(
        self,
        partial_optimizer: partial[Optimizer],
        gamma: float,
        gae_lambda: float = 0.95,
        seq_len: int = 1,
        max_samples: int = 1,
        batch_size: int = 1,
        max_epochs: int = 1,
        norm_advantage: bool = True,
        clip_coef: float = 0.1,
        entropy_coef: float = 0.0,
        vfunc_coef: float = 0.5,
        imagination_length: int = 1,
        imagination_average_method: Callable[[Tensor], Tensor] = average_exponentially,
        model_name: str = ModelName.FD_POLICY_VALUE,
        data_user_name: str = BufferName.FD_POLICY_VALUE,
        log_prefix: str = "fd-ppo-policy",
        include_upper_action: bool = False,
        min_buffer_size: int | None = None,
        min_new_data_count: int = 0,
    ) -> None:
        """Initialize the PPO Policy trainer.

        Args:
            partial_optimizer: Partially initialized optimizer lacking with model parameters.
            gamma: Discount factor for future rewards (0 <= gamma <= 1).
            gae_lambda: Lambda parameter for Generalized Advantage Estimation (GAE)
                to balance bias-variance tradeoff (0 <= gae_lambda <= 1).
            seq_len: Sequence length per batch.
            max_samples: Number of samples from entire dataset.
            batch_size: Data size of 1 batch.
            max_epochs: Maximum number of epochs to train per training session.
            norm_advantage: Whether to normalize advantages.
            clip_coef: Clipping coefficient for PPO.
            entropy_coef: Coefficient for entropy regularization.
            vfunc_coef: Coefficient for value function loss.
            imagination_length: Length of the imagination sequence.
            imagination_average_method: Method to average the loss over the imagination sequence.
            data_user_name: Name of the data user providing training data.
            min_buffer_size: Minimum buffer size required before training starts.
            min_new_data_count: Minimum number of new data points required for training.
        """
        if not (0 <= gamma <= 1):
            raise ValueError(f"gamma must be in range [0, 1], got {gamma}")
        if not (0 <= gae_lambda <= 1):
            raise ValueError(f"gae_lambda must be in range [0, 1], got {gae_lambda}")
        if min_buffer_size is None:
            min_buffer_size = seq_len + 1
        if min_buffer_size < seq_len + 1:
            raise ValueError(
                f"min_buffer_size ({min_buffer_size}) must be at least seq_len ({seq_len} + 1)"
            )
        super().__init__(data_user_name, min_buffer_size, min_new_data_count)

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.model_name = model_name
        self.data_user_name = data_user_name
        self.log_prefix = log_prefix
        self.partial_optimizer = partial_optimizer
        self.partial_sampler = partial(
            RandomTimeSeriesSampler, sequence_length=seq_len, max_samples=max_samples
        )
        self.partial_dataloader = partial(DataLoader, batch_size=batch_size)
        self.max_epochs = max_epochs
        self.norm_advantage = norm_advantage
        self.clip_coef = clip_coef
        self.entropy_coef = entropy_coef
        self.vfunc_coef = vfunc_coef
        self.imagination_length = imagination_length
        self.imagination_average_method = imagination_average_method
        self.global_step = 0

        self.include_upper_action = include_upper_action

    @override
    def on_data_users_attached(self) -> None:
        """Set up data user references when they are attached to the
        trainer."""
        super().on_data_users_attached()
        self.policy_data_user: DataUser[dict[str, list[Tensor]]] = self.get_data_user(
            self.data_user_name
        )

    @override
    def on_training_models_attached(self) -> None:
        """Set up model references when they are attached to the trainer."""
        super().on_training_models_attached()
        self.fd_piv = self.get_torch_training_model(self.model_name, HiddenStateFDPiV)

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
            hiddens,
            actions,
            action_log_probs,
            values,
            advantages,
            returns,
            upper_action,
        ) = batch

        # Get new distributions and values
        _, new_dist, new_values, _ = self.fd_piv.model(
            observations,
            actions,
            upper_action,
            hiddens[:, 0],
        )
        new_log_probs = new_dist.log_prob(actions)
        entropy = new_dist.entropy()

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

        entropy_loss = entropy.mean()

        # Imagination loss

        device = get_device(self.fd_piv.model)
        obs_imaginations, hiddens = (
            observations[:, : -self.imagination_length],
            hiddens[:, 0].to(device),
        )

        loss_imaginations: list[Tensor] = []
        for i in range(self.imagination_length):
            action_imaginations = actions[
                :, i : -self.imagination_length + i
            ]  # a_i:i+T-H, (B, T-H, *)
            obs_targets = observations[
                :,
                i + 1 : observations.size(1) - self.imagination_length + i + 1,
            ]  # o_i+1:T-H+i+1, (B, T-H, *)
            if i > 0:
                action_imaginations = action_imaginations.flatten(0, 1)  # (B', *)
                obs_targets = obs_targets.flatten(0, 1)  # (B', *)

            if i == 0:
                forward_method = self.fd_piv.model.__call__
            else:
                forward_method = partial(self.fd_piv.model, no_len=True)

            obses_next_hat, _, _, next_hiddens = forward_method(
                obs_imaginations, action_imaginations, hidden=hiddens
            )

            loss = torch.nn.functional.mse_loss(obses_next_hat, obs_targets)
            loss_imaginations.append(loss)
            obs_imaginations = obses_next_hat

            if i == 0:
                obs_imaginations = obs_imaginations.flatten(
                    0, 1
                )  # (B, T-H, *) -> (B', *)
                hiddens = next_hiddens.movedim(2, 1).flatten(
                    0, 1
                )  # h'_i, (B, D, T-H, *) -> (B, T-H, D, *) -> (B', D, *)

        fd_loss = self.imagination_average_method(torch.stack(loss_imaginations))

        # Total loss
        loss = (
            pg_loss
            - self.entropy_coef * entropy_loss
            + v_loss * self.vfunc_coef
            + fd_loss
        )

        return {
            "loss": loss,
            "policy_loss": pg_loss,
            "value_loss": v_loss,
            "fd_loss": fd_loss,
            "entropy": entropy_loss,
            "approx_kl": approx_kl,
            "clipfrac": clipfracs,
        }

    @override
    def train(self) -> None:
        """Execute PPO training process."""

        # Get dataset from data user
        data = self.policy_data_user.get_data()

        keys = [
            DataKey.OBSERVATION,
            DataKey.HIDDEN,
            DataKey.ACTION,
            DataKey.ACTION_LOG_PROB,
            DataKey.REWARD,
            DataKey.VALUE,
        ]
        if self.include_upper_action:
            keys.append(DataKey.UPPER_ACTION)

        tensors = {key: torch.stack(data[key][:-1]) for key in keys}

        # compute advantages and returns
        advantages = compute_advantage(
            rewards=tensors[DataKey.REWARD],
            values=tensors[DataKey.VALUE],
            final_next_value=data[DataKey.VALUE][-1],
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        returns = advantages + tensors[DataKey.VALUE]

        tensor_list = [
            tensors[DataKey.OBSERVATION],
            tensors[DataKey.HIDDEN],
            tensors[DataKey.ACTION],
            tensors[DataKey.ACTION_LOG_PROB],
            tensors[DataKey.VALUE],
            advantages,
            returns,
        ]

        if self.include_upper_action:
            tensor_list.append(tensors[DataKey.UPPER_ACTION])

        dataset = TensorDataset(*tensor_list)
        sampler = self.partial_sampler(dataset)
        dataloader = self.partial_dataloader(dataset=dataset, sampler=sampler)
        device = get_device(self.fd_piv.model)

        for _ in range(self.max_epochs):
            batch: tuple[Tensor, ...]
            for batch in dataloader:
                self.optimizers[OPTIMIZER_NAME].zero_grad()

                data_list: list[Tensor | None] = [d.to(device) for d in batch]
                if not self.include_upper_action:
                    data_list.append(None)

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

                self.optimizers[OPTIMIZER_NAME].step()

                # Logging
                metrics = {k: v.item() for k, v in outputs.items()}
                metrics["grad_norm"] = grad_norm.item()

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
        max_size: int, include_upper_action: bool = False
    ) -> DictSequentialBuffer[Tensor]:
        """Create data buffer for this trainer."""
        keys = [
            DataKey.OBSERVATION,
            DataKey.HIDDEN,
            DataKey.ACTION,
            DataKey.ACTION_LOG_PROB,
            DataKey.REWARD,
            DataKey.VALUE,
        ]
        if include_upper_action:
            keys.append(DataKey.UPPER_ACTION)

        return DictSequentialBuffer(
            keys,
            max_size=max_size,
        )

    @classmethod
    def create_hierarchical_buffers(
        cls, max_size: int, num: int
    ) -> list[DictSequentialBuffer[Tensor]]:
        """Create multiple buffer instances."""
        bufs = []
        for i in range(num):
            bufs.append(
                cls.create_buffer(
                    max_size,
                    include_upper_action=(i + 1) < num,  #  without top.
                )
            )
        return bufs

    @classmethod
    def create_multiple(
        cls, num_trainers: int, hierarchical: bool = False, **trainer_params: Any
    ) -> list[Self]:
        """Create multiple trainer instances.

        Args:
            num_trainers: Number of trainers to create.
            hierarchical: If True, enable upper_action for all trainers except the last.
            **trainer_params: Parameters to pass to each trainer constructor.

        Returns:
            List of configured trainer instances with indexed names.
        """
        trainers = list[Self]()

        def include_upper_action(idx: int) -> bool:
            return idx < num_trainers - 1 and hierarchical

        for i in range(num_trainers):
            trainers.append(
                cls(
                    **trainer_params,
                    model_name=ModelName.FD_POLICY_VALUE + str(i),
                    data_user_name=BufferName.FD_POLICY_VALUE + str(i),
                    log_prefix="fd-ppo-policy" + str(i),
                    include_upper_action=include_upper_action(i),
                )
            )
        return trainers
