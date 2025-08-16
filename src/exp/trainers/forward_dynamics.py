from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Self, override

import torch
import torch.nn.functional as F
from pamiq_core import DataUser
from pamiq_core.data.impls import DictSequentialBuffer
from pamiq_core.torch import OptimizersSetup, TorchTrainer, get_device
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from exp.aim_utils import get_global_run
from exp.data import BufferName, DataKey
from exp.models import ModelName
from exp.models.forward_dynamics import StackedHiddenFD
from exp.models.latent_fd import LatentFD
from exp.utils import average_exponentially

from .sampler import RandomTimeSeriesSampler

OPTIMIZER_NAME = "optimizer"


class StackedHiddenFDTrainer(TorchTrainer):
    """Trainer for the StackedHiddenFD model.

    This trainer implements the training loop for the StackedHiddenFD
    model, which predicts the next observation distribution given the
    current observation and action. It uses a recurrent core model to
    maintain hidden state across sequential predictions.
    """

    def __init__(
        self,
        partial_optimizer: partial[Optimizer],
        seq_len: int = 1,
        max_samples: int = 1,
        batch_size: int = 1,
        max_epochs: int = 1,
        imagination_length: int = 1,
        imagination_average_method: Callable[[Tensor], Tensor] = average_exponentially,
        model_name: str = ModelName.FORWARD_DYNAMICS,
        data_user_name: str = BufferName.FORWARD_DYNAMICS,
        log_prefix: str = "forward-dynamics",
        min_buffer_size: int | None = None,
        min_new_data_count: int = 0,
    ) -> None:
        """Initialize the StackedHiddenFDTrainer.

        Args:
            partial_optimizer: Partially configured optimizer to be used with
                the model parameters.
            seq_len: Sequence length per batch.
            max_samples: Max number of sample from dataset in 1 epoch.
            batch_size: Data sample size of 1 batch.
            max_epochs: Maximum number of epochs to train per training session.
            imagination_length: Length of the imagination sequence.
            imagenation_average_method: Method to average the loss over the imagination sequence.
            data_user_name: Name of the data user providing training data.
            min_buffer_size: Minimum buffer size required before training starts.
            min_new_data_count: Minimum number of new data points required for training.
        """
        if imagination_length < 1:
            raise ValueError("Imagination length must be greater than 0")

        if min_buffer_size is None:
            min_buffer_size = imagination_length + seq_len
        if min_buffer_size < imagination_length + seq_len:
            raise ValueError(
                "Buffer size must be greater than imagination length + sequence length."
            )
        super().__init__(data_user_name, min_buffer_size, min_new_data_count)

        self.partial_optimizer = partial_optimizer
        self.partial_sampler = partial(
            RandomTimeSeriesSampler,
            sequence_length=seq_len + imagination_length,
            max_samples=max_samples,
        )
        self.partial_dataloader = partial(DataLoader, batch_size=batch_size)
        self.max_epochs = max_epochs
        self.model_name = model_name
        self.data_user_name = data_user_name
        self.log_prefix = log_prefix
        self.imagination_length = imagination_length
        self.imagination_average_method = imagination_average_method

        self.global_step = 0

    @override
    def on_data_users_attached(self) -> None:
        """Set up data user references when they are attached to the trainer.

        This method is called automatically by the PAMIQ framework when
        data users are attached to the trainer. It retrieves and stores
        references to the required data users for convenient access
        during training.
        """
        super().on_data_users_attached()
        self.forward_dynamics_data_user: DataUser[dict[str, list[Tensor]]] = (
            self.get_data_user(self.data_user_name)
        )

    @override
    def on_training_models_attached(self) -> None:
        """Set up model references when they are attached to the trainer.

        This method is called automatically by the PAMIQ framework when
        training models are attached to the trainer. It retrieves and
        stores references to the StackedHiddenFD model for convenient
        access during training.
        """

        super().on_training_models_attached()
        self.forward_dynamics = self.get_torch_training_model(
            self.model_name, StackedHiddenFD
        )

    @override
    def create_optimizers(self) -> OptimizersSetup:
        """Create optimizers for the StackedHiddenFD model. This method is
        called automatically by the PAMIQ framework to set up optimizers for
        the training process. It uses the `partial_optimizer` function to
        create an optimizer for the StackedHiddenFD model's parameters.

        Returns:
            Dictionary mapping optimizer name to configured optimizer instance.
        """
        return {
            OPTIMIZER_NAME: self.partial_optimizer(
                self.forward_dynamics.model.parameters()
            )
        }

    @override
    def train(self) -> None:
        """Execute StackedHiddenFD training process.

        This method implements the core StackedHiddenFD training loop:
        1. Creates a dataset and dataloader
        2. For each batch:
            - Moves data to the appropriate device
            - Splits observations, actions, and hidden states
            - Computes the next observation distribution
            - Calculates the loss
            - Backpropagates the loss
            - Updates the model parameters
        3. Logs the loss to Aim
        4. Increments the global step counter
        """

        data = self.forward_dynamics_data_user.get_data()
        dataset = TensorDataset(
            torch.stack(data[DataKey.OBSERVATION]),
            torch.stack(data[DataKey.ACTION]),
            torch.stack(data[DataKey.HIDDEN]),
        )
        sampler = self.partial_sampler(dataset=dataset)
        dataloader = self.partial_dataloader(dataset=dataset, sampler=sampler)
        device = get_device(self.forward_dynamics.model)

        for _ in range(self.max_epochs):
            batch: tuple[Tensor, Tensor, Tensor]
            for batch in dataloader:
                observations, actions, hiddens = batch
                observations = observations.to(device)
                actions = actions.to(device)
                obs_imaginations, hiddens = (
                    observations[:, : -self.imagination_length],
                    hiddens[:, 0].to(device),
                )

                self.optimizers[OPTIMIZER_NAME].zero_grad()

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
                        action_imaginations = action_imaginations.flatten(
                            0, 1
                        )  # (B', *)
                        obs_targets = obs_targets.flatten(0, 1)  # (B', *)

                    if i == 0:
                        forward_method = self.forward_dynamics.model.__call__
                    else:
                        forward_method = self.forward_dynamics.model.forward_with_no_len

                    obses_next_hat, next_hiddens = forward_method(
                        obs_imaginations, action_imaginations, hiddens
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

                loss = self.imagination_average_method(torch.stack(loss_imaginations))
                loss.backward()

                metrics = {"loss/average": loss.item()}
                for i, loss_item in enumerate(loss_imaginations, start=1):
                    metrics[f"loss/imagination_{i}"] = loss_item.item()

                metrics["grad norm"] = (
                    torch.cat(
                        [
                            p.grad.flatten()
                            for p in self.forward_dynamics.model.parameters()
                            if p.grad is not None
                        ]
                    )
                    .norm()
                    .item()
                )

                if run := get_global_run():
                    for k, v in metrics.items():
                        run.track(
                            v,
                            name=k,
                            step=self.global_step,
                            context={
                                "namespace": "trainer",
                                "trainer_type": self.log_prefix,
                            },
                        )
                self.optimizers[OPTIMIZER_NAME].step()
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
    def create_buffer(max_size: int) -> DictSequentialBuffer[Tensor]:
        """Create data buffer for this trainer."""
        return DictSequentialBuffer(
            [DataKey.OBSERVATION, DataKey.ACTION, DataKey.HIDDEN], max_size=max_size
        )

    @classmethod
    def create_multiple(cls, num_trainers: int, **trainer_params: Any) -> list[Self]:
        """Create multiple trainer instances.

        Each trainer is assigned a unique model_name and data_user_name by
        appending an index (0 to num_trainers-1) to the base names.

        Args:
            num_trainers: Number of trainers to create.
            **trainer_params: Parameters to pass to each trainer constructor.

        Returns:
            List of configured trainer instances.
        """
        trainers = list[Self]()
        for i in range(num_trainers):
            trainers.append(
                cls(
                    **trainer_params,
                    model_name=ModelName.FORWARD_DYNAMICS + str(i),
                    data_user_name=BufferName.FORWARD_DYNAMICS + str(i),
                    log_prefix="forward-dynamics" + str(i),
                )
            )
        return trainers


class StackedHiddenFDTrainerExplicitTarget(TorchTrainer):
    """Trainer for the StackedHiddenFD model with explicit targets.

    This trainer is a simplified version of StackedHiddenFDTrainer that uses
    explicitly provided targets rather than imagination sequences. Instead of
    predicting multiple steps into the future using imagination, this trainer
    directly uses target observations from the buffer to compute the loss.
    This approach is more straightforward and suitable when you have explicit
    target data available.

    Key differences from StackedHiddenFDTrainer:
    - Uses explicit target observations from DataKey.TARGET
    - No imagination sequence generation
    - Single-step prediction loss computation
    - Supports multiple model instances via model_index parameter
    """

    def __init__(
        self,
        partial_optimizer: partial[Optimizer],
        seq_len: int = 1,
        max_samples: int = 1,
        batch_size: int = 1,
        max_epochs: int = 1,
        model_name: str = ModelName.FORWARD_DYNAMICS,
        data_user_name: str = BufferName.FORWARD_DYNAMICS,
        log_prefix: str = "forward-dynamics",
        min_buffer_size: int | None = None,
        min_new_data_count: int = 0,
    ) -> None:
        """Initialize the StackedHiddenFDTrainerExplicitTarget.

        Args:
            partial_optimizer: Partially configured optimizer to be used with
                the model parameters.
            seq_len: Sequence length per batch.
            max_samples: Max number of samples from dataset in 1 epoch.
            batch_size: Data sample size of 1 batch.
            max_epochs: Maximum number of epochs to train per training session.
            data_user_name: Name of the data user providing training data.
            min_buffer_size: Minimum buffer size required before training starts.
            min_new_data_count: Minimum number of new data points required for training.
            model_index: Optional index to support multiple model instances.
        """

        if min_buffer_size is None:
            min_buffer_size = seq_len
        if min_buffer_size < seq_len:
            raise ValueError("Buffer size must be greater than sequence length.")

        super().__init__(data_user_name, min_buffer_size, min_new_data_count)

        self.partial_optimizer = partial_optimizer
        self.partial_sampler = partial(
            RandomTimeSeriesSampler,
            sequence_length=seq_len,
            max_samples=max_samples,
        )
        self.partial_dataloader = partial(DataLoader, batch_size=batch_size)
        self.max_epochs = max_epochs
        self.data_user_name = data_user_name
        self.model_name = model_name
        self.log_prefix = log_prefix

        self.global_step = 0

    @override
    def on_data_users_attached(self) -> None:
        """Set up data user references when they are attached to the trainer.

        This method is called automatically by the PAMIQ framework when
        data users are attached to the trainer. It retrieves and stores
        references to the required data users for convenient access
        during training.
        """
        super().on_data_users_attached()
        self.forward_dynamics_data_user: DataUser[dict[str, list[Tensor]]] = (
            self.get_data_user(self.data_user_name)
        )

    @override
    def on_training_models_attached(self) -> None:
        """Set up model references when they are attached to the trainer.

        This method is called automatically by the PAMIQ framework when
        training models are attached to the trainer. It retrieves and
        stores references to the StackedHiddenFD model for convenient
        access during training.
        """

        super().on_training_models_attached()
        self.forward_dynamics = self.get_torch_training_model(
            self.model_name, StackedHiddenFD
        )

    @override
    def create_optimizers(self) -> OptimizersSetup:
        """Create optimizers for the StackedHiddenFD model. This method is
        called automatically by the PAMIQ framework to set up optimizers for
        the training process. It uses the `partial_optimizer` function to
        create an optimizer for the StackedHiddenFD model's parameters.

        Returns:
            Dictionary mapping optimizer name to configured optimizer instance.
        """
        return {
            OPTIMIZER_NAME: self.partial_optimizer(
                self.forward_dynamics.model.parameters()
            )
        }

    @override
    def train(self) -> None:
        """Execute StackedHiddenFD training process with explicit targets.

        This method implements a simplified StackedHiddenFD training loop:
        1. Creates a dataset and dataloader with explicit targets
        2. For each batch:
            - Moves data to the appropriate device
            - Extracts observations, actions, hidden states, and targets
            - Computes the predicted next observation
            - Calculates MSE loss between prediction and explicit target
            - Backpropagates the loss
            - Updates the model parameters
        3. Logs the loss and gradient norm to Aim
        4. Increments the global step counter
        """

        data = self.forward_dynamics_data_user.get_data()
        dataset = TensorDataset(
            torch.stack(data[DataKey.OBSERVATION]),
            torch.stack(data[DataKey.ACTION]),
            torch.stack(data[DataKey.HIDDEN]),
            torch.stack(data[DataKey.TARGET]),
        )
        sampler = self.partial_sampler(dataset=dataset)
        dataloader = self.partial_dataloader(dataset=dataset, sampler=sampler)
        device = get_device(self.forward_dynamics.model)

        for _ in range(self.max_epochs):
            batch: tuple[Tensor, Tensor, Tensor, Tensor]
            for batch in dataloader:
                observations, actions, hiddens, targets = batch
                observations = observations.to(device)
                actions = actions.to(device)
                hiddens = hiddens[:, 0].to(device)
                targets = targets.to(device)

                self.optimizers[OPTIMIZER_NAME].zero_grad()

                preds, _ = self.forward_dynamics.model(observations, actions, hiddens)

                loss = F.mse_loss(targets, preds)
                loss.backward()

                metrics = {"loss": loss.item()}
                metrics["grad norm"] = (
                    torch.cat(
                        [
                            p.grad.flatten()
                            for p in self.forward_dynamics.model.parameters()
                            if p.grad is not None
                        ]
                    )
                    .norm()
                    .item()
                )

                if run := get_global_run():
                    for k, v in metrics.items():
                        run.track(
                            v,
                            name=k,
                            step=self.global_step,
                            context={
                                "namespace": "trainer",
                                "trainer_type": self.log_prefix,
                            },
                        )
                self.optimizers[OPTIMIZER_NAME].step()
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
    def create_buffer(max_size: int) -> DictSequentialBuffer[Tensor]:
        """Create data buffer for this trainer."""
        return DictSequentialBuffer(
            [DataKey.OBSERVATION, DataKey.ACTION, DataKey.HIDDEN, DataKey.TARGET],
            max_size=max_size,
        )

    @classmethod
    def create_multiple(cls, num_trainers: int, **trainer_params: Any) -> list[Self]:
        """Create multiple StackedHiddenFDTrainerExplicitTarget instances.

        Each trainer is assigned a unique model_name and data_user_name by
        appending an index (0 to num_trainers-1) to the base names.

        Args:
            num_trainers: Number of trainers to create.
            **trainer_params: Parameters to pass to each trainer constructor.

        Returns:
            List of configured trainer instances.
        """
        trainers = list[Self]()
        for i in range(num_trainers):
            trainers.append(
                cls(
                    **trainer_params,
                    model_name=ModelName.FORWARD_DYNAMICS + str(i),
                    data_user_name=BufferName.FORWARD_DYNAMICS + str(i),
                    log_prefix="forward-dynamics" + str(i),
                )
            )
        return trainers


class LatentFDTrainer(TorchTrainer):
    """Trainer for the LatentFD model.

    This trainer implements the training loop for the LatentFD model,
    which predicts the next observation distribution given the current
    observation and action. It uses a recurrent core model to maintain
    hidden state across sequential predictions.
    """

    def __init__(
        self,
        partial_optimizer: partial[Optimizer],
        seq_len: int = 1,
        max_samples: int = 1,
        batch_size: int = 1,
        max_epochs: int = 1,
        imagination_length: int = 1,
        imagination_average_method: Callable[[Tensor], Tensor] = average_exponentially,
        model_name: str = ModelName.FORWARD_DYNAMICS,
        data_user_name: str = BufferName.FORWARD_DYNAMICS,
        log_prefix: str = "forward-dynamics",
        min_buffer_size: int | None = None,
        min_new_data_count: int = 0,
    ) -> None:
        """Initialize the StackedHiddenFDTrainer.

        Args:
            partial_optimizer: Partially configured optimizer to be used with
                the model parameters.
            seq_len: Sequence length per batch.
            max_samples: Max number of sample from dataset in 1 epoch.
            batch_size: Data sample size of 1 batch.
            max_epochs: Maximum number of epochs to train per training session.
            imagination_length: Length of the imagination sequence.
            imagenation_average_method: Method to average the loss over the imagination sequence.
            data_user_name: Name of the data user providing training data.
            min_buffer_size: Minimum buffer size required before training starts.
            min_new_data_count: Minimum number of new data points required for training.
        """
        if imagination_length < 1:
            raise ValueError("Imagination length must be greater than 0")

        if min_buffer_size is None:
            min_buffer_size = imagination_length + seq_len
        if min_buffer_size < imagination_length + seq_len:
            raise ValueError(
                "Buffer size must be greater than imagination length + sequence length."
            )
        super().__init__(data_user_name, min_buffer_size, min_new_data_count)

        self.partial_optimizer = partial_optimizer
        self.partial_sampler = partial(
            RandomTimeSeriesSampler,
            sequence_length=seq_len + imagination_length,
            max_samples=max_samples,
        )
        self.partial_dataloader = partial(DataLoader, batch_size=batch_size)
        self.max_epochs = max_epochs
        self.model_name = model_name
        self.data_user_name = data_user_name
        self.imagination_length = imagination_length
        self.imagination_average_method = imagination_average_method
        self.log_prefix = log_prefix

        self.global_step = 0

    @override
    def on_data_users_attached(self) -> None:
        """Set up data user references when they are attached to the trainer.

        This method is called automatically by the PAMIQ framework when
        data users are attached to the trainer. It retrieves and stores
        references to the required data users for convenient access
        during training.
        """
        super().on_data_users_attached()
        self.forward_dynamics_data_user: DataUser[dict[str, list[Tensor]]] = (
            self.get_data_user(self.data_user_name)
        )

    @override
    def on_training_models_attached(self) -> None:
        """Set up model references when they are attached to the trainer.

        This method is called automatically by the PAMIQ framework when
        training models are attached to the trainer. It retrieves and
        stores references to the LatentFD model for convenient access
        during training.
        """

        super().on_training_models_attached()
        self.forward_dynamics = self.get_torch_training_model(self.model_name, LatentFD)

    @override
    def create_optimizers(self) -> OptimizersSetup:
        """Create optimizers for the LatentFD model. This method is called
        automatically by the PAMIQ framework to set up optimizers for the
        training process. It uses the `partial_optimizer` function to create an
        optimizer for the LatentFD model's parameters.

        Returns:
            Dictionary mapping optimizer name to configured optimizer instance.
        """
        return {
            OPTIMIZER_NAME: self.partial_optimizer(
                self.forward_dynamics.model.parameters()
            )
        }

    @override
    def train(self) -> None:
        """Execute LatentFD training process.

        This method implements the core LatentFD training loop:
        1. Creates a dataset and dataloader
        2. For each batch:
            - Moves data to the appropriate device
            - Splits observations, actions, and hidden states
            - Computes the next observation distribution
            - Calculates the loss
            - Backpropagates the loss
            - Updates the model parameters
        3. Logs the loss to Aim
        4. Increments the global step counter
        """

        data = self.forward_dynamics_data_user.get_data()
        dataset = TensorDataset(
            torch.stack(data[DataKey.OBSERVATION]),
            torch.stack(data[DataKey.ACTION]),
            torch.stack(data[DataKey.ENCODER_HIDDEN]),
            torch.stack(data[DataKey.PREDICTOR_HIDDEN]),
        )
        sampler = self.partial_sampler(dataset=dataset)
        dataloader = self.partial_dataloader(dataset=dataset, sampler=sampler)
        device = get_device(self.forward_dynamics.model)

        for _ in range(self.max_epochs):
            batch: tuple[Tensor, Tensor, Tensor, Tensor]
            for batch in dataloader:
                observations, actions, encoder_hiddens, predictor_hiddens = batch
                observations = observations.to(device)
                actions = actions.to(device)
                obs_imaginations, encoder_hiddens, predictor_hiddens = (
                    observations[:, : -self.imagination_length],
                    encoder_hiddens[:, 0].to(device),
                    predictor_hiddens[:, 0].to(device),
                )

                self.optimizers[OPTIMIZER_NAME].zero_grad()

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
                        action_imaginations = action_imaginations.flatten(
                            0, 1
                        )  # (B', *)
                        obs_targets = obs_targets.flatten(0, 1)  # (B', *)

                    if i == 0:
                        forward_method = self.forward_dynamics.model.__call__
                    else:
                        forward_method = self.forward_dynamics.model.forward_with_no_len

                    obses_next_hat, _, next_encoder_hiddens, next_predictor_hiddens = (
                        forward_method(
                            obs_imaginations,
                            action_imaginations,
                            encoder_hiddens,
                            predictor_hiddens,
                        )
                    )

                    loss = torch.nn.functional.mse_loss(obses_next_hat, obs_targets)
                    loss_imaginations.append(loss)
                    obs_imaginations = obses_next_hat

                    if i == 0:
                        obs_imaginations = obs_imaginations.flatten(
                            0, 1
                        )  # (B, T-H, *) -> (B', *)
                        encoder_hiddens = next_encoder_hiddens.movedim(2, 1).flatten(
                            0, 1
                        )  # h'_i, (B, D, T-H, *) -> (B, T-H, D, *) -> (B', D, *)
                        predictor_hiddens = next_predictor_hiddens.movedim(
                            2, 1
                        ).flatten(
                            0, 1
                        )  # h'_i, (B, D, T-H, *) -> (B, T-H, D, *) -> (B', D, *)

                loss = self.imagination_average_method(torch.stack(loss_imaginations))
                loss.backward()

                metrics = {"loss/average": loss.item()}
                for i, loss_item in enumerate(loss_imaginations, start=1):
                    metrics[f"loss/imagination_{i}"] = loss_item.item()

                metrics["grad norm"] = (
                    torch.cat(
                        [
                            p.grad.flatten()
                            for p in self.forward_dynamics.model.parameters()
                            if p.grad is not None
                        ]
                    )
                    .norm()
                    .item()
                )

                if run := get_global_run():
                    for k, v in metrics.items():
                        run.track(
                            v,
                            name=k,
                            step=self.global_step,
                            context={
                                "namespace": "trainer",
                                "trainer_type": self.log_prefix,
                            },
                        )
                self.optimizers[OPTIMIZER_NAME].step()
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
    def create_buffer(max_size: int) -> DictSequentialBuffer[Tensor]:
        """Create data buffer for this trainer."""
        return DictSequentialBuffer(
            [
                DataKey.OBSERVATION,
                DataKey.ACTION,
                DataKey.ENCODER_HIDDEN,
                DataKey.PREDICTOR_HIDDEN,
            ],
            max_size=max_size,
        )

    @classmethod
    def create_multiple(cls, num_trainers: int, **trainer_params: Any) -> list[Self]:
        """Create multiple LatentFDTrainer instances.

        Each trainer is assigned a unique model_name and data_user_name by
        appending an index (0 to num_trainers-1) to the base names.

        Args:
            num_trainers: Number of trainers to create.
            **trainer_params: Parameters to pass to each trainer constructor.

        Returns:
            List of configured trainer instances.
        """
        trainers = list[Self]()
        for i in range(num_trainers):
            trainers.append(
                cls(
                    **trainer_params,
                    model_name=ModelName.FORWARD_DYNAMICS + str(i),
                    data_user_name=BufferName.FORWARD_DYNAMICS + str(i),
                    log_prefix="forward-dynamics" + str(i),
                )
            )
        return trainers
