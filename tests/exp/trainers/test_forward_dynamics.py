from functools import partial
from pathlib import Path

import pytest
import torch
from pamiq_core.data.impls import DictSequentialBuffer
from pamiq_core.testing import connect_components
from pamiq_core.torch import TorchTrainingModel
from pytest_mock import MockerFixture
from torch.optim import AdamW

from exp.data import BufferName, DataKey
from exp.models import ModelName
from exp.models.components.qlstm import QLSTM
from exp.models.forward_dynamics import StackedHiddenFD
from exp.models.latent_fd import LatentFD, ObsActionFlattenHead, ObsPredictionHead
from exp.models.utils import ActionInfo, ObsInfo
from exp.trainers.forward_dynamics import (
    HiddenStateFDTrainer,
    HiddenStateFDTrainerExplicitTarget,
    LatentFDTrainer,
)
from tests.helpers import parametrize_device


class TestHiddenStateFDTrainer:
    BATCH = 4
    DEPTH = 8
    DIM = 16
    DIM_FF_HIDDEN = 32
    LEN = 64
    LEN_SEQ = 16
    DROPOUT = 0.1
    DIM_OBS = 32
    DIM_ACTION = 8
    ACTION_CHOICES = [4, 9, 2]

    @pytest.fixture
    def forward_dynamics(
        self,
    ):
        obs_info = ObsInfo(dim=self.DIM_OBS, dim_hidden=self.DIM, num_tokens=1)
        action_info = ActionInfo(choices=self.ACTION_CHOICES, dim=self.DIM_ACTION)
        core_model = QLSTM(
            depth=self.DEPTH,
            dim=self.DIM,
            dim_ff_hidden=self.DIM_FF_HIDDEN,
            dropout=self.DROPOUT,
        )
        return StackedHiddenFD(
            obs_info=obs_info,
            action_info=action_info,
            dim=self.DIM,
            core_model=core_model,
        )

    @pytest.fixture
    def data_buffers(self):
        return {
            BufferName.FORWARD_DYNAMICS: DictSequentialBuffer(
                [DataKey.OBSERVATION, DataKey.ACTION, DataKey.HIDDEN], max_size=self.LEN
            )
        }

    @pytest.fixture
    def trainer(
        self,
        mocker: MockerFixture,
    ):
        mocker.patch("exp.trainers.forward_dynamics.get_global_run")
        trainer = HiddenStateFDTrainer(
            partial(AdamW, lr=1e-4, weight_decay=0.04),
            seq_len=self.LEN_SEQ,
            max_samples=4,
            batch_size=2,
            max_epochs=1,
            imagination_length=2,
            min_buffer_size=self.LEN,
            min_new_data_count=4,
        )
        return trainer

    def create_action(self) -> torch.Tensor:
        actions = []
        for choice in self.ACTION_CHOICES:
            actions.append(torch.randint(0, choice, ()))
        return torch.stack(actions, dim=-1)

    @parametrize_device
    def test_run(self, device, data_buffers, forward_dynamics, trainer):
        models = {
            str(ModelName.FORWARD_DYNAMICS): TorchTrainingModel(
                forward_dynamics, has_inference_model=False, device=device
            ),
        }
        components = connect_components(
            trainers=trainer, buffers=data_buffers, models=models
        )
        collector = components.data_collectors[BufferName.FORWARD_DYNAMICS]
        for _ in range(self.LEN):
            collector.collect(
                {
                    DataKey.OBSERVATION: torch.randn(1, self.DIM_OBS),
                    DataKey.ACTION: self.create_action(),
                    DataKey.HIDDEN: torch.randn(self.DEPTH, self.DIM),
                }
            )

        assert trainer.global_step == 0
        assert trainer.run() is True
        assert trainer.global_step > 0
        global_step = trainer.global_step
        assert trainer.run() is False
        assert trainer.global_step == global_step

    def test_save_and_load_state(self, trainer, tmp_path: Path):
        trainer_path = tmp_path / "trainer"
        trainer.save_state(trainer_path)
        global_step = trainer.global_step
        assert (trainer_path / "global_step").is_file()

        trainer.global_step = -1
        trainer.load_state(trainer_path)
        assert trainer.global_step == global_step

    def test_imagination_length_validation(self, mocker: MockerFixture):
        """Test that imagination_length must be greater than 0."""
        mocker.patch("exp.trainers.forward_dynamics.get_global_run")
        with pytest.raises(
            ValueError, match="Imagination length must be greater than 0"
        ):
            HiddenStateFDTrainer(
                partial(AdamW, lr=1e-4, weight_decay=0.04),
                seq_len=self.LEN_SEQ,
                imagination_length=0,
                min_buffer_size=self.LEN,
            )

    def test_buffer_size_validation(self, mocker: MockerFixture):
        """Test that buffer size must be greater than imagination_length +
        seq_len."""
        mocker.patch("exp.trainers.forward_dynamics.get_global_run")
        with pytest.raises(
            ValueError,
            match="Buffer size must be greater than imagination length \\+ sequence length",
        ):
            HiddenStateFDTrainer(
                partial(AdamW, lr=1e-4, weight_decay=0.04),
                seq_len=10,
                imagination_length=5,
                min_buffer_size=14,  # Less than seq_len + imagination_length
            )


class TestHiddenStateFDTrainerExplicitTarget:
    """Test class for HiddenStateFDTrainerExplicitTarget."""

    BATCH = 4
    DEPTH = 8
    DIM = 16
    DIM_FF_HIDDEN = 32
    LEN = 64
    LEN_SEQ = 16
    DROPOUT = 0.1
    DIM_OBS = 32
    DIM_ACTION = 8
    ACTION_CHOICES = [4, 9, 2]

    @pytest.fixture
    def forward_dynamics(self):
        """Create a StackedHiddenFD model for testing."""
        obs_info = ObsInfo(dim=self.DIM_OBS, dim_hidden=self.DIM, num_tokens=1)
        action_info = ActionInfo(choices=self.ACTION_CHOICES, dim=self.DIM_ACTION)
        core_model = QLSTM(
            depth=self.DEPTH,
            dim=self.DIM,
            dim_ff_hidden=self.DIM_FF_HIDDEN,
            dropout=self.DROPOUT,
        )
        return StackedHiddenFD(
            obs_info=obs_info,
            action_info=action_info,
            dim=self.DIM,
            core_model=core_model,
        )

    @pytest.fixture
    def data_buffers(self):
        """Create data buffers with TARGET key for explicit target training."""
        return {
            BufferName.FORWARD_DYNAMICS: DictSequentialBuffer(
                [DataKey.OBSERVATION, DataKey.ACTION, DataKey.HIDDEN, DataKey.TARGET],
                max_size=self.LEN,
            )
        }

    @pytest.fixture
    def trainer(self, mocker: MockerFixture):
        """Create a HiddenStateFDTrainerExplicitTarget for testing."""
        mocker.patch("exp.trainers.forward_dynamics.get_global_run")
        trainer = HiddenStateFDTrainerExplicitTarget(
            partial(AdamW, lr=1e-4, weight_decay=0.04),
            seq_len=self.LEN_SEQ,
            max_samples=4,
            batch_size=2,
            max_epochs=1,
            min_buffer_size=self.LEN,
            min_new_data_count=4,
        )
        return trainer

    @pytest.fixture
    def trainer_with_index(self, mocker: MockerFixture):
        """Create a trainer with model_index for testing multi-model
        scenarios."""
        mocker.patch("exp.trainers.forward_dynamics.get_global_run")
        trainer = HiddenStateFDTrainerExplicitTarget(
            partial(AdamW, lr=1e-4, weight_decay=0.04),
            seq_len=self.LEN_SEQ,
            max_samples=4,
            batch_size=2,
            max_epochs=1,
            min_buffer_size=self.LEN,
            min_new_data_count=4,
        )
        return trainer

    def create_action(self) -> torch.Tensor:
        """Create a random action tensor for testing."""
        actions = []
        for choice in self.ACTION_CHOICES:
            actions.append(torch.randint(0, choice, ()))
        return torch.stack(actions, dim=-1)

    @parametrize_device
    def test_run(self, device, data_buffers, forward_dynamics, trainer):
        """Test the complete training run."""
        models = {
            str(ModelName.FORWARD_DYNAMICS): TorchTrainingModel(
                forward_dynamics, has_inference_model=False, device=device
            ),
        }
        components = connect_components(
            trainers=trainer, buffers=data_buffers, models=models
        )
        collector = components.data_collectors[BufferName.FORWARD_DYNAMICS]

        # Collect training data including explicit targets
        for _ in range(self.LEN):
            collector.collect(
                {
                    DataKey.OBSERVATION: torch.randn(1, self.DIM_OBS),
                    DataKey.ACTION: self.create_action(),
                    DataKey.HIDDEN: torch.randn(self.DEPTH, self.DIM),
                    DataKey.TARGET: torch.randn(1, self.DIM_OBS),  # Explicit target
                }
            )

        assert trainer.global_step == 0
        assert trainer.run() is True
        assert trainer.global_step > 0
        global_step = trainer.global_step
        assert trainer.run() is False
        assert trainer.global_step == global_step

    def test_save_and_load_state(self, trainer, tmp_path: Path):
        """Test saving and loading trainer state."""
        trainer_path = tmp_path / "trainer"
        trainer.save_state(trainer_path)
        global_step = trainer.global_step
        assert (trainer_path / "global_step").is_file()

        trainer.global_step = -1
        trainer.load_state(trainer_path)
        assert trainer.global_step == global_step

    def test_buffer_size_validation(self, mocker: MockerFixture):
        """Test that buffer size must be greater than seq_len."""
        mocker.patch("exp.trainers.forward_dynamics.get_global_run")
        with pytest.raises(
            ValueError,
            match="Buffer size must be greater than sequence length",
        ):
            HiddenStateFDTrainerExplicitTarget(
                partial(AdamW, lr=1e-4, weight_decay=0.04),
                seq_len=10,
                min_buffer_size=9,  # Less than seq_len
            )


class TestLatentFDTrainer:
    BATCH = 4
    DEPTH = 8
    DIM = 16
    DIM_FF_HIDDEN = 32
    LEN = 64
    LEN_SEQ = 16
    DROPOUT = 0.1
    DIM_OBS = 32
    DIM_ACTION = 8
    ACTION_CHOICES = [4, 9, 2]

    @pytest.fixture
    def forward_dynamics(
        self,
    ):
        obs_info = ObsInfo(dim=self.DIM_OBS, dim_hidden=self.DIM, num_tokens=1)
        action_info = ActionInfo(choices=self.ACTION_CHOICES, dim=self.DIM_ACTION)
        encoder = QLSTM(
            depth=self.DEPTH,
            dim=self.DIM,
            dim_ff_hidden=self.DIM_FF_HIDDEN,
            dropout=self.DROPOUT,
        )
        predictor = QLSTM(
            depth=self.DEPTH,
            dim=self.DIM,
            dim_ff_hidden=self.DIM_FF_HIDDEN,
            dropout=self.DROPOUT,
        )
        obs_action_flatten_head = ObsActionFlattenHead(
            obs_info=obs_info,
            action_info=action_info,
            output_dim=self.DIM,
        )
        obs_prediction_head = ObsPredictionHead(
            obs_info=obs_info,
            input_dim=self.DIM,
        )

        return LatentFD(
            obs_action_flatten_head=obs_action_flatten_head,
            encoder=encoder,
            predictor=predictor,
            obs_prediction_head=obs_prediction_head,
        )

    @pytest.fixture
    def data_buffers(self):
        return {
            BufferName.FORWARD_DYNAMICS: DictSequentialBuffer(
                [
                    DataKey.OBSERVATION,
                    DataKey.ACTION,
                    DataKey.ENCODER_HIDDEN,
                    DataKey.PREDICTOR_HIDDEN,
                ],
                max_size=self.LEN,
            )
        }

    @pytest.fixture
    def trainer(
        self,
        mocker: MockerFixture,
    ):
        mocker.patch("exp.trainers.forward_dynamics.get_global_run")
        trainer = LatentFDTrainer(
            partial(AdamW, lr=1e-4, weight_decay=0.04),
            seq_len=self.LEN_SEQ,
            max_samples=4,
            batch_size=2,
            max_epochs=1,
            imagination_length=2,
            min_buffer_size=self.LEN,
            min_new_data_count=4,
        )
        return trainer

    def create_action(self) -> torch.Tensor:
        actions = []
        for choice in self.ACTION_CHOICES:
            actions.append(torch.randint(0, choice, ()))
        return torch.stack(actions, dim=-1)

    @parametrize_device
    def test_run(self, device, data_buffers, forward_dynamics, trainer):
        models = {
            str(ModelName.FORWARD_DYNAMICS): TorchTrainingModel(
                forward_dynamics, has_inference_model=False, device=device
            ),
        }
        components = connect_components(
            trainers=trainer, buffers=data_buffers, models=models
        )
        collector = components.data_collectors[BufferName.FORWARD_DYNAMICS]
        for _ in range(self.LEN):
            collector.collect(
                {
                    DataKey.OBSERVATION: torch.randn(1, self.DIM_OBS),
                    DataKey.ACTION: self.create_action(),
                    DataKey.ENCODER_HIDDEN: torch.randn(self.DEPTH, self.DIM),
                    DataKey.PREDICTOR_HIDDEN: torch.randn(self.DEPTH, self.DIM),
                }
            )

        assert trainer.global_step == 0
        assert trainer.run() is True
        assert trainer.global_step > 0
        global_step = trainer.global_step
        assert trainer.run() is False
        assert trainer.global_step == global_step

    def test_save_and_load_state(self, trainer, tmp_path: Path):
        trainer_path = tmp_path / "trainer"
        trainer.save_state(trainer_path)
        global_step = trainer.global_step
        assert (trainer_path / "global_step").is_file()

        trainer.global_step = -1
        trainer.load_state(trainer_path)
        assert trainer.global_step == global_step

    def test_imagination_length_validation(self, mocker: MockerFixture):
        """Test that imagination_length must be greater than 0."""
        mocker.patch("exp.trainers.forward_dynamics.get_global_run")
        with pytest.raises(
            ValueError, match="Imagination length must be greater than 0"
        ):
            LatentFDTrainer(
                partial(AdamW, lr=1e-4, weight_decay=0.04),
                seq_len=self.LEN_SEQ,
                imagination_length=0,
                min_buffer_size=self.LEN,
            )

    def test_buffer_size_validation(self, mocker: MockerFixture):
        """Test that buffer size must be greater than imagination_length +
        seq_len."""
        mocker.patch("exp.trainers.forward_dynamics.get_global_run")
        with pytest.raises(
            ValueError,
            match="Buffer size must be greater than imagination length \\+ sequence length",
        ):
            LatentFDTrainer(
                partial(AdamW, lr=1e-4, weight_decay=0.04),
                seq_len=10,
                imagination_length=5,
                min_buffer_size=14,  # Less than seq_len + imagination_length
            )
