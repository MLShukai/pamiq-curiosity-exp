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
from exp.models.fd_policy import StackedHiddenFDPiV
from exp.models.utils import ActionInfo, ObsInfo
from exp.trainers.fd_ppo_policy import (
    PPOHiddenStateFDPiVTrainer,
    compute_advantage,
)
from tests.helpers import parametrize_device


class TestPPOHiddenStateFDPiVTrainer:
    DEPTH = 2
    DIM = 8
    OBS_DIM = 16
    OBS_DIM_HIDDEN = 12
    OBS_NUM_TOKENS = 4
    ACTION_CHOICES = [3, 4]  # Multiple discrete actions
    DIM_ACTION = 8
    SEQ_LEN = 10
    DIM_FF_HIDDEN = 16

    @pytest.fixture
    def fd_policy_value_model(self):
        obs_info = ObsInfo(
            dim=self.OBS_DIM,
            dim_hidden=self.OBS_DIM_HIDDEN,
            num_tokens=self.OBS_NUM_TOKENS,
        )
        action_info = ActionInfo(choices=self.ACTION_CHOICES, dim=self.DIM_ACTION)
        core_model = QLSTM(
            depth=self.DEPTH,
            dim=self.DIM,
            dim_ff_hidden=self.DIM_FF_HIDDEN,
            dropout=0.0,
        )
        return StackedHiddenFDPiV(
            obs_info=obs_info,
            action_info=action_info,
            dim=self.DIM,
            core_model=core_model,
        )

    @pytest.fixture
    def models(self, fd_policy_value_model):
        return {ModelName.FD_POLICY_VALUE: fd_policy_value_model}

    @pytest.fixture
    def data_buffers(self):
        return {
            BufferName.FD_POLICY_VALUE: DictSequentialBuffer(
                [
                    DataKey.OBSERVATION,
                    DataKey.HIDDEN,
                    DataKey.ACTION,
                    DataKey.ACTION_LOG_PROB,
                    DataKey.REWARD,
                    DataKey.VALUE,
                ],
                max_size=32,
            )
        }

    @pytest.fixture
    def trainer(
        self,
        mocker: MockerFixture,
    ):
        mocker.patch("exp.trainers.ppo_policy.get_global_run")
        return PPOHiddenStateFDPiVTrainer(
            partial_optimizer=partial(AdamW, lr=3e-4),
            gamma=0.99,
            gae_lambda=0.95,
            seq_len=self.SEQ_LEN,
            max_samples=4,
            batch_size=2,
            min_new_data_count=1,
        )

    def test_init_validation(self, mocker: MockerFixture):
        """Test initialization parameter validation."""
        mocker.patch("exp.trainers.ppo_policy.get_global_run")

        # Test invalid gamma
        with pytest.raises(ValueError, match="gamma must be in range"):
            PPOHiddenStateFDPiVTrainer(
                partial_optimizer=partial(AdamW),
                gamma=1.5,
                gae_lambda=0.95,
            )

        # Test invalid gae_lambda
        with pytest.raises(ValueError, match="gae_lambda must be in range"):
            PPOHiddenStateFDPiVTrainer(
                partial_optimizer=partial(AdamW),
                gamma=0.99,
                gae_lambda=-0.1,
            )

        # Test min_buffer_size < seq_len
        with pytest.raises(
            ValueError, match="min_buffer_size .* must be at least seq_len"
        ):
            PPOHiddenStateFDPiVTrainer(
                partial_optimizer=partial(AdamW),
                gamma=0.99,
                gae_lambda=0.95,
                seq_len=10,
                min_buffer_size=5,
            )

    @parametrize_device
    def test_run(
        self, device, data_buffers, models, trainer: PPOHiddenStateFDPiVTrainer
    ):
        """Test PPO Policy Trainer workflow."""
        models = {
            name: TorchTrainingModel(m, has_inference_model=False, device=device)
            for name, m in models.items()
        }

        components = connect_components(
            trainers=trainer, buffers=data_buffers, models=models
        )
        collector = components.data_collectors[BufferName.FD_POLICY_VALUE]

        # Collect policy data
        for _ in range(20):
            observations = torch.randn(self.OBS_NUM_TOKENS, self.OBS_DIM)
            hidden = torch.randn(self.DEPTH, self.DIM)
            actions = torch.stack(
                [torch.randint(0, dim, ()) for dim in self.ACTION_CHOICES], dim=-1
            )
            action_log_probs = torch.randn(len(self.ACTION_CHOICES))
            rewards = torch.randn(())
            values = torch.randn(())

            collector.collect(
                {
                    DataKey.OBSERVATION: observations,
                    DataKey.HIDDEN: hidden,
                    DataKey.ACTION: actions,
                    DataKey.ACTION_LOG_PROB: action_log_probs,
                    DataKey.REWARD: rewards,
                    DataKey.VALUE: values,
                }
            )

        assert trainer.global_step == 0
        assert trainer.run() is True
        assert trainer.global_step > 0
        global_step = trainer.global_step
        assert trainer.run() is False
        assert trainer.global_step == global_step

    @parametrize_device
    def test_run_with_upper_action(self, device, models, mocker: MockerFixture):
        """Test PPO Policy Trainer with include_upper_action=True."""
        mocker.patch("exp.trainers.ppo_policy.get_global_run")

        # Create trainer with include_upper_action=True
        trainer = PPOHiddenStateFDPiVTrainer(
            partial_optimizer=partial(AdamW, lr=3e-4),
            gamma=0.99,
            gae_lambda=0.95,
            seq_len=self.SEQ_LEN,
            max_samples=4,
            batch_size=2,
            min_new_data_count=1,
            include_upper_action=True,
        )

        # Create buffer with UPPER_ACTION key
        data_buffers = {
            BufferName.FD_POLICY_VALUE.value: DictSequentialBuffer(
                [
                    DataKey.OBSERVATION,
                    DataKey.HIDDEN,
                    DataKey.ACTION,
                    DataKey.ACTION_LOG_PROB,
                    DataKey.REWARD,
                    DataKey.VALUE,
                    DataKey.UPPER_ACTION,
                ],
                max_size=32,
            )
        }

        models = {
            name: TorchTrainingModel(m, has_inference_model=False, device=device)
            for name, m in models.items()
        }

        components = connect_components(
            trainers=trainer, buffers=data_buffers, models=models
        )
        collector = components.data_collectors[BufferName.FD_POLICY_VALUE]

        # Collect policy data with upper action
        for _ in range(20):
            observations = torch.randn(self.OBS_NUM_TOKENS, self.OBS_DIM)
            hidden = torch.randn(self.DEPTH, self.DIM)
            actions = torch.stack(
                [torch.randint(0, dim, ()) for dim in self.ACTION_CHOICES], dim=-1
            )
            action_log_probs = torch.randn(len(self.ACTION_CHOICES))
            rewards = torch.randn(())
            values = torch.randn(())
            upper_action = torch.randn(2)  # Example upper action dimension

            collector.collect(
                {
                    DataKey.OBSERVATION: observations,
                    DataKey.HIDDEN: hidden,
                    DataKey.ACTION: actions,
                    DataKey.ACTION_LOG_PROB: action_log_probs,
                    DataKey.REWARD: rewards,
                    DataKey.VALUE: values,
                    DataKey.UPPER_ACTION: upper_action,
                }
            )

        assert trainer.global_step == 0
        assert trainer.run() is True
        assert trainer.global_step > 0

    def test_save_and_load_state(
        self, trainer: PPOHiddenStateFDPiVTrainer, tmp_path: Path
    ):
        """Test saving and loading trainer state."""
        trainer.global_step = 42
        trainer_path = tmp_path / "trainer"
        trainer.save_state(trainer_path)
        assert (trainer_path / "global_step").is_file()

        trainer.global_step = -1
        trainer.load_state(trainer_path)
        assert trainer.global_step == 42

    def test_create_hierarchical_buffers(self):
        """Test creating hierarchical buffers."""
        max_size = 100
        num_buffers = 3

        buffers = PPOHiddenStateFDPiVTrainer.create_hierarchical_buffers(
            max_size, num_buffers
        )

        # Check correct number of buffers created
        assert len(buffers) == num_buffers

        # Check each buffer has correct configuration
        for i, buffer in enumerate(buffers):
            assert isinstance(buffer, DictSequentialBuffer)
            assert buffer.max_size == max_size

            # Check keys
            expected_keys = [
                DataKey.OBSERVATION,
                DataKey.HIDDEN,
                DataKey.ACTION,
                DataKey.ACTION_LOG_PROB,
                DataKey.REWARD,
                DataKey.VALUE,
            ]

            # Only the last buffer should have UPPER_ACTION
            if i < num_buffers - 1:
                expected_keys.append(DataKey.UPPER_ACTION)

            assert buffer.keys == set(expected_keys)

    def test_create_multiple_hierarchical(self, mocker: MockerFixture):
        """Test creating multiple trainers with hierarchical flag."""
        mocker.patch("exp.trainers.ppo_policy.get_global_run")

        num_trainers = 3
        trainers = PPOHiddenStateFDPiVTrainer.create_multiple(
            num_trainers=num_trainers,
            hierarchical=True,
            partial_optimizer=partial(AdamW, lr=3e-4),
            gamma=0.99,
        )

        assert len(trainers) == num_trainers

        # Check hierarchical setup: all except last should have upper_action
        for i, trainer in enumerate(trainers):
            if i < num_trainers - 1:
                assert trainer.include_upper_action is True
            else:
                assert trainer.include_upper_action is False

            # Check unique names
            assert trainer.model_name == ModelName.FD_POLICY_VALUE + str(i)
            assert trainer.data_user_name == BufferName.FD_POLICY_VALUE + str(i)
            assert trainer.log_prefix == "ppo-policy" + str(i)


class TestComputeAdvantage:
    """Test compute_advantage function."""

    @pytest.mark.parametrize("shape", [(3,), (5,), (10,)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @parametrize_device
    def test_compute_advantage(self, shape, dtype, device):
        """Test advantage computation maintains shape, dtype, and device."""
        # Create test tensors with specified properties
        rewards = torch.randn(shape, dtype=dtype, device=device)
        values = torch.randn(shape, dtype=dtype, device=device)
        final_next_value = torch.randn((), dtype=dtype, device=device)
        gamma = 0.99
        gae_lambda = 0.95

        advantages = compute_advantage(
            rewards, values, final_next_value, gamma, gae_lambda
        )

        # Verify shape, dtype, and device are preserved
        assert advantages.shape == values.shape
        assert advantages.dtype == dtype
        assert advantages.device == device
