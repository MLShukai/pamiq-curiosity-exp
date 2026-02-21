from collections.abc import Mapping
from functools import partial
from pathlib import Path

import pytest
import torch
from pamiq_core.testing import connect_components
from pamiq_core.torch import TorchTrainingModel
from pytest_mock import MockerFixture
from torch.optim import AdamW

from exp.data import BufferName, DataKey
from exp.data.dict_intermittent_chunk_buffer import DictIntermittentChunkBuffer
from exp.models import ModelName
from exp.models.components.qgru import QGRU
from exp.models.components.stacked_features import LerpStackedFeatures
from exp.models.components.ttt import TTT
from exp.models.fd_policy import TTTFDPiV
from exp.models.utils import ActionInfo, ObsInfo
from exp.trainers.ppo_policy import compute_advantage
from exp.trainers.ttt import TTTFDPiVTrainer
from tests.helpers import parametrize_device


class TestTTTFDPiVTrainer:
    DEPTH = 2
    DIM = 8
    OBS_DIM = 16
    OBS_NUM_TOKENS = 4
    ACTION_CHOICES = [3, 4]  # Multiple discrete actions
    DIM_ACTION = 8
    DIM_INTERNAL_ACTION = 8
    DIM_HIDDEN = 16
    NUM_HEAD = 4
    GET_INTERVAL = 16
    DIM_EXTERNAL_ACTION = 16
    DIM_ATTENTION = 17
    DIM_SURPRISAL = 18
    DIM_VALUE = 19
    DIM_BODY_STATE = 1

    @pytest.fixture
    def fd_policy_value_model(self):
        action_info = ActionInfo(choices=self.ACTION_CHOICES, dim=self.DIM_ACTION)
        obs_encoder = LerpStackedFeatures(
            dim_in=self.OBS_DIM, dim_out=self.DIM, num_stack=self.OBS_NUM_TOKENS
        )
        time_mixer = QGRU(
            depth=self.DEPTH,
            dim=self.DIM,
            dim_hidden=self.DIM_HIDDEN,
            dropout=0.1,
        )
        core_model = TTT(
            depth=self.DEPTH,
            dim=self.DIM,
            dim_hidden=self.DIM_HIDDEN,
            num_head=self.NUM_HEAD,
            base_lr=(0.0001, 0.01),
            chunk_size=16,
            dropout=0.1,
        )
        return TTTFDPiV(
            obs_dim_hidden=self.DIM,
            action_info=action_info,
            external_action_dim=self.DIM_EXTERNAL_ACTION,
            attention_dim=self.DIM_ATTENTION,
            surprisal_dim=self.DIM_SURPRISAL,
            value_dim=self.DIM_VALUE,
            internal_action_dim=self.DIM_INTERNAL_ACTION,
            body_state_dim=self.DIM_BODY_STATE,
            dim=self.DIM,
            obs_encoder=obs_encoder,
            obs_time_mixer=time_mixer,
            core_model=core_model,
            surprisal_shape=torch.Size(
                [self.DEPTH, self.NUM_HEAD, self.DIM // self.NUM_HEAD]
            ),
        )

    @pytest.fixture
    def models(self, fd_policy_value_model):
        return {ModelName.FD_POLICY_VALUE: fd_policy_value_model}

    @pytest.fixture
    def data_buffers(self):
        intermittent_keys_first_add_steps: Mapping[str, int] = {
            DataKey.HIDDEN: 0,
        }
        chunk_keys_first_add_steps: Mapping[str, int] = {
            DataKey.OBSERVATION: 0,
            DataKey.TARGET: 1,
            DataKey.PREVIOUS_ACTION: 0,
            DataKey.ACTION: 0,
            DataKey.INTERNAL_STATE: 0,
            DataKey.ACTION_LOG_PROB: 0,
            DataKey.REWARD: 0,
            DataKey.VALUE: 0,
        }
        return {
            BufferName.FD_POLICY_VALUE: DictIntermittentChunkBuffer(
                intermittent_keys_first_add_steps,
                chunk_keys_first_add_steps,
                get_interval=self.GET_INTERVAL,
                max_size=16,
            )
        }

    @pytest.fixture
    def trainer(
        self,
        mocker: MockerFixture,
    ):
        mocker.patch("exp.trainers.ppo_policy.get_global_run")
        return TTTFDPiVTrainer(
            partial_optimizer=partial(AdamW, lr=3e-4),
            gamma=0.99,
            gae_lambda=0.95,
            min_new_data_count=self.GET_INTERVAL,
        )

    def test_init_validation(self, mocker: MockerFixture):
        """Test initialization parameter validation."""
        mocker.patch("exp.trainers.ppo_policy.get_global_run")

        # Test invalid gamma
        with pytest.raises(ValueError, match="gamma must be in range"):
            TTTFDPiVTrainer(
                partial_optimizer=partial(AdamW),
                gamma=1.5,
                gae_lambda=0.95,
            )

        # Test invalid gae_lambda
        with pytest.raises(ValueError, match="gae_lambda must be in range"):
            TTTFDPiVTrainer(
                partial_optimizer=partial(AdamW),
                gamma=0.99,
                gae_lambda=-0.1,
            )

    @parametrize_device
    def test_run(self, device, data_buffers, models, trainer: TTTFDPiVTrainer):
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
            obs_embeddings = torch.randn(self.DIM)
            hidden = {
                "time": torch.randn(self.DEPTH, self.DIM_HIDDEN),
                "ttt": [
                    {
                        "W1": torch.randn(
                            self.NUM_HEAD,
                            self.DIM_HIDDEN // self.NUM_HEAD,
                            self.DIM // self.NUM_HEAD,
                        ),
                        "W2": torch.randn(
                            self.NUM_HEAD,
                            self.DIM // self.NUM_HEAD,
                            self.DIM_HIDDEN // self.NUM_HEAD,
                        ),
                    }
                    for _ in range(self.DEPTH)
                ],
            }
            actions = {
                "external_action": torch.stack(
                    [torch.randint(0, dim, ()) for dim in self.ACTION_CHOICES], dim=-1
                ),
                "internal_action": torch.rand(
                    self.DIM_VALUE
                    + self.DIM_ATTENTION
                    + self.DIM_SURPRISAL
                    + self.DIM_INTERNAL_ACTION
                ),
            }
            previous_actions = actions
            action_log_probs = torch.randn(())
            rewards = torch.randn(())
            values = torch.randn(())
            internal_state = torch.randn(())

            collector.collect(
                {
                    DataKey.OBSERVATION: observations,
                    DataKey.TARGET: obs_embeddings,
                    DataKey.HIDDEN: hidden,
                    DataKey.PREVIOUS_ACTION: previous_actions,
                    DataKey.ACTION: actions,
                    DataKey.ACTION_LOG_PROB: action_log_probs,
                    DataKey.REWARD: rewards,
                    DataKey.VALUE: values,
                    DataKey.INTERNAL_STATE: internal_state,
                }
            )

        assert trainer.global_step == 0
        assert trainer.run() is True
        assert trainer.global_step > 0
        global_step = trainer.global_step
        assert trainer.run() is False
        assert trainer.global_step == global_step

    def test_save_and_load_state(self, trainer: TTTFDPiVTrainer, tmp_path: Path):
        """Test saving and loading trainer state."""
        trainer.global_step = 42
        trainer_path = tmp_path / "trainer"
        trainer.save_state(trainer_path)
        assert (trainer_path / "global_step").is_file()

        trainer.global_step = -1
        trainer.load_state(trainer_path)
        assert trainer.global_step == 42
