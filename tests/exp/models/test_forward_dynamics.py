import pytest
import torch

from exp.models.components.qlstm import QLSTM
from exp.models.forward_dynamics import StackedHiddenFD
from exp.models.utils import ActionInfo, ObsInfo


class TestStackedHiddenFD:
    # Test hyperparameters
    BATCH_SIZE = 2
    SEQ_LEN = 3
    DEPTH = 2
    DIM = 8
    OBS_DIM = 16
    OBS_DIM_HIDDEN = 12
    OBS_NUM_TOKENS = 4
    ACTION_DIM = 4
    ACTION_CHOICES = [2, 3, 4]
    DIM_FF_HIDDEN = 16

    @pytest.fixture
    def obs_info(self):
        return ObsInfo(
            dim=self.OBS_DIM,
            dim_hidden=self.OBS_DIM_HIDDEN,
            num_tokens=self.OBS_NUM_TOKENS,
        )

    @pytest.fixture
    def action_info(self):
        return ActionInfo(
            choices=self.ACTION_CHOICES,
            dim=self.ACTION_DIM,
        )

    @pytest.fixture
    def core_model(self):
        return QLSTM(
            depth=self.DEPTH,
            dim=self.DIM,
            dim_ff_hidden=self.DIM_FF_HIDDEN,
            dropout=0.0,
        )

    @pytest.fixture
    def dynamics_model(self, obs_info, action_info, core_model):
        return StackedHiddenFD(
            obs_info=obs_info,
            action_info=action_info,
            dim=self.DIM,
            core_model=core_model,
        )

    @pytest.fixture
    def obs(self):
        return torch.randn(
            self.BATCH_SIZE, self.SEQ_LEN, self.OBS_NUM_TOKENS, self.OBS_DIM
        )

    @pytest.fixture
    def action(self):
        actions = []
        for choice in self.ACTION_CHOICES:
            actions.append(torch.randint(0, choice, (self.BATCH_SIZE, self.SEQ_LEN)))
        return torch.stack(actions, dim=-1)

    @pytest.fixture
    def hidden(self):
        return torch.randn(self.BATCH_SIZE, self.DEPTH, self.DIM)

    def test_forward(self, dynamics_model, obs, action, hidden):
        """Test forward pass of StackedHiddenFD model."""
        # Run forward pass
        obs_hat, next_hidden = dynamics_model(obs, action, hidden)

        # Check output types and shapes
        assert obs_hat.shape == (
            self.BATCH_SIZE,
            self.SEQ_LEN,
            self.OBS_NUM_TOKENS,
            self.OBS_DIM,
        )
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM,
        )

        # Check tensor properties
        assert obs_hat.dtype == torch.float32
        assert obs_hat.requires_grad

    def test_single_batch(self, dynamics_model, obs, action, hidden):
        """Test with single batch size."""
        single_obs = obs[:1]
        single_action = action[:1]
        single_hidden = hidden[:1]

        single_obs_hat, single_next_hidden = dynamics_model(
            single_obs, single_action, single_hidden
        )

        assert single_obs_hat.shape == (
            1,
            self.SEQ_LEN,
            self.OBS_NUM_TOKENS,
            self.OBS_DIM,
        )
        assert single_next_hidden.shape == (1, self.DEPTH, self.SEQ_LEN, self.DIM)

    def test_forward_with_no_len(self, dynamics_model, hidden):
        """Test forward_with_no_len for inference without sequence
        dimension."""
        # Create inputs without sequence length dimension
        obs_no_len = torch.randn(self.BATCH_SIZE, self.OBS_NUM_TOKENS, self.OBS_DIM)
        action_no_len = torch.stack(
            [
                torch.randint(0, choice, (self.BATCH_SIZE,))
                for choice in self.ACTION_CHOICES
            ],
            dim=-1,
        )

        # Run forward pass
        obs_hat, next_hidden = dynamics_model.forward_with_no_len(
            obs_no_len, action_no_len, hidden
        )

        # Check output shapes
        assert obs_hat.shape == (self.BATCH_SIZE, self.OBS_NUM_TOKENS, self.OBS_DIM)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.DIM,
        )

        # Check tensor properties
        assert obs_hat.dtype == torch.float32
        assert obs_hat.requires_grad

    def test_forward_no_hidden(self, dynamics_model, obs, action):
        """Test forward pass without providing hidden state."""
        # Run forward pass without hidden
        obs_hat, next_hidden = dynamics_model(obs, action)

        # Check output types and shapes
        assert obs_hat.shape == (
            self.BATCH_SIZE,
            self.SEQ_LEN,
            self.OBS_NUM_TOKENS,
            self.OBS_DIM,
        )
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM,
        )

    def test_forward_with_no_len_no_hidden(self, dynamics_model):
        """Test forward_with_no_len without providing hidden state."""
        # Create inputs without sequence length dimension
        obs_no_len = torch.randn(self.BATCH_SIZE, self.OBS_NUM_TOKENS, self.OBS_DIM)
        action_no_len = torch.stack(
            [
                torch.randint(0, choice, (self.BATCH_SIZE,))
                for choice in self.ACTION_CHOICES
            ],
            dim=-1,
        )

        # Run forward pass without hidden
        obs_hat, next_hidden = dynamics_model.forward_with_no_len(
            obs_no_len, action_no_len
        )

        # Check output shapes
        assert obs_hat.shape == (self.BATCH_SIZE, self.OBS_NUM_TOKENS, self.OBS_DIM)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.DIM,
        )
