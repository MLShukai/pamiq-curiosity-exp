import pytest
import torch
import torch.nn as nn
from torch.distributions import Normal

from exp.models.components.stacked_hidden_state import StackedHiddenState
from exp.models.latent_policy import (
    FCDeterministicNormalHead,
    LatentPolicy,
    ObsUpperActionFlattenHead,
)
from exp.models.utils import ActionInfo, ObsInfo


class TestLatentPolicy:
    """Test suite for the LatentPolicy model components."""

    # Test hyperparameters
    BATCH_SIZE = 2
    SEQ_LEN = 3
    DEPTH = 2
    DIM_CORE_MODEL = 16
    DIM_EMBED = 12
    OBS_DIM = 8
    OBS_DIM_HIDDEN = 10
    OBS_NUM_TOKENS = 4
    UPPER_ACTION_DIM = 12
    ACTION_DIM = 6
    ACTION_CHOICES = [2, 3, 4]

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
    def mock_encoder(self, mocker):
        """Mock StackedHiddenState for isolated testing."""
        mock = mocker.Mock(spec=StackedHiddenState)
        # Set up the mock to return predictable outputs
        output_tensor = torch.randn(self.BATCH_SIZE, self.SEQ_LEN, self.DIM_CORE_MODEL)
        hidden_tensor = torch.randn(
            self.BATCH_SIZE, self.DEPTH, self.SEQ_LEN, self.DIM_CORE_MODEL
        )
        output_tensor_no_len = torch.randn(self.BATCH_SIZE, self.DIM_CORE_MODEL)
        hidden_tensor_no_len = torch.randn(
            self.BATCH_SIZE, self.DEPTH, self.DIM_CORE_MODEL
        )

        # Configure return values based on no_len parameter
        def side_effect(x, hidden=None, *, no_len=False):
            if no_len:
                return output_tensor_no_len, hidden_tensor_no_len
            else:
                return output_tensor, hidden_tensor

        mock.side_effect = side_effect
        mock.forward_with_no_len.return_value = (
            output_tensor_no_len,
            hidden_tensor_no_len,
        )
        return mock

    @pytest.fixture
    def mock_predictor(self, mocker):
        """Mock StackedHiddenState for isolated testing."""
        mock = mocker.Mock(spec=StackedHiddenState)
        # Set up the mock to return predictable outputs
        output_tensor = torch.randn(self.BATCH_SIZE, self.SEQ_LEN, self.DIM_CORE_MODEL)
        hidden_tensor = torch.randn(
            self.BATCH_SIZE, self.DEPTH, self.SEQ_LEN, self.DIM_CORE_MODEL
        )
        output_tensor_no_len = torch.randn(self.BATCH_SIZE, self.DIM_CORE_MODEL)
        hidden_tensor_no_len = torch.randn(
            self.BATCH_SIZE, self.DEPTH, self.DIM_CORE_MODEL
        )

        # Configure return values based on no_len parameter
        def side_effect(x, hidden=None, *, no_len=False):
            if no_len:
                return output_tensor_no_len, hidden_tensor_no_len
            else:
                return output_tensor, hidden_tensor

        mock.side_effect = side_effect
        mock.forward_with_no_len.return_value = (
            output_tensor_no_len,
            hidden_tensor_no_len,
        )
        return mock

    @pytest.fixture
    def obs_upper_action_flatten_head(self, obs_info):
        return ObsUpperActionFlattenHead(
            obs_info=obs_info,
            action_dim=self.UPPER_ACTION_DIM,
            output_dim=self.DIM_CORE_MODEL,
        )

    @pytest.fixture
    def action_dist_head(self):
        return FCDeterministicNormalHead(
            dim_in=self.DIM_CORE_MODEL,
            dim_out=self.ACTION_DIM,
        )

    @pytest.fixture
    def value_head(self):
        return nn.Linear(
            in_features=self.DIM_CORE_MODEL,
            out_features=1,  # Single value output
        )

    @pytest.fixture
    def latent_policy(
        self,
        obs_upper_action_flatten_head,
        mock_encoder,
        mock_predictor,
        action_dist_head,
        value_head,
    ):
        """Fixture for LatentPolicy model."""
        return LatentPolicy(
            obs_upper_action_flatten_head=obs_upper_action_flatten_head,
            encoder=mock_encoder,
            predictor=mock_predictor,
            action_dist_head=action_dist_head,
            value_head=value_head,
        )

    @pytest.fixture
    def obs_tensor(self):
        """Observation tensor for testing."""
        return torch.randn(
            self.BATCH_SIZE, self.SEQ_LEN, self.OBS_NUM_TOKENS, self.OBS_DIM
        )

    @pytest.fixture
    def upper_action_tensor(self):
        """Action tensor for testing."""
        return torch.randn(self.BATCH_SIZE, self.SEQ_LEN, self.UPPER_ACTION_DIM)

    @pytest.fixture
    def hidden_encoder_tensor(self):
        """Hidden state tensor for testing."""
        return torch.randn(
            self.BATCH_SIZE, self.DEPTH, self.SEQ_LEN, self.DIM_CORE_MODEL
        )

    @pytest.fixture
    def hidden_predictor_tensor(self):
        """Hidden state tensor for testing."""
        return torch.randn(
            self.BATCH_SIZE, self.DEPTH, self.SEQ_LEN, self.DIM_CORE_MODEL
        )

    def test_forward(
        self,
        latent_policy,
        obs_tensor,
        upper_action_tensor,
        hidden_encoder_tensor,
        hidden_predictor_tensor,
    ):
        """Test the forward pass of LatentPolicy."""
        action_dist, value, latent, hidden_predictor, hidden_encoder = latent_policy(
            obs=obs_tensor,
            upper_action=upper_action_tensor,
            hidden_encoder=hidden_encoder_tensor,
            hidden_predictor=hidden_predictor_tensor,
        )
        assert isinstance(
            action_dist, Normal
        )  # Check if action_dist is a Normal distribution
        assert action_dist.mean.shape == (
            self.BATCH_SIZE,
            self.SEQ_LEN,
            self.ACTION_DIM,
        )
        assert value.shape == (self.BATCH_SIZE, self.SEQ_LEN, 1)
        assert latent.shape == (self.BATCH_SIZE, self.SEQ_LEN, self.DIM_CORE_MODEL)
        assert hidden_predictor.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM_CORE_MODEL,
        )
        assert hidden_encoder.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM_CORE_MODEL,
        )

    def test_forward_with_no_len(
        self,
        latent_policy,
        obs_tensor,
        upper_action_tensor,
        hidden_encoder_tensor,
        hidden_predictor_tensor,
    ):
        """Test the forward pass of LatentPolicy with no_len."""
        action_dist, value, latent, hidden_predictor, hidden_encoder = (
            latent_policy.forward_with_no_len(
                obs=obs_tensor[:, 0],  # Use only the first time step for no_len
                upper_action=upper_action_tensor[:, 0],
                hidden_encoder=hidden_encoder_tensor[:, 0],
                hidden_predictor=hidden_predictor_tensor[:, 0],
            )
        )
        assert isinstance(action_dist, Normal)
        assert action_dist.mean.shape == (self.BATCH_SIZE, self.ACTION_DIM)
        assert value.shape == (self.BATCH_SIZE, 1)
        assert latent.shape == (self.BATCH_SIZE, self.DIM_CORE_MODEL)
        assert hidden_predictor.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.DIM_CORE_MODEL,
        )
        assert hidden_encoder.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.DIM_CORE_MODEL,
        )
