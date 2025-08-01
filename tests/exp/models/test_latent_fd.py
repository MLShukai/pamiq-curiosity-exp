import pytest
import torch
import torch.nn as nn
from torch.distributions import Normal

from exp.models.components.stacked_hidden_state import StackedHiddenState
from exp.models.latent_fd import (
    Encoder,
    LatentFD,
    ObsActionFlattenHead,
    ObsPredictionHead,
    Predictor,
)
from exp.models.utils import ActionInfo, ObsInfo


class TestLatentFD:
    """Test suite for the LatentFD model components."""

    # Test hyperparameters
    BATCH_SIZE = 2
    SEQ_LEN = 3
    DEPTH = 2
    DIM_CORE_MODEL = 16
    DIM_EMBED = 12
    OBS_DIM = 8
    OBS_DIM_HIDDEN = 10
    OBS_NUM_TOKENS = 4
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
    def obs_action_flatten_head(self, obs_info, action_info):
        return ObsActionFlattenHead(
            obs_info=obs_info,
            action_info=action_info,
            output_dim=self.DIM_CORE_MODEL,
        )

    @pytest.fixture
    def obs_prediction_head(self, obs_info):
        return ObsPredictionHead(
            input_dim=self.DIM_CORE_MODEL,
            obs_info=obs_info,
        )

    @pytest.fixture
    def latent_fd(
        self, obs_action_flatten_head, obs_prediction_head, mock_encoder, mock_predictor
    ):
        """LatentFD model initialized with ObsInfo and ActionInfo."""
        return LatentFD(
            obs_action_flatten_head=obs_action_flatten_head,
            obs_predict_head=obs_prediction_head,
            encoder=mock_encoder,
            predictor=mock_predictor,
        )

    @pytest.fixture
    def obs_tensor(self):
        """Observation tensor for testing."""
        return torch.randn(
            self.BATCH_SIZE, self.SEQ_LEN, self.OBS_NUM_TOKENS, self.OBS_DIM
        )

    @pytest.fixture
    def action_tensor(self):
        """Action tensor for testing."""
        actions = []
        for choice in self.ACTION_CHOICES:
            actions.append(torch.randint(0, choice, (self.BATCH_SIZE, self.SEQ_LEN)))
        return torch.stack(actions, dim=-1)

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
        latent_fd,
        obs_tensor,
        action_tensor,
        hidden_encoder_tensor,
        hidden_predictor_tensor,
        mock_encoder,
        mock_predictor,
    ):
        """Test forward pass with all inputs."""
        obs_dist, latent, next_hidden_encoder, next_hidden_predictor = latent_fd(
            obs_tensor, action_tensor, hidden_encoder_tensor, hidden_predictor_tensor
        )
        # Check output types
        assert isinstance(obs_dist, Normal)
        assert isinstance(latent, torch.Tensor)
        assert isinstance(next_hidden_encoder, torch.Tensor)
        assert isinstance(next_hidden_predictor, torch.Tensor)

        # Check shapes
        assert obs_dist.mean.shape == (
            self.BATCH_SIZE,
            self.SEQ_LEN,
            self.OBS_NUM_TOKENS,
            self.OBS_DIM,
        )
        assert latent.shape == (self.BATCH_SIZE, self.SEQ_LEN, self.DIM_CORE_MODEL)
        assert next_hidden_encoder.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM_CORE_MODEL,
        )
        assert next_hidden_predictor.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM_CORE_MODEL,
        )

        # Verify encoder and predictor were called
        mock_encoder.assert_called_once()
        mock_predictor.assert_called_once()


class TestEncoder:
    """Test suite for the Encoder class."""

    # Test hyperparameters
    BATCH_SIZE = 2
    SEQ_LEN = 3
    DEPTH = 2
    DIM_CORE_MODEL = 16
    DIM_EMBED = 12
    OBS_DIM = 8
    OBS_DIM_HIDDEN = 10
    OBS_NUM_TOKENS = 4
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
    def mock_core_model(self, mocker):
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
    def encoder_with_info(self, obs_info, action_info, mock_core_model):
        """Encoder initialized with ObsInfo and ActionInfo."""
        return Encoder(
            obs_info=obs_info,
            action_info=action_info,
            core_model=mock_core_model,
            core_model_dim=self.DIM_CORE_MODEL,
            embed_dim=self.DIM_EMBED,
        )

    @pytest.fixture
    def encoder_with_int(self, mock_core_model):
        """Encoder initialized with integer dimensions."""
        return Encoder(
            obs_info=self.OBS_DIM,
            action_info=self.ACTION_DIM,
            core_model=mock_core_model,
            core_model_dim=self.DIM_CORE_MODEL,
            embed_dim=None,  # No output projection
        )

    @pytest.fixture
    def obs_tensor(self):
        """Observation tensor for testing."""
        return torch.randn(
            self.BATCH_SIZE, self.SEQ_LEN, self.OBS_NUM_TOKENS, self.OBS_DIM
        )

    @pytest.fixture
    def action_tensor(self):
        """Action tensor for testing."""
        actions = []
        for choice in self.ACTION_CHOICES:
            actions.append(torch.randint(0, choice, (self.BATCH_SIZE, self.SEQ_LEN)))
        return torch.stack(actions, dim=-1)

    @pytest.fixture
    def hidden_tensor(self):
        """Hidden state tensor for testing."""
        return torch.randn(self.BATCH_SIZE, self.DEPTH, self.DIM_CORE_MODEL)

    def test_forward(
        self,
        encoder_with_info,
        obs_tensor,
        action_tensor,
        hidden_tensor,
        mock_core_model,
    ):
        """Test forward pass with all inputs."""
        output, next_hidden = encoder_with_info(
            obs_tensor, action_tensor, hidden_tensor
        )

        # Check output shapes
        assert output.shape == (self.BATCH_SIZE, self.SEQ_LEN, self.DIM_EMBED)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM_CORE_MODEL,
        )

        # Verify core model was called
        mock_core_model.assert_called_once()

    def test_forward_no_hidden(
        self, encoder_with_info, obs_tensor, action_tensor, mock_core_model
    ):
        """Test forward pass without hidden state."""
        output, next_hidden = encoder_with_info(obs_tensor, action_tensor)

        # Check output shapes
        assert output.shape == (self.BATCH_SIZE, self.SEQ_LEN, self.DIM_EMBED)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM_CORE_MODEL,
        )

        # Verify None was passed as hidden
        args = mock_core_model.call_args[0]
        assert args[1] is None  # Second argument is hidden

    def test_forward_with_no_len(
        self,
        encoder_with_info,
        obs_tensor,
        action_tensor,
        hidden_tensor,
        mock_core_model,
    ):
        """Test forward_with_no_len for single-step inference."""
        # Remove sequence dimension
        obs_no_len = obs_tensor[:, 0, :, :]
        action_no_len = action_tensor[:, 0, :]

        output, next_hidden = encoder_with_info.forward_with_no_len(
            obs_no_len, action_no_len, hidden_tensor
        )

        # Check output shapes (no sequence dimension)
        assert output.shape == (self.BATCH_SIZE, self.DIM_EMBED)
        assert next_hidden.shape == (self.BATCH_SIZE, self.DEPTH, self.DIM_CORE_MODEL)

        # Verify core model was called with no_len=True
        mock_core_model.assert_called_once()
        # Check that no_len=True was passed
        args, kwargs = mock_core_model.call_args
        assert kwargs.get("no_len") is True

    def test_output_projection(self, obs_info, action_info, mock_core_model):
        """Test output projection when embed_dim differs from
        core_model_dim."""
        # Create encoder with different embed dimension
        encoder = Encoder(
            obs_info=obs_info,
            action_info=action_info,
            core_model=mock_core_model,
            core_model_dim=self.DIM_CORE_MODEL,
            embed_dim=24,  # Different from core_model_dim
        )

        obs = torch.randn(
            self.BATCH_SIZE, self.SEQ_LEN, self.OBS_NUM_TOKENS, self.OBS_DIM
        )
        actions = []
        for choice in self.ACTION_CHOICES:
            actions.append(torch.randint(0, choice, (self.BATCH_SIZE, self.SEQ_LEN)))
        action = torch.stack(actions, dim=-1)

        output, _ = encoder(obs, action)
        assert output.shape == (self.BATCH_SIZE, self.SEQ_LEN, 24)


class TestPredictor:
    """Test suite for the Predictor class."""

    # Test hyperparameters
    BATCH_SIZE = 2
    SEQ_LEN = 3
    DEPTH = 2
    DIM_CORE_MODEL = 16
    DIM_EMBED = 20
    OBS_DIM = 8
    OBS_DIM_HIDDEN = 10
    OBS_NUM_TOKENS = 4

    @pytest.fixture
    def obs_info(self):
        return ObsInfo(
            dim=self.OBS_DIM,
            dim_hidden=self.OBS_DIM_HIDDEN,
            num_tokens=self.OBS_NUM_TOKENS,
        )

    @pytest.fixture
    def mock_core_model(self, mocker):
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
    def predictor_with_obsinfo(self, obs_info, mock_core_model):
        """Predictor initialized with ObsInfo."""
        return Predictor(
            obs_info=obs_info,
            core_model=mock_core_model,
            core_model_dim=self.DIM_CORE_MODEL,
            embed_dim=self.DIM_EMBED,
        )

    @pytest.fixture
    def predictor_with_int(self, mock_core_model):
        """Predictor initialized with integer dimension."""
        return Predictor(
            obs_info=self.OBS_DIM,
            core_model=mock_core_model,
            core_model_dim=self.DIM_CORE_MODEL,
            embed_dim=None,
        )

    @pytest.fixture
    def latent_tensor(self):
        """Latent representation tensor for testing."""
        return torch.randn(self.BATCH_SIZE, self.SEQ_LEN, self.DIM_EMBED)

    @pytest.fixture
    def hidden_tensor(self):
        """Hidden state tensor for testing."""
        return torch.randn(self.BATCH_SIZE, self.DEPTH, self.DIM_CORE_MODEL)

    def test_forward(
        self, predictor_with_obsinfo, latent_tensor, hidden_tensor, mock_core_model
    ):
        """Test forward pass."""
        obs_dist, next_hidden = predictor_with_obsinfo(latent_tensor, hidden_tensor)

        # Check output types
        assert isinstance(obs_dist, Normal)  # FCDeterministicNormalHead returns Normal
        assert isinstance(next_hidden, torch.Tensor)

        # Check shapes
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM_CORE_MODEL,
        )

        # Verify core model was called
        mock_core_model.assert_called_once()

    def test_forward_no_hidden(
        self, predictor_with_obsinfo, latent_tensor, mock_core_model
    ):
        """Test forward pass without hidden state."""
        obs_dist, next_hidden = predictor_with_obsinfo(latent_tensor)

        # Check output types
        assert isinstance(obs_dist, Normal)
        assert isinstance(next_hidden, torch.Tensor)

        # Verify None was passed as hidden
        args = mock_core_model.call_args[0]
        assert args[1] is None

    def test_forward_with_no_len(
        self, predictor_with_obsinfo, latent_tensor, hidden_tensor, mock_core_model
    ):
        """Test forward_with_no_len for single-step inference."""
        # Remove sequence dimension
        latent_no_len = latent_tensor[:, 0, :]

        obs_dist, next_hidden = predictor_with_obsinfo.forward_with_no_len(
            latent_no_len, hidden_tensor
        )

        # Check output types
        assert isinstance(obs_dist, Normal)
        assert next_hidden.shape == (self.BATCH_SIZE, self.DEPTH, self.DIM_CORE_MODEL)

        # Verify core model was called with no_len=True
        mock_core_model.assert_called_once()
        # Check that no_len=True was passed
        args, kwargs = mock_core_model.call_args
        assert kwargs.get("no_len") is True

    def test_input_projection(self, obs_info, mock_core_model):
        """Test input projection when embed_dim is provided."""
        # Create predictor with input projection
        predictor = Predictor(
            obs_info=obs_info,
            core_model=mock_core_model,
            core_model_dim=self.DIM_CORE_MODEL,
            embed_dim=32,  # Different from core_model_dim
        )

        assert predictor.input_proj is not None
        assert isinstance(predictor.input_proj, nn.Linear)

        # Test that projection is applied
        latent = torch.randn(self.BATCH_SIZE, self.SEQ_LEN, 32)
        obs_dist, _ = predictor(latent)

        # Verify the distribution was created
        assert isinstance(obs_dist, Normal)
