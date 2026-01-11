import pytest
import torch
import torch.nn as nn

from exp.models.components.qlstm import QLSTM
from exp.models.latent_fd import Encoder, LatentFDFramework, Predictor
from exp.models.utils import ActionInfo, ObsInfo
from tests.helpers import parametrize_device


class TestEncoder:
    """Test suite for the Encoder module."""

    @pytest.fixture
    def obs_info(self):
        return ObsInfo(dim=8, dim_hidden=10, num_tokens=4)

    @pytest.fixture
    def action_info(self):
        return ActionInfo(choices=[2, 3, 4], dim=6)

    @pytest.fixture
    def core_model(self):
        return QLSTM(depth=2, dim=16, dim_ff_hidden=32, dropout=0.1)

    @parametrize_device
    def test_encoder_with_obs_action_info(
        self, obs_info, action_info, core_model, device
    ):
        """Test encoder with ObsInfo and ActionInfo configurations."""
        encoder = Encoder(
            obs_info=obs_info,
            action_info=action_info,
            core_model_dim=16,
            core_model=core_model,
            out_dim=12,
        ).to(device)

        batch_size, seq_len = 2, 3
        obs = torch.randn(
            batch_size, seq_len, obs_info.num_tokens, obs_info.dim, device=device
        )
        action = torch.randint(
            0, 2, (batch_size, seq_len, len(action_info.choices)), device=device
        )

        output, hidden = encoder(obs, action)

        assert output.shape == (batch_size, seq_len, 12)
        assert hidden.shape == (batch_size, 2, seq_len, 16)  # depth=2, dim=16

    @parametrize_device
    def test_encoder_with_int_dimensions(self, core_model, device):
        """Test encoder with direct integer dimensions."""
        encoder = Encoder(
            obs_info=10,
            action_info=5,
            core_model_dim=16,
            core_model=core_model,
        ).to(device)

        batch_size, seq_len = 2, 3
        obs = torch.randn(batch_size, seq_len, 10, device=device)
        action = torch.randn(batch_size, seq_len, 5, device=device)

        output, hidden = encoder(obs, action)

        assert output.shape == (batch_size, seq_len, 16)
        assert hidden.shape == (batch_size, 2, seq_len, 16)  # depth=2, dim=16

    @parametrize_device
    def test_encoder_no_len(self, obs_info, action_info, core_model, device):
        """Test encoder with no_len=True for single-step processing."""
        encoder = Encoder(
            obs_info=obs_info,
            action_info=action_info,
            core_model_dim=16,
            core_model=core_model,
        ).to(device)

        batch_size = 2
        obs = torch.randn(batch_size, obs_info.num_tokens, obs_info.dim, device=device)
        action = torch.randint(
            0, 2, (batch_size, len(action_info.choices)), device=device
        )

        output, hidden = encoder(obs, action, no_len=True)

        assert output.shape == (batch_size, 16)
        assert hidden.shape == (batch_size, 2, 16)  # depth=2, dim=16


class TestPredictor:
    """Test suite for the Predictor module."""

    @pytest.fixture
    def obs_info(self):
        return ObsInfo(dim=8, dim_hidden=10, num_tokens=4)

    @parametrize_device
    def test_predictor_with_obs_info(self, obs_info, device):
        """Test predictor with ObsInfo configuration."""
        predictor = Predictor(
            latent_dim=12,
            obs_info=obs_info,
            core_model=nn.Linear(16, 16),
            core_model_dim=16,
        ).to(device)

        batch_size, seq_len = 2, 3
        latent = torch.randn(batch_size, seq_len, 12, device=device)

        output = predictor(latent)

        assert output.shape == (batch_size, seq_len, obs_info.num_tokens, obs_info.dim)

    @parametrize_device
    def test_predictor_with_int_dimension(self, device):
        """Test predictor with direct integer dimension."""
        predictor = Predictor(
            latent_dim=12,
            obs_info=10,
        ).to(device)

        batch_size, seq_len = 2, 3
        latent = torch.randn(batch_size, seq_len, 12, device=device)

        output = predictor(latent)

        assert output.shape == (batch_size, seq_len, 10)

    @parametrize_device
    def test_predictor_without_core_model(self, device):
        """Test predictor without core model (identity transformation)."""
        predictor = Predictor(
            latent_dim=12,
            obs_info=10,
        ).to(device)

        batch_size = 2
        latent = torch.randn(batch_size, 12, device=device)

        output = predictor(latent)

        assert output.shape == (batch_size, 10)


class TestLatentFDFramework:
    """Test suite for the LatentFDFramework."""

    @pytest.fixture
    def obs_info(self):
        return ObsInfo(dim=8, dim_hidden=10, num_tokens=4)

    @pytest.fixture
    def action_info(self):
        return ActionInfo(choices=[2, 3], dim=6)

    @pytest.fixture
    def encoder(self, obs_info, action_info):
        return Encoder(
            obs_info=obs_info,
            action_info=action_info,
            core_model_dim=16,
            core_model=QLSTM(depth=2, dim=16, dim_ff_hidden=32, dropout=0.1),
            out_dim=12,
        )

    @pytest.fixture
    def predictor(self, obs_info):
        return Predictor(
            latent_dim=12,
            obs_info=obs_info,
        )

    @parametrize_device
    def test_framework_forward(self, encoder, predictor, obs_info, action_info, device):
        """Test forward pass through the complete framework."""
        framework = LatentFDFramework(
            encoder=encoder.to(device),
            predictor=predictor.to(device),
        )

        batch_size, seq_len = 2, 3
        obs = torch.randn(
            batch_size, seq_len, obs_info.num_tokens, obs_info.dim, device=device
        )
        action = torch.randint(
            0, 2, (batch_size, seq_len, len(action_info.choices)), device=device
        )

        output, hidden = framework(obs, action)

        assert output.shape == (batch_size, seq_len, obs_info.num_tokens, obs_info.dim)
        assert hidden.shape == (batch_size, 2, seq_len, 16)  # depth=2, dim=16

    @parametrize_device
    def test_framework_with_hidden_state(
        self, encoder, predictor, obs_info, action_info, device
    ):
        """Test framework with provided hidden state."""
        framework = LatentFDFramework(
            encoder=encoder.to(device),
            predictor=predictor.to(device),
        )

        batch_size, seq_len = 2, 3
        obs = torch.randn(
            batch_size, seq_len, obs_info.num_tokens, obs_info.dim, device=device
        )
        action = torch.randint(
            0, 2, (batch_size, seq_len, len(action_info.choices)), device=device
        )
        hidden = torch.randn(
            batch_size,
            2,
            16,
            device=device,  # depth=2, dim=16 (no seq_len for QLSTM hidden state)
        )

        output, next_hidden = framework(obs, action, hidden)

        assert output.shape == (batch_size, seq_len, obs_info.num_tokens, obs_info.dim)
        assert next_hidden.shape == (
            batch_size,
            2,
            seq_len,
            16,
        )  # QLSTM outputs include seq_len

    @parametrize_device
    def test_framework_gradient_flow(
        self, encoder, predictor, obs_info, action_info, device
    ):
        """Test that gradients flow through the framework."""
        framework = LatentFDFramework(
            encoder=encoder.to(device),
            predictor=predictor.to(device),
        )

        batch_size, seq_len = 2, 3
        obs = torch.randn(
            batch_size,
            seq_len,
            obs_info.num_tokens,
            obs_info.dim,
            device=device,
            requires_grad=True,
        )
        action = torch.randint(
            0, 2, (batch_size, seq_len, len(action_info.choices)), device=device
        )

        output, _ = framework(obs, action)
        loss = output.mean()
        loss.backward()

        assert obs.grad is not None
        assert obs.grad.shape == obs.shape
