import pytest
import torch
import torch.nn as nn

from exp.models.components.multi_discretes import MultiCategoricals
from exp.models.components.qlstm import QLSTM
from exp.models.latent_policy_new import Encoder, Generator, LatentPiVFramework
from exp.models.utils import ObsInfo
from tests.helpers import parametrize_device


class TestEncoder:
    """Test suite for the Encoder module."""

    @pytest.fixture
    def obs_info(self):
        return ObsInfo(dim=8, dim_hidden=10, num_tokens=4)

    @pytest.fixture
    def core_model(self):
        return QLSTM(depth=2, dim=16, dim_ff_hidden=32, dropout=0.1)

    @parametrize_device
    def test_encoder_with_obs_info(self, obs_info, core_model, device):
        """Test encoder with ObsInfo configuration."""
        encoder = Encoder(
            obs_info=obs_info,
            core_model_dim=16,
            core_model=core_model,
            out_dim=12,
        ).to(device)

        batch_size, seq_len = 2, 3
        obs = torch.randn(
            batch_size, seq_len, obs_info.num_tokens, obs_info.dim, device=device
        )

        output, hidden = encoder(obs)

        assert output.shape == (batch_size, seq_len, 12)
        assert hidden.shape == (batch_size, 2, seq_len, 16)  # depth=2, dim=16

    @parametrize_device
    def test_encoder_with_int_dimension(self, core_model, device):
        """Test encoder with direct integer dimension."""
        encoder = Encoder(
            obs_info=10,
            core_model_dim=16,
            core_model=core_model,
        ).to(device)

        batch_size, seq_len = 2, 3
        obs = torch.randn(batch_size, seq_len, 10, device=device)

        output, hidden = encoder(obs)

        assert output.shape == (batch_size, seq_len, 16)
        assert hidden.shape == (batch_size, 2, seq_len, 16)  # depth=2, dim=16

    @parametrize_device
    def test_encoder_without_out_dim(self, obs_info, core_model, device):
        """Test encoder without explicit output dimension."""
        encoder = Encoder(
            obs_info=obs_info,
            core_model_dim=16,
            core_model=core_model,
        ).to(device)

        batch_size, seq_len = 2, 3
        obs = torch.randn(
            batch_size, seq_len, obs_info.num_tokens, obs_info.dim, device=device
        )

        output, hidden = encoder(obs)

        assert output.shape == (batch_size, seq_len, 16)  # Uses core_model_dim
        assert hidden.shape == (batch_size, 2, seq_len, 16)

    @parametrize_device
    def test_encoder_with_hidden_state(self, obs_info, core_model, device):
        """Test encoder with provided hidden state."""
        encoder = Encoder(
            obs_info=obs_info,
            core_model_dim=16,
            core_model=core_model,
        ).to(device)

        batch_size, seq_len = 2, 3
        obs = torch.randn(
            batch_size, seq_len, obs_info.num_tokens, obs_info.dim, device=device
        )
        hidden = torch.randn(batch_size, 2, 16, device=device)  # Initial hidden state

        output, next_hidden = encoder(obs, hidden)

        assert output.shape == (batch_size, seq_len, 16)
        assert next_hidden.shape == (batch_size, 2, seq_len, 16)


class TestGenerator:
    """Test suite for the Generator module."""

    @pytest.fixture
    def action_choices(self):
        return [2, 3, 4]  # Multi-discrete action space

    @parametrize_device
    def test_generator_basic(self, action_choices, device):
        """Test basic generator functionality."""
        generator = Generator(
            latent_dim=12,
            action_choices=action_choices,
        ).to(device)

        batch_size, seq_len = 2, 3
        latent = torch.randn(batch_size, seq_len, 12, device=device)

        policy_dist, value = generator(latent)

        # Check policy distribution
        assert isinstance(policy_dist, MultiCategoricals)
        assert len(policy_dist.dists) == len(action_choices)
        for i, num_choices in enumerate(action_choices):
            assert policy_dist.dists[i].logits.shape == (
                batch_size,
                seq_len,
                num_choices,
            )

        # Check value output
        assert value.shape == (batch_size, seq_len)

    @parametrize_device
    def test_generator_with_core_model(self, action_choices, device):
        """Test generator with core model."""
        generator = Generator(
            latent_dim=12,
            action_choices=action_choices,
            core_model=nn.Linear(16, 16),
            core_model_dim=16,
        ).to(device)

        batch_size, seq_len = 2, 3
        latent = torch.randn(batch_size, seq_len, 12, device=device)

        policy_dist, value = generator(latent)

        # Check policy distribution
        assert isinstance(policy_dist, MultiCategoricals)
        assert len(policy_dist.dists) == len(action_choices)
        for i, num_choices in enumerate(action_choices):
            assert policy_dist.dists[i].logits.shape == (
                batch_size,
                seq_len,
                num_choices,
            )

        # Check value output
        assert value.shape == (batch_size, seq_len)

    @parametrize_device
    def test_generator_single_step(self, action_choices, device):
        """Test generator with single-step input."""
        generator = Generator(
            latent_dim=12,
            action_choices=action_choices,
        ).to(device)

        batch_size = 2
        latent = torch.randn(batch_size, 12, device=device)

        policy_dist, value = generator(latent)

        # Check policy distribution
        assert isinstance(policy_dist, MultiCategoricals)
        assert len(policy_dist.dists) == len(action_choices)
        for i, num_choices in enumerate(action_choices):
            assert policy_dist.dists[i].logits.shape == (batch_size, num_choices)

        # Check value output
        assert value.shape == (batch_size,)

    @parametrize_device
    def test_generator_action_sampling(self, action_choices, device):
        """Test that generator can sample valid actions."""
        generator = Generator(
            latent_dim=12,
            action_choices=action_choices,
        ).to(device)

        batch_size, seq_len = 2, 3
        latent = torch.randn(batch_size, seq_len, 12, device=device)

        policy_dist, _ = generator(latent)

        # Check policy distribution type
        assert isinstance(policy_dist, MultiCategoricals)

        # Sample actions
        actions = policy_dist.sample()
        assert actions.shape == (batch_size, seq_len, len(action_choices))

        # Check that sampled actions are within valid range
        for i, num_choices in enumerate(action_choices):
            assert (actions[:, :, i] >= 0).all()
            assert (actions[:, :, i] < num_choices).all()


class TestLatentPiVFramework:
    """Test suite for the LatentPiVFramework."""

    @pytest.fixture
    def obs_info(self):
        return ObsInfo(dim=8, dim_hidden=10, num_tokens=4)

    @pytest.fixture
    def action_choices(self):
        return [2, 3]

    @pytest.fixture
    def encoder(self, obs_info):
        return Encoder(
            obs_info=obs_info,
            core_model_dim=16,
            core_model=QLSTM(depth=2, dim=16, dim_ff_hidden=32, dropout=0.1),
            out_dim=12,
        )

    @pytest.fixture
    def generator(self, action_choices):
        return Generator(
            latent_dim=12,
            action_choices=action_choices,
        )

    @parametrize_device
    def test_framework_forward(
        self, encoder, generator, obs_info, action_choices, device
    ):
        """Test forward pass through the complete framework."""
        framework = LatentPiVFramework(
            encoder=encoder.to(device),
            generator=generator.to(device),
        )

        batch_size, seq_len = 2, 3
        obs = torch.randn(
            batch_size, seq_len, obs_info.num_tokens, obs_info.dim, device=device
        )

        policy_dist, value, hidden = framework(obs)

        # Check policy distribution
        assert isinstance(policy_dist, MultiCategoricals)
        assert len(policy_dist.dists) == len(action_choices)
        for i, num_choices in enumerate(action_choices):
            assert policy_dist.dists[i].logits.shape == (
                batch_size,
                seq_len,
                num_choices,
            )

        # Check value output
        assert value.shape == (batch_size, seq_len)

        # Check hidden state
        assert hidden.shape == (batch_size, 2, seq_len, 16)  # depth=2, dim=16

    @parametrize_device
    def test_framework_with_hidden_state(
        self, encoder, generator, obs_info, action_choices, device
    ):
        """Test framework with provided hidden state."""
        framework = LatentPiVFramework(
            encoder=encoder.to(device),
            generator=generator.to(device),
        )

        batch_size, seq_len = 2, 3
        obs = torch.randn(
            batch_size, seq_len, obs_info.num_tokens, obs_info.dim, device=device
        )
        hidden = torch.randn(
            batch_size,
            2,
            16,
            device=device,  # Initial hidden state
        )

        policy_dist, value, next_hidden = framework(obs, hidden)

        # Check outputs
        assert isinstance(policy_dist, MultiCategoricals)
        assert len(policy_dist.dists) == len(action_choices)
        assert value.shape == (batch_size, seq_len)
        assert next_hidden.shape == (batch_size, 2, seq_len, 16)
