import pytest
import torch
from torch.distributions import Distribution

from exp.models.components.qlstm import QLSTM
from exp.models.policy import (
    StackedHiddenContinuousPiVLatent,
    StackedHiddenPiV,
    StackedHiddenPiVLatent,
)
from exp.models.utils import ObsInfo


class TestStackedHiddenPiV:
    # Test hyperparameters
    BATCH_SIZE = 2
    SEQ_LEN = 3
    DEPTH = 2
    DIM = 8
    OBS_DIM = 16
    OBS_DIM_HIDDEN = 12
    OBS_NUM_TOKENS = 4
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
    def core_model(self):
        return QLSTM(
            depth=self.DEPTH,
            dim=self.DIM,
            dim_ff_hidden=self.DIM_FF_HIDDEN,
            dropout=0.0,
        )

    @pytest.fixture
    def policy_value_model(self, obs_info, core_model):
        return StackedHiddenPiV(
            obs_info=obs_info,
            action_choices=self.ACTION_CHOICES,
            dim=self.DIM,
            core_model=core_model,
        )

    @pytest.fixture
    def observation(self):
        return torch.randn(
            self.BATCH_SIZE, self.SEQ_LEN, self.OBS_NUM_TOKENS, self.OBS_DIM
        )

    @pytest.fixture
    def hidden(self):
        return torch.randn(self.BATCH_SIZE, self.DEPTH, self.DIM)

    def test_forward(self, policy_value_model, observation, hidden):
        """Test forward pass of StackedHiddenPiV model."""
        # Run forward pass
        policy_dist, value, next_hidden = policy_value_model(observation, hidden)

        # Check output types
        assert isinstance(policy_dist, Distribution)
        assert isinstance(value, torch.Tensor)
        assert isinstance(next_hidden, torch.Tensor)

        # Check output shapes
        sample_action = policy_dist.sample()
        assert sample_action.shape == (
            self.BATCH_SIZE,
            self.SEQ_LEN,
            len(self.ACTION_CHOICES),
        )
        assert value.shape == (self.BATCH_SIZE, self.SEQ_LEN)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM,
        )

        # Check distribution properties
        log_prob = policy_dist.log_prob(sample_action)
        assert log_prob.shape == (
            self.BATCH_SIZE,
            self.SEQ_LEN,
            len(self.ACTION_CHOICES),
        )

        entropy = policy_dist.entropy()
        assert entropy.shape == (
            self.BATCH_SIZE,
            self.SEQ_LEN,
            len(self.ACTION_CHOICES),
        )

    def test_single_batch(self, policy_value_model, observation, hidden):
        """Test with single batch size."""
        single_obs = observation[:1]
        single_hidden = hidden[:1]

        policy_dist, value, next_hidden = policy_value_model(single_obs, single_hidden)

        sample_action = policy_dist.sample()
        assert sample_action.shape == (1, self.SEQ_LEN, len(self.ACTION_CHOICES))
        assert value.shape == (1, self.SEQ_LEN)
        assert next_hidden.shape == (1, self.DEPTH, self.SEQ_LEN, self.DIM)

    def test_forward_with_no_len(self, policy_value_model, hidden):
        """Test forward_with_no_len for inference without sequence
        dimension."""
        # Create input without sequence length dimension
        obs_no_len = torch.randn(self.BATCH_SIZE, self.OBS_NUM_TOKENS, self.OBS_DIM)

        # Run forward pass
        policy_dist, value, next_hidden = policy_value_model.forward_with_no_len(
            obs_no_len, hidden
        )

        # Check output shapes
        sample_action = policy_dist.sample()
        assert sample_action.shape == (self.BATCH_SIZE, len(self.ACTION_CHOICES))
        assert value.shape == (self.BATCH_SIZE,)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.DIM,
        )

        # Check distribution properties
        log_prob = policy_dist.log_prob(sample_action)
        assert log_prob.shape == (self.BATCH_SIZE, len(self.ACTION_CHOICES))

    def test_forward_no_hidden(self, policy_value_model, observation):
        """Test forward pass without providing hidden state."""
        # Run forward pass without hidden
        policy_dist, value, next_hidden = policy_value_model(observation)

        # Check output types
        assert isinstance(policy_dist, Distribution)
        assert isinstance(value, torch.Tensor)
        assert isinstance(next_hidden, torch.Tensor)

        # Check output shapes
        sample_action = policy_dist.sample()
        assert sample_action.shape == (
            self.BATCH_SIZE,
            self.SEQ_LEN,
            len(self.ACTION_CHOICES),
        )
        assert value.shape == (self.BATCH_SIZE, self.SEQ_LEN)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM,
        )

    def test_forward_with_no_len_no_hidden(self, policy_value_model):
        """Test forward_with_no_len without providing hidden state."""
        # Create input without sequence length dimension
        obs_no_len = torch.randn(self.BATCH_SIZE, self.OBS_NUM_TOKENS, self.OBS_DIM)

        # Run forward pass without hidden
        policy_dist, value, next_hidden = policy_value_model.forward_with_no_len(
            obs_no_len
        )

        # Check output shapes
        sample_action = policy_dist.sample()
        assert sample_action.shape == (self.BATCH_SIZE, len(self.ACTION_CHOICES))
        assert value.shape == (self.BATCH_SIZE,)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.DIM,
        )


class TestStackedHiddenPiVLatent:
    # Test hyperparameters
    BATCH_SIZE = 2
    SEQ_LEN = 3
    DEPTH = 2
    DIM = 8
    OBS_DIM = 16
    OBS_DIM_HIDDEN = 12
    ACTION_CHOICES = [2, 3, 4]
    DIM_FF_HIDDEN = 16

    @pytest.fixture
    def core_model(self):
        return QLSTM(
            depth=self.DEPTH,
            dim=self.DIM,
            dim_ff_hidden=self.DIM_FF_HIDDEN,
            dropout=0.0,
        )

    @pytest.fixture
    def policy_value_model(self, core_model):
        return StackedHiddenPiVLatent(
            obs_dim=self.OBS_DIM,
            action_choices=self.ACTION_CHOICES,
            dim=self.DIM,
            core_model=core_model,
        )

    @pytest.fixture
    def observation(self):
        return torch.randn(self.BATCH_SIZE, self.SEQ_LEN, self.OBS_DIM)

    @pytest.fixture
    def hidden(self):
        return torch.randn(self.BATCH_SIZE, self.DEPTH, self.DIM)

    def test_forward(self, policy_value_model, observation, hidden):
        """Test forward pass of StackedHiddenPiV model."""
        # Run forward pass
        policy_dist, value, latent, next_hidden = policy_value_model(
            observation, hidden
        )

        # Check output types
        assert isinstance(policy_dist, Distribution)
        assert isinstance(value, torch.Tensor)
        assert isinstance(latent, torch.Tensor)
        assert isinstance(next_hidden, torch.Tensor)

        # Check output shapes
        sample_action = policy_dist.sample()
        assert sample_action.shape == (
            self.BATCH_SIZE,
            self.SEQ_LEN,
            len(self.ACTION_CHOICES),
        )
        assert value.shape == (self.BATCH_SIZE, self.SEQ_LEN)
        assert latent.shape == (self.BATCH_SIZE, self.SEQ_LEN, self.DIM)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM,
        )

        # Check distribution properties
        log_prob = policy_dist.log_prob(sample_action)
        assert log_prob.shape == (
            self.BATCH_SIZE,
            self.SEQ_LEN,
            len(self.ACTION_CHOICES),
        )

        entropy = policy_dist.entropy()
        assert entropy.shape == (
            self.BATCH_SIZE,
            self.SEQ_LEN,
            len(self.ACTION_CHOICES),
        )

    def test_single_batch(self, policy_value_model, observation, hidden):
        """Test with single batch size."""
        single_obs = observation[:1]
        single_hidden = hidden[:1]

        policy_dist, value, latent, next_hidden = policy_value_model(
            single_obs, single_hidden
        )

        sample_action = policy_dist.sample()
        assert sample_action.shape == (1, self.SEQ_LEN, len(self.ACTION_CHOICES))
        assert value.shape == (1, self.SEQ_LEN)
        assert latent.shape == (1, self.SEQ_LEN, self.DIM)
        assert next_hidden.shape == (1, self.DEPTH, self.SEQ_LEN, self.DIM)

    def test_forward_with_no_len(self, policy_value_model, hidden):
        """Test forward_with_no_len for inference without sequence
        dimension."""
        # Create input without sequence length dimension
        obs_no_len = torch.randn(self.BATCH_SIZE, self.OBS_DIM)

        # Run forward pass
        policy_dist, value, latent, next_hidden = (
            policy_value_model.forward_with_no_len(obs_no_len, hidden)
        )

        # Check output shapes
        sample_action = policy_dist.sample()
        assert sample_action.shape == (self.BATCH_SIZE, len(self.ACTION_CHOICES))
        assert value.shape == (self.BATCH_SIZE,)
        assert latent.shape == (self.BATCH_SIZE, self.DIM)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.DIM,
        )

        # Check distribution properties
        log_prob = policy_dist.log_prob(sample_action)
        assert log_prob.shape == (self.BATCH_SIZE, len(self.ACTION_CHOICES))

    def test_forward_no_hidden(self, policy_value_model, observation):
        """Test forward pass without providing hidden state."""
        # Run forward pass without hidden
        policy_dist, value, latent, next_hidden = policy_value_model(observation)

        # Check output types
        assert isinstance(policy_dist, Distribution)
        assert isinstance(value, torch.Tensor)
        assert isinstance(latent, torch.Tensor)
        assert isinstance(next_hidden, torch.Tensor)

        # Check output shapes
        sample_action = policy_dist.sample()
        assert sample_action.shape == (
            self.BATCH_SIZE,
            self.SEQ_LEN,
            len(self.ACTION_CHOICES),
        )
        assert value.shape == (self.BATCH_SIZE, self.SEQ_LEN)
        assert latent.shape == (self.BATCH_SIZE, self.SEQ_LEN, self.DIM)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM,
        )

    def test_forward_with_no_len_no_hidden(self, policy_value_model):
        """Test forward_with_no_len without providing hidden state."""
        # Create input without sequence length dimension
        obs_no_len = torch.randn(self.BATCH_SIZE, self.OBS_DIM)

        # Run forward pass without hidden
        policy_dist, value, latent, next_hidden = (
            policy_value_model.forward_with_no_len(obs_no_len)
        )

        # Check output shapes
        sample_action = policy_dist.sample()
        assert sample_action.shape == (self.BATCH_SIZE, len(self.ACTION_CHOICES))
        assert value.shape == (self.BATCH_SIZE,)
        assert latent.shape == (self.BATCH_SIZE, self.DIM)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.DIM,
        )


class TestStackedHiddenContinuousPiVLatent:
    # Test hyperparameters
    BATCH_SIZE = 2
    SEQ_LEN = 3
    DEPTH = 2
    DIM = 8
    OBS_DIM = 16
    OBS_DIM_HIDDEN = 12
    ACTION_DIM = 32
    DIM_FF_HIDDEN = 16

    @pytest.fixture
    def core_model(self):
        return QLSTM(
            depth=self.DEPTH,
            dim=self.DIM,
            dim_ff_hidden=self.DIM_FF_HIDDEN,
            dropout=0.0,
        )

    @pytest.fixture
    def policy_value_model(self, core_model):
        return StackedHiddenContinuousPiVLatent(
            obs_dim=self.OBS_DIM,
            action_dim=self.ACTION_DIM,
            dim=self.DIM,
            core_model=core_model,
        )

    @pytest.fixture
    def observation(self):
        return torch.randn(self.BATCH_SIZE, self.SEQ_LEN, self.OBS_DIM)

    @pytest.fixture
    def hidden(self):
        return torch.randn(self.BATCH_SIZE, self.DEPTH, self.DIM)

    def test_forward(self, policy_value_model, observation, hidden):
        """Test forward pass of StackedHiddenPiV model."""
        # Run forward pass
        policy_dist, value, latent, next_hidden = policy_value_model(
            observation, hidden
        )

        # Check output types
        assert isinstance(policy_dist, Distribution)
        assert isinstance(value, torch.Tensor)
        assert isinstance(latent, torch.Tensor)
        assert isinstance(next_hidden, torch.Tensor)

        # Check output shapes
        sample_action = policy_dist.sample()
        assert sample_action.shape == (
            self.BATCH_SIZE,
            self.SEQ_LEN,
            self.ACTION_DIM,
        )
        assert value.shape == (self.BATCH_SIZE, self.SEQ_LEN)
        assert latent.shape == (self.BATCH_SIZE, self.SEQ_LEN, self.DIM)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM,
        )

        # Check distribution properties
        log_prob = policy_dist.log_prob(sample_action)
        assert log_prob.shape == (
            self.BATCH_SIZE,
            self.SEQ_LEN,
            self.ACTION_DIM,
        )

        entropy = policy_dist.entropy()
        assert entropy.shape == (
            self.BATCH_SIZE,
            self.SEQ_LEN,
            self.ACTION_DIM,
        )

    def test_single_batch(self, policy_value_model, observation, hidden):
        """Test with single batch size."""
        single_obs = observation[:1]
        single_hidden = hidden[:1]

        policy_dist, value, latent, next_hidden = policy_value_model(
            single_obs, single_hidden
        )

        sample_action = policy_dist.sample()
        assert sample_action.shape == (1, self.SEQ_LEN, self.ACTION_DIM)
        assert value.shape == (1, self.SEQ_LEN)
        assert latent.shape == (1, self.SEQ_LEN, self.DIM)
        assert next_hidden.shape == (1, self.DEPTH, self.SEQ_LEN, self.DIM)

    def test_forward_with_no_len(self, policy_value_model, hidden):
        """Test forward_with_no_len for inference without sequence
        dimension."""
        # Create input without sequence length dimension
        obs_no_len = torch.randn(self.BATCH_SIZE, self.OBS_DIM)

        # Run forward pass
        policy_dist, value, latent, next_hidden = (
            policy_value_model.forward_with_no_len(obs_no_len, hidden)
        )

        # Check output shapes
        sample_action = policy_dist.sample()
        assert sample_action.shape == (self.BATCH_SIZE, self.ACTION_DIM)
        assert value.shape == (self.BATCH_SIZE,)
        assert latent.shape == (self.BATCH_SIZE, self.DIM)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.DIM,
        )

        # Check distribution properties
        log_prob = policy_dist.log_prob(sample_action)
        assert log_prob.shape == (self.BATCH_SIZE, self.ACTION_DIM)

    def test_forward_no_hidden(self, policy_value_model, observation):
        """Test forward pass without providing hidden state."""
        # Run forward pass without hidden
        policy_dist, value, latent, next_hidden = policy_value_model(observation)

        # Check output types
        assert isinstance(policy_dist, Distribution)
        assert isinstance(value, torch.Tensor)
        assert isinstance(latent, torch.Tensor)
        assert isinstance(next_hidden, torch.Tensor)

        # Check output shapes
        sample_action = policy_dist.sample()
        assert sample_action.shape == (
            self.BATCH_SIZE,
            self.SEQ_LEN,
            self.ACTION_DIM,
        )
        assert value.shape == (self.BATCH_SIZE, self.SEQ_LEN)
        assert latent.shape == (self.BATCH_SIZE, self.SEQ_LEN, self.DIM)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM,
        )

    def test_forward_with_no_len_no_hidden(self, policy_value_model):
        """Test forward_with_no_len without providing hidden state."""
        # Create input without sequence length dimension
        obs_no_len = torch.randn(self.BATCH_SIZE, self.OBS_DIM)

        # Run forward pass without hidden
        policy_dist, value, latent, next_hidden = (
            policy_value_model.forward_with_no_len(obs_no_len)
        )

        # Check output shapes
        sample_action = policy_dist.sample()
        assert sample_action.shape == (self.BATCH_SIZE, self.ACTION_DIM)
        assert value.shape == (self.BATCH_SIZE,)
        assert latent.shape == (self.BATCH_SIZE, self.DIM)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.DIM,
        )
