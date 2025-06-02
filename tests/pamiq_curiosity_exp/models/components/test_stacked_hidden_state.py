import pytest
import torch

from pamiq_curiosity_exp.models.components.qlstm import QLSTM

BATCH = 4
DEPTH = 8
DIM = 16
DIM_FF_HIDDEN = 32
LEN = 64
DROPOUT = 0.1


class TestStackedHiddenState:
    @pytest.fixture
    def qlstm(self):
        qlstm = QLSTM(DEPTH, DIM, DIM_FF_HIDDEN, DROPOUT)
        return qlstm

    def test_batch_len_with_hidden(self, qlstm):
        """Test with batch and length dimensions, providing hidden state."""
        x_shape = (BATCH, LEN, DIM)
        x = torch.randn(*x_shape)
        hidden_shape = (BATCH, DEPTH, DIM)
        hidden = torch.randn(*hidden_shape)

        x, hidden_out = qlstm(x, hidden)
        assert x.shape == x_shape
        assert hidden_out.shape == (BATCH, DEPTH, LEN, DIM)

    def test_batch_len_without_hidden(self, qlstm):
        """Test with batch and length dimensions, without hidden state."""
        x_shape = (BATCH, LEN, DIM)
        x = torch.randn(*x_shape)

        x, hidden_out = qlstm(x, None)
        assert x.shape == x_shape
        assert hidden_out.shape == (BATCH, DEPTH, LEN, DIM)

    def test_batch_len_default_hidden(self, qlstm):
        """Test with batch and length dimensions, using default hidden
        parameter."""
        x_shape = (BATCH, LEN, DIM)
        x = torch.randn(*x_shape)

        x, hidden_out = qlstm(x)  # No hidden parameter
        assert x.shape == x_shape
        assert hidden_out.shape == (BATCH, DEPTH, LEN, DIM)

    def test_no_batch_len_with_hidden(self, qlstm):
        """Test without batch dimension, with hidden state."""
        x_shape = (LEN, DIM)
        x = torch.randn(*x_shape)
        hidden_shape = (DEPTH, DIM)
        hidden = torch.randn(*hidden_shape)

        x, hidden_out = qlstm(x, hidden)
        assert x.shape == x_shape
        assert hidden_out.shape == (DEPTH, LEN, DIM)

    def test_no_batch_len_without_hidden(self, qlstm):
        """Test without batch dimension, without hidden state."""
        x_shape = (LEN, DIM)
        x = torch.randn(*x_shape)

        x, hidden_out = qlstm(x, None)
        assert x.shape == x_shape
        assert hidden_out.shape == (DEPTH, LEN, DIM)

    def test_many_batch_shape_with_hidden(self, qlstm):
        """Test with multiple batch dimensions, providing hidden state."""
        x_shape = (1, 2, 3, BATCH, LEN, DIM)
        x = torch.randn(*x_shape)
        hidden_shape = (1, 2, 3, BATCH, DEPTH, DIM)
        hidden = torch.randn(*hidden_shape)

        x, hidden_out = qlstm(x, hidden)
        assert x.shape == x_shape
        assert hidden_out.shape == (1, 2, 3, BATCH, DEPTH, LEN, DIM)

    def test_many_batch_shape_without_hidden(self, qlstm):
        """Test with multiple batch dimensions, without hidden state."""
        x_shape = (1, 2, 3, BATCH, LEN, DIM)
        x = torch.randn(*x_shape)

        x, hidden_out = qlstm(x, None)
        assert x.shape == x_shape
        assert hidden_out.shape == (1, 2, 3, BATCH, DEPTH, LEN, DIM)

    def test_batch_shape_mismatch(self, qlstm):
        """Test error when batch shapes don't match."""
        x = torch.randn(BATCH, LEN, DIM)
        hidden = torch.randn(BATCH + 1, DEPTH, DIM)  # Different batch size

        with pytest.raises(ValueError, match="Batch shape mismatch"):
            qlstm(x, hidden)

    def test_feature_dim_mismatch(self, qlstm):
        """Test error when feature dimensions don't match."""
        x = torch.randn(BATCH, LEN, DIM)
        hidden = torch.randn(BATCH, DEPTH, DIM + 1)  # Different feature dim

        with pytest.raises(ValueError, match="Feature dim mismatch"):
            qlstm(x, hidden)

    def test_multi_batch_shape_mismatch(self, qlstm):
        """Test error when multi-dimensional batch shapes don't match."""
        x = torch.randn(2, 3, BATCH, LEN, DIM)
        hidden = torch.randn(2, 4, BATCH, DEPTH, DIM)  # Different batch shape

        with pytest.raises(ValueError, match="Batch shape mismatch"):
            qlstm(x, hidden)

    def test_forward_with_no_len(self, qlstm):
        x = torch.randn(DIM)
        hidden = torch.randn(DEPTH, DIM)

        x, hidden_out = qlstm.forward_with_no_len(x, hidden)
        assert x.shape == (DIM,)
        assert hidden_out.shape == (DEPTH, DIM)

        x = torch.randn(1, 2, 3, DIM)
        hidden = torch.randn(1, 2, 3, DEPTH, DIM)

        x, hidden_out = qlstm.forward_with_no_len(x, hidden)
        assert x.shape == (1, 2, 3, DIM)
        assert hidden_out.shape == (1, 2, 3, DEPTH, DIM)
